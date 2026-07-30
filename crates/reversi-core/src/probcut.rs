//! ProbCut forward pruning implementation.
//!
//! Uses a statistical model to predict whether a shallow search result can
//! substitute for a deeper search, allowing subtrees to be pruned with
//! controlled error probability. Pre-computed lookup tables for mean and sigma
//! values are initialized once via [`init`] and accessed through the `get_*`
//! functions.

use std::sync::OnceLock;

use crate::types::Depth;
use crate::types::ScaledScore;

/// Selectivity level for search pruning (ProbCut confidence levels).
///
/// Lower levels are more aggressive (prune more), higher levels are more conservative.
/// `None` disables ProbCut entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Selectivity {
    /// Midgame-only level: 63% confidence (t=0.9). Never part of the
    /// endgame selectivity ladder.
    Mid = 0,
    /// Most aggressive endgame-ladder level: 73% confidence (t=1.1)
    #[default]
    Level1 = 1,
    /// 95% confidence (t=2.0)
    Level2 = 2,
    /// Most conservative: 99% confidence (t=3.3)
    Level3 = 3,
    /// ProbCut disabled.
    None = 4,
}

impl Selectivity {
    /// Returns the statistical confidence multiplier (t-value).
    #[inline]
    pub fn t_value(self) -> f64 {
        match self {
            Selectivity::Mid => 0.9,
            Selectivity::Level1 => 1.1,
            Selectivity::Level2 => 2.0,
            Selectivity::Level3 => 3.3,
            Selectivity::None => 999.0,
        }
    }

    /// Returns the probability percentage for this level.
    #[inline]
    pub fn probability(self) -> i32 {
        match self {
            Selectivity::Mid => 63,
            Selectivity::Level1 => 73,
            Selectivity::Level2 => 95,
            Selectivity::Level3 => 99,
            Selectivity::None => 100,
        }
    }

    /// Converts to [`u8`].
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Creates a [`Selectivity`] from a [`u8`] value, clamping to valid range.
    ///
    /// Values > 4 are clamped to [`Selectivity::None`].
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Selectivity::Mid,
            1 => Selectivity::Level1,
            2 => Selectivity::Level2,
            3 => Selectivity::Level3,
            _ => Selectivity::None,
        }
    }

    /// Returns `true` if ProbCut is enabled at this selectivity level.
    #[inline]
    pub fn is_enabled(self) -> bool {
        self != Selectivity::None
    }
}

/// Fraction of the fitted odd-deep parity offset credited in the runtime mean.
///
/// The tempo optimism of odd-depth searches must not be credited in full: a
/// ProbCut fail-high freezes that optimism into the tree as a bound instead
/// of letting deeper iterations correct it.
const PARITY_CREDIT: f64 = 0.5;

/// Statistical parameters for the midgame ProbCut prediction model.
///
/// - `mean = mean_intercept + mean_coef_shallow * shallow + mean_coef_deep * deep
///   + PARITY_CREDIT * mean_coef_parity * (deep & 1)`
/// - `sigma = exp(std_intercept + std_coef_shallow * shallow + std_coef_deep * sqrt(deep))`
struct ProbcutMidgameParams {
    mean_intercept: f64,
    mean_coef_shallow: f64,
    mean_coef_deep: f64,
    mean_coef_parity: f64,
    std_intercept: f64,
    std_coef_shallow: f64,
    std_coef_deep: f64,
}

impl ProbcutMidgameParams {
    fn mean(&self, shallow: f64, deep: f64) -> f64 {
        let deep_parity = ((deep as usize) & 1) as f64;
        self.mean_intercept
            + self.mean_coef_shallow * shallow
            + self.mean_coef_deep * deep
            + PARITY_CREDIT * self.mean_coef_parity * deep_parity
    }

    fn sigma(&self, shallow: f64, deep: f64) -> f64 {
        (self.std_intercept + self.std_coef_shallow * shallow + self.std_coef_deep * deep.sqrt())
            .exp()
    }
}

/// Statistical parameters for ProbCut prediction models.
///
/// - `mean = mean_intercept + mean_coef_shallow * shallow + mean_coef_deep * deep`
/// - `sigma = exp(std_intercept + std_coef_shallow * shallow + std_coef_deep * deep)`
struct ProbcutParams {
    mean_intercept: f64,
    mean_coef_shallow: f64,
    mean_coef_deep: f64,
    std_intercept: f64,
    std_coef_shallow: f64,
    std_coef_deep: f64,
}

impl ProbcutParams {
    fn mean(&self, shallow: f64, deep: f64) -> f64 {
        self.mean_intercept + self.mean_coef_shallow * shallow + self.mean_coef_deep * deep
    }

    fn sigma(&self, shallow: f64, deep: f64) -> f64 {
        (self.std_intercept + self.std_coef_shallow * shallow + self.std_coef_deep * deep).exp()
    }
}

const NUM_PLY: usize = 60;
const NUM_DEPTH: usize = 60;

type MidTable = [[[f64; NUM_DEPTH]; NUM_DEPTH]; NUM_PLY];
type EndTable = [[f64; NUM_DEPTH]; NUM_DEPTH];

const SCORE_SCALE_F64: f64 = ScaledScore::SCALE as f64;

static MEAN_TABLE: OnceLock<Box<MidTable>> = OnceLock::new();
static SIGMA_TABLE: OnceLock<Box<MidTable>> = OnceLock::new();
static MEAN_TABLE_END: OnceLock<EndTable> = OnceLock::new();
static SIGMA_TABLE_END: OnceLock<EndTable> = OnceLock::new();

/// Builds a 3D `[ply][shallow][deep]` table from midgame ProbCut parameters.
///
/// Only populates entries where `shallow <= deep` (callers always satisfy this).
fn build_mid_table(f: impl Fn(&ProbcutMidgameParams, f64, f64) -> f64) -> Box<MidTable> {
    let mut tbl: Box<MidTable> = vec![[[0.0f64; NUM_DEPTH]; NUM_DEPTH]; NUM_PLY]
        .into_boxed_slice()
        .try_into()
        .unwrap();
    for ply in 0..NUM_PLY {
        let params = &PROBCUT_PARAMS[ply];
        for shallow in 0..NUM_DEPTH {
            for deep in shallow..NUM_DEPTH {
                tbl[ply][shallow][deep] = f(params, shallow as f64, deep as f64) * SCORE_SCALE_F64;
            }
        }
    }
    tbl
}

/// Builds a 2D `[shallow][deep]` table from endgame ProbCut parameters.
///
/// Only populates entries where `shallow <= deep` (callers always satisfy this).
fn build_end_table(f: impl Fn(&ProbcutParams, f64, f64) -> f64) -> EndTable {
    let mut tbl = [[0.0f64; NUM_DEPTH]; NUM_DEPTH];
    #[allow(clippy::needless_range_loop)]
    for shallow in 0..NUM_DEPTH {
        for deep in shallow..NUM_DEPTH {
            tbl[shallow][deep] =
                f(&PROBCUT_ENDGAME_PARAMS, shallow as f64, deep as f64) * SCORE_SCALE_F64;
        }
    }
    tbl
}

/// Initializes the ProbCut lookup tables.
///
/// Must be called before any `get_*` functions. Called automatically by
/// [`Search::new`](crate::search::Search::new).
pub fn init() {
    MEAN_TABLE.get_or_init(|| build_mid_table(ProbcutMidgameParams::mean));
    SIGMA_TABLE.get_or_init(|| build_mid_table(ProbcutMidgameParams::sigma));
    MEAN_TABLE_END.get_or_init(|| build_end_table(ProbcutParams::mean));
    SIGMA_TABLE_END.get_or_init(|| build_end_table(ProbcutParams::sigma));
}

#[inline(always)]
fn lookup_mid(tbl: &MidTable, ply: usize, shallow: Depth, deep: Depth) -> f64 {
    debug_assert!(ply < NUM_PLY);
    debug_assert!((shallow as usize) <= (deep as usize));
    debug_assert!((deep as usize) < NUM_DEPTH);
    tbl[ply][shallow as usize][deep as usize]
}

#[inline(always)]
fn lookup_end(tbl: &EndTable, shallow: Depth, deep: Depth) -> f64 {
    debug_assert!((shallow as usize) <= (deep as usize));
    debug_assert!((deep as usize) < NUM_DEPTH);
    tbl[shallow as usize][deep as usize]
}

/// Returns the pre-computed mean value for midgame positions.
///
/// # Safety
///
/// [`init`] must have been called before this function.
#[inline(always)]
pub fn get_mean(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    // SAFETY: `init()` is called once at startup before any search begins,
    // guaranteeing the OnceLock is initialized.
    let tbl = unsafe { MEAN_TABLE.get().unwrap_unchecked() };
    lookup_mid(tbl, ply, shallow, deep)
}

/// Returns the pre-computed sigma value for midgame positions.
///
/// # Safety
///
/// [`init`] must have been called before this function.
#[inline(always)]
pub fn get_sigma(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    // SAFETY: `init()` is called once at startup before any search begins,
    // guaranteeing the OnceLock is initialized.
    let tbl = unsafe { SIGMA_TABLE.get().unwrap_unchecked() };
    lookup_mid(tbl, ply, shallow, deep)
}

/// Returns the pre-computed mean value for endgame positions.
///
/// # Safety
///
/// [`init`] must have been called before this function.
#[inline(always)]
pub fn get_mean_end(shallow: Depth, deep: Depth) -> f64 {
    // SAFETY: `init()` is called once at startup before any search begins,
    // guaranteeing the OnceLock is initialized.
    let tbl = unsafe { MEAN_TABLE_END.get().unwrap_unchecked() };
    lookup_end(tbl, shallow, deep)
}

/// Returns the pre-computed sigma value for endgame positions.
///
/// # Safety
///
/// [`init`] must have been called before this function.
#[inline(always)]
pub fn get_sigma_end(shallow: Depth, deep: Depth) -> f64 {
    // SAFETY: `init()` is called once at startup before any search begins,
    // guaranteeing the OnceLock is initialized.
    let tbl = unsafe { SIGMA_TABLE_END.get().unwrap_unchecked() };
    lookup_end(tbl, shallow, deep)
}

/// Computes the ProbCut beta threshold for verification search.
#[inline]
pub fn compute_probcut_beta(beta: ScaledScore, t: f64, mean: f64, sigma: f64) -> ScaledScore {
    ScaledScore::from_raw((beta.value() as f64 + t * sigma - mean).ceil() as i32)
}

/// Computes the evaluation threshold for ProbCut pre-screening.
#[inline(always)]
pub fn compute_eval_beta(
    beta: ScaledScore,
    t: f64,
    mean: f64,
    sigma: f64,
    mean0: f64,
    sigma0: f64,
    cut_node: bool,
) -> ScaledScore {
    let eval_mean = 0.5 * mean0 + mean;
    let eval_sigma = t * 0.5 * sigma0 + sigma;
    let all_node_margin = if cut_node { 0.0 } else { sigma0 * 1.5 };
    ScaledScore::from_raw(
        (beta.value() as f64 - eval_sigma - eval_mean + all_node_margin).floor() as i32,
    )
}

/// Statistical parameters for endgame ProbCut.
#[rustfmt::skip]
const PROBCUT_ENDGAME_PARAMS: ProbcutParams = ProbcutParams {
    mean_intercept: -0.2142968574,
    mean_coef_shallow: 0.0258202257,
    mean_coef_deep: 0.0074065736,
    std_intercept: 0.8452019175,
    std_coef_shallow: -0.0616277893,
    std_coef_deep: 0.0333471614,
};

/// Statistical parameters for midgame ProbCut indexed by ply.
#[rustfmt::skip]
const PROBCUT_PARAMS: [ProbcutMidgameParams; 60] = [
    ProbcutMidgameParams {
        mean_intercept: 0.0000000000,
        mean_coef_shallow: 0.0000000000,
        mean_coef_deep: 0.0000000000,
        mean_coef_parity: 0.0000000000,
        std_intercept: -18.4206807440,
        std_coef_shallow: 0.0000000000,
        std_coef_deep: 0.0000000000,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.1936797957,
        mean_coef_shallow: -0.0128700815,
        mean_coef_deep: 0.0064389782,
        mean_coef_parity: 0.1087248614,
        std_intercept: -1.6691406530,
        std_coef_shallow: 0.0626250178,
        std_coef_deep: 0.3008221322,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.3106663105,
        mean_coef_shallow: -0.0662605933,
        mean_coef_deep: -0.0000612726,
        mean_coef_parity: 0.4748868606,
        std_intercept: -0.8983856957,
        std_coef_shallow: 0.0584934606,
        std_coef_deep: 0.1389003619,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.3184243402,
        mean_coef_shallow: -0.0476883639,
        mean_coef_deep: -0.0062796737,
        mean_coef_parity: 0.6681891345,
        std_intercept: -0.3673164510,
        std_coef_shallow: 0.0428799731,
        std_coef_deep: 0.0508810580,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.2774474991,
        mean_coef_shallow: -0.0259283722,
        mean_coef_deep: -0.0059836339,
        mean_coef_parity: 0.7246074209,
        std_intercept: -0.3766454164,
        std_coef_shallow: 0.0331495910,
        std_coef_deep: 0.0678413121,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.0137697552,
        mean_coef_shallow: -0.0876504906,
        mean_coef_deep: 0.0573330430,
        mean_coef_parity: 0.8152174274,
        std_intercept: 0.0461232941,
        std_coef_shallow: -0.0032705827,
        std_coef_deep: 0.0302935563,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.5229131630,
        mean_coef_shallow: -0.0361444826,
        mean_coef_deep: -0.0283738430,
        mean_coef_parity: 0.8275443091,
        std_intercept: 0.3645428749,
        std_coef_shallow: -0.0013236059,
        std_coef_deep: -0.0353121629,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.3988327890,
        mean_coef_shallow: -0.0492463871,
        mean_coef_deep: 0.1022265240,
        mean_coef_parity: 0.8050722023,
        std_intercept: 0.4051367987,
        std_coef_shallow: -0.0186157442,
        std_coef_deep: -0.0306527814,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.9669359560,
        mean_coef_shallow: -0.1473040137,
        mean_coef_deep: -0.0351437699,
        mean_coef_parity: 0.6779563323,
        std_intercept: 0.4105785981,
        std_coef_shallow: -0.0270014912,
        std_coef_deep: 0.0023529004,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.3057204603,
        mean_coef_shallow: -0.0311021220,
        mean_coef_deep: 0.0168195488,
        mean_coef_parity: 0.7947153837,
        std_intercept: 0.4690848176,
        std_coef_shallow: -0.0469339268,
        std_coef_deep: 0.0491759097,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.4355199675,
        mean_coef_shallow: -0.1526847671,
        mean_coef_deep: -0.0011204783,
        mean_coef_parity: 0.9175539648,
        std_intercept: 0.4652657487,
        std_coef_shallow: -0.0865455177,
        std_coef_deep: 0.0972096105,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.6777874201,
        mean_coef_shallow: -0.0835461306,
        mean_coef_deep: -0.0068348530,
        mean_coef_parity: 1.0174751521,
        std_intercept: 0.5481433103,
        std_coef_shallow: -0.0825149253,
        std_coef_deep: 0.1012381543,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.2058046140,
        mean_coef_shallow: -0.0674682759,
        mean_coef_deep: 0.0411438121,
        mean_coef_parity: 1.1282066084,
        std_intercept: 0.6403405450,
        std_coef_shallow: -0.0950655216,
        std_coef_deep: 0.0780241721,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.6144319442,
        mean_coef_shallow: -0.1067809206,
        mean_coef_deep: -0.0297322362,
        mean_coef_parity: 1.2240571605,
        std_intercept: 0.6825624602,
        std_coef_shallow: -0.0987512025,
        std_coef_deep: 0.0864169594,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.9539363744,
        mean_coef_shallow: 0.1019725858,
        mean_coef_deep: 0.0569209093,
        mean_coef_parity: 1.4014254142,
        std_intercept: 0.6811351994,
        std_coef_shallow: -0.1012447307,
        std_coef_deep: 0.0950630230,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.1748714425,
        mean_coef_shallow: -0.1088436277,
        mean_coef_deep: -0.0205649534,
        mean_coef_parity: 1.3598659915,
        std_intercept: 0.6900179195,
        std_coef_shallow: -0.0885430844,
        std_coef_deep: 0.0875844510,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.5087798000,
        mean_coef_shallow: 0.0448440259,
        mean_coef_deep: 0.0226095289,
        mean_coef_parity: 1.4464094108,
        std_intercept: 0.7290677450,
        std_coef_shallow: -0.1043537171,
        std_coef_deep: 0.0977518753,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.4060011874,
        mean_coef_shallow: -0.0649346614,
        mean_coef_deep: 0.0281650777,
        mean_coef_parity: 1.4423657322,
        std_intercept: 0.7601286326,
        std_coef_shallow: -0.1046820812,
        std_coef_deep: 0.0926496923,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.2079026127,
        mean_coef_shallow: -0.0133739148,
        mean_coef_deep: -0.0353549502,
        mean_coef_parity: 1.4896185721,
        std_intercept: 0.7555119208,
        std_coef_shallow: -0.1039467401,
        std_coef_deep: 0.1001580252,
    },
    ProbcutMidgameParams {
        mean_intercept: -1.3715141282,
        mean_coef_shallow: 0.1689529726,
        mean_coef_deep: 0.0508911601,
        mean_coef_parity: 1.6220923703,
        std_intercept: 0.7023338232,
        std_coef_shallow: -0.0974451171,
        std_coef_deep: 0.1176588191,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.2149275005,
        mean_coef_shallow: 0.0556122919,
        mean_coef_deep: -0.0326589028,
        mean_coef_parity: 1.6008709644,
        std_intercept: 0.6400292370,
        std_coef_shallow: -0.0952406155,
        std_coef_deep: 0.1359054468,
    },
    ProbcutMidgameParams {
        mean_intercept: -1.0164954950,
        mean_coef_shallow: 0.2776032488,
        mean_coef_deep: -0.0126061013,
        mean_coef_parity: 1.7571997436,
        std_intercept: 0.6244608108,
        std_coef_shallow: -0.0939289222,
        std_coef_deep: 0.1393003418,
    },
    ProbcutMidgameParams {
        mean_intercept: -1.0637976027,
        mean_coef_shallow: 0.1317233181,
        mean_coef_deep: 0.0201841946,
        mean_coef_parity: 1.8501734322,
        std_intercept: 0.6050758569,
        std_coef_shallow: -0.0943613257,
        std_coef_deep: 0.1491455338,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.3057062414,
        mean_coef_shallow: 0.0534528202,
        mean_coef_deep: 0.0086055966,
        mean_coef_parity: 1.7912898168,
        std_intercept: 0.5722186921,
        std_coef_shallow: -0.0912900444,
        std_coef_deep: 0.1621919322,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.6434071157,
        mean_coef_shallow: 0.0762401560,
        mean_coef_deep: 0.0102931746,
        mean_coef_parity: 1.7908639631,
        std_intercept: 0.5818584689,
        std_coef_shallow: -0.0898172411,
        std_coef_deep: 0.1662842139,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.1131433703,
        mean_coef_shallow: -0.0445341135,
        mean_coef_deep: 0.0095509488,
        mean_coef_parity: 1.7907007233,
        std_intercept: 0.5876017926,
        std_coef_shallow: -0.0925362724,
        std_coef_deep: 0.1740004066,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.6128262201,
        mean_coef_shallow: 0.1220994790,
        mean_coef_deep: 0.0086951988,
        mean_coef_parity: 1.7915263134,
        std_intercept: 0.6081210584,
        std_coef_shallow: -0.0946312317,
        std_coef_deep: 0.1804973981,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.1952629423,
        mean_coef_shallow: -0.0033225043,
        mean_coef_deep: 0.0002443148,
        mean_coef_parity: 1.8051386505,
        std_intercept: 0.6103962705,
        std_coef_shallow: -0.0927619427,
        std_coef_deep: 0.1805550836,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.5550654630,
        mean_coef_shallow: 0.0964746873,
        mean_coef_deep: 0.0169977412,
        mean_coef_parity: 1.7678516775,
        std_intercept: 0.6601009270,
        std_coef_shallow: -0.0899296722,
        std_coef_deep: 0.1623524222,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.6791846915,
        mean_coef_shallow: 0.1872519641,
        mean_coef_deep: 0.0081090883,
        mean_coef_parity: 1.6911693746,
        std_intercept: 0.6536252436,
        std_coef_shallow: -0.0920003921,
        std_coef_deep: 0.1730219193,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.2217794863,
        mean_coef_shallow: 0.0805339494,
        mean_coef_deep: 0.0054637105,
        mean_coef_parity: 1.6188074056,
        std_intercept: 0.6815035134,
        std_coef_shallow: -0.0880767033,
        std_coef_deep: 0.1640600558,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.4485801848,
        mean_coef_shallow: 0.1349894164,
        mean_coef_deep: 0.0088701282,
        mean_coef_parity: 1.5260108562,
        std_intercept: 0.7141214268,
        std_coef_shallow: -0.0935936388,
        std_coef_deep: 0.1649149721,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.2134516025,
        mean_coef_shallow: 0.0331412392,
        mean_coef_deep: -0.0259525877,
        mean_coef_parity: 1.5423126600,
        std_intercept: 0.7672093321,
        std_coef_shallow: -0.0917404988,
        std_coef_deep: 0.1467673243,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.0043811255,
        mean_coef_shallow: -0.0116545400,
        mean_coef_deep: 0.0195162127,
        mean_coef_parity: 1.5121838421,
        std_intercept: 0.7693075071,
        std_coef_shallow: -0.0891705355,
        std_coef_deep: 0.1511074975,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.1195833718,
        mean_coef_shallow: 0.0536055013,
        mean_coef_deep: -0.0389940957,
        mean_coef_parity: 1.5507972537,
        std_intercept: 0.7670349967,
        std_coef_shallow: -0.0888419315,
        std_coef_deep: 0.1607566032,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.0659650713,
        mean_coef_shallow: -0.0582863437,
        mean_coef_deep: 0.0406371213,
        mean_coef_parity: 1.6048159399,
        std_intercept: 0.7933957509,
        std_coef_shallow: -0.0884918404,
        std_coef_deep: 0.1533379495,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.0817922507,
        mean_coef_shallow: 0.0027673306,
        mean_coef_deep: -0.0064555770,
        mean_coef_parity: 1.5356468468,
        std_intercept: 0.7955398207,
        std_coef_shallow: -0.0879285302,
        std_coef_deep: 0.1606616912,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.3244800242,
        mean_coef_shallow: 0.1090803081,
        mean_coef_deep: -0.0202321036,
        mean_coef_parity: 1.4800483182,
        std_intercept: 0.8395992021,
        std_coef_shallow: -0.0891970609,
        std_coef_deep: 0.1532182705,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.3046200927,
        mean_coef_shallow: 0.0125569618,
        mean_coef_deep: 0.0133430672,
        mean_coef_parity: 1.4108717133,
        std_intercept: 0.9259499055,
        std_coef_shallow: -0.0910363237,
        std_coef_deep: 0.1331863390,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.1746003840,
        mean_coef_shallow: 0.0536711381,
        mean_coef_deep: -0.0549349222,
        mean_coef_parity: 1.3415846679,
        std_intercept: 0.9214471855,
        std_coef_shallow: -0.0892617055,
        std_coef_deep: 0.1360464237,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.4845672774,
        mean_coef_shallow: 0.0517536675,
        mean_coef_deep: 0.0394864705,
        mean_coef_parity: 1.2370208897,
        std_intercept: 0.8881891864,
        std_coef_shallow: -0.0843915796,
        std_coef_deep: 0.1447577190,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.4688838354,
        mean_coef_shallow: 0.0146361677,
        mean_coef_deep: -0.0645637771,
        mean_coef_parity: 1.0676975869,
        std_intercept: 0.8920672782,
        std_coef_shallow: -0.0808970774,
        std_coef_deep: 0.1453686955,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.3001304193,
        mean_coef_shallow: 0.1365940785,
        mean_coef_deep: 0.0199100810,
        mean_coef_parity: 0.9631317670,
        std_intercept: 0.8563974617,
        std_coef_shallow: -0.0765058374,
        std_coef_deep: 0.1608477776,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.3651436102,
        mean_coef_shallow: 0.0782657321,
        mean_coef_deep: -0.0710066847,
        mean_coef_parity: 0.8507917036,
        std_intercept: 0.8713186457,
        std_coef_shallow: -0.0737533181,
        std_coef_deep: 0.1595633712,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.1537456041,
        mean_coef_shallow: 0.0776460731,
        mean_coef_deep: -0.0055627135,
        mean_coef_parity: 0.7629420410,
        std_intercept: 0.8556698259,
        std_coef_shallow: -0.0698435029,
        std_coef_deep: 0.1727535199,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.3302717781,
        mean_coef_shallow: 0.0455512719,
        mean_coef_deep: -0.0536750014,
        mean_coef_parity: 0.6243306819,
        std_intercept: 0.8204510401,
        std_coef_shallow: -0.0711098953,
        std_coef_deep: 0.1932180930,
    },
    ProbcutMidgameParams {
        mean_intercept: 1.0550879914,
        mean_coef_shallow: -0.0716476672,
        mean_coef_deep: -0.0473714211,
        mean_coef_parity: 0.5501106166,
        std_intercept: 0.8516299018,
        std_coef_shallow: -0.0802243379,
        std_coef_deep: 0.1982170463,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.2582046168,
        mean_coef_shallow: 0.0265737189,
        mean_coef_deep: -0.0443958307,
        mean_coef_parity: 0.5132465801,
        std_intercept: 0.8784556011,
        std_coef_shallow: -0.0871228768,
        std_coef_deep: 0.1984768179,
    },
    ProbcutMidgameParams {
        mean_intercept: 1.0873302464,
        mean_coef_shallow: -0.1290539805,
        mean_coef_deep: -0.0648334861,
        mean_coef_parity: 0.4743311346,
        std_intercept: 0.8743228513,
        std_coef_shallow: -0.0889566560,
        std_coef_deep: 0.2003810199,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.2176347322,
        mean_coef_shallow: -0.0106458275,
        mean_coef_deep: -0.0353077982,
        mean_coef_parity: 0.4415105361,
        std_intercept: 0.8555182092,
        std_coef_shallow: -0.0980801476,
        std_coef_deep: 0.2188487750,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.6344202142,
        mean_coef_shallow: -0.0213964596,
        mean_coef_deep: -0.0946385218,
        mean_coef_parity: 0.4203822328,
        std_intercept: 0.8583781709,
        std_coef_shallow: -0.1034312550,
        std_coef_deep: 0.2179956164,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.1236553168,
        mean_coef_shallow: 0.0443181373,
        mean_coef_deep: -0.0096640027,
        mean_coef_parity: 0.4014733685,
        std_intercept: 0.7961268352,
        std_coef_shallow: -0.1098205740,
        std_coef_deep: 0.2441625587,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.3596524079,
        mean_coef_shallow: -0.0184847222,
        mean_coef_deep: -0.0994279068,
        mean_coef_parity: 0.4119018081,
        std_intercept: 0.7272997204,
        std_coef_shallow: -0.1063104713,
        std_coef_deep: 0.2636415459,
    },
    ProbcutMidgameParams {
        mean_intercept: -0.3341883857,
        mean_coef_shallow: -0.1337773432,
        mean_coef_deep: 0.0507776731,
        mean_coef_parity: 0.4745841029,
        std_intercept: 0.7376400487,
        std_coef_shallow: -0.1224061310,
        std_coef_deep: 0.2666076413,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.6250169712,
        mean_coef_shallow: 0.0363997349,
        mean_coef_deep: -0.2417428479,
        mean_coef_parity: 0.5019264383,
        std_intercept: 0.7134237240,
        std_coef_shallow: -0.1289164632,
        std_coef_deep: 0.2646755919,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.6250169712,
        mean_coef_shallow: 0.0363997349,
        mean_coef_deep: -0.2417428479,
        mean_coef_parity: 0.5019264383,
        std_intercept: 0.7134237240,
        std_coef_shallow: -0.1289164632,
        std_coef_deep: 0.2646755919,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.6250169712,
        mean_coef_shallow: 0.0363997349,
        mean_coef_deep: -0.2417428479,
        mean_coef_parity: 0.5019264383,
        std_intercept: 0.7134237240,
        std_coef_shallow: -0.1289164632,
        std_coef_deep: 0.2646755919,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.6250169712,
        mean_coef_shallow: 0.0363997349,
        mean_coef_deep: -0.2417428479,
        mean_coef_parity: 0.5019264383,
        std_intercept: 0.7134237240,
        std_coef_shallow: -0.1289164632,
        std_coef_deep: 0.2646755919,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.6250169712,
        mean_coef_shallow: 0.0363997349,
        mean_coef_deep: -0.2417428479,
        mean_coef_parity: 0.5019264383,
        std_intercept: 0.7134237240,
        std_coef_shallow: -0.1289164632,
        std_coef_deep: 0.2646755919,
    },
    ProbcutMidgameParams {
        mean_intercept: 0.0000000000,
        mean_coef_shallow: 0.0000000000,
        mean_coef_deep: 0.0000000000,
        mean_coef_parity: 0.0000000000,
        std_intercept: -18.4206807440,
        std_coef_shallow: 0.0000000000,
        std_coef_deep: 0.0000000000,
    },
];

const _: () = assert!(PROBCUT_PARAMS.len() == NUM_PLY);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midgame_model_credits_half_the_deep_parity_and_uses_sqrt_for_sigma() {
        let params = ProbcutMidgameParams {
            mean_intercept: 0.25,
            mean_coef_shallow: -0.5,
            mean_coef_deep: 0.125,
            mean_coef_parity: 1.25,
            std_intercept: 0.75,
            std_coef_shallow: -0.25,
            std_coef_deep: 0.5,
        };

        // Even deep: no parity contribution.
        assert_eq!(params.mean(2.0, 8.0), 0.25);
        // Odd deep: the fitted parity offset enters at PARITY_CREDIT (0.5),
        // so 0.25 - 1.0 + 1.125 + 0.5 * 1.25 = 1.0.
        assert_eq!(params.mean(2.0, 9.0), 1.0);
        assert_eq!(params.sigma(2.0, 9.0), (0.75_f64 - 0.5 + 0.5 * 3.0).exp());
    }

    #[test]
    fn endgame_model_keeps_linear_deep_features() {
        let params = ProbcutParams {
            mean_intercept: 0.25,
            mean_coef_shallow: -0.5,
            mean_coef_deep: 0.125,
            std_intercept: 0.75,
            std_coef_shallow: -0.25,
            std_coef_deep: 0.5,
        };

        assert_eq!(params.mean(2.0, 9.0), 0.375);
        assert_eq!(params.sigma(2.0, 9.0), (0.75_f64 - 0.5 + 0.5 * 9.0).exp());
    }

    #[test]
    fn from_u8_maps_to_the_five_supported_selectivities() {
        assert_eq!(Selectivity::from_u8(0), Selectivity::Mid);
        assert_eq!(Selectivity::from_u8(1), Selectivity::Level1);
        assert_eq!(Selectivity::from_u8(2), Selectivity::Level2);
        assert_eq!(Selectivity::from_u8(3), Selectivity::Level3);
        assert_eq!(Selectivity::from_u8(4), Selectivity::None);
        assert_eq!(Selectivity::from_u8(5), Selectivity::None);
    }

    #[test]
    fn selectivity_orders_from_aggressive_to_disabled() {
        // The derived ordering is load-bearing for TT cutoff/replacement decisions.
        assert!(Selectivity::Mid < Selectivity::Level1);
        assert!(Selectivity::Level1 < Selectivity::Level2);
        assert!(Selectivity::Level2 < Selectivity::Level3);
        assert!(Selectivity::Level3 < Selectivity::None);
    }

    #[test]
    fn mid_selectivity_reports_confidence_for_its_t_value() {
        assert_eq!(Selectivity::Mid.t_value(), 0.9);
        assert_eq!(Selectivity::Mid.probability(), 63);
    }

    #[test]
    fn compute_probcut_beta_applies_t_sigma_minus_mean_and_rounds_up() {
        // raw value 2560; 2560 + 2.0*1.0 - 0.5 = 2561.5 -> ceil -> 2562
        assert_eq!(
            compute_probcut_beta(ScaledScore::from_disc_diff(10), 2.0, 0.5, 1.0),
            ScaledScore::from_raw(2562)
        );
        // 0 + 1.0*0.4 - 0.0 = 0.4 -> ceil -> 1
        assert_eq!(
            compute_probcut_beta(ScaledScore::ZERO, 1.0, 0.0, 0.4),
            ScaledScore::from_raw(1)
        );
    }

    #[test]
    fn compute_eval_beta_blends_models_and_gates_the_all_node_margin() {
        // Named so the two calls below obviously differ only in `cut_node`, and a
        // future signature reorder can't silently shuffle these positional args.
        let beta = ScaledScore::from_disc_diff(20); // raw value 5120
        let (t, mean, sigma, mean0, sigma0) = (2.0, 1.0, 2.0, 4.0, 3.0);
        // eval_mean = 0.5*mean0 + mean = 3; eval_sigma = t*0.5*sigma0 + sigma = 5

        // cut node: margin 0 -> floor(5120 - 5 - 3 + 0) = 5112
        assert_eq!(
            compute_eval_beta(beta, t, mean, sigma, mean0, sigma0, true),
            ScaledScore::from_raw(5112)
        );
        // all node: margin sigma0*1.5 = 4.5 -> floor(5120 - 5 - 3 + 4.5) = 5116
        assert_eq!(
            compute_eval_beta(beta, t, mean, sigma, mean0, sigma0, false),
            ScaledScore::from_raw(5116)
        );
    }

    #[test]
    fn init_builds_tables_matching_the_parameter_formulas() {
        init();

        let approx = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(1.0);

        // Midgame tables, including the shallow == deep diagonal.
        for (ply, shallow, deep) in [(5usize, 3u32, 10u32), (5, 7, 7), (40, 0, 12)] {
            let params = &PROBCUT_PARAMS[ply];
            assert!(approx(
                get_mean(ply, shallow, deep),
                params.mean(shallow as f64, deep as f64) * SCORE_SCALE_F64
            ));
            assert!(approx(
                get_sigma(ply, shallow, deep),
                params.sigma(shallow as f64, deep as f64) * SCORE_SCALE_F64
            ));
        }

        // Endgame tables.
        assert!(approx(
            get_mean_end(4, 9),
            PROBCUT_ENDGAME_PARAMS.mean(4.0, 9.0) * SCORE_SCALE_F64
        ));
        assert!(approx(
            get_sigma_end(4, 9),
            PROBCUT_ENDGAME_PARAMS.sigma(4.0, 9.0) * SCORE_SCALE_F64
        ));
    }
}
