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
    /// Most aggressive: 73% confidence (t=1.1)
    #[default]
    Level1 = 0,
    /// 87% confidence (t=1.5)
    Level2 = 1,
    /// 95% confidence (t=2.0)
    Level3 = 2,
    /// 98% confidence (t=2.6)
    Level4 = 3,
    /// Most conservative: 99% confidence (t=3.3)
    Level5 = 4,
    /// ProbCut disabled.
    None = 5,
}

impl Selectivity {
    /// Selectivity configuration: (t_multiplier, probability_percent)
    const CONFIG: [(f64, i32); 6] = [
        (1.1, 73),    // Level1: Most aggressive
        (1.5, 87),    // Level2
        (2.0, 95),    // Level3
        (2.6, 98),    // Level4
        (3.3, 99),    // Level5: Most conservative
        (999.0, 100), // None: Effectively disabled
    ];

    /// Returns the statistical confidence multiplier (t-value).
    #[inline]
    pub fn t_value(self) -> f64 {
        Self::CONFIG[self as usize].0
    }

    /// Returns the probability percentage for this level.
    #[inline]
    pub fn probability(self) -> i32 {
        Self::CONFIG[self as usize].1
    }

    /// Converts to [`u8`].
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Creates a [`Selectivity`] from a [`u8`] value, clamping to valid range.
    ///
    /// Values > 5 are clamped to [`Selectivity::None`].
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Selectivity::Level1,
            1 => Selectivity::Level2,
            2 => Selectivity::Level3,
            3 => Selectivity::Level4,
            4 => Selectivity::Level5,
            _ => Selectivity::None,
        }
    }

    /// Returns `true` if ProbCut is enabled at this selectivity level.
    #[inline]
    pub fn is_enabled(self) -> bool {
        self != Selectivity::None
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

/// Builds a 3D [ply][shallow][deep] table from midgame ProbCut parameters.
///
/// Only populates entries where `shallow <= deep` (callers always satisfy this).
fn build_mid_table(f: impl Fn(&ProbcutParams, f64, f64) -> f64) -> Box<MidTable> {
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

/// Builds a 2D [shallow][deep] table from endgame ProbCut parameters.
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
    MEAN_TABLE.get_or_init(|| build_mid_table(ProbcutParams::mean));
    SIGMA_TABLE.get_or_init(|| build_mid_table(ProbcutParams::sigma));
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
#[inline]
pub fn compute_eval_beta(
    beta: ScaledScore,
    t: f64,
    mean: f64,
    sigma: f64,
    mean0: f64,
    sigma0: f64,
) -> ScaledScore {
    let eval_mean = 0.5 * mean0 + mean;
    let eval_sigma = t * 0.5 * sigma0 + sigma;
    ScaledScore::from_raw((beta.value() as f64 - eval_sigma - eval_mean).floor() as i32)
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
const PROBCUT_PARAMS: [ProbcutParams; 60] = [
    ProbcutParams {
        mean_intercept: 0.0000000000,
        mean_coef_shallow: 0.0000000000,
        mean_coef_deep: 0.0000000000,
        std_intercept: -18.4206807440,
        std_coef_shallow: -0.0000000000,
        std_coef_deep: -0.0000000000,
    },
    ProbcutParams {
        mean_intercept: 0.8781971386,
        mean_coef_shallow: 0.0016336644,
        mean_coef_deep: -0.0381230560,
        std_intercept: -1.2794101361,
        std_coef_shallow: 0.0079717112,
        std_coef_deep: 0.0094355667,
    },
    ProbcutParams {
        mean_intercept: -1.4771794089,
        mean_coef_shallow: 0.1593721857,
        mean_coef_deep: 0.0689377578,
        std_intercept: -0.8122622339,
        std_coef_shallow: 0.0645485629,
        std_coef_deep: 0.0239454757,
    },
    ProbcutParams {
        mean_intercept: -0.1476461405,
        mean_coef_shallow: 0.0428556575,
        mean_coef_deep: -0.0076493022,
        std_intercept: -0.3643950872,
        std_coef_shallow: 0.0470178624,
        std_coef_deep: 0.0137807587,
    },
    ProbcutParams {
        mean_intercept: -0.2240807499,
        mean_coef_shallow: 0.0948150811,
        mean_coef_deep: 0.0111354011,
        std_intercept: -0.2092700839,
        std_coef_shallow: 0.0247943887,
        std_coef_deep: 0.0230855460,
    },
    ProbcutParams {
        mean_intercept: -0.3602889857,
        mean_coef_shallow: 0.0065152102,
        mean_coef_deep: 0.0533259775,
        std_intercept: 0.0857688093,
        std_coef_shallow: 0.0055862504,
        std_coef_deep: 0.0120759200,
    },
    ProbcutParams {
        mean_intercept: 0.4434472386,
        mean_coef_shallow: -0.0183167641,
        mean_coef_deep: -0.0530929845,
        std_intercept: 0.3166527961,
        std_coef_shallow: -0.0078062140,
        std_coef_deep: 0.0031285027,
    },
    ProbcutParams {
        mean_intercept: -0.7419346358,
        mean_coef_shallow: 0.0961719780,
        mean_coef_deep: 0.0853339557,
        std_intercept: 0.4057015204,
        std_coef_shallow: -0.0041915114,
        std_coef_deep: -0.0060408828,
    },
    ProbcutParams {
        mean_intercept: 0.2381844257,
        mean_coef_shallow: -0.0234154094,
        mean_coef_deep: -0.0403518733,
        std_intercept: 0.5610233736,
        std_coef_shallow: -0.0418217325,
        std_coef_deep: -0.0022265754,
    },
    ProbcutParams {
        mean_intercept: 0.0097003852,
        mean_coef_shallow: 0.0669118728,
        mean_coef_deep: -0.0056332124,
        std_intercept: 0.5699077463,
        std_coef_shallow: -0.0602498657,
        std_coef_deep: 0.0102066103,
    },
    ProbcutParams {
        mean_intercept: -0.0591758559,
        mean_coef_shallow: -0.0257166812,
        mean_coef_deep: -0.0050402564,
        std_intercept: 0.5437663497,
        std_coef_shallow: -0.0668204799,
        std_coef_deep: 0.0184794711,
    },
    ProbcutParams {
        mean_intercept: 0.1854770343,
        mean_coef_shallow: 0.0505973584,
        mean_coef_deep: -0.0398137907,
        std_intercept: 0.5984367925,
        std_coef_shallow: -0.0535778195,
        std_coef_deep: 0.0167856339,
    },
    ProbcutParams {
        mean_intercept: -0.4537276493,
        mean_coef_shallow: 0.0728938856,
        mean_coef_deep: 0.0384134356,
        std_intercept: 0.5972552583,
        std_coef_shallow: -0.0595272316,
        std_coef_deep: 0.0190575825,
    },
    ProbcutParams {
        mean_intercept: 0.3997407003,
        mean_coef_shallow: -0.0196100610,
        mean_coef_deep: -0.0726431666,
        std_intercept: 0.6531370603,
        std_coef_shallow: -0.0573020171,
        std_coef_deep: 0.0163707018,
    },
    ProbcutParams {
        mean_intercept: -0.2489455268,
        mean_coef_shallow: 0.0660679992,
        mean_coef_deep: 0.0403054573,
        std_intercept: 0.6728071923,
        std_coef_shallow: -0.0651730733,
        std_coef_deep: 0.0208820865,
    },
    ProbcutParams {
        mean_intercept: 0.2557365076,
        mean_coef_shallow: -0.0407567825,
        mean_coef_deep: -0.0510061967,
        std_intercept: 0.7342895085,
        std_coef_shallow: -0.0667062261,
        std_coef_deep: 0.0153212648,
    },
    ProbcutParams {
        mean_intercept: -0.0745696055,
        mean_coef_shallow: 0.0583855224,
        mean_coef_deep: -0.0071743116,
        std_intercept: 0.7805237946,
        std_coef_shallow: -0.0644905410,
        std_coef_deep: 0.0146097628,
    },
    ProbcutParams {
        mean_intercept: -0.3454939507,
        mean_coef_shallow: 0.0503416731,
        mean_coef_deep: 0.0186789894,
        std_intercept: 0.8131070705,
        std_coef_shallow: -0.0685338645,
        std_coef_deep: 0.0119557765,
    },
    ProbcutParams {
        mean_intercept: 0.2417653237,
        mean_coef_shallow: 0.0355777585,
        mean_coef_deep: -0.0769797845,
        std_intercept: 0.8198764500,
        std_coef_shallow: -0.0631220894,
        std_coef_deep: 0.0116546685,
    },
    ProbcutParams {
        mean_intercept: -0.4400876776,
        mean_coef_shallow: 0.0946181552,
        mean_coef_deep: 0.0418068280,
        std_intercept: 0.7683124303,
        std_coef_shallow: -0.0540578488,
        std_coef_deep: 0.0167376096,
    },
    ProbcutParams {
        mean_intercept: 0.1902041657,
        mean_coef_shallow: -0.0174543789,
        mean_coef_deep: -0.0597655923,
        std_intercept: 0.7812893859,
        std_coef_shallow: -0.0516106823,
        std_coef_deep: 0.0146076037,
    },
    ProbcutParams {
        mean_intercept: 0.0699527872,
        mean_coef_shallow: 0.0816954843,
        mean_coef_deep: -0.0480500865,
        std_intercept: 0.7427519586,
        std_coef_shallow: -0.0537232560,
        std_coef_deep: 0.0228417059,
    },
    ProbcutParams {
        mean_intercept: -0.2826398970,
        mean_coef_shallow: 0.0526538201,
        mean_coef_deep: 0.0156533334,
        std_intercept: 0.7587045827,
        std_coef_shallow: -0.0564581612,
        std_coef_deep: 0.0253417041,
    },
    ProbcutParams {
        mean_intercept: -0.1278578835,
        mean_coef_shallow: 0.0789765177,
        mean_coef_deep: -0.0120588553,
        std_intercept: 0.8170566093,
        std_coef_shallow: -0.0521477391,
        std_coef_deep: 0.0193052243,
    },
    ProbcutParams {
        mean_intercept: -0.3564657081,
        mean_coef_shallow: 0.1309397274,
        mean_coef_deep: 0.0000298692,
        std_intercept: 0.8464839635,
        std_coef_shallow: -0.0584809887,
        std_coef_deep: 0.0191871555,
    },
    ProbcutParams {
        mean_intercept: -0.0857621738,
        mean_coef_shallow: 0.0124222448,
        mean_coef_deep: -0.0088097770,
        std_intercept: 0.8786245167,
        std_coef_shallow: -0.0589956523,
        std_coef_deep: 0.0188086917,
    },
    ProbcutParams {
        mean_intercept: -0.3137952831,
        mean_coef_shallow: 0.1500605418,
        mean_coef_deep: -0.0020749594,
        std_intercept: 0.9012072205,
        std_coef_shallow: -0.0524648538,
        std_coef_deep: 0.0179556924,
    },
    ProbcutParams {
        mean_intercept: -0.1650004566,
        mean_coef_shallow: 0.0678055142,
        mean_coef_deep: -0.0161890049,
        std_intercept: 0.9149459087,
        std_coef_shallow: -0.0618956687,
        std_coef_deep: 0.0193254553,
    },
    ProbcutParams {
        mean_intercept: -0.1458099625,
        mean_coef_shallow: 0.0337190676,
        mean_coef_deep: 0.0022454918,
        std_intercept: 0.9383313469,
        std_coef_shallow: -0.0472151061,
        std_coef_deep: 0.0152359555,
    },
    ProbcutParams {
        mean_intercept: -0.2747041526,
        mean_coef_shallow: 0.1273447526,
        mean_coef_deep: -0.0017490244,
        std_intercept: 0.9710626979,
        std_coef_shallow: -0.0456346255,
        std_coef_deep: 0.0128850030,
    },
    ProbcutParams {
        mean_intercept: -0.1039423978,
        mean_coef_shallow: 0.0418743573,
        mean_coef_deep: -0.0065739583,
        std_intercept: 0.9991049111,
        std_coef_shallow: -0.0595939822,
        std_coef_deep: 0.0153907348,
    },
    ProbcutParams {
        mean_intercept: -0.2732496000,
        mean_coef_shallow: 0.0992998886,
        mean_coef_deep: 0.0057764650,
        std_intercept: 1.0220917585,
        std_coef_shallow: -0.0570197278,
        std_coef_deep: 0.0131929657,
    },
    ProbcutParams {
        mean_intercept: -0.1541687820,
        mean_coef_shallow: 0.1182102170,
        mean_coef_deep: -0.0351965394,
        std_intercept: 1.0218851568,
        std_coef_shallow: -0.0653689642,
        std_coef_deep: 0.0156949286,
    },
    ProbcutParams {
        mean_intercept: -0.1984031804,
        mean_coef_shallow: 0.0435221378,
        mean_coef_deep: 0.0183507280,
        std_intercept: 1.0059028049,
        std_coef_shallow: -0.0661922355,
        std_coef_deep: 0.0190793548,
    },
    ProbcutParams {
        mean_intercept: -0.2706377910,
        mean_coef_shallow: 0.1745333572,
        mean_coef_deep: -0.0417320632,
        std_intercept: 1.0020153409,
        std_coef_shallow: -0.0605412911,
        std_coef_deep: 0.0211209170,
    },
    ProbcutParams {
        mean_intercept: -0.3945369190,
        mean_coef_shallow: 0.1057746251,
        mean_coef_deep: 0.0497197266,
        std_intercept: 0.9806288670,
        std_coef_shallow: -0.0643495662,
        std_coef_deep: 0.0237608982,
    },
    ProbcutParams {
        mean_intercept: -0.1373561437,
        mean_coef_shallow: 0.0435951241,
        mean_coef_deep: -0.0109629115,
        std_intercept: 1.0237998989,
        std_coef_shallow: -0.0667296943,
        std_coef_deep: 0.0216928657,
    },
    ProbcutParams {
        mean_intercept: -0.1805897025,
        mean_coef_shallow: 0.1210463285,
        mean_coef_deep: -0.0295270534,
        std_intercept: 1.0704636306,
        std_coef_shallow: -0.0713705333,
        std_coef_deep: 0.0197504645,
    },
    ProbcutParams {
        mean_intercept: -0.2694398811,
        mean_coef_shallow: 0.0580362423,
        mean_coef_deep: 0.0189237616,
        std_intercept: 1.1262923540,
        std_coef_shallow: -0.0747503360,
        std_coef_deep: 0.0164496648,
    },
    ProbcutParams {
        mean_intercept: -0.0638135538,
        mean_coef_shallow: 0.1000136929,
        mean_coef_deep: -0.0649964310,
        std_intercept: 1.0895627049,
        std_coef_shallow: -0.0687161644,
        std_coef_deep: 0.0190893660,
    },
    ProbcutParams {
        mean_intercept: -0.4005951275,
        mean_coef_shallow: 0.0888023128,
        mean_coef_deep: 0.0556283669,
        std_intercept: 1.0986819732,
        std_coef_shallow: -0.0667388717,
        std_coef_deep: 0.0190770959,
    },
    ProbcutParams {
        mean_intercept: -0.0532351028,
        mean_coef_shallow: 0.0641432303,
        mean_coef_deep: -0.0672380996,
        std_intercept: 1.0851450757,
        std_coef_shallow: -0.0624730912,
        std_coef_deep: 0.0197057702,
    },
    ProbcutParams {
        mean_intercept: -0.2607105065,
        mean_coef_shallow: 0.0846409702,
        mean_coef_deep: 0.0344551239,
        std_intercept: 1.0731649655,
        std_coef_shallow: -0.0597513922,
        std_coef_deep: 0.0231316915,
    },
    ProbcutParams {
        mean_intercept: -0.0624988352,
        mean_coef_shallow: 0.0809697583,
        mean_coef_deep: -0.0658709328,
        std_intercept: 1.0759424127,
        std_coef_shallow: -0.0616108277,
        std_coef_deep: 0.0239988728,
    },
    ProbcutParams {
        mean_intercept: -0.1274317030,
        mean_coef_shallow: 0.0603023344,
        mean_coef_deep: 0.0069884887,
        std_intercept: 1.0762485989,
        std_coef_shallow: -0.0637769739,
        std_coef_deep: 0.0277180451,
    },
    ProbcutParams {
        mean_intercept: -0.2527357122,
        mean_coef_shallow: 0.1061509806,
        mean_coef_deep: -0.0202711919,
        std_intercept: 1.0713234024,
        std_coef_shallow: -0.0700790296,
        std_coef_deep: 0.0320075302,
    },
    ProbcutParams {
        mean_intercept: -0.0644243399,
        mean_coef_shallow: 0.0672004707,
        mean_coef_deep: -0.0274314105,
        std_intercept: 1.0974216236,
        std_coef_shallow: -0.0726685628,
        std_coef_deep: 0.0306565205,
    },
    ProbcutParams {
        mean_intercept: -0.3600853280,
        mean_coef_shallow: 0.1242696450,
        mean_coef_deep: 0.0099180738,
        std_intercept: 1.1494486678,
        std_coef_shallow: -0.0781133653,
        std_coef_deep: 0.0269351308,
    },
    ProbcutParams {
        mean_intercept: -0.0897356638,
        mean_coef_shallow: 0.0874644160,
        mean_coef_deep: -0.0430239078,
        std_intercept: 1.1425693502,
        std_coef_shallow: -0.0785012598,
        std_coef_deep: 0.0252379150,
    },
    ProbcutParams {
        mean_intercept: -0.2452158311,
        mean_coef_shallow: 0.0601019694,
        mean_coef_deep: 0.0137499736,
        std_intercept: 1.1807339354,
        std_coef_shallow: -0.0899091218,
        std_coef_deep: 0.0230256580,
    },
    ProbcutParams {
        mean_intercept: -0.0569624690,
        mean_coef_shallow: 0.0309609517,
        mean_coef_deep: -0.0352864897,
        std_intercept: 1.2289942813,
        std_coef_shallow: -0.1076841409,
        std_coef_deep: 0.0192636019,
    },
    ProbcutParams {
        mean_intercept: -0.1678024492,
        mean_coef_shallow: 0.0298359037,
        mean_coef_deep: 0.0150322570,
        std_intercept: 1.2642590900,
        std_coef_shallow: -0.1348290236,
        std_coef_deep: 0.0152762488,
    },
    ProbcutParams {
        mean_intercept: -0.1724880666,
        mean_coef_shallow: 0.0245152619,
        mean_coef_deep: -0.0076207915,
        std_intercept: 1.3735011729,
        std_coef_shallow: -0.2015008197,
        std_coef_deep: 0.0099355624,
    },
    ProbcutParams {
        mean_intercept: -0.1056871497,
        mean_coef_shallow: -0.0009725156,
        mean_coef_deep: 0.0088979237,
        std_intercept: 1.4807274226,
        std_coef_shallow: -0.3126687633,
        std_coef_deep: 0.0061099226,
    },
    ProbcutParams {
        mean_intercept: -0.1911585077,
        mean_coef_shallow: 0.0103586936,
        mean_coef_deep: -0.0048496078,
        std_intercept: 1.7551440006,
        std_coef_shallow: -0.6161882234,
        std_coef_deep: 0.0100132898,
    },
    ProbcutParams {
        mean_intercept: -0.0067086171,
        mean_coef_shallow: -0.0031970589,
        mean_coef_deep: 0.0019878358,
        std_intercept: 1.4294640856,
        std_coef_shallow: -0.5915429446,
        std_coef_deep: 0.0005571824,
    },
    ProbcutParams {
        mean_intercept: -0.2346769930,
        mean_coef_shallow: -0.0023419478,
        mean_coef_deep: 0.0025682595,
        std_intercept: 0.8454568660,
        std_coef_shallow: -0.5829595232,
        std_coef_deep: 0.0119112925,
    },
    ProbcutParams {
        mean_intercept: -0.1001925134,
        mean_coef_shallow: -0.0015764096,
        mean_coef_deep: 0.0011655513,
        std_intercept: 0.1863429772,
        std_coef_shallow: -0.5712852987,
        std_coef_deep: 0.0000019249,
    },
    ProbcutParams {
        mean_intercept: -0.3410933064,
        mean_coef_shallow: -0.0019097601,
        mean_coef_deep: 0.0017505073,
        std_intercept: -0.6311956538,
        std_coef_shallow: -0.4978091489,
        std_coef_deep: 0.0000123274,
    },
    ProbcutParams {
        mean_intercept: 0.0000000000,
        mean_coef_shallow: 0.0000000000,
        mean_coef_deep: 0.0000000000,
        std_intercept: -18.4206807440,
        std_coef_shallow: -0.0000000000,
        std_coef_deep: -0.0000000000,
    },
];

const _: () = assert!(PROBCUT_PARAMS.len() == NUM_PLY);
