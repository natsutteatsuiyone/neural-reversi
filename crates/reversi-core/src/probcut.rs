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

const MAX_PLY: usize = 60;
const MAX_DEPTH: usize = 60;

type MeanTable = [[[f64; MAX_DEPTH]; MAX_DEPTH]; MAX_PLY];
type SigmaTable = [[[f64; MAX_DEPTH]; MAX_DEPTH]; MAX_PLY];

const SCORE_SCALE_F64: f64 = ScaledScore::SCALE as f64;

static MEAN_TABLE: OnceLock<Box<MeanTable>> = OnceLock::new();
static SIGMA_TABLE: OnceLock<Box<SigmaTable>> = OnceLock::new();
static MEAN_TABLE_END: OnceLock<Box<[[f64; MAX_DEPTH]; MAX_DEPTH]>> = OnceLock::new();
static SIGMA_TABLE_END: OnceLock<Box<[[f64; MAX_DEPTH]; MAX_DEPTH]>> = OnceLock::new();

fn alloc_3d_table() -> Box<MeanTable> {
    vec![[[0.0f64; MAX_DEPTH]; MAX_DEPTH]; MAX_PLY]
        .into_boxed_slice()
        .try_into()
        .unwrap()
}

fn alloc_2d_table() -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    vec![[0.0f64; MAX_DEPTH]; MAX_DEPTH]
        .into_boxed_slice()
        .try_into()
        .unwrap()
}

/// Builds a 3D [ply][shallow][deep] table from midgame ProbCut parameters.
///
/// Only populates entries where `shallow <= deep` (callers always satisfy this).
fn build_mid_table(f: impl Fn(&ProbcutParams, f64, f64) -> f64) -> Box<MeanTable> {
    let mut tbl = alloc_3d_table();
    for ply in 0..MAX_PLY {
        let params = &PROBCUT_PARAMS[ply];
        for shallow in 0..MAX_DEPTH {
            for deep in shallow..MAX_DEPTH {
                tbl[ply][shallow][deep] = f(params, shallow as f64, deep as f64) * SCORE_SCALE_F64;
            }
        }
    }
    tbl
}

/// Builds a 2D [shallow][deep] table from endgame ProbCut parameters.
///
/// Only populates entries where `shallow <= deep` (callers always satisfy this).
fn build_end_table(
    f: impl Fn(&ProbcutParams, f64, f64) -> f64,
) -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    let mut tbl = alloc_2d_table();
    for shallow in 0..MAX_DEPTH {
        for deep in shallow..MAX_DEPTH {
            tbl[shallow][deep] =
                f(&PROBCUT_ENDGAME_PARAMS, shallow as f64, deep as f64) * SCORE_SCALE_F64;
        }
    }
    tbl
}

#[cold]
#[inline(never)]
fn probcut_not_initialized() -> ! {
    panic!("probcut not initialized");
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

/// Returns the pre-computed mean value for midgame positions.
///
/// # Panics
///
/// Panics if [`init`] has not been called.
#[inline]
pub fn get_mean(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    debug_assert!(ply < MAX_PLY);
    debug_assert!((shallow as usize) <= (deep as usize));
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = MEAN_TABLE
        .get()
        .unwrap_or_else(|| probcut_not_initialized());
    tbl[ply][shallow as usize][deep as usize]
}

/// Returns the pre-computed sigma value for midgame positions.
///
/// # Panics
///
/// Panics if [`init`] has not been called.
#[inline]
pub fn get_sigma(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    debug_assert!(ply < MAX_PLY);
    debug_assert!((shallow as usize) <= (deep as usize));
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = SIGMA_TABLE
        .get()
        .unwrap_or_else(|| probcut_not_initialized());
    tbl[ply][shallow as usize][deep as usize]
}

/// Returns the pre-computed mean value for endgame positions.
///
/// # Panics
///
/// Panics if [`init`] has not been called.
#[inline]
pub fn get_mean_end(shallow: Depth, deep: Depth) -> f64 {
    debug_assert!((shallow as usize) <= (deep as usize));
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = MEAN_TABLE_END
        .get()
        .unwrap_or_else(|| probcut_not_initialized());
    tbl[shallow as usize][deep as usize]
}

/// Returns the pre-computed sigma value for endgame positions.
///
/// # Panics
///
/// Panics if [`init`] has not been called.
#[inline]
pub fn get_sigma_end(shallow: Depth, deep: Depth) -> f64 {
    debug_assert!((shallow as usize) <= (deep as usize));
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = SIGMA_TABLE_END
        .get()
        .unwrap_or_else(|| probcut_not_initialized());
    tbl[shallow as usize][deep as usize]
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
    mean_intercept: -0.2031031679,
    mean_coef_shallow: 0.0263614864,
    mean_coef_deep: 0.0059992592,
    std_intercept: 0.9077005940,
    std_coef_shallow: -0.0612455789,
    std_coef_deep: 0.0312307898,
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
        mean_intercept: 0.8943067222,
        mean_coef_shallow: 0.0352078708,
        mean_coef_deep: -0.0842961308,
        std_intercept: -1.5341650210,
        std_coef_shallow: 0.0817155641,
        std_coef_deep: 0.0967317952,
    },
    ProbcutParams {
        mean_intercept: 1.0357346463,
        mean_coef_shallow: -0.0576334562,
        mean_coef_deep: -0.1038299377,
        std_intercept: -0.6614978687,
        std_coef_shallow: 0.0752786476,
        std_coef_deep: 0.0439325551,
    },
    ProbcutParams {
        mean_intercept: -0.9708211858,
        mean_coef_shallow: 0.2257722256,
        mean_coef_deep: 0.0953341551,
        std_intercept: 0.0477927136,
        std_coef_shallow: 0.0775416612,
        std_coef_deep: -0.0212895796,
    },
    ProbcutParams {
        mean_intercept: 0.2991848417,
        mean_coef_shallow: -0.0463726006,
        mean_coef_deep: -0.0744327640,
        std_intercept: 0.3190034185,
        std_coef_shallow: 0.0868776765,
        std_coef_deep: -0.0383472616,
    },
    ProbcutParams {
        mean_intercept: 0.3033317748,
        mean_coef_shallow: -0.0875338014,
        mean_coef_deep: 0.0792704221,
        std_intercept: 0.3311986121,
        std_coef_shallow: 0.0118998638,
        std_coef_deep: -0.0098954055,
    },
    ProbcutParams {
        mean_intercept: -0.2344950613,
        mean_coef_shallow: 0.1128943370,
        mean_coef_deep: -0.0464865484,
        std_intercept: 0.5147401904,
        std_coef_shallow: 0.0096502345,
        std_coef_deep: -0.0183533038,
    },
    ProbcutParams {
        mean_intercept: 0.7542841861,
        mean_coef_shallow: -0.1790800570,
        mean_coef_deep: 0.0171389782,
        std_intercept: 0.4512695360,
        std_coef_shallow: 0.0145653747,
        std_coef_deep: -0.0136595050,
    },
    ProbcutParams {
        mean_intercept: -0.5129907638,
        mean_coef_shallow: 0.0640718979,
        mean_coef_deep: 0.0412650851,
        std_intercept: 0.5152894147,
        std_coef_shallow: -0.0245484301,
        std_coef_deep: 0.0000465664,
    },
    ProbcutParams {
        mean_intercept: 0.8189588930,
        mean_coef_shallow: -0.1604758332,
        mean_coef_deep: -0.0276576940,
        std_intercept: 0.6133676694,
        std_coef_shallow: -0.0594611025,
        std_coef_deep: 0.0067581191,
    },
    ProbcutParams {
        mean_intercept: -0.0251723829,
        mean_coef_shallow: -0.0206286454,
        mean_coef_deep: 0.0295088219,
        std_intercept: 0.6256540028,
        std_coef_shallow: -0.0745743871,
        std_coef_deep: 0.0118636965,
    },
    ProbcutParams {
        mean_intercept: 0.3314019905,
        mean_coef_shallow: -0.0451570539,
        mean_coef_deep: -0.0361256774,
        std_intercept: 0.6906248071,
        std_coef_shallow: -0.0868379165,
        std_coef_deep: 0.0117457561,
    },
    ProbcutParams {
        mean_intercept: 0.0985685246,
        mean_coef_shallow: -0.0791758347,
        mean_coef_deep: 0.0440185218,
        std_intercept: 0.7247618297,
        std_coef_shallow: -0.0905077556,
        std_coef_deep: 0.0133124833,
    },
    ProbcutParams {
        mean_intercept: 0.4266320023,
        mean_coef_shallow: -0.0459275661,
        mean_coef_deep: -0.0651865469,
        std_intercept: 0.7025371717,
        std_coef_shallow: -0.0827184341,
        std_coef_deep: 0.0149238529,
    },
    ProbcutParams {
        mean_intercept: 0.1359160634,
        mean_coef_shallow: -0.0422631855,
        mean_coef_deep: 0.0195909379,
        std_intercept: 0.7096454481,
        std_coef_shallow: -0.0872429536,
        std_coef_deep: 0.0184546119,
    },
    ProbcutParams {
        mean_intercept: 0.0080606977,
        mean_coef_shallow: 0.0213360033,
        mean_coef_deep: -0.0279853521,
        std_intercept: 0.7033807916,
        std_coef_shallow: -0.0811870810,
        std_coef_deep: 0.0182332852,
    },
    ProbcutParams {
        mean_intercept: 0.1616024019,
        mean_coef_shallow: 0.0128628051,
        mean_coef_deep: -0.0173170461,
        std_intercept: 0.7562948586,
        std_coef_shallow: -0.0924003201,
        std_coef_deep: 0.0195808213,
    },
    ProbcutParams {
        mean_intercept: -0.2035723943,
        mean_coef_shallow: 0.0145786196,
        mean_coef_deep: 0.0154668666,
        std_intercept: 0.7910124097,
        std_coef_shallow: -0.0956474312,
        std_coef_deep: 0.0186689586,
    },
    ProbcutParams {
        mean_intercept: 0.1733705319,
        mean_coef_shallow: 0.0723342707,
        mean_coef_deep: -0.0750499537,
        std_intercept: 0.7540715545,
        std_coef_shallow: -0.0905220217,
        std_coef_deep: 0.0218693341,
    },
    ProbcutParams {
        mean_intercept: -0.2518917325,
        mean_coef_shallow: 0.0713270149,
        mean_coef_deep: 0.0164152919,
        std_intercept: 0.6727711952,
        std_coef_shallow: -0.0827447603,
        std_coef_deep: 0.0318628519,
    },
    ProbcutParams {
        mean_intercept: -0.2835720052,
        mean_coef_shallow: -0.0206551376,
        mean_coef_deep: 0.0527776701,
        std_intercept: 0.7352947726,
        std_coef_shallow: -0.0757288096,
        std_coef_deep: 0.0220859584,
    },
    ProbcutParams {
        mean_intercept: 0.0807305221,
        mean_coef_shallow: 0.1344020336,
        mean_coef_deep: -0.0981210025,
        std_intercept: 0.7644904618,
        std_coef_shallow: -0.0739785920,
        std_coef_deep: 0.0209545564,
    },
    ProbcutParams {
        mean_intercept: -0.2995722841,
        mean_coef_shallow: -0.0479752611,
        mean_coef_deep: 0.0878225171,
        std_intercept: 0.7739835909,
        std_coef_shallow: -0.0724705890,
        std_coef_deep: 0.0218749627,
    },
    ProbcutParams {
        mean_intercept: 0.1482703076,
        mean_coef_shallow: 0.0443957927,
        mean_coef_deep: -0.0816081025,
        std_intercept: 0.8386093490,
        std_coef_shallow: -0.0695570433,
        std_coef_deep: 0.0164300905,
    },
    ProbcutParams {
        mean_intercept: -0.4044627125,
        mean_coef_shallow: 0.0552987769,
        mean_coef_deep: 0.0781193477,
        std_intercept: 0.8413481022,
        std_coef_shallow: -0.0706897690,
        std_coef_deep: 0.0176889948,
    },
    ProbcutParams {
        mean_intercept: 0.0172709543,
        mean_coef_shallow: -0.0509143828,
        mean_coef_deep: -0.0393939572,
        std_intercept: 0.8863299854,
        std_coef_shallow: -0.0703948731,
        std_coef_deep: 0.0162697400,
    },
    ProbcutParams {
        mean_intercept: -0.1327062950,
        mean_coef_shallow: 0.0946082678,
        mean_coef_deep: 0.0114708956,
        std_intercept: 0.8689732837,
        std_coef_shallow: -0.0556553775,
        std_coef_deep: 0.0190367137,
    },
    ProbcutParams {
        mean_intercept: -0.4044953668,
        mean_coef_shallow: 0.0628527159,
        mean_coef_deep: 0.0308646625,
        std_intercept: 0.9471871209,
        std_coef_shallow: -0.0770502927,
        std_coef_deep: 0.0177748260,
    },
    ProbcutParams {
        mean_intercept: 0.1700703791,
        mean_coef_shallow: 0.0066578565,
        mean_coef_deep: -0.0530360122,
        std_intercept: 0.9670500904,
        std_coef_shallow: -0.0693549059,
        std_coef_deep: 0.0145194658,
    },
    ProbcutParams {
        mean_intercept: -0.5068424838,
        mean_coef_shallow: 0.1376151900,
        mean_coef_deep: 0.0435733136,
        std_intercept: 0.9423947791,
        std_coef_shallow: -0.0750174008,
        std_coef_deep: 0.0204616617,
    },
    ProbcutParams {
        mean_intercept: 0.0466894848,
        mean_coef_shallow: -0.0389422597,
        mean_coef_deep: -0.0062240435,
        std_intercept: 0.9606123726,
        std_coef_shallow: -0.0820223473,
        std_coef_deep: 0.0207687403,
    },
    ProbcutParams {
        mean_intercept: -0.5300610091,
        mean_coef_shallow: 0.1458493183,
        mean_coef_deep: 0.0579144338,
        std_intercept: 0.9691270619,
        std_coef_shallow: -0.0743459248,
        std_coef_deep: 0.0202649182,
    },
    ProbcutParams {
        mean_intercept: -0.2293250567,
        mean_coef_shallow: 0.1107591732,
        mean_coef_deep: -0.0385169640,
        std_intercept: 0.9504393131,
        std_coef_shallow: -0.0822129327,
        std_coef_deep: 0.0236252347,
    },
    ProbcutParams {
        mean_intercept: -0.0122770718,
        mean_coef_shallow: -0.0184421321,
        mean_coef_deep: 0.0204940210,
        std_intercept: 0.9242132592,
        std_coef_shallow: -0.0755708389,
        std_coef_deep: 0.0257628278,
    },
    ProbcutParams {
        mean_intercept: -0.4085058349,
        mean_coef_shallow: 0.1092866221,
        mean_coef_deep: 0.0263213068,
        std_intercept: 0.9652739247,
        std_coef_shallow: -0.0771961004,
        std_coef_deep: 0.0239554959,
    },
    ProbcutParams {
        mean_intercept: -0.0959162607,
        mean_coef_shallow: 0.0423828382,
        mean_coef_deep: -0.0121360066,
        std_intercept: 0.9800689904,
        std_coef_shallow: -0.0797340368,
        std_coef_deep: 0.0237031815,
    },
    ProbcutParams {
        mean_intercept: -0.2464865796,
        mean_coef_shallow: 0.0661003932,
        mean_coef_deep: 0.0103654852,
        std_intercept: 0.9824230587,
        std_coef_shallow: -0.0780997052,
        std_coef_deep: 0.0261527419,
    },
    ProbcutParams {
        mean_intercept: -0.0854873841,
        mean_coef_shallow: 0.0556673659,
        mean_coef_deep: -0.0189032540,
        std_intercept: 1.0352823006,
        std_coef_shallow: -0.0840209809,
        std_coef_deep: 0.0230669907,
    },
    ProbcutParams {
        mean_intercept: -0.0892252884,
        mean_coef_shallow: 0.0736084388,
        mean_coef_deep: -0.0243038558,
        std_intercept: 1.0620872504,
        std_coef_shallow: -0.0862242953,
        std_coef_deep: 0.0229056280,
    },
    ProbcutParams {
        mean_intercept: -0.1869694140,
        mean_coef_shallow: 0.0301746709,
        mean_coef_deep: -0.0011878455,
        std_intercept: 1.0613429249,
        std_coef_shallow: -0.0837798249,
        std_coef_deep: 0.0246371529,
    },
    ProbcutParams {
        mean_intercept: -0.0811749271,
        mean_coef_shallow: 0.0505803831,
        mean_coef_deep: -0.0173968145,
        std_intercept: 1.0471615817,
        std_coef_shallow: -0.0774743251,
        std_coef_deep: 0.0256269417,
    },
    ProbcutParams {
        mean_intercept: -0.2006770942,
        mean_coef_shallow: 0.0834796455,
        mean_coef_deep: -0.0045255727,
        std_intercept: 1.0529611274,
        std_coef_shallow: -0.0770201213,
        std_coef_deep: 0.0253639775,
    },
    ProbcutParams {
        mean_intercept: -0.2633183826,
        mean_coef_shallow: 0.0363705279,
        mean_coef_deep: 0.0280639739,
        std_intercept: 1.0570033277,
        std_coef_shallow: -0.0682926659,
        std_coef_deep: 0.0255141843,
    },
    ProbcutParams {
        mean_intercept: -0.0718449421,
        mean_coef_shallow: 0.0429779897,
        mean_coef_deep: -0.0236996532,
        std_intercept: 1.0578443703,
        std_coef_shallow: -0.0671125541,
        std_coef_deep: 0.0264493435,
    },
    ProbcutParams {
        mean_intercept: -0.0853323973,
        mean_coef_shallow: -0.0035588968,
        mean_coef_deep: 0.0067971390,
        std_intercept: 1.0655845420,
        std_coef_shallow: -0.0682383310,
        std_coef_deep: 0.0289481899,
    },
    ProbcutParams {
        mean_intercept: -0.0888858157,
        mean_coef_shallow: 0.0877987379,
        mean_coef_deep: -0.0529424577,
        std_intercept: 1.0615749345,
        std_coef_shallow: -0.0729806019,
        std_coef_deep: 0.0339416339,
    },
    ProbcutParams {
        mean_intercept: -0.2618357204,
        mean_coef_shallow: 0.0866304135,
        mean_coef_deep: 0.0231227170,
        std_intercept: 1.1024524657,
        std_coef_shallow: -0.0753097715,
        std_coef_deep: 0.0312515119,
    },
    ProbcutParams {
        mean_intercept: 0.0014800232,
        mean_coef_shallow: 0.0224123961,
        mean_coef_deep: -0.0519053662,
        std_intercept: 1.1471839432,
        std_coef_shallow: -0.0810537818,
        std_coef_deep: 0.0286726745,
    },
    ProbcutParams {
        mean_intercept: -0.0996040308,
        mean_coef_shallow: 0.0910862861,
        mean_coef_deep: -0.0222152368,
        std_intercept: 1.1750648328,
        std_coef_shallow: -0.0825556866,
        std_coef_deep: 0.0247292565,
    },
    ProbcutParams {
        mean_intercept: -0.2908450379,
        mean_coef_shallow: 0.0913032509,
        mean_coef_deep: 0.0036729163,
        std_intercept: 1.2076545707,
        std_coef_shallow: -0.0911124943,
        std_coef_deep: 0.0222534961,
    },
    ProbcutParams {
        mean_intercept: -0.0061301516,
        mean_coef_shallow: 0.0447641544,
        mean_coef_deep: -0.0362877273,
        std_intercept: 1.2454700704,
        std_coef_shallow: -0.1076828113,
        std_coef_deep: 0.0188871779,
    },
    ProbcutParams {
        mean_intercept: -0.2611763830,
        mean_coef_shallow: 0.0656257540,
        mean_coef_deep: 0.0174646310,
        std_intercept: 1.2771592966,
        std_coef_shallow: -0.1296422287,
        std_coef_deep: 0.0143684025,
    },
    ProbcutParams {
        mean_intercept: -0.0557506635,
        mean_coef_shallow: 0.0041497204,
        mean_coef_deep: -0.0168389846,
        std_intercept: 1.3471428168,
        std_coef_shallow: -0.1806534313,
        std_coef_deep: 0.0100540198,
    },
    ProbcutParams {
        mean_intercept: -0.1897938679,
        mean_coef_shallow: 0.0045315176,
        mean_coef_deep: 0.0195922604,
        std_intercept: 1.3923847412,
        std_coef_shallow: -0.2502827521,
        std_coef_deep: 0.0064084723,
    },
    ProbcutParams {
        mean_intercept: -0.1537332304,
        mean_coef_shallow: 0.0266994857,
        mean_coef_deep: -0.0137267418,
        std_intercept: 1.5276402454,
        std_coef_shallow: -0.4444851929,
        std_coef_deep: 0.0089293651,
    },
    ProbcutParams {
        mean_intercept: 0.1301017850,
        mean_coef_shallow: -0.0013119475,
        mean_coef_deep: 0.0005249412,
        std_intercept: 1.7209977250,
        std_coef_shallow: -0.8455981984,
        std_coef_deep: 0.0193103989,
    },
    ProbcutParams {
        mean_intercept: -0.1648897408,
        mean_coef_shallow: 0.0021339966,
        mean_coef_deep: -0.0002451127,
        std_intercept: 1.0568809144,
        std_coef_shallow: -0.5796965993,
        std_coef_deep: 0.0054121185,
    },
    ProbcutParams {
        mean_intercept: 0.2336957916,
        mean_coef_shallow: 0.0019863234,
        mean_coef_deep: -0.0017797848,
        std_intercept: 0.4614887946,
        std_coef_shallow: -0.6026743203,
        std_coef_deep: 0.0000158683,
    },
    ProbcutParams {
        mean_intercept: -0.4063469657,
        mean_coef_shallow: -0.0031597689,
        mean_coef_deep: 0.0030499636,
        std_intercept: -0.4009385958,
        std_coef_shallow: -0.4649963095,
        std_coef_deep: -0.0000135637,
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

const _: () = assert!(PROBCUT_PARAMS.len() == MAX_PLY);
