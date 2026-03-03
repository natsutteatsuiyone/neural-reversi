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
        mean_intercept: -2.9298984582,
        mean_coef_shallow: 0.1468850718,
        mean_coef_deep: 0.0528082086,
        std_intercept: -1.2338037896,
        std_coef_shallow: 0.0518454912,
        std_coef_deep: 0.0186088074,
    },
    ProbcutParams {
        mean_intercept: 1.7025568570,
        mean_coef_shallow: -0.0813776266,
        mean_coef_deep: -0.0702953176,
        std_intercept: -0.6652063802,
        std_coef_shallow: 0.0314412281,
        std_coef_deep: 0.0195622127,
    },
    ProbcutParams {
        mean_intercept: -0.8258060607,
        mean_coef_shallow: 0.1258961158,
        mean_coef_deep: 0.0543992018,
        std_intercept: -0.0396301239,
        std_coef_shallow: 0.0667980097,
        std_coef_deep: -0.0187832518,
    },
    ProbcutParams {
        mean_intercept: 0.4108594788,
        mean_coef_shallow: 0.0069422494,
        mean_coef_deep: -0.0709988747,
        std_intercept: 0.3938057604,
        std_coef_shallow: 0.0455778257,
        std_coef_deep: -0.0383665059,
    },
    ProbcutParams {
        mean_intercept: -0.6046388420,
        mean_coef_shallow: 0.1127245691,
        mean_coef_deep: 0.0836518367,
        std_intercept: 0.3056495612,
        std_coef_shallow: 0.0192120867,
        std_coef_deep: -0.0121902874,
    },
    ProbcutParams {
        mean_intercept: 0.4552069931,
        mean_coef_shallow: -0.0404964566,
        mean_coef_deep: -0.0869900606,
        std_intercept: 0.3908325892,
        std_coef_shallow: 0.0066114963,
        std_coef_deep: -0.0080766065,
    },
    ProbcutParams {
        mean_intercept: -0.3939692246,
        mean_coef_shallow: 0.0937298352,
        mean_coef_deep: 0.0763207108,
        std_intercept: 0.5199485053,
        std_coef_shallow: -0.0144101719,
        std_coef_deep: -0.0081000522,
    },
    ProbcutParams {
        mean_intercept: -0.6254743354,
        mean_coef_shallow: 0.1522080539,
        mean_coef_deep: -0.0158665266,
        std_intercept: 0.5300404019,
        std_coef_shallow: -0.0519998118,
        std_coef_deep: 0.0031051179,
    },
    ProbcutParams {
        mean_intercept: 0.6925909478,
        mean_coef_shallow: -0.1480419445,
        mean_coef_deep: 0.0115833027,
        std_intercept: 0.5552028046,
        std_coef_shallow: -0.0630009553,
        std_coef_deep: 0.0110400349,
    },
    ProbcutParams {
        mean_intercept: -0.1062687710,
        mean_coef_shallow: -0.0263382782,
        mean_coef_deep: 0.0155012728,
        std_intercept: 0.5751462211,
        std_coef_shallow: -0.0705560367,
        std_coef_deep: 0.0135486382,
    },
    ProbcutParams {
        mean_intercept: 0.3678261289,
        mean_coef_shallow: -0.0739053009,
        mean_coef_deep: -0.0041879170,
        std_intercept: 0.6641655282,
        std_coef_shallow: -0.0715466599,
        std_coef_deep: 0.0110823429,
    },
    ProbcutParams {
        mean_intercept: -0.0379609278,
        mean_coef_shallow: -0.0337940155,
        mean_coef_deep: 0.0225310236,
        std_intercept: 0.7031141370,
        std_coef_shallow: -0.0843019713,
        std_coef_deep: 0.0123437058,
    },
    ProbcutParams {
        mean_intercept: 0.4586045213,
        mean_coef_shallow: -0.0337963426,
        mean_coef_deep: -0.0657337052,
        std_intercept: 0.7197120527,
        std_coef_shallow: -0.0819450978,
        std_coef_deep: 0.0153223633,
    },
    ProbcutParams {
        mean_intercept: -0.1361027698,
        mean_coef_shallow: -0.0113484858,
        mean_coef_deep: 0.0539989071,
        std_intercept: 0.7147039732,
        std_coef_shallow: -0.0817327397,
        std_coef_deep: 0.0183389064,
    },
    ProbcutParams {
        mean_intercept: 0.2676183790,
        mean_coef_shallow: -0.0211975668,
        mean_coef_deep: -0.0497572451,
        std_intercept: 0.7660603080,
        std_coef_shallow: -0.0878292452,
        std_coef_deep: 0.0165885228,
    },
    ProbcutParams {
        mean_intercept: -0.1116292142,
        mean_coef_shallow: 0.0363205613,
        mean_coef_deep: 0.0153665716,
        std_intercept: 0.7743662737,
        std_coef_shallow: -0.0874292172,
        std_coef_deep: 0.0170585035,
    },
    ProbcutParams {
        mean_intercept: -0.2031591898,
        mean_coef_shallow: 0.0130861979,
        mean_coef_deep: 0.0091486273,
        std_intercept: 0.8226561336,
        std_coef_shallow: -0.0942444973,
        std_coef_deep: 0.0153828680,
    },
    ProbcutParams {
        mean_intercept: -0.0683959464,
        mean_coef_shallow: 0.0194771934,
        mean_coef_deep: 0.0097294112,
        std_intercept: 0.8269521817,
        std_coef_shallow: -0.0927362862,
        std_coef_deep: 0.0139923055,
    },
    ProbcutParams {
        mean_intercept: -0.2848747318,
        mean_coef_shallow: 0.0853784632,
        mean_coef_deep: -0.0123861601,
        std_intercept: 0.7896758875,
        std_coef_shallow: -0.0890608104,
        std_coef_deep: 0.0175997775,
    },
    ProbcutParams {
        mean_intercept: 0.1082634913,
        mean_coef_shallow: -0.0074780352,
        mean_coef_deep: -0.0286395890,
        std_intercept: 0.7282141192,
        std_coef_shallow: -0.0792750040,
        std_coef_deep: 0.0216201360,
    },
    ProbcutParams {
        mean_intercept: -0.0555539265,
        mean_coef_shallow: 0.0641839492,
        mean_coef_deep: -0.0220913661,
        std_intercept: 0.7134636902,
        std_coef_shallow: -0.0772417920,
        std_coef_deep: 0.0249661492,
    },
    ProbcutParams {
        mean_intercept: -0.0697975487,
        mean_coef_shallow: 0.0268740161,
        mean_coef_deep: -0.0044439154,
        std_intercept: 0.7145761216,
        std_coef_shallow: -0.0789765851,
        std_coef_deep: 0.0278382979,
    },
    ProbcutParams {
        mean_intercept: -0.1620334385,
        mean_coef_shallow: 0.0376345951,
        mean_coef_deep: 0.0040459893,
        std_intercept: 0.7305874138,
        std_coef_shallow: -0.0718156281,
        std_coef_deep: 0.0263746844,
    },
    ProbcutParams {
        mean_intercept: -0.1133017325,
        mean_coef_shallow: 0.0643530323,
        mean_coef_deep: -0.0124598117,
        std_intercept: 0.7449379628,
        std_coef_shallow: -0.0743962464,
        std_coef_deep: 0.0279893718,
    },
    ProbcutParams {
        mean_intercept: -0.0656479039,
        mean_coef_shallow: 0.0357236684,
        mean_coef_deep: -0.0082929031,
        std_intercept: 0.7873754453,
        std_coef_shallow: -0.0757387229,
        std_coef_deep: 0.0269550555,
    },
    ProbcutParams {
        mean_intercept: -0.2864532161,
        mean_coef_shallow: 0.0521104438,
        mean_coef_deep: 0.0253074401,
        std_intercept: 0.8668120417,
        std_coef_shallow: -0.0771999464,
        std_coef_deep: 0.0232071352,
    },
    ProbcutParams {
        mean_intercept: 0.0009335299,
        mean_coef_shallow: 0.0890012923,
        mean_coef_deep: -0.0660734928,
        std_intercept: 0.8603460886,
        std_coef_shallow: -0.0779209675,
        std_coef_deep: 0.0268594681,
    },
    ProbcutParams {
        mean_intercept: -0.2177384320,
        mean_coef_shallow: -0.0109161352,
        mean_coef_deep: 0.0592749418,
        std_intercept: 0.8743733052,
        std_coef_shallow: -0.0661264669,
        std_coef_deep: 0.0234575124,
    },
    ProbcutParams {
        mean_intercept: -0.1255151977,
        mean_coef_shallow: 0.0806314372,
        mean_coef_deep: -0.0347853818,
        std_intercept: 0.9229089277,
        std_coef_shallow: -0.0685895014,
        std_coef_deep: 0.0209695861,
    },
    ProbcutParams {
        mean_intercept: -0.1643800401,
        mean_coef_shallow: 0.0364950612,
        mean_coef_deep: 0.0252624998,
        std_intercept: 0.9410745230,
        std_coef_shallow: -0.0713103159,
        std_coef_deep: 0.0205980623,
    },
    ProbcutParams {
        mean_intercept: 0.0101038299,
        mean_coef_shallow: 0.0074971746,
        mean_coef_deep: -0.0387845935,
        std_intercept: 0.9633806909,
        std_coef_shallow: -0.0711877895,
        std_coef_deep: 0.0199089162,
    },
    ProbcutParams {
        mean_intercept: -0.1007265308,
        mean_coef_shallow: 0.0597770810,
        mean_coef_deep: 0.0077306610,
        std_intercept: 0.9798883603,
        std_coef_shallow: -0.0695183507,
        std_coef_deep: 0.0199766319,
    },
    ProbcutParams {
        mean_intercept: -0.3857816755,
        mean_coef_shallow: 0.0477768486,
        mean_coef_deep: 0.0450906324,
        std_intercept: 1.0171489957,
        std_coef_shallow: -0.0735951445,
        std_coef_deep: 0.0176388857,
    },
    ProbcutParams {
        mean_intercept: -0.0081926975,
        mean_coef_shallow: 0.0917325021,
        mean_coef_deep: -0.0686523099,
        std_intercept: 1.0040240308,
        std_coef_shallow: -0.0647452180,
        std_coef_deep: 0.0203020881,
    },
    ProbcutParams {
        mean_intercept: -0.5280781088,
        mean_coef_shallow: 0.0856816709,
        mean_coef_deep: 0.0989153360,
        std_intercept: 1.0235234743,
        std_coef_shallow: -0.0661357813,
        std_coef_deep: 0.0192536943,
    },
    ProbcutParams {
        mean_intercept: 0.1209463679,
        mean_coef_shallow: 0.0798952928,
        mean_coef_deep: -0.1092497027,
        std_intercept: 1.0363530261,
        std_coef_shallow: -0.0750367407,
        std_coef_deep: 0.0223482272,
    },
    ProbcutParams {
        mean_intercept: -0.3709841125,
        mean_coef_shallow: 0.0752819157,
        mean_coef_deep: 0.0638483625,
        std_intercept: 1.0535798945,
        std_coef_shallow: -0.0745025120,
        std_coef_deep: 0.0225324307,
    },
    ProbcutParams {
        mean_intercept: -0.1802194419,
        mean_coef_shallow: 0.0390235138,
        mean_coef_deep: -0.0214340455,
        std_intercept: 1.0938015946,
        std_coef_shallow: -0.0762094149,
        std_coef_deep: 0.0200604054,
    },
    ProbcutParams {
        mean_intercept: -0.0934470286,
        mean_coef_shallow: 0.0976243980,
        mean_coef_deep: -0.0222688965,
        std_intercept: 1.0936763164,
        std_coef_shallow: -0.0778785821,
        std_coef_deep: 0.0216565422,
    },
    ProbcutParams {
        mean_intercept: -0.3651114633,
        mean_coef_shallow: 0.1151172864,
        mean_coef_deep: -0.0029042543,
        std_intercept: 1.0953414082,
        std_coef_shallow: -0.0816848039,
        std_coef_deep: 0.0247093261,
    },
    ProbcutParams {
        mean_intercept: 0.0231728040,
        mean_coef_shallow: 0.0682999476,
        mean_coef_deep: -0.0527121073,
        std_intercept: 1.0610637921,
        std_coef_shallow: -0.0726273484,
        std_coef_deep: 0.0265669977,
    },
    ProbcutParams {
        mean_intercept: -0.3174243666,
        mean_coef_shallow: 0.0842934584,
        mean_coef_deep: 0.0331740664,
        std_intercept: 1.0756928466,
        std_coef_shallow: -0.0709114449,
        std_coef_deep: 0.0270215218,
    },
    ProbcutParams {
        mean_intercept: 0.1082817476,
        mean_coef_shallow: 0.0143731674,
        mean_coef_deep: -0.0722326901,
        std_intercept: 1.0871720466,
        std_coef_shallow: -0.0679353659,
        std_coef_deep: 0.0250302110,
    },
    ProbcutParams {
        mean_intercept: -0.4601653358,
        mean_coef_shallow: 0.1190505350,
        mean_coef_deep: 0.0749079870,
        std_intercept: 1.0785713423,
        std_coef_shallow: -0.0627179122,
        std_coef_deep: 0.0278057505,
    },
    ProbcutParams {
        mean_intercept: -0.0301275962,
        mean_coef_shallow: 0.0832914392,
        mean_coef_deep: -0.0774881635,
        std_intercept: 1.0904894507,
        std_coef_shallow: -0.0686096746,
        std_coef_deep: 0.0296898537,
    },
    ProbcutParams {
        mean_intercept: -0.0625200168,
        mean_coef_shallow: 0.0300185394,
        mean_coef_deep: 0.0010017465,
        std_intercept: 1.1350106804,
        std_coef_shallow: -0.0721485635,
        std_coef_deep: 0.0278773259,
    },
    ProbcutParams {
        mean_intercept: -0.2293389999,
        mean_coef_shallow: 0.0785658852,
        mean_coef_deep: -0.0224897045,
        std_intercept: 1.1834953163,
        std_coef_shallow: -0.0756177008,
        std_coef_deep: 0.0243671034,
    },
    ProbcutParams {
        mean_intercept: 0.0218909520,
        mean_coef_shallow: 0.0450777385,
        mean_coef_deep: -0.0435759788,
        std_intercept: 1.1955630288,
        std_coef_shallow: -0.0760624649,
        std_coef_deep: 0.0220733390,
    },
    ProbcutParams {
        mean_intercept: -0.3284833552,
        mean_coef_shallow: 0.0884949554,
        mean_coef_deep: 0.0141278945,
        std_intercept: 1.2365028297,
        std_coef_shallow: -0.0857961919,
        std_coef_deep: 0.0198681680,
    },
    ProbcutParams {
        mean_intercept: -0.0314283183,
        mean_coef_shallow: 0.0485199421,
        mean_coef_deep: -0.0376900344,
        std_intercept: 1.2662757734,
        std_coef_shallow: -0.1017520139,
        std_coef_deep: 0.0171795030,
    },
    ProbcutParams {
        mean_intercept: -0.1586013083,
        mean_coef_shallow: 0.0423369933,
        mean_coef_deep: 0.0115087007,
        std_intercept: 1.3205469471,
        std_coef_shallow: -0.1340054188,
        std_coef_deep: 0.0134916284,
    },
    ProbcutParams {
        mean_intercept: -0.0478293161,
        mean_coef_shallow: 0.0056403759,
        mean_coef_deep: -0.0196426187,
        std_intercept: 1.3763111506,
        std_coef_shallow: -0.1864189534,
        std_coef_deep: 0.0107413532,
    },
    ProbcutParams {
        mean_intercept: -0.2343644350,
        mean_coef_shallow: 0.0072748702,
        mean_coef_deep: 0.0266983010,
        std_intercept: 1.3791798581,
        std_coef_shallow: -0.2355642255,
        std_coef_deep: 0.0076455146,
    },
    ProbcutParams {
        mean_intercept: -0.2536923490,
        mean_coef_shallow: 0.0231011579,
        mean_coef_deep: -0.0102182329,
        std_intercept: 1.7453798003,
        std_coef_shallow: -0.4916085456,
        std_coef_deep: -0.0077120174,
    },
    ProbcutParams {
        mean_intercept: -0.0325081977,
        mean_coef_shallow: -0.0164768449,
        mean_coef_deep: 0.0106109394,
        std_intercept: 1.2703651012,
        std_coef_shallow: -0.4477636081,
        std_coef_deep: 0.0074341438,
    },
    ProbcutParams {
        mean_intercept: -0.2307783010,
        mean_coef_shallow: -0.0005347364,
        mean_coef_deep: 0.0009603316,
        std_intercept: 1.2589433518,
        std_coef_shallow: -0.7255310035,
        std_coef_deep: 0.0106076502,
    },
    ProbcutParams {
        mean_intercept: 0.2378648036,
        mean_coef_shallow: 0.0017527132,
        mean_coef_deep: -0.0017508701,
        std_intercept: 0.4840866408,
        std_coef_shallow: -0.5865618842,
        std_coef_deep: 0.0000039622,
    },
    ProbcutParams {
        mean_intercept: -0.3485910202,
        mean_coef_shallow: -0.0025394841,
        mean_coef_deep: 0.0023420765,
        std_intercept: -0.3820687919,
        std_coef_shallow: -0.4933840161,
        std_coef_deep: -0.0000405036,
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
