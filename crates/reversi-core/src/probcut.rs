//! ProbCut forward pruning implementation for search optimization.

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

    /// Gets the statistical confidence multiplier (t-value).
    #[inline]
    pub fn t_value(self) -> f64 {
        Self::CONFIG[self as usize].0
    }

    /// Gets the expected success probability percentage.
    #[inline]
    pub fn probability(self) -> i32 {
        Self::CONFIG[self as usize].1
    }

    /// Converts to u8.
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Creates a Selectivity from a u8 value, clamping to valid range.
    ///
    /// Values > 5 are clamped to `Selectivity::None` (5).
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        // SAFETY: Selectivity enum has repr(u8) with contiguous values 0-5.
        unsafe { std::mem::transmute(value.min(5)) }
    }

    /// Checks if ProbCut is enabled for this selectivity level.
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

/// Safely allocates a 3D table on the heap to avoid stack overflow.
fn alloc_3d_table() -> Box<MeanTable> {
    let tbl = vec![[0.0f64; MAX_DEPTH]; MAX_PLY * MAX_DEPTH].into_boxed_slice();
    unsafe { Box::from_raw(Box::into_raw(tbl) as *mut MeanTable) }
}

/// Safely allocates a 2D table on the heap to avoid stack overflow.
fn alloc_2d_table() -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    let tbl = vec![0.0f64; MAX_DEPTH * MAX_DEPTH].into_boxed_slice();
    unsafe { Box::from_raw(Box::into_raw(tbl) as *mut [[f64; MAX_DEPTH]; MAX_DEPTH]) }
}

/// Builds the pre-computed mean table for midgame positions.
fn build_mean_table() -> Box<MeanTable> {
    let mut tbl = alloc_3d_table();

    for ply in 0..MAX_PLY {
        let params = &PROBCUT_PARAMS[ply];
        for shallow in 0..MAX_DEPTH {
            for deep in shallow..MAX_DEPTH {
                let v = params.mean(shallow as f64, deep as f64) * SCORE_SCALE_F64;
                tbl[ply][shallow][deep] = v;
                tbl[ply][deep][shallow] = v; // Mirror for symmetric table access
            }
        }
    }
    tbl
}

/// Builds the pre-computed sigma table for midgame positions.
fn build_sigma_table() -> Box<SigmaTable> {
    let mut tbl = unsafe { Box::from_raw(Box::into_raw(alloc_3d_table()) as *mut SigmaTable) };

    for ply in 0..MAX_PLY {
        let params = &PROBCUT_PARAMS[ply];
        for shallow in 0..MAX_DEPTH {
            for deep in shallow..MAX_DEPTH {
                let v = params.sigma(shallow as f64, deep as f64) * SCORE_SCALE_F64;
                tbl[ply][shallow][deep] = v;
                tbl[ply][deep][shallow] = v; // Mirror for symmetric table access
            }
        }
    }
    tbl
}

/// Builds the pre-computed mean table for endgame positions.
fn build_mean_table_end() -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    let mut tbl = alloc_2d_table();

    for shallow in 0..MAX_DEPTH {
        for deep in shallow..MAX_DEPTH {
            let v = PROBCUT_ENDGAME_PARAMS.mean(shallow as f64, deep as f64) * SCORE_SCALE_F64;
            tbl[shallow][deep] = v;
            tbl[deep][shallow] = v;
        }
    }
    tbl
}

/// Builds the pre-computed sigma table for endgame positions.
fn build_sigma_table_end() -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    let mut tbl = alloc_2d_table();

    for shallow in 0..MAX_DEPTH {
        for deep in shallow..MAX_DEPTH {
            let v = PROBCUT_ENDGAME_PARAMS.sigma(shallow as f64, deep as f64) * SCORE_SCALE_F64;
            tbl[shallow][deep] = v;
            tbl[deep][shallow] = v;
        }
    }
    tbl
}

/// Initializes probcut tables. Called from Search::new().
pub fn init() {
    MEAN_TABLE.get_or_init(build_mean_table);
    SIGMA_TABLE.get_or_init(build_sigma_table);
    MEAN_TABLE_END.get_or_init(build_mean_table_end);
    SIGMA_TABLE_END.get_or_init(build_sigma_table_end);
}

/// Returns the pre-computed mean value for midgame positions.
#[inline]
pub fn get_mean(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    debug_assert!(ply < MAX_PLY);
    debug_assert!((shallow as usize) < MAX_DEPTH);
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = MEAN_TABLE.get().expect("probcut not initialized");
    tbl[ply][shallow as usize][deep as usize]
}

/// Returns the pre-computed sigma value for midgame positions.
#[inline]
pub fn get_sigma(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    debug_assert!(ply < MAX_PLY);
    debug_assert!((shallow as usize) < MAX_DEPTH);
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = SIGMA_TABLE.get().expect("probcut not initialized");
    tbl[ply][shallow as usize][deep as usize]
}

/// Returns the pre-computed mean value for endgame positions.
#[inline]
pub fn get_mean_end(shallow: Depth, deep: Depth) -> f64 {
    debug_assert!((shallow as usize) < MAX_DEPTH);
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = MEAN_TABLE_END.get().expect("probcut not initialized");
    tbl[shallow as usize][deep as usize]
}

/// Returns the pre-computed sigma value for endgame positions.
#[inline]
pub fn get_sigma_end(shallow: Depth, deep: Depth) -> f64 {
    debug_assert!((shallow as usize) < MAX_DEPTH);
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = SIGMA_TABLE_END.get().expect("probcut not initialized");
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
    mean_intercept: -0.2069276080,
    mean_coef_shallow: 0.0261871778,
    mean_coef_deep: 0.0067046108,
    std_intercept: 0.9052427490,
    std_coef_shallow: -0.0609488670,
    std_coef_deep: 0.0312314258,
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
        mean_intercept: -0.9713940596,
        mean_coef_shallow: 0.0824417124,
        mean_coef_deep: 0.0313890227,
        std_intercept: -0.3153389082,
        std_coef_shallow: 0.0620567641,
        std_coef_deep: -0.0301182035,
    },
    ProbcutParams {
        mean_intercept: 0.2399974693,
        mean_coef_shallow: -0.0130203439,
        mean_coef_deep: -0.0035872251,
        std_intercept: -0.2675536007,
        std_coef_shallow: 0.0621246983,
        std_coef_deep: -0.0146921636,
    },
    ProbcutParams {
        mean_intercept: -0.7389064123,
        mean_coef_shallow: 0.1020252881,
        mean_coef_deep: 0.0520729735,
        std_intercept: -0.2020967587,
        std_coef_shallow: 0.0511338497,
        std_coef_deep: 0.0002693368,
    },
    ProbcutParams {
        mean_intercept: -0.2029417267,
        mean_coef_shallow: 0.1565606038,
        mean_coef_deep: -0.0380119061,
        std_intercept: 0.1631328675,
        std_coef_shallow: 0.0448051115,
        std_coef_deep: -0.0179651179,
    },
    ProbcutParams {
        mean_intercept: -0.4922076804,
        mean_coef_shallow: -0.0654561296,
        mean_coef_deep: 0.1262066381,
        std_intercept: 0.2512649361,
        std_coef_shallow: 0.0021326187,
        std_coef_deep: -0.0040067834,
    },
    ProbcutParams {
        mean_intercept: 0.7531448451,
        mean_coef_shallow: -0.0284427561,
        mean_coef_deep: -0.1003950504,
        std_intercept: 0.4253599051,
        std_coef_shallow: -0.0212478251,
        std_coef_deep: -0.0075299931,
    },
    ProbcutParams {
        mean_intercept: -0.2536586957,
        mean_coef_shallow: -0.0262436017,
        mean_coef_deep: 0.1016402884,
        std_intercept: 0.2989477844,
        std_coef_shallow: -0.0077333021,
        std_coef_deep: 0.0029390034,
    },
    ProbcutParams {
        mean_intercept: 0.3488137845,
        mean_coef_shallow: -0.0712677669,
        mean_coef_deep: -0.0487435861,
        std_intercept: 0.5092736633,
        std_coef_shallow: -0.0571356812,
        std_coef_deep: 0.0075694336,
    },
    ProbcutParams {
        mean_intercept: 0.6718457038,
        mean_coef_shallow: 0.0138769908,
        mean_coef_deep: -0.0607823254,
        std_intercept: 0.5318224602,
        std_coef_shallow: -0.0712404077,
        std_coef_deep: 0.0194428874,
    },
    ProbcutParams {
        mean_intercept: -0.1233338409,
        mean_coef_shallow: -0.1054405425,
        mean_coef_deep: 0.0553530849,
        std_intercept: 0.5940814836,
        std_coef_shallow: -0.0780561047,
        std_coef_deep: 0.0182218768,
    },
    ProbcutParams {
        mean_intercept: 0.4070830376,
        mean_coef_shallow: 0.0004556560,
        mean_coef_deep: -0.0504489491,
        std_intercept: 0.7243220452,
        std_coef_shallow: -0.0774627360,
        std_coef_deep: 0.0092269544,
    },
    ProbcutParams {
        mean_intercept: -0.4732067189,
        mean_coef_shallow: 0.0668531534,
        mean_coef_deep: 0.0635252688,
        std_intercept: 0.7056579541,
        std_coef_shallow: -0.0761103291,
        std_coef_deep: 0.0109932698,
    },
    ProbcutParams {
        mean_intercept: 0.3790426685,
        mean_coef_shallow: -0.1564679263,
        mean_coef_deep: -0.0140911662,
        std_intercept: 0.7475985444,
        std_coef_shallow: -0.0758640200,
        std_coef_deep: 0.0095110163,
    },
    ProbcutParams {
        mean_intercept: -0.0110214228,
        mean_coef_shallow: 0.1069885409,
        mean_coef_deep: -0.0311225910,
        std_intercept: 0.7502933051,
        std_coef_shallow: -0.0734175552,
        std_coef_deep: 0.0128951778,
    },
    ProbcutParams {
        mean_intercept: 0.0001303134,
        mean_coef_shallow: -0.0352919769,
        mean_coef_deep: -0.0010193227,
        std_intercept: 0.7448781295,
        std_coef_shallow: -0.0796626514,
        std_coef_deep: 0.0174350935,
    },
    ProbcutParams {
        mean_intercept: 0.2978142297,
        mean_coef_shallow: 0.0125989924,
        mean_coef_deep: -0.0717281200,
        std_intercept: 0.7872152769,
        std_coef_shallow: -0.0750942662,
        std_coef_deep: 0.0164932594,
    },
    ProbcutParams {
        mean_intercept: -0.4824110722,
        mean_coef_shallow: 0.0521994306,
        mean_coef_deep: 0.0834721565,
        std_intercept: 0.8070561741,
        std_coef_shallow: -0.0796746878,
        std_coef_deep: 0.0155859494,
    },
    ProbcutParams {
        mean_intercept: 0.3681895349,
        mean_coef_shallow: -0.0333391070,
        mean_coef_deep: -0.0821715702,
        std_intercept: 0.8020001539,
        std_coef_shallow: -0.0822382782,
        std_coef_deep: 0.0168987666,
    },
    ProbcutParams {
        mean_intercept: -0.1638501158,
        mean_coef_shallow: 0.1049460715,
        mean_coef_deep: 0.0042494901,
        std_intercept: 0.7658943104,
        std_coef_shallow: -0.0758940604,
        std_coef_deep: 0.0221614815,
    },
    ProbcutParams {
        mean_intercept: -0.2138663588,
        mean_coef_shallow: 0.0367880101,
        mean_coef_deep: -0.0042524467,
        std_intercept: 0.7608986777,
        std_coef_shallow: -0.0797184032,
        std_coef_deep: 0.0228694408,
    },
    ProbcutParams {
        mean_intercept: 0.2998459558,
        mean_coef_shallow: 0.0506368065,
        mean_coef_deep: -0.0827441486,
        std_intercept: 0.7428841600,
        std_coef_shallow: -0.0796801960,
        std_coef_deep: 0.0272388311,
    },
    ProbcutParams {
        mean_intercept: -0.6536361734,
        mean_coef_shallow: 0.0945797893,
        mean_coef_deep: 0.0745862657,
        std_intercept: 0.7347126033,
        std_coef_shallow: -0.0681197641,
        std_coef_deep: 0.0274590942,
    },
    ProbcutParams {
        mean_intercept: 0.1697234102,
        mean_coef_shallow: 0.0082789369,
        mean_coef_deep: -0.0601159961,
        std_intercept: 0.7569151642,
        std_coef_shallow: -0.0622665104,
        std_coef_deep: 0.0253120276,
    },
    ProbcutParams {
        mean_intercept: -0.4107294239,
        mean_coef_shallow: 0.0753778072,
        mean_coef_deep: 0.0693817838,
        std_intercept: 0.8160664803,
        std_coef_shallow: -0.0637042744,
        std_coef_deep: 0.0218491237,
    },
    ProbcutParams {
        mean_intercept: 0.0285234997,
        mean_coef_shallow: -0.0160578656,
        mean_coef_deep: -0.0370000873,
        std_intercept: 0.8952495095,
        std_coef_shallow: -0.0689793583,
        std_coef_deep: 0.0173168103,
    },
    ProbcutParams {
        mean_intercept: 0.0492443922,
        mean_coef_shallow: 0.0141735658,
        mean_coef_deep: -0.0107659691,
        std_intercept: 0.8723533296,
        std_coef_shallow: -0.0632797688,
        std_coef_deep: 0.0217525821,
    },
    ProbcutParams {
        mean_intercept: -0.3236144281,
        mean_coef_shallow: 0.0480457436,
        mean_coef_deep: 0.0282007672,
        std_intercept: 0.9436466557,
        std_coef_shallow: -0.0695248046,
        std_coef_deep: 0.0179538790,
    },
    ProbcutParams {
        mean_intercept: -0.0019995081,
        mean_coef_shallow: 0.0619577437,
        mean_coef_deep: -0.0448740437,
        std_intercept: 0.9677525331,
        std_coef_shallow: -0.0649935930,
        std_coef_deep: 0.0168643365,
    },
    ProbcutParams {
        mean_intercept: -0.4972564864,
        mean_coef_shallow: 0.1500818424,
        mean_coef_deep: 0.0369738117,
        std_intercept: 0.9175074510,
        std_coef_shallow: -0.0580117526,
        std_coef_deep: 0.0206003529,
    },
    ProbcutParams {
        mean_intercept: 0.0290338261,
        mean_coef_shallow: 0.0068492339,
        mean_coef_deep: -0.0263895527,
        std_intercept: 0.9553483530,
        std_coef_shallow: -0.0680519794,
        std_coef_deep: 0.0202071171,
    },
    ProbcutParams {
        mean_intercept: -0.1755063653,
        mean_coef_shallow: 0.0884875445,
        mean_coef_deep: 0.0068220859,
        std_intercept: 0.9929568802,
        std_coef_shallow: -0.0764945391,
        std_coef_deep: 0.0205171271,
    },
    ProbcutParams {
        mean_intercept: -0.2340576211,
        mean_coef_shallow: 0.0636391672,
        mean_coef_deep: -0.0009638869,
        std_intercept: 1.0093705742,
        std_coef_shallow: -0.0800159476,
        std_coef_deep: 0.0197973450,
    },
    ProbcutParams {
        mean_intercept: -0.1469088406,
        mean_coef_shallow: 0.0756880802,
        mean_coef_deep: -0.0078042729,
        std_intercept: 1.0019336223,
        std_coef_shallow: -0.0745966775,
        std_coef_deep: 0.0213573088,
    },
    ProbcutParams {
        mean_intercept: -0.4223229463,
        mean_coef_shallow: 0.1158650260,
        mean_coef_deep: 0.0344888778,
        std_intercept: 1.0044771116,
        std_coef_shallow: -0.0732524350,
        std_coef_deep: 0.0214118155,
    },
    ProbcutParams {
        mean_intercept: -0.1445522779,
        mean_coef_shallow: 0.0463664629,
        mean_coef_deep: -0.0089610126,
        std_intercept: 1.0295243178,
        std_coef_shallow: -0.0759257077,
        std_coef_deep: 0.0203384779,
    },
    ProbcutParams {
        mean_intercept: -0.3248832178,
        mean_coef_shallow: 0.0843207369,
        mean_coef_deep: 0.0313094348,
        std_intercept: 1.0232462866,
        std_coef_shallow: -0.0705632052,
        std_coef_deep: 0.0212334323,
    },
    ProbcutParams {
        mean_intercept: -0.0201020064,
        mean_coef_shallow: 0.0450259904,
        mean_coef_deep: -0.0564193670,
        std_intercept: 1.0702182167,
        std_coef_shallow: -0.0746147641,
        std_coef_deep: 0.0196974853,
    },
    ProbcutParams {
        mean_intercept: -0.1571847330,
        mean_coef_shallow: 0.0367914341,
        mean_coef_deep: 0.0254786551,
        std_intercept: 1.0894871182,
        std_coef_shallow: -0.0753374751,
        std_coef_deep: 0.0201190171,
    },
    ProbcutParams {
        mean_intercept: -0.2348856015,
        mean_coef_shallow: 0.1133349108,
        mean_coef_deep: -0.0407081191,
        std_intercept: 1.0920310948,
        std_coef_shallow: -0.0765671775,
        std_coef_deep: 0.0217884928,
    },
    ProbcutParams {
        mean_intercept: -0.0332169819,
        mean_coef_shallow: 0.0374107726,
        mean_coef_deep: -0.0184309221,
        std_intercept: 1.0910539802,
        std_coef_shallow: -0.0763084311,
        std_coef_deep: 0.0237162547,
    },
    ProbcutParams {
        mean_intercept: -0.2400223040,
        mean_coef_shallow: 0.1080700943,
        mean_coef_deep: -0.0147294034,
        std_intercept: 1.0905983066,
        std_coef_shallow: -0.0739049051,
        std_coef_deep: 0.0238137587,
    },
    ProbcutParams {
        mean_intercept: -0.1351644713,
        mean_coef_shallow: 0.0729821329,
        mean_coef_deep: -0.0084428625,
        std_intercept: 1.0665336201,
        std_coef_shallow: -0.0699420714,
        std_coef_deep: 0.0273468858,
    },
    ProbcutParams {
        mean_intercept: -0.2031205192,
        mean_coef_shallow: 0.0867849012,
        mean_coef_deep: -0.0110029832,
        std_intercept: 1.0803670081,
        std_coef_shallow: -0.0665142894,
        std_coef_deep: 0.0256921958,
    },
    ProbcutParams {
        mean_intercept: -0.1720996060,
        mean_coef_shallow: 0.0351359363,
        mean_coef_deep: 0.0139525672,
        std_intercept: 1.1068642806,
        std_coef_shallow: -0.0688112924,
        std_coef_deep: 0.0262648390,
    },
    ProbcutParams {
        mean_intercept: -0.0990136609,
        mean_coef_shallow: 0.1162143104,
        mean_coef_deep: -0.0591022634,
        std_intercept: 1.0931210943,
        std_coef_shallow: -0.0696068199,
        std_coef_deep: 0.0298613950,
    },
    ProbcutParams {
        mean_intercept: -0.2026584120,
        mean_coef_shallow: 0.0524465048,
        mean_coef_deep: 0.0164799260,
        std_intercept: 1.1310694804,
        std_coef_shallow: -0.0744926717,
        std_coef_deep: 0.0286789119,
    },
    ProbcutParams {
        mean_intercept: -0.1242565818,
        mean_coef_shallow: 0.0657807206,
        mean_coef_deep: -0.0391810423,
        std_intercept: 1.1629056048,
        std_coef_shallow: -0.0736441994,
        std_coef_deep: 0.0256883152,
    },
    ProbcutParams {
        mean_intercept: -0.0702153837,
        mean_coef_shallow: 0.0637272589,
        mean_coef_deep: -0.0233737037,
        std_intercept: 1.1840468097,
        std_coef_shallow: -0.0780415997,
        std_coef_deep: 0.0235521568,
    },
    ProbcutParams {
        mean_intercept: -0.1712261264,
        mean_coef_shallow: 0.0377612174,
        mean_coef_deep: -0.0091975608,
        std_intercept: 1.2347169868,
        std_coef_shallow: -0.0904433334,
        std_coef_deep: 0.0210887951,
    },
    ProbcutParams {
        mean_intercept: -0.0949590185,
        mean_coef_shallow: 0.0690966153,
        mean_coef_deep: -0.0235086113,
        std_intercept: 1.2628624653,
        std_coef_shallow: -0.1040452832,
        std_coef_deep: 0.0176581344,
    },
    ProbcutParams {
        mean_intercept: -0.1965153416,
        mean_coef_shallow: 0.0465310459,
        mean_coef_deep: 0.0073884765,
        std_intercept: 1.3124366810,
        std_coef_shallow: -0.1349296855,
        std_coef_deep: 0.0142072178,
    },
    ProbcutParams {
        mean_intercept: -0.0612994614,
        mean_coef_shallow: -0.0056009642,
        mean_coef_deep: -0.0096077661,
        std_intercept: 1.3519937640,
        std_coef_shallow: -0.1765126020,
        std_coef_deep: 0.0103406089,
    },
    ProbcutParams {
        mean_intercept: -0.2723106937,
        mean_coef_shallow: 0.0196946548,
        mean_coef_deep: 0.0231016538,
        std_intercept: 1.3878932129,
        std_coef_shallow: -0.2368920962,
        std_coef_deep: 0.0065971336,
    },
    ProbcutParams {
        mean_intercept: -0.2010281822,
        mean_coef_shallow: 0.0257595432,
        mean_coef_deep: -0.0142589213,
        std_intercept: 1.7463583152,
        std_coef_shallow: -0.5801185655,
        std_coef_deep: 0.0103062701,
    },
    ProbcutParams {
        mean_intercept: -0.0566970849,
        mean_coef_shallow: -0.0195703316,
        mean_coef_deep: 0.0131500974,
        std_intercept: 1.2616323167,
        std_coef_shallow: -0.4382175705,
        std_coef_deep: 0.0071391143,
    },
    ProbcutParams {
        mean_intercept: -0.2473265346,
        mean_coef_shallow: 0.0000528475,
        mean_coef_deep: 0.0005200912,
        std_intercept: 1.3779491542,
        std_coef_shallow: -0.7714412362,
        std_coef_deep: 0.0091022066,
    },
    ProbcutParams {
        mean_intercept: 0.2242282155,
        mean_coef_shallow: 0.0011567846,
        mean_coef_deep: -0.0012168419,
        std_intercept: 0.5790585508,
        std_coef_shallow: -0.6338765301,
        std_coef_deep: 0.0000032872,
    },
    ProbcutParams {
        mean_intercept: -0.6257328397,
        mean_coef_shallow: -0.0054106968,
        mean_coef_deep: 0.0066578710,
        std_intercept: -0.3532976660,
        std_coef_shallow: -0.3838596862,
        std_coef_deep: -0.0000100560,
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
