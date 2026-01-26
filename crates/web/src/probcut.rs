use std::sync::OnceLock;

use reversi_core::{
    board::Board,
    probcut::Selectivity,
    search::node_type::NonPV,
    types::{Depth, ScaledScore},
};

use crate::search::{self, search_context::SearchContext};

/// Statistical parameters for ProbCut prediction models
/// - `depth_gap = max(deep_depth - shallow_depth, 0)`
/// - `depth_gap_is_even = 1` when `depth_gap` is even, `0` otherwise
/// - `depth_feat = 1 / (1 + ln(1 + depth_gap))`
/// - `mean = mean_intercept + mean_coef_shallow * shallow_depth + mean_coef_deep * deep_depth + mean_coef_depth_diff * depth_feat + mean_coef_depth_gap_is_even * depth_gap_is_even`
/// - `sigma = exp(std_intercept + std_coef_shallow * shallow_depth + std_coef_deep * deep_depth + std_coef_depth_diff * depth_feat + std_coef_depth_gap_is_even * depth_gap_is_even)`
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

/// Safely allocate a 3D table on the heap to avoid stack overflow
fn alloc_3d_table() -> Box<MeanTable> {
    let tbl = vec![[0.0f64; MAX_DEPTH]; MAX_PLY * MAX_DEPTH].into_boxed_slice();
    unsafe { Box::from_raw(Box::into_raw(tbl) as *mut MeanTable) }
}

/// Build the pre-computed mean table for midgame positions
fn build_mean_table() -> Box<MeanTable> {
    let mut tbl = alloc_3d_table();

    for ply in 0..MAX_PLY {
        let params = &PROBCUT_PARAMS[ply];
        for shallow in 0..MAX_DEPTH {
            for deep in shallow..MAX_DEPTH {
                let v = params.mean(shallow as f64, deep as f64) * SCORE_SCALE_F64;
                tbl[ply][shallow][deep] = v;
                tbl[ply][deep][shallow] = v; // Symmetric: mean(a,b) = mean(b,a)
            }
        }
    }
    tbl
}

/// Build the pre-computed sigma table for midgame positions
fn build_sigma_table() -> Box<SigmaTable> {
    let mut tbl = unsafe { Box::from_raw(Box::into_raw(alloc_3d_table()) as *mut SigmaTable) };

    for ply in 0..MAX_PLY {
        let params = &PROBCUT_PARAMS[ply];
        for shallow in 0..MAX_DEPTH {
            for deep in shallow..MAX_DEPTH {
                let v = params.sigma(shallow as f64, deep as f64) * SCORE_SCALE_F64;
                tbl[ply][shallow][deep] = v;
                tbl[ply][deep][shallow] = v; // Symmetric: sigma(a,b) = sigma(b,a)
            }
        }
    }
    tbl
}

/// Initialize probcut tables. Called from lib.rs init().
pub fn init() {
    MEAN_TABLE.set(build_mean_table()).ok();
    SIGMA_TABLE.set(build_sigma_table()).ok();
}

/// Fast lookup of pre-computed mean value for midgame positions
#[inline]
fn calc_mean(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    let tbl = MEAN_TABLE.get().expect("probcut not initialized");
    tbl[ply][shallow as usize][deep as usize]
}

/// Fast lookup of pre-computed sigma value for midgame positions
#[inline]
fn calc_sigma(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    let tbl = SIGMA_TABLE.get().expect("probcut not initialized");
    tbl[ply][shallow as usize][deep as usize]
}

/// Returns the pre-computed sigma value for midgame positions.
/// Public API for Score-Based Reduction in move ordering.
#[inline]
pub fn get_sigma(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    calc_sigma(ply, shallow, deep)
}

/// Determines the depth of the shallow search in probcut.
///
/// # Arguments
///
/// * `depth` - The depth of the deep search.
///
/// # Returns
///
/// The depth of the shallow search.
fn determine_probcut_depth(depth: Depth) -> Depth {
    let mut probcut_depth = 2 * (depth as f64 * 0.2).floor() as Depth + (depth & 1);
    if probcut_depth == 0 {
        probcut_depth = depth - 2;
    }
    probcut_depth
}

/// Attempts ProbCut pruning for midgame positions
///
/// # Arguments
///
/// * `ctx` - Search context containing selectivity settings and search state
/// * `board` - Current board position to evaluate
/// * `depth` - Depth of the deep search that would be performed
/// * `alpha` - Alpha bound for the search window
/// * `beta` - Beta bound for the search window
///
/// # Returns
///
/// * `Some(score)` - If probcut triggers, returns the predicted bound (alpha or beta)
/// * `None` - If probcut doesn't trigger, deep search should be performed
pub fn probcut_midgame(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    beta: ScaledScore,
) -> Option<ScaledScore> {
    if depth >= 3 && ctx.selectivity.is_enabled() {
        let ply = ctx.ply();
        let pc_depth = determine_probcut_depth(depth);
        let mean = calc_mean(ply, pc_depth, depth);
        let sigma = calc_sigma(ply, pc_depth, depth);
        let t = ctx.selectivity.t_value();

        let eval_score = search::evaluate(ctx, board);
        let eval_mean = 0.5 * calc_mean(ply, 0, depth) + mean;
        let eval_sigma = t * 0.5 * calc_sigma(ply, 0, depth) + sigma;

        let beta_raw = beta.value() as f64;
        let eval_beta = ScaledScore::from_raw((beta_raw - eval_sigma - eval_mean).floor() as i32);
        let pc_beta = ScaledScore::from_raw((beta_raw + t * sigma - mean).ceil() as i32);
        if eval_score >= eval_beta && pc_beta < ScaledScore::MAX {
            let current_selectivity = ctx.selectivity;
            ctx.selectivity = Selectivity::None; // Disable nested ProbCut
            let score = search::search::<NonPV>(ctx, board, pc_depth, pc_beta - 1, pc_beta, false);
            ctx.selectivity = current_selectivity; // Restore selectivity
            if score >= pc_beta {
                return Some((beta + pc_beta) / 2);
            }
        }
    }
    None
}

/// Statistical parameters for midgame ProbCut indexed by ply
#[rustfmt::skip]
const PROBCUT_PARAMS: [ProbcutParams; 60] = [
    ProbcutParams {
        mean_intercept: 0.0000000000,
        mean_coef_shallow: 0.0000000000,
        mean_coef_deep: 0.0000000000,
        std_intercept: -18.4206807440,
        std_coef_shallow: 0.0000000000,
        std_coef_deep: 0.0000000000,
    },
    ProbcutParams {
        mean_intercept: 2.3458462861,
        mean_coef_shallow: -0.2056347168,
        mean_coef_deep: -0.1850982377,
        std_intercept: -0.5475629692,
        std_coef_shallow: 0.0333804089,
        std_coef_deep: -0.0024308354,
    },
    ProbcutParams {
        mean_intercept: -0.1046574542,
        mean_coef_shallow: 0.0979886617,
        mean_coef_deep: 0.0295504781,
        std_intercept: 0.4819324831,
        std_coef_shallow: 0.0404812313,
        std_coef_deep: -0.0942682807,
    },
    ProbcutParams {
        mean_intercept: 0.5131156279,
        mean_coef_shallow: -0.3452073957,
        mean_coef_deep: -0.0282734497,
        std_intercept: 0.7368566186,
        std_coef_shallow: 0.2065592040,
        std_coef_deep: -0.1027559149,
    },
    ProbcutParams {
        mean_intercept: -0.0251044126,
        mean_coef_shallow: 0.0769004979,
        mean_coef_deep: 0.0386906155,
        std_intercept: 0.5941880813,
        std_coef_shallow: 0.1161838012,
        std_coef_deep: -0.0519974117,
    },
    ProbcutParams {
        mean_intercept: -0.2071864492,
        mean_coef_shallow: -0.0010763200,
        mean_coef_deep: 0.0318614556,
        std_intercept: 0.9299862041,
        std_coef_shallow: -0.0835986603,
        std_coef_deep: -0.0072323888,
    },
    ProbcutParams {
        mean_intercept: 0.3701683033,
        mean_coef_shallow: -0.0795695281,
        mean_coef_deep: -0.0314234342,
        std_intercept: 1.0403523275,
        std_coef_shallow: -0.1633379645,
        std_coef_deep: -0.0037554133,
    },
    ProbcutParams {
        mean_intercept: -0.2662708339,
        mean_coef_shallow: 0.0427375995,
        mean_coef_deep: 0.0453387148,
        std_intercept: 1.0783987418,
        std_coef_shallow: -0.2016821916,
        std_coef_deep: 0.0081896561,
    },
    ProbcutParams {
        mean_intercept: 0.0729503230,
        mean_coef_shallow: 0.0045077481,
        mean_coef_deep: -0.0174119466,
        std_intercept: 0.7057034304,
        std_coef_shallow: -0.1063439378,
        std_coef_deep: 0.0124109100,
    },
    ProbcutParams {
        mean_intercept: 0.2253785165,
        mean_coef_shallow: -0.0328915936,
        mean_coef_deep: -0.0034751781,
        std_intercept: 0.7021788037,
        std_coef_shallow: -0.0960517501,
        std_coef_deep: 0.0118447172,
    },
    ProbcutParams {
        mean_intercept: 0.1159196346,
        mean_coef_shallow: -0.0063262941,
        mean_coef_deep: -0.0155573243,
        std_intercept: 0.7196196783,
        std_coef_shallow: -0.1072802095,
        std_coef_deep: 0.0200966729,
    },
    ProbcutParams {
        mean_intercept: 0.3204243531,
        mean_coef_shallow: -0.0369217868,
        mean_coef_deep: -0.0534013607,
        std_intercept: 0.7846594027,
        std_coef_shallow: -0.1175908681,
        std_coef_deep: 0.0237065390,
    },
    ProbcutParams {
        mean_intercept: 0.1685202774,
        mean_coef_shallow: -0.0200381874,
        mean_coef_deep: -0.0288440974,
        std_intercept: 0.8451459913,
        std_coef_shallow: -0.1213453796,
        std_coef_deep: 0.0205993702,
    },
    ProbcutParams {
        mean_intercept: 0.1460933473,
        mean_coef_shallow: -0.0386555622,
        mean_coef_deep: -0.0173341695,
        std_intercept: 0.8636606655,
        std_coef_shallow: -0.1125530014,
        std_coef_deep: 0.0215770369,
    },
    ProbcutParams {
        mean_intercept: 0.1623310339,
        mean_coef_shallow: -0.0183479194,
        mean_coef_deep: -0.0371680778,
        std_intercept: 0.8147895949,
        std_coef_shallow: -0.1085855193,
        std_coef_deep: 0.0258094629,
    },
    ProbcutParams {
        mean_intercept: 0.1718437763,
        mean_coef_shallow: -0.0041450121,
        mean_coef_deep: -0.0502570963,
        std_intercept: 0.8243335293,
        std_coef_shallow: -0.1082124902,
        std_coef_deep: 0.0256923621,
    },
    ProbcutParams {
        mean_intercept: 0.0881333059,
        mean_coef_shallow: 0.0233603329,
        mean_coef_deep: -0.0484342638,
        std_intercept: 0.8749468322,
        std_coef_shallow: -0.1180242618,
        std_coef_deep: 0.0269471781,
    },
    ProbcutParams {
        mean_intercept: 0.1380234161,
        mean_coef_shallow: 0.0127809576,
        mean_coef_deep: -0.0640640589,
        std_intercept: 0.9132225323,
        std_coef_shallow: -0.1212018044,
        std_coef_deep: 0.0254443995,
    },
    ProbcutParams {
        mean_intercept: 0.1486190513,
        mean_coef_shallow: 0.0318696909,
        mean_coef_deep: -0.0698232761,
        std_intercept: 0.9505107521,
        std_coef_shallow: -0.1162802977,
        std_coef_deep: 0.0197864151,
    },
    ProbcutParams {
        mean_intercept: 0.0734488488,
        mean_coef_shallow: 0.0427543007,
        mean_coef_deep: -0.0625484152,
        std_intercept: 0.9537308515,
        std_coef_shallow: -0.1005206458,
        std_coef_deep: 0.0160962477,
    },
    ProbcutParams {
        mean_intercept: 0.1409605034,
        mean_coef_shallow: 0.0353450492,
        mean_coef_deep: -0.0908795074,
        std_intercept: 0.9131889015,
        std_coef_shallow: -0.0899087783,
        std_coef_deep: 0.0192887682,
    },
    ProbcutParams {
        mean_intercept: 0.1080983822,
        mean_coef_shallow: 0.0390953581,
        mean_coef_deep: -0.0718482378,
        std_intercept: 0.9348944243,
        std_coef_shallow: -0.0922728179,
        std_coef_deep: 0.0186010301,
    },
    ProbcutParams {
        mean_intercept: 0.1542860485,
        mean_coef_shallow: 0.0419742916,
        mean_coef_deep: -0.1033791540,
        std_intercept: 0.9939713268,
        std_coef_shallow: -0.1062715913,
        std_coef_deep: 0.0184952088,
    },
    ProbcutParams {
        mean_intercept: 0.1483806364,
        mean_coef_shallow: 0.0191365790,
        mean_coef_deep: -0.0758482771,
        std_intercept: 0.9589670815,
        std_coef_shallow: -0.0961835187,
        std_coef_deep: 0.0215419989,
    },
    ProbcutParams {
        mean_intercept: 0.1218474431,
        mean_coef_shallow: 0.0419590190,
        mean_coef_deep: -0.0916105546,
        std_intercept: 0.9472251527,
        std_coef_shallow: -0.0852486842,
        std_coef_deep: 0.0223676935,
    },
    ProbcutParams {
        mean_intercept: 0.1475662015,
        mean_coef_shallow: 0.0282871833,
        mean_coef_deep: -0.0948232634,
        std_intercept: 0.9924984125,
        std_coef_shallow: -0.0880140943,
        std_coef_deep: 0.0213475632,
    },
    ProbcutParams {
        mean_intercept: 0.1344057743,
        mean_coef_shallow: 0.0078434825,
        mean_coef_deep: -0.0811336967,
        std_intercept: 0.9998867946,
        std_coef_shallow: -0.0893627228,
        std_coef_deep: 0.0253524696,
    },
    ProbcutParams {
        mean_intercept: 0.1694717594,
        mean_coef_shallow: 0.0064285525,
        mean_coef_deep: -0.0978881592,
        std_intercept: 1.0480125646,
        std_coef_shallow: -0.0886401072,
        std_coef_deep: 0.0228741786,
    },
    ProbcutParams {
        mean_intercept: 0.1082489288,
        mean_coef_shallow: 0.0241593736,
        mean_coef_deep: -0.0793727794,
        std_intercept: 1.0700148127,
        std_coef_shallow: -0.0935755424,
        std_coef_deep: 0.0207942030,
    },
    ProbcutParams {
        mean_intercept: 0.1830767233,
        mean_coef_shallow: -0.0072030007,
        mean_coef_deep: -0.0955457575,
        std_intercept: 1.0645777757,
        std_coef_shallow: -0.0885169698,
        std_coef_deep: 0.0232679658,
    },
    ProbcutParams {
        mean_intercept: 0.1579927788,
        mean_coef_shallow: -0.0222806776,
        mean_coef_deep: -0.0784708564,
        std_intercept: 1.0529370078,
        std_coef_shallow: -0.0865638277,
        std_coef_deep: 0.0253383634,
    },
    ProbcutParams {
        mean_intercept: 0.1757299911,
        mean_coef_shallow: 0.0224555314,
        mean_coef_deep: -0.1134883689,
        std_intercept: 1.0682212025,
        std_coef_shallow: -0.0834714263,
        std_coef_deep: 0.0254614361,
    },
    ProbcutParams {
        mean_intercept: 0.1259124871,
        mean_coef_shallow: -0.0255788720,
        mean_coef_deep: -0.0548153949,
        std_intercept: 1.0994061368,
        std_coef_shallow: -0.0840677210,
        std_coef_deep: 0.0234650318,
    },
    ProbcutParams {
        mean_intercept: 0.1988967648,
        mean_coef_shallow: 0.0060617730,
        mean_coef_deep: -0.1244524314,
        std_intercept: 1.1166910726,
        std_coef_shallow: -0.0825311795,
        std_coef_deep: 0.0230932391,
    },
    ProbcutParams {
        mean_intercept: 0.1260170568,
        mean_coef_shallow: -0.0014348362,
        mean_coef_deep: -0.0645415860,
        std_intercept: 1.1363411958,
        std_coef_shallow: -0.0857239383,
        std_coef_deep: 0.0249978940,
    },
    ProbcutParams {
        mean_intercept: 0.1550478828,
        mean_coef_shallow: -0.0048896276,
        mean_coef_deep: -0.1041724311,
        std_intercept: 1.1770710313,
        std_coef_shallow: -0.0995175839,
        std_coef_deep: 0.0247535715,
    },
    ProbcutParams {
        mean_intercept: 0.1742439121,
        mean_coef_shallow: -0.0000871614,
        mean_coef_deep: -0.0953768794,
        std_intercept: 1.1817761317,
        std_coef_shallow: -0.0982475224,
        std_coef_deep: 0.0238425048,
    },
    ProbcutParams {
        mean_intercept: 0.1680079822,
        mean_coef_shallow: 0.0472980750,
        mean_coef_deep: -0.1348361243,
        std_intercept: 1.2042373876,
        std_coef_shallow: -0.1073101046,
        std_coef_deep: 0.0274605999,
    },
    ProbcutParams {
        mean_intercept: 0.1338795327,
        mean_coef_shallow: -0.0052276686,
        mean_coef_deep: -0.0854534709,
        std_intercept: 1.1932605489,
        std_coef_shallow: -0.1009875169,
        std_coef_deep: 0.0278184908,
    },
    ProbcutParams {
        mean_intercept: 0.1669079519,
        mean_coef_shallow: 0.0289193709,
        mean_coef_deep: -0.1290900536,
        std_intercept: 1.1885239407,
        std_coef_shallow: -0.0998333792,
        std_coef_deep: 0.0312195915,
    },
    ProbcutParams {
        mean_intercept: 0.1621461475,
        mean_coef_shallow: -0.0159302831,
        mean_coef_deep: -0.0762802764,
        std_intercept: 1.1784064280,
        std_coef_shallow: -0.0976573295,
        std_coef_deep: 0.0308698488,
    },
    ProbcutParams {
        mean_intercept: 0.1262683996,
        mean_coef_shallow: 0.0521302621,
        mean_coef_deep: -0.1297483066,
        std_intercept: 1.1881394550,
        std_coef_shallow: -0.0946440555,
        std_coef_deep: 0.0277891807,
    },
    ProbcutParams {
        mean_intercept: 0.1796904932,
        mean_coef_shallow: -0.0562822999,
        mean_coef_deep: -0.0690623338,
        std_intercept: 1.2013978680,
        std_coef_shallow: -0.0907086253,
        std_coef_deep: 0.0267449685,
    },
    ProbcutParams {
        mean_intercept: 0.1069457315,
        mean_coef_shallow: 0.0303443015,
        mean_coef_deep: -0.1056342134,
        std_intercept: 1.1829444545,
        std_coef_shallow: -0.0855205847,
        std_coef_deep: 0.0297477644,
    },
    ProbcutParams {
        mean_intercept: 0.1489576586,
        mean_coef_shallow: -0.0301539409,
        mean_coef_deep: -0.0760812284,
        std_intercept: 1.2121213079,
        std_coef_shallow: -0.0930920218,
        std_coef_deep: 0.0321537803,
    },
    ProbcutParams {
        mean_intercept: 0.0689186461,
        mean_coef_shallow: 0.0334790999,
        mean_coef_deep: -0.0935860799,
        std_intercept: 1.2100984012,
        std_coef_shallow: -0.0974407845,
        std_coef_deep: 0.0343774440,
    },
    ProbcutParams {
        mean_intercept: 0.1394405288,
        mean_coef_shallow: -0.0189250174,
        mean_coef_deep: -0.0727633887,
        std_intercept: 1.1798812877,
        std_coef_shallow: -0.0936049525,
        std_coef_deep: 0.0383747470,
    },
    ProbcutParams {
        mean_intercept: 0.1067721578,
        mean_coef_shallow: 0.0036149585,
        mean_coef_deep: -0.1074780062,
        std_intercept: 1.1795156425,
        std_coef_shallow: -0.0887815349,
        std_coef_deep: 0.0389860450,
    },
    ProbcutParams {
        mean_intercept: 0.1724280463,
        mean_coef_shallow: -0.0370159456,
        mean_coef_deep: -0.0801168453,
        std_intercept: 1.1515923018,
        std_coef_shallow: -0.0831548216,
        std_coef_deep: 0.0419662442,
    },
    ProbcutParams {
        mean_intercept: -0.0014557883,
        mean_coef_shallow: 0.0048859109,
        mean_coef_deep: -0.0543642692,
        std_intercept: 1.1939501873,
        std_coef_shallow: -0.0907574421,
        std_coef_deep: 0.0412328862,
    },
    ProbcutParams {
        mean_intercept: 0.1585007953,
        mean_coef_shallow: -0.0114611991,
        mean_coef_deep: -0.1086802758,
        std_intercept: 1.2075247912,
        std_coef_shallow: -0.0950131574,
        std_coef_deep: 0.0398714397,
    },
    ProbcutParams {
        mean_intercept: 0.0012806090,
        mean_coef_shallow: -0.0010977377,
        mean_coef_deep: -0.0280602172,
        std_intercept: 1.2290594962,
        std_coef_shallow: -0.0944178495,
        std_coef_deep: 0.0355346294,
    },
    ProbcutParams {
        mean_intercept: 0.0553657320,
        mean_coef_shallow: -0.0068901945,
        mean_coef_deep: -0.0673891653,
        std_intercept: 1.2731917451,
        std_coef_shallow: -0.1062584460,
        std_coef_deep: 0.0264332107,
    },
    ProbcutParams {
        mean_intercept: -0.0174034262,
        mean_coef_shallow: 0.0140745685,
        mean_coef_deep: -0.0159889193,
        std_intercept: 1.3251712814,
        std_coef_shallow: -0.1390963915,
        std_coef_deep: 0.0198635335,
    },
    ProbcutParams {
        mean_intercept: -0.0769854189,
        mean_coef_shallow: 0.0472603686,
        mean_coef_deep: -0.0286997258,
        std_intercept: 1.3779856437,
        std_coef_shallow: -0.2135626900,
        std_coef_deep: 0.0106994925,
    },
    ProbcutParams {
        mean_intercept: -0.0657746963,
        mean_coef_shallow: 0.0378135234,
        mean_coef_deep: -0.0185572779,
        std_intercept: 1.5647518695,
        std_coef_shallow: -0.4647020828,
        std_coef_deep: 0.0058562981,
    },
    ProbcutParams {
        mean_intercept: -0.0632612908,
        mean_coef_shallow: -0.0006391944,
        mean_coef_deep: 0.0010953133,
        std_intercept: 1.6797884138,
        std_coef_shallow: -0.8287932078,
        std_coef_deep: 0.0022270984,
    },
    ProbcutParams {
        mean_intercept: -0.0723921488,
        mean_coef_shallow: -0.0009036488,
        mean_coef_deep: 0.0008430712,
        std_intercept: 1.2760853128,
        std_coef_shallow: -0.9352153702,
        std_coef_deep: -0.0000149210,
    },
    ProbcutParams {
        mean_intercept: 0.0776954322,
        mean_coef_shallow: 0.0009506678,
        mean_coef_deep: -0.0006548215,
        std_intercept: 0.3748467216,
        std_coef_shallow: -0.9613216209,
        std_coef_deep: 0.0000989280,
    },
    ProbcutParams {
        mean_intercept: 0.0850793386,
        mean_coef_shallow: 0.0002343356,
        mean_coef_deep: -0.0001563062,
        std_intercept: -1.6314987979,
        std_coef_shallow: -0.8912326478,
        std_coef_deep: 0.0001295672,
    },
];
