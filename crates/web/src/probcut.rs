use std::sync::OnceLock;

use reversi_core::{
    board::Board,
    probcut::Selectivity,
    search::node_type::NonPV,
    types::{Depth, ScaledScore},
};

use crate::search::{self, search_context::SearchContext, search_strategy::MidGameStrategy};

/// Holds statistical parameters for ProbCut prediction models.
///
/// - `mean = mean_intercept + mean_coef_shallow * shallow_depth + mean_coef_deep * deep_depth`
/// - `sigma = exp(std_intercept + std_coef_shallow * shallow_depth + std_coef_deep * deep_depth)`
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
type EndTable = [[f64; MAX_DEPTH]; MAX_DEPTH];

const SCORE_SCALE_F64: f64 = ScaledScore::SCALE as f64;

static MEAN_TABLE: OnceLock<Box<MeanTable>> = OnceLock::new();
static SIGMA_TABLE: OnceLock<Box<SigmaTable>> = OnceLock::new();
static MEAN_TABLE_END: OnceLock<EndTable> = OnceLock::new();
static SIGMA_TABLE_END: OnceLock<EndTable> = OnceLock::new();

/// Allocates a zeroed 3D table on the heap to avoid stack overflow.
fn alloc_3d_table() -> Box<MeanTable> {
    let tbl = vec![[0.0f64; MAX_DEPTH]; MAX_PLY * MAX_DEPTH].into_boxed_slice();
    unsafe { Box::from_raw(Box::into_raw(tbl) as *mut MeanTable) }
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
                tbl[ply][deep][shallow] = v; // Symmetric: mean(a,b) = mean(b,a)
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
                tbl[ply][deep][shallow] = v; // Symmetric: sigma(a,b) = sigma(b,a)
            }
        }
    }
    tbl
}

/// Builds a 2D `[shallow][deep]` table from the endgame ProbCut parameters.
///
/// Only populates entries where `shallow <= deep` (callers always satisfy this).
fn build_end_table(f: impl Fn(&ProbcutParams, f64, f64) -> f64) -> EndTable {
    let mut tbl = [[0.0f64; MAX_DEPTH]; MAX_DEPTH];
    #[allow(clippy::needless_range_loop)]
    for shallow in 0..MAX_DEPTH {
        for deep in shallow..MAX_DEPTH {
            tbl[shallow][deep] =
                f(&PROBCUT_ENDGAME_PARAMS, shallow as f64, deep as f64) * SCORE_SCALE_F64;
        }
    }
    tbl
}

/// Initializes the ProbCut lookup tables.
pub fn init() {
    MEAN_TABLE.set(build_mean_table()).ok();
    SIGMA_TABLE.set(build_sigma_table()).ok();
    MEAN_TABLE_END
        .set(build_end_table(ProbcutParams::mean))
        .ok();
    SIGMA_TABLE_END
        .set(build_end_table(ProbcutParams::sigma))
        .ok();
}

/// Returns the pre-computed mean value for midgame positions.
#[inline]
fn calc_mean(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    let tbl = MEAN_TABLE.get().expect("probcut not initialized");
    tbl[ply][shallow as usize][deep as usize]
}

/// Returns the pre-computed sigma value for midgame positions.
#[inline]
fn calc_sigma(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    let tbl = SIGMA_TABLE.get().expect("probcut not initialized");
    tbl[ply][shallow as usize][deep as usize]
}

/// Returns the pre-computed mean value for endgame positions.
#[inline]
fn calc_mean_end(shallow: Depth, deep: Depth) -> f64 {
    let tbl = MEAN_TABLE_END.get().expect("probcut not initialized");
    tbl[shallow as usize][deep as usize]
}

/// Returns the pre-computed sigma value for endgame positions.
#[inline]
fn calc_sigma_end(shallow: Depth, deep: Depth) -> f64 {
    let tbl = SIGMA_TABLE_END.get().expect("probcut not initialized");
    tbl[shallow as usize][deep as usize]
}

/// Determines the shallow search depth for ProbCut from the given deep search depth.
fn determine_probcut_depth(depth: Depth) -> Depth {
    let mut probcut_depth = 2 * (depth as f64 * 0.2).floor() as Depth + (depth & 1);
    if probcut_depth == 0 {
        probcut_depth = depth - 2;
    }
    probcut_depth
}

/// Attempts ProbCut pruning for midgame positions.
///
/// Returns [`Some`] with the cutoff score if a beta cutoff is predicted, or
/// [`None`] if the deep search should proceed.
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
            let score = search::search::<NonPV, MidGameStrategy>(
                ctx,
                board,
                pc_depth,
                pc_beta - 1,
                pc_beta,
            );
            ctx.selectivity = current_selectivity; // Restore selectivity
            if score >= pc_beta {
                return Some((beta + pc_beta) / 2);
            }
        }
    }
    None
}

/// Attempts ProbCut pruning for endgame positions.
///
/// Mirrors `reversi_core::search::endgame::try_probcut`: it predicts the deep
/// (exact) endgame result from a fixed-depth shallow search using the dedicated
/// endgame statistical model, then verifies with a null-window shallow search.
///
/// Returns [`Some`] with the cutoff score (`beta`) if a beta cutoff is predicted,
/// or [`None`] if the deep search should proceed.
pub fn probcut_endgame(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    beta: ScaledScore,
) -> Option<ScaledScore> {
    if !ctx.selectivity.is_enabled() {
        return None;
    }

    const PC_DEPTH: Depth = 2;
    let mean = calc_mean_end(PC_DEPTH, depth);
    let sigma = calc_sigma_end(PC_DEPTH, depth);
    let t = ctx.selectivity.t_value();

    let beta_raw = beta.value() as f64;
    let pc_beta = ScaledScore::from_raw((beta_raw + t * sigma - mean).ceil() as i32);
    if pc_beta >= ScaledScore::MAX {
        return None;
    }

    let eval_score = search::evaluate(ctx, board);
    let eval_mean = 0.5 * calc_mean_end(0, depth) + mean;
    let eval_sigma = t * 0.5 * calc_sigma_end(0, depth) + sigma;
    let eval_beta = ScaledScore::from_raw((beta_raw - eval_sigma - eval_mean).floor() as i32);

    if eval_score >= eval_beta {
        let current_selectivity = ctx.selectivity;
        ctx.selectivity = Selectivity::None; // Disable nested ProbCut
        let score =
            search::search::<NonPV, MidGameStrategy>(ctx, board, PC_DEPTH, pc_beta - 1, pc_beta);
        ctx.selectivity = current_selectivity; // Restore selectivity
        if score >= pc_beta {
            return Some(beta);
        }
    }

    None
}

/// Statistical parameters for endgame ProbCut
const PROBCUT_ENDGAME_PARAMS: ProbcutParams = ProbcutParams {
    mean_intercept: 0.9974461987,
    mean_coef_shallow: -1.0744099285,
    mean_coef_deep: -0.0791259693,
    std_intercept: 0.8120017267,
    std_coef_shallow: 0.0000000000,
    std_coef_deep: 0.0262075611,
};

/// Statistical parameters for midgame ProbCut, indexed by ply.
#[rustfmt::skip]
const PROBCUT_PARAMS: [ProbcutParams; 60] = [
    ProbcutParams {
        mean_intercept: 0.0000000000,
        mean_coef_shallow: 0.0000000000,
        mean_coef_deep: 0.0000000000,
        std_intercept: -18.4206807440,
        std_coef_shallow: -0.0000000000,
        std_coef_deep: 0.0000000000,
    },
    ProbcutParams {
        mean_intercept: -0.9036091854,
        mean_coef_shallow: -0.2852039798,
        mean_coef_deep: 0.1096670898,
        std_intercept: 0.4341907645,
        std_coef_shallow: 0.0665272765,
        std_coef_deep: -0.1450528926,
    },
    ProbcutParams {
        mean_intercept: 2.2478028630,
        mean_coef_shallow: 0.0018656789,
        mean_coef_deep: -0.3193507889,
        std_intercept: 0.0007727933,
        std_coef_shallow: -0.0364762131,
        std_coef_deep: -0.0083169264,
    },
    ProbcutParams {
        mean_intercept: -0.0436764832,
        mean_coef_shallow: 0.0790915170,
        mean_coef_deep: -0.0017731402,
        std_intercept: 0.3785564066,
        std_coef_shallow: 0.0614701791,
        std_coef_deep: -0.0411536031,
    },
    ProbcutParams {
        mean_intercept: 0.1207798388,
        mean_coef_shallow: -0.2218283772,
        mean_coef_deep: -0.0188591969,
        std_intercept: 0.3274842606,
        std_coef_shallow: 0.0450060026,
        std_coef_deep: -0.0231925149,
    },
    ProbcutParams {
        mean_intercept: 0.6271058926,
        mean_coef_shallow: -0.1883392166,
        mean_coef_deep: -0.0443327985,
        std_intercept: 0.5348261122,
        std_coef_shallow: 0.0924159317,
        std_coef_deep: -0.0439414235,
    },
    ProbcutParams {
        mean_intercept: -0.6396160740,
        mean_coef_shallow: 0.2457577747,
        mean_coef_deep: 0.0640574256,
        std_intercept: 0.5602581357,
        std_coef_shallow: 0.0545402301,
        std_coef_deep: -0.0276023324,
    },
    ProbcutParams {
        mean_intercept: 0.0756348640,
        mean_coef_shallow: -0.2495813322,
        mean_coef_deep: 0.0637415048,
        std_intercept: 0.7078561291,
        std_coef_shallow: -0.0239648907,
        std_coef_deep: -0.0199453176,
    },
    ProbcutParams {
        mean_intercept: -0.3222360672,
        mean_coef_shallow: 0.0044267989,
        mean_coef_deep: 0.1306552199,
        std_intercept: 0.7834848280,
        std_coef_shallow: -0.0491143900,
        std_coef_deep: -0.0103018741,
    },
    ProbcutParams {
        mean_intercept: -0.2700090932,
        mean_coef_shallow: 0.0507842027,
        mean_coef_deep: 0.0544551782,
        std_intercept: 0.7239633008,
        std_coef_shallow: -0.0591291371,
        std_coef_deep: 0.0031448765,
    },
    ProbcutParams {
        mean_intercept: 0.3605966345,
        mean_coef_shallow: -0.1322084506,
        mean_coef_deep: -0.0204520904,
        std_intercept: 0.7824217641,
        std_coef_shallow: -0.0992637336,
        std_coef_deep: 0.0110048553,
    },
    ProbcutParams {
        mean_intercept: -0.2384378922,
        mean_coef_shallow: -0.1125263961,
        mean_coef_deep: 0.1525032279,
        std_intercept: 0.8260776545,
        std_coef_shallow: -0.1260767174,
        std_coef_deep: 0.0155064364,
    },
    ProbcutParams {
        mean_intercept: -0.0502440774,
        mean_coef_shallow: -0.0264473003,
        mean_coef_deep: 0.0313326807,
        std_intercept: 0.8228495567,
        std_coef_shallow: -0.1334427118,
        std_coef_deep: 0.0183401839,
    },
    ProbcutParams {
        mean_intercept: -0.0595431235,
        mean_coef_shallow: 0.0353831866,
        mean_coef_deep: 0.0541827506,
        std_intercept: 0.8045962507,
        std_coef_shallow: -0.1249243101,
        std_coef_deep: 0.0231416918,
    },
    ProbcutParams {
        mean_intercept: -0.2590816106,
        mean_coef_shallow: -0.0390082107,
        mean_coef_deep: 0.0796373261,
        std_intercept: 0.7896858813,
        std_coef_shallow: -0.1140620027,
        std_coef_deep: 0.0246929139,
    },
    ProbcutParams {
        mean_intercept: -0.1231395928,
        mean_coef_shallow: -0.0791932391,
        mean_coef_deep: 0.0929387914,
        std_intercept: 0.7758883371,
        std_coef_shallow: -0.1106818640,
        std_coef_deep: 0.0273912758,
    },
    ProbcutParams {
        mean_intercept: -0.2021840787,
        mean_coef_shallow: 0.0771505117,
        mean_coef_deep: 0.0314895348,
        std_intercept: 0.8258011240,
        std_coef_shallow: -0.1236789606,
        std_coef_deep: 0.0292950843,
    },
    ProbcutParams {
        mean_intercept: 0.1505727556,
        mean_coef_shallow: 0.0164694044,
        mean_coef_deep: -0.0397423186,
        std_intercept: 0.9266180206,
        std_coef_shallow: -0.1169514714,
        std_coef_deep: 0.0157997748,
    },
    ProbcutParams {
        mean_intercept: -0.2315994403,
        mean_coef_shallow: -0.0420398668,
        mean_coef_deep: 0.1012681887,
        std_intercept: 0.9199466832,
        std_coef_shallow: -0.1092082034,
        std_coef_deep: 0.0125080619,
    },
    ProbcutParams {
        mean_intercept: -0.0820577740,
        mean_coef_shallow: 0.0987361388,
        mean_coef_deep: -0.0319118268,
        std_intercept: 0.9338421905,
        std_coef_shallow: -0.1104516652,
        std_coef_deep: 0.0116580620,
    },
    ProbcutParams {
        mean_intercept: -0.2884870113,
        mean_coef_shallow: 0.0724127275,
        mean_coef_deep: 0.0920077182,
        std_intercept: 0.8903306596,
        std_coef_shallow: -0.0989246849,
        std_coef_deep: 0.0159600773,
    },
    ProbcutParams {
        mean_intercept: -0.2586437206,
        mean_coef_shallow: -0.0576332130,
        mean_coef_deep: 0.0653125600,
        std_intercept: 0.8610413294,
        std_coef_shallow: -0.0946502092,
        std_coef_deep: 0.0225823589,
    },
    ProbcutParams {
        mean_intercept: 0.3410073153,
        mean_coef_shallow: 0.1561334245,
        mean_coef_deep: -0.1898154120,
        std_intercept: 0.8898794046,
        std_coef_shallow: -0.1132175499,
        std_coef_deep: 0.0227226297,
    },
    ProbcutParams {
        mean_intercept: -0.3914982663,
        mean_coef_shallow: 0.0754477337,
        mean_coef_deep: 0.1236563531,
        std_intercept: 0.8967476343,
        std_coef_shallow: -0.1138930625,
        std_coef_deep: 0.0228702113,
    },
    ProbcutParams {
        mean_intercept: -0.0864953354,
        mean_coef_shallow: -0.1097951762,
        mean_coef_deep: 0.0189076938,
        std_intercept: 0.8539311627,
        std_coef_shallow: -0.1004801001,
        std_coef_deep: 0.0279541916,
    },
    ProbcutParams {
        mean_intercept: -0.2761626391,
        mean_coef_shallow: -0.0011199172,
        mean_coef_deep: 0.1405470154,
        std_intercept: 0.8129525825,
        std_coef_shallow: -0.1048866685,
        std_coef_deep: 0.0369789494,
    },
    ProbcutParams {
        mean_intercept: -0.6222829559,
        mean_coef_shallow: -0.0200598529,
        mean_coef_deep: 0.1790820955,
        std_intercept: 0.8079942986,
        std_coef_shallow: -0.0969859997,
        std_coef_deep: 0.0429321749,
    },
    ProbcutParams {
        mean_intercept: 0.2268895027,
        mean_coef_shallow: -0.0589144610,
        mean_coef_deep: -0.0809677986,
        std_intercept: 0.8466568411,
        std_coef_shallow: -0.0974830314,
        std_coef_deep: 0.0403350141,
    },
    ProbcutParams {
        mean_intercept: -0.5155179484,
        mean_coef_shallow: 0.0683301717,
        mean_coef_deep: 0.1781044069,
        std_intercept: 0.9083522203,
        std_coef_shallow: -0.0981913933,
        std_coef_deep: 0.0333132130,
    },
    ProbcutParams {
        mean_intercept: -0.1817633675,
        mean_coef_shallow: -0.0285925349,
        mean_coef_deep: 0.0454287949,
        std_intercept: 0.9538211207,
        std_coef_shallow: -0.0936512635,
        std_coef_deep: 0.0310044025,
    },
    ProbcutParams {
        mean_intercept: -0.1817479407,
        mean_coef_shallow: 0.0373078295,
        mean_coef_deep: 0.1019867695,
        std_intercept: 0.9685676388,
        std_coef_shallow: -0.0822355925,
        std_coef_deep: 0.0278089952,
    },
    ProbcutParams {
        mean_intercept: -0.3145352824,
        mean_coef_shallow: 0.1681225592,
        mean_coef_deep: 0.0241105490,
        std_intercept: 1.0016854176,
        std_coef_shallow: -0.0738857194,
        std_coef_deep: 0.0245129255,
    },
    ProbcutParams {
        mean_intercept: -0.0180681246,
        mean_coef_shallow: 0.0773292353,
        mean_coef_deep: -0.0115299339,
        std_intercept: 1.0640324659,
        std_coef_shallow: -0.0843632970,
        std_coef_deep: 0.0204935447,
    },
    ProbcutParams {
        mean_intercept: -0.4936949210,
        mean_coef_shallow: 0.0440931798,
        mean_coef_deep: 0.1976223572,
        std_intercept: 1.1095437516,
        std_coef_shallow: -0.0873409821,
        std_coef_deep: 0.0172450234,
    },
    ProbcutParams {
        mean_intercept: 0.1041346022,
        mean_coef_shallow: 0.0047108640,
        mean_coef_deep: -0.0800062098,
        std_intercept: 1.0875539789,
        std_coef_shallow: -0.0887740782,
        std_coef_deep: 0.0254967382,
    },
    ProbcutParams {
        mean_intercept: -0.1036649267,
        mean_coef_shallow: 0.0125639423,
        mean_coef_deep: 0.0778851267,
        std_intercept: 1.0992593997,
        std_coef_shallow: -0.0856049412,
        std_coef_deep: 0.0227580838,
    },
    ProbcutParams {
        mean_intercept: -0.1886050340,
        mean_coef_shallow: -0.0401253391,
        mean_coef_deep: 0.0405811655,
        std_intercept: 1.1154958837,
        std_coef_shallow: -0.0943198270,
        std_coef_deep: 0.0245795787,
    },
    ProbcutParams {
        mean_intercept: 0.0406220979,
        mean_coef_shallow: 0.0403639687,
        mean_coef_deep: -0.0166129292,
        std_intercept: 1.1056600000,
        std_coef_shallow: -0.1020030668,
        std_coef_deep: 0.0291257781,
    },
    ProbcutParams {
        mean_intercept: -0.3605804075,
        mean_coef_shallow: 0.0760687520,
        mean_coef_deep: 0.1248364485,
        std_intercept: 1.1505333965,
        std_coef_shallow: -0.0946485380,
        std_coef_deep: 0.0243708616,
    },
    ProbcutParams {
        mean_intercept: 0.0872215231,
        mean_coef_shallow: -0.0849712551,
        mean_coef_deep: -0.0450136637,
        std_intercept: 1.1096695195,
        std_coef_shallow: -0.0832689353,
        std_coef_deep: 0.0297690495,
    },
    ProbcutParams {
        mean_intercept: -0.0841058455,
        mean_coef_shallow: 0.1001379590,
        mean_coef_deep: 0.0122781243,
        std_intercept: 1.1544346692,
        std_coef_shallow: -0.0975935120,
        std_coef_deep: 0.0264000327,
    },
    ProbcutParams {
        mean_intercept: -0.0770997107,
        mean_coef_shallow: 0.1504643051,
        mean_coef_deep: -0.0545141189,
        std_intercept: 1.1920923716,
        std_coef_shallow: -0.1019120222,
        std_coef_deep: 0.0237812143,
    },
    ProbcutParams {
        mean_intercept: -0.0687792036,
        mean_coef_shallow: -0.0959514107,
        mean_coef_deep: 0.0886980569,
        std_intercept: 1.1637445839,
        std_coef_shallow: -0.0952072586,
        std_coef_deep: 0.0283895850,
    },
    ProbcutParams {
        mean_intercept: -0.0178536122,
        mean_coef_shallow: 0.0786918795,
        mean_coef_deep: -0.0529529354,
        std_intercept: 1.1497004480,
        std_coef_shallow: -0.0959540540,
        std_coef_deep: 0.0308260303,
    },
    ProbcutParams {
        mean_intercept: -0.1026822489,
        mean_coef_shallow: -0.0047768433,
        mean_coef_deep: 0.0680847832,
        std_intercept: 1.1265331037,
        std_coef_shallow: -0.0907449073,
        std_coef_deep: 0.0351844454,
    },
    ProbcutParams {
        mean_intercept: -0.0245836416,
        mean_coef_shallow: -0.0017542249,
        mean_coef_deep: -0.0418475933,
        std_intercept: 1.0964372846,
        std_coef_shallow: -0.0897977526,
        std_coef_deep: 0.0405937469,
    },
    ProbcutParams {
        mean_intercept: 0.0611168135,
        mean_coef_shallow: -0.0077357178,
        mean_coef_deep: -0.0120402141,
        std_intercept: 1.0985280389,
        std_coef_shallow: -0.0966609391,
        std_coef_deep: 0.0448627113,
    },
    ProbcutParams {
        mean_intercept: -0.1073512988,
        mean_coef_shallow: 0.0607385705,
        mean_coef_deep: -0.0114007198,
        std_intercept: 1.0998979063,
        std_coef_shallow: -0.0977337179,
        std_coef_deep: 0.0479445875,
    },
    ProbcutParams {
        mean_intercept: 0.1709494098,
        mean_coef_shallow: 0.0096116764,
        mean_coef_deep: -0.0957875698,
        std_intercept: 1.1041343606,
        std_coef_shallow: -0.0906604011,
        std_coef_deep: 0.0447159264,
    },
    ProbcutParams {
        mean_intercept: -0.1170701465,
        mean_coef_shallow: 0.0763599672,
        mean_coef_deep: 0.0128586261,
        std_intercept: 1.1072594548,
        std_coef_shallow: -0.0912317763,
        std_coef_deep: 0.0468008047,
    },
    ProbcutParams {
        mean_intercept: -0.0239099488,
        mean_coef_shallow: 0.0368272559,
        mean_coef_deep: -0.0382569235,
        std_intercept: 1.1100838761,
        std_coef_shallow: -0.0912674294,
        std_coef_deep: 0.0468893920,
    },
    ProbcutParams {
        mean_intercept: 0.0460164648,
        mean_coef_shallow: 0.0847471119,
        mean_coef_deep: -0.0690580794,
        std_intercept: 1.1679065761,
        std_coef_shallow: -0.0976858872,
        std_coef_deep: 0.0393837581,
    },
    ProbcutParams {
        mean_intercept: -0.0340656720,
        mean_coef_shallow: 0.0699519014,
        mean_coef_deep: -0.0504603520,
        std_intercept: 1.2062269598,
        std_coef_shallow: -0.1012029702,
        std_coef_deep: 0.0279057524,
    },
    ProbcutParams {
        mean_intercept: -0.0005273524,
        mean_coef_shallow: -0.0026047318,
        mean_coef_deep: -0.0328342274,
        std_intercept: 1.2613776999,
        std_coef_shallow: -0.1231042010,
        std_coef_deep: 0.0164854583,
    },
    ProbcutParams {
        mean_intercept: -0.0377628492,
        mean_coef_shallow: 0.0868255942,
        mean_coef_deep: -0.0368638598,
        std_intercept: 1.3021482651,
        std_coef_shallow: -0.1815553831,
        std_coef_deep: 0.0074031571,
    },
    ProbcutParams {
        mean_intercept: -0.0919941950,
        mean_coef_shallow: 0.0952243686,
        mean_coef_deep: -0.0236953125,
        std_intercept: 1.3196863981,
        std_coef_shallow: -0.2789145890,
        std_coef_deep: -0.0001713567,
    },
    ProbcutParams {
        mean_intercept: 0.0327358730,
        mean_coef_shallow: 0.0475645907,
        mean_coef_deep: -0.0345849953,
        std_intercept: 1.3616281933,
        std_coef_shallow: -0.5945563300,
        std_coef_deep: 0.0020518651,
    },
    ProbcutParams {
        mean_intercept: -0.1090568218,
        mean_coef_shallow: -0.0306735082,
        mean_coef_deep: 0.0173556860,
        std_intercept: 1.0788743533,
        std_coef_shallow: -0.8369462961,
        std_coef_deep: 0.0000322550,
    },
    ProbcutParams {
        mean_intercept: -0.2169587900,
        mean_coef_shallow: 0.0047815722,
        mean_coef_deep: -0.0005898726,
        std_intercept: 0.2415986973,
        std_coef_shallow: -0.8002968227,
        std_coef_deep: -0.0000484745,
    },
    ProbcutParams {
        mean_intercept: 0.4002704065,
        mean_coef_shallow: 0.0063112350,
        mean_coef_deep: -0.0045565813,
        std_intercept: -1.0287348343,
        std_coef_shallow: -0.6871206780,
        std_coef_deep: 0.0000148974,
    },
];
