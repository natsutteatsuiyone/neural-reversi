use std::sync::OnceLock;

use crate::{
    board::Board,
    constants::{EVAL_SCORE_SCALE, EVAL_SCORE_SCALE_BITS, MID_SCORE_MAX, MID_SCORE_MIN},
    search::{midgame, search_context::SearchContext},
    types::{Depth, NonPV, Score},
};

/// Maximum selectivity level (disables ProbCut when `selectivity >= NO_SELECTIVITY`)
pub const NO_SELECTIVITY: u8 = 6;

/// Selectivity configuration table: (level, t_multiplier, probability_percent)
///
/// - `level`: Selectivity level (0-6)
/// - `t_multiplier`: Statistical confidence multiplier (higher = more conservative)
/// - `probability_percent`: Expected success probability percentage
const SELECTIVITY: [(u8, f64, i32); NO_SELECTIVITY as usize + 1] = [
    (0, 1.0, 68),   // Most aggressive: 68% confidence
    (1, 1.1, 73),
    (2, 1.5, 87),
    (3, 2.0, 95),
    (4, 2.6, 98),
    (5, 3.3, 99),   // Most conservative: 99% confidence
    (6, 999.0, 100), // Effectively disabled
];

/// Statistical parameters for ProbCut prediction models
/// - `mean = intercept + coef_shallow * shallow_depth + coef_deep * deep_depth`
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

static MEAN_TABLE: OnceLock<Box<MeanTable>> = OnceLock::new();
static SIGMA_TABLE: OnceLock<Box<SigmaTable>> = OnceLock::new();
static MEAN_TABLE_END: OnceLock<Box<[[f64; MAX_DEPTH]; MAX_DEPTH]>> = OnceLock::new();
static SIGMA_TABLE_END: OnceLock<Box<[[f64; MAX_DEPTH]; MAX_DEPTH]>> = OnceLock::new();

/// Safely allocate a 3D table on the heap to avoid stack overflow
fn alloc_3d_table() -> Box<MeanTable> {
    let tbl = vec![[0.0f64; MAX_DEPTH]; MAX_PLY * MAX_DEPTH].into_boxed_slice();
    unsafe {
        Box::from_raw(Box::into_raw(tbl) as *mut MeanTable)
    }
}

/// Safely allocate a 2D table on the heap to avoid stack overflow
fn alloc_2d_table() -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    let tbl = vec![0.0f64; MAX_DEPTH * MAX_DEPTH].into_boxed_slice();
    unsafe {
        Box::from_raw(Box::into_raw(tbl) as *mut [[f64; MAX_DEPTH]; MAX_DEPTH])
    }
}

/// Build the pre-computed mean table for midgame positions
fn build_mean_table() -> Box<MeanTable> {
    let mut tbl = alloc_3d_table();

    for ply in 0..MAX_PLY {
        let params = &PROBCUT_PARAMS[ply];
        for shallow in 0..MAX_DEPTH {
            for deep in shallow..MAX_DEPTH {
                let v = params.mean(shallow as f64, deep as f64) * EVAL_SCORE_SCALE as f64;
                tbl[ply][shallow][deep] = v;
                tbl[ply][deep][shallow] = v; // Symmetric: mean(a,b) = mean(b,a)
            }
        }
    }
    tbl
}

/// Build the pre-computed sigma table for midgame positions
fn build_sigma_table() -> Box<SigmaTable> {
    let mut tbl = unsafe {
        Box::from_raw(Box::into_raw(alloc_3d_table()) as *mut SigmaTable)
    };

    for ply in 0..MAX_PLY {
        let params = &PROBCUT_PARAMS[ply];
        for shallow in 0..MAX_DEPTH {
            for deep in shallow..MAX_DEPTH {
                let v = params.sigma(shallow as f64, deep as f64) * EVAL_SCORE_SCALE as f64;
                tbl[ply][shallow][deep] = v;
                tbl[ply][deep][shallow] = v; // Symmetric: sigma(a,b) = sigma(b,a)
            }
        }
    }
    tbl
}

/// Build the pre-computed mean table for endgame positions
fn build_mean_table_end() -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    let mut tbl = alloc_2d_table();

    for shallow in 0..MAX_DEPTH {
        for deep in shallow..MAX_DEPTH {
            let v = PROBCUT_ENDGAME_PARAMS.mean(shallow as f64, deep as f64) * EVAL_SCORE_SCALE as f64;
            tbl[shallow][deep] = v;
            tbl[deep][shallow] = v;
        }
    }
    tbl
}

/// Build the pre-computed sigma table for endgame positions
fn build_sigma_table_end() -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    let mut tbl = alloc_2d_table();

    for shallow in 0..MAX_DEPTH {
        for deep in shallow..MAX_DEPTH {
            let v = PROBCUT_ENDGAME_PARAMS.sigma(shallow as f64, deep as f64) * EVAL_SCORE_SCALE as f64;
            tbl[shallow][deep] = v;
            tbl[deep][shallow] = v;
        }
    }
    tbl
}

/// Initialize probcut tables. Called from lib.rs init().
pub fn init() {
    MEAN_TABLE.set(build_mean_table()).ok();
    SIGMA_TABLE.set(build_sigma_table()).ok();
    MEAN_TABLE_END.set(build_mean_table_end()).ok();
    SIGMA_TABLE_END.set(build_sigma_table_end()).ok();
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

/// Fast lookup of pre-computed mean value for endgame positions
#[inline]
fn calc_mean_end(shallow: Depth, deep: Depth) -> f64 {
    let tbl = MEAN_TABLE_END.get().expect("probcut not initialized");
    tbl[shallow as usize][deep as usize]
}

/// Fast lookup of pre-computed sigma value for endgame positions
#[inline]
fn calc_sigma_end(shallow: Depth, deep: Depth) -> f64 {
    let tbl = SIGMA_TABLE_END.get().expect("probcut not initialized");
    tbl[shallow as usize][deep as usize]
}

/// Determines the depth of the shallow search in probcut.
///
/// # Arguments
///
/// * `depth` - The depth of the deep search.
///
/// # Returns
///
/// The depth of the shallow search (typically 20% of deep search depth).
fn determine_probcut_depth(depth: Depth) -> Depth {
    let mut probcut_depth = 2 * (depth as f64 * 0.20).floor() as Depth + (depth & 1);
    if probcut_depth == 0 {
        probcut_depth = depth - 2;
    }
    probcut_depth
}

/// Get the expected success probability percentage for a given selectivity level
///
/// # Arguments
///
/// * `selectivity` - Selectivity level (0-6)
///
/// # Returns
///
/// Expected success probability as a percentage (68-100)
#[inline]
pub fn get_probability(selectivity: u8) -> i32 {
    SELECTIVITY[selectivity as usize].2
}

/// Get the statistical confidence multiplier (t-value) for a given selectivity level
///
/// # Arguments
///
/// * `selectivity` - Selectivity level (0-6)
///
/// # Returns
///
/// The t-multiplier for statistical confidence calculations
#[inline]
fn get_t(selectivity: u8) -> f64 {
    SELECTIVITY[selectivity as usize].1
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
    alpha: Score,
    beta: Score,
) -> Option<Score> {
    if depth >= 3 && ctx.selectivity < NO_SELECTIVITY {
        let ply = ctx.ply();
        let pc_depth = determine_probcut_depth(depth);
        let mean = calc_mean(ply, pc_depth, depth);
        let sigma = calc_sigma(ply, pc_depth, depth);
        let t = get_t(ctx.selectivity);
        let current_selectivity = ctx.selectivity;

        let eval_score = midgame::evaluate(ctx, board);
        let eval_mean = 0.5 * calc_mean(ply, 0, depth) + mean;
        let eval_sigma = t * 0.5 * calc_sigma(ply, 0, depth) + sigma;

        let eval_beta = (beta as f64 - eval_sigma - eval_mean).floor() as Score;
        let pc_beta = (beta as f64 + t * sigma - mean).ceil() as Score;
        if eval_score >= eval_beta && pc_beta < MID_SCORE_MAX {
            ctx.update_probcut();
            let score = midgame::search::<NonPV, false>(ctx, board, pc_depth, pc_beta - 1, pc_beta, None);
            ctx.undo_probcut(current_selectivity);
            if score >= pc_beta {
                return Some(beta);
            }
        }

        let eval_alpha = (alpha as f64 + eval_sigma - eval_mean).ceil() as Score;
        let pc_alpha = (alpha as f64 - t * sigma - mean).floor() as Score;
        if eval_score < eval_alpha && pc_alpha > MID_SCORE_MIN {
            ctx.update_probcut();
            let score = midgame::search::<NonPV, false>(ctx, board, pc_depth, pc_alpha, pc_alpha + 1, None);
            ctx.undo_probcut(current_selectivity);
            if score <= pc_alpha {
                return Some(alpha);
            }
        }
    }
    None
}

/// Attempts ProbCut pruning for endgame positions
///
/// # Arguments
///
/// * `ctx` - Search context containing selectivity settings and search state
/// * `board` - Current board position to evaluate
/// * `depth` - Depth of the deep search that would be performed
/// * `alpha` - Alpha bound for the search window (will be scaled internally)
/// * `beta` - Beta bound for the search window (will be scaled internally)
///
/// # Returns
///
/// * `Some(score)` - If probcut triggers, returns the predicted bound
/// * `None` - If probcut doesn't trigger, full endgame search should be performed
pub fn probcut_endgame(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    alpha: Score,
    beta: Score,
) -> Option<Score> {
    if depth >= 10 && ctx.selectivity < NO_SELECTIVITY {
        let scaled_alpha = alpha << EVAL_SCORE_SCALE_BITS;
        let scaled_beta = beta << EVAL_SCORE_SCALE_BITS;
        if let Some(score) = probcut_endgame_internal(ctx, board, depth, scaled_alpha, scaled_beta) {
            return Some(score >> EVAL_SCORE_SCALE_BITS);
        }
    }
    None
}

/// Internal implementation of endgame probcut with scaled score values
fn probcut_endgame_internal(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    alpha: Score,
    beta: Score,
) -> Option<Score> {
    let pc_depth = determine_probcut_depth(depth);
    let mean: f64 = calc_mean_end(pc_depth, depth);
    let sigma: f64 = calc_sigma_end(pc_depth, depth);
    let t = get_t(ctx.selectivity);
    let current_selectivity = ctx.selectivity;

    let eval_score = midgame::evaluate(ctx, board);

    let pc_beta = (beta as f64 + t * sigma - mean).ceil() as Score;
    if eval_score > alpha && pc_beta < MID_SCORE_MAX {
        ctx.update_probcut();
        let score = midgame::search::<NonPV, false>(ctx, board, pc_depth, pc_beta - 1, pc_beta, None);
        ctx.undo_probcut(current_selectivity);
        if score >= pc_beta {
            return Some(beta);
        }
    }

    let pc_alpha = (alpha as f64 - t * sigma - mean).floor() as Score;
    if eval_score < beta &&  pc_alpha > MID_SCORE_MIN {
        ctx.update_probcut();
        let score = midgame::search::<NonPV, false>(ctx, board, pc_depth, pc_alpha, pc_alpha + 1, None);
        ctx.undo_probcut(current_selectivity);
        if score <= pc_alpha {
            return Some(alpha);
        }
    }

    None
}

/// Statistical parameters for endgame ProbCut
#[rustfmt::skip]
const PROBCUT_ENDGAME_PARAMS: ProbcutParams = ProbcutParams {
    mean_intercept: -1.192882481,
    mean_coef_shallow: 0.1433213255,
    mean_coef_deep: 0.005958478026,
    std_intercept: 1.65272932,
    std_coef_shallow: -0.07145816068,
    std_coef_deep: 0.001187215727,
};

/// Statistical parameters for midgame ProbCut indexed by ply
#[rustfmt::skip]
const PROBCUT_PARAMS: [ProbcutParams; MAX_PLY] = [
    ProbcutParams {
        mean_intercept: 0.0,
        mean_coef_shallow: 0.0,
        mean_coef_deep: 0.0,
        std_intercept: -18.42068074,
        std_coef_shallow: 0.0,
        std_coef_deep: 0.0,
    },
    ProbcutParams {
        mean_intercept: -0.937127826,
        mean_coef_shallow: 0.1102971036,
        mean_coef_deep: 0.04557656245,
        std_intercept: -1.312578756,
        std_coef_shallow: 0.02307551146,
        std_coef_deep: 0.1154310461,
    },
    ProbcutParams {
        mean_intercept: -0.2469738803,
        mean_coef_shallow: -0.0516801701,
        mean_coef_deep: 0.1006349511,
        std_intercept: -0.924698672,
        std_coef_shallow: 0.05333252651,
        std_coef_deep: 0.04727662001,
    },
    ProbcutParams {
        mean_intercept: -0.8998163552,
        mean_coef_shallow: 0.09620006207,
        mean_coef_deep: 0.05432404312,
        std_intercept: -0.4633870403,
        std_coef_shallow: 0.03206731591,
        std_coef_deep: 0.04191869214,
    },
    ProbcutParams {
        mean_intercept: -0.03970965049,
        mean_coef_shallow: -0.05886407573,
        mean_coef_deep: 0.05307945104,
        std_intercept: -0.1517172776,
        std_coef_shallow: 0.02559935627,
        std_coef_deep: 0.02625317754,
    },
    ProbcutParams {
        mean_intercept: -0.3654123992,
        mean_coef_shallow: -0.01774609994,
        mean_coef_deep: 0.07457173693,
        std_intercept: 0.1259640475,
        std_coef_shallow: 0.0316966373,
        std_coef_deep: 0.00238505587,
    },
    ProbcutParams {
        mean_intercept: -0.328226007,
        mean_coef_shallow: 0.0007392299725,
        mean_coef_deep: 0.05175979314,
        std_intercept: 0.228038991,
        std_coef_shallow: 0.0240073078,
        std_coef_deep: 0.003629798191,
    },
    ProbcutParams {
        mean_intercept: -0.6263867544,
        mean_coef_shallow: -0.007859731633,
        mean_coef_deep: 0.1136012145,
        std_intercept: 0.3721396065,
        std_coef_shallow: -0.003181446423,
        std_coef_deep: 0.009728546146,
    },
    ProbcutParams {
        mean_intercept: -0.5658242095,
        mean_coef_shallow: -0.009646050211,
        mean_coef_deep: 0.08657775621,
        std_intercept: 0.4506964185,
        std_coef_shallow: -0.02558118528,
        std_coef_deep: 0.02449000893,
    },
    ProbcutParams {
        mean_intercept: -0.2636992768,
        mean_coef_shallow: -0.09157524156,
        mean_coef_deep: 0.1208489448,
        std_intercept: 0.4580436171,
        std_coef_shallow: -0.04137254443,
        std_coef_deep: 0.03859106944,
    },
    ProbcutParams {
        mean_intercept: -0.5541614082,
        mean_coef_shallow: 0.01902661235,
        mean_coef_deep: 0.06983062518,
        std_intercept: 0.5916485508,
        std_coef_shallow: -0.03945877891,
        std_coef_deep: 0.02795057096,
    },
    ProbcutParams {
        mean_intercept: -0.6539759407,
        mean_coef_shallow: -0.0196306911,
        mean_coef_deep: 0.1239229271,
        std_intercept: 0.6097322889,
        std_coef_shallow: -0.04146215253,
        std_coef_deep: 0.03200191658,
    },
    ProbcutParams {
        mean_intercept: -0.6934404821,
        mean_coef_shallow: 0.01713716008,
        mean_coef_deep: 0.09790848704,
        std_intercept: 0.6538991691,
        std_coef_shallow: -0.04069701796,
        std_coef_deep: 0.03474228754,
    },
    ProbcutParams {
        mean_intercept: -0.6857607622,
        mean_coef_shallow: 0.03707098619,
        mean_coef_deep: 0.0907158371,
        std_intercept: 0.7207442333,
        std_coef_shallow: -0.03601957085,
        std_coef_deep: 0.02936180174,
    },
    ProbcutParams {
        mean_intercept: -1.023775794,
        mean_coef_shallow: 0.06401051959,
        mean_coef_deep: 0.1182046269,
        std_intercept: 0.824235815,
        std_coef_shallow: -0.0434374178,
        std_coef_deep: 0.0224952961,
    },
    ProbcutParams {
        mean_intercept: -0.5843931301,
        mean_coef_shallow: 0.02798201884,
        mean_coef_deep: 0.07105587588,
        std_intercept: 0.8773157186,
        std_coef_shallow: -0.04080588685,
        std_coef_deep: 0.01929085383,
    },
    ProbcutParams {
        mean_intercept: -0.7656389191,
        mean_coef_shallow: 0.07226412782,
        mean_coef_deep: 0.0701453065,
        std_intercept: 0.9907488886,
        std_coef_shallow: -0.04434909781,
        std_coef_deep: 0.007653875135,
    },
    ProbcutParams {
        mean_intercept: -0.9941712473,
        mean_coef_shallow: 0.1068672146,
        mean_coef_deep: 0.07410592308,
        std_intercept: 1.072835955,
        std_coef_shallow: -0.05042112313,
        std_coef_deep: 0.002762413634,
    },
    ProbcutParams {
        mean_intercept: -1.106828896,
        mean_coef_shallow: 0.15402027,
        mean_coef_deep: 0.06172099574,
        std_intercept: 1.054740458,
        std_coef_shallow: -0.0494109543,
        std_coef_deep: 0.005508258083,
    },
    ProbcutParams {
        mean_intercept: -1.443545798,
        mean_coef_shallow: 0.2001361971,
        mean_coef_deep: 0.07427632145,
        std_intercept: 1.042080217,
        std_coef_shallow: -0.04769556226,
        std_coef_deep: 0.007047109608,
    },
    ProbcutParams {
        mean_intercept: -1.183575362,
        mean_coef_shallow: 0.1698697543,
        mean_coef_deep: 0.05520884793,
        std_intercept: 0.9930158285,
        std_coef_shallow: -0.04358157825,
        std_coef_deep: 0.01170418503,
    },
    ProbcutParams {
        mean_intercept: -1.459943462,
        mean_coef_shallow: 0.2360718022,
        mean_coef_deep: 0.05369678441,
        std_intercept: 0.9785690739,
        std_coef_shallow: -0.05029655487,
        std_coef_deep: 0.01751918105,
    },
    ProbcutParams {
        mean_intercept: -1.663590185,
        mean_coef_shallow: 0.2504164048,
        mean_coef_deep: 0.07154632102,
        std_intercept: 0.9911704589,
        std_coef_shallow: -0.0474161115,
        std_coef_deep: 0.01557083123,
    },
    ProbcutParams {
        mean_intercept: -1.406190113,
        mean_coef_shallow: 0.191161507,
        mean_coef_deep: 0.07386162455,
        std_intercept: 0.9744018748,
        std_coef_shallow: -0.04402783807,
        std_coef_deep: 0.01836300629,
    },
    ProbcutParams {
        mean_intercept: -1.411942439,
        mean_coef_shallow: 0.2185302647,
        mean_coef_deep: 0.05825828692,
        std_intercept: 0.9680184515,
        std_coef_shallow: -0.03974640663,
        std_coef_deep: 0.02028010855,
    },
    ProbcutParams {
        mean_intercept: -1.285671679,
        mean_coef_shallow: 0.2474971507,
        mean_coef_deep: 0.01408713335,
        std_intercept: 0.9306457351,
        std_coef_shallow: -0.03684504005,
        std_coef_deep: 0.02597259144,
    },
    ProbcutParams {
        mean_intercept: -1.135861589,
        mean_coef_shallow: 0.2061372198,
        mean_coef_deep: 0.0296169219,
        std_intercept: 0.9430166069,
        std_coef_shallow: -0.04361649336,
        std_coef_deep: 0.03244193198,
    },
    ProbcutParams {
        mean_intercept: -1.372062124,
        mean_coef_shallow: 0.2354638048,
        mean_coef_deep: 0.03523019679,
        std_intercept: 0.9959575368,
        std_coef_shallow: -0.04597977329,
        std_coef_deep: 0.02815314973,
    },
    ProbcutParams {
        mean_intercept: -1.370726425,
        mean_coef_shallow: 0.2286892985,
        mean_coef_deep: 0.04479307031,
        std_intercept: 1.028660579,
        std_coef_shallow: -0.05490057061,
        std_coef_deep: 0.0269598485,
    },
    ProbcutParams {
        mean_intercept: -1.55171919,
        mean_coef_shallow: 0.2355203233,
        mean_coef_deep: 0.06404621812,
        std_intercept: 1.058095712,
        std_coef_shallow: -0.06492085364,
        std_coef_deep: 0.02773342155,
    },
    ProbcutParams {
        mean_intercept: -1.190070644,
        mean_coef_shallow: 0.228396248,
        mean_coef_deep: 0.01360981192,
        std_intercept: 1.074622625,
        std_coef_shallow: -0.0732436547,
        std_coef_deep: 0.02775703784,
    },
    ProbcutParams {
        mean_intercept: -1.514391694,
        mean_coef_shallow: 0.2393730489,
        mean_coef_deep: 0.06208186427,
        std_intercept: 1.118379845,
        std_coef_shallow: -0.07616999696,
        std_coef_deep: 0.02376753425,
    },
    ProbcutParams {
        mean_intercept: -1.32847733,
        mean_coef_shallow: 0.2408427001,
        mean_coef_deep: 0.0237779927,
        std_intercept: 1.119534084,
        std_coef_shallow: -0.08154488409,
        std_coef_deep: 0.02460377246,
    },
    ProbcutParams {
        mean_intercept: -1.341609142,
        mean_coef_shallow: 0.2556529506,
        mean_coef_deep: 0.02409427302,
        std_intercept: 1.149198346,
        std_coef_shallow: -0.07963423756,
        std_coef_deep: 0.02081188471,
    },
    ProbcutParams {
        mean_intercept: -1.423365986,
        mean_coef_shallow: 0.2734014505,
        mean_coef_deep: 0.01321768987,
        std_intercept: 1.124451968,
        std_coef_shallow: -0.07838934542,
        std_coef_deep: 0.02260008864,
    },
    ProbcutParams {
        mean_intercept: -1.185954156,
        mean_coef_shallow: 0.1609717716,
        mean_coef_deep: 0.06065724554,
        std_intercept: 1.132135497,
        std_coef_shallow: -0.08296832425,
        std_coef_deep: 0.02366269907,
    },
    ProbcutParams {
        mean_intercept: -0.8002505913,
        mean_coef_shallow: 0.1715613865,
        mean_coef_deep: -0.01590932488,
        std_intercept: 1.122180216,
        std_coef_shallow: -0.08013150584,
        std_coef_deep: 0.02381104487,
    },
    ProbcutParams {
        mean_intercept: -0.8388281201,
        mean_coef_shallow: 0.1538614047,
        mean_coef_deep: 0.0150753224,
        std_intercept: 1.131912298,
        std_coef_shallow: -0.08134309447,
        std_coef_deep: 0.02474747696,
    },
    ProbcutParams {
        mean_intercept: -1.144438027,
        mean_coef_shallow: 0.1992409972,
        mean_coef_deep: 0.01784347773,
        std_intercept: 1.169736341,
        std_coef_shallow: -0.08604672246,
        std_coef_deep: 0.02355253548,
    },
    ProbcutParams {
        mean_intercept: -0.8768689091,
        mean_coef_shallow: 0.1986723945,
        mean_coef_deep: -0.01815851823,
        std_intercept: 1.142205072,
        std_coef_shallow: -0.08678100343,
        std_coef_deep: 0.02927426327,
    },
    ProbcutParams {
        mean_intercept: -1.213291497,
        mean_coef_shallow: 0.1861601128,
        mean_coef_deep: 0.04873829391,
        std_intercept: 1.142496813,
        std_coef_shallow: -0.08984425918,
        std_coef_deep: 0.0303754637,
    },
    ProbcutParams {
        mean_intercept: -0.7252156417,
        mean_coef_shallow: 0.1891052401,
        mean_coef_deep: -0.04303486013,
        std_intercept: 1.099368644,
        std_coef_shallow: -0.07873484574,
        std_coef_deep: 0.03054729651,
    },
    ProbcutParams {
        mean_intercept: -1.102832417,
        mean_coef_shallow: 0.1687234952,
        mean_coef_deep: 0.04890403694,
        std_intercept: 1.087539637,
        std_coef_shallow: -0.07299972546,
        std_coef_deep: 0.03341075812,
    },
    ProbcutParams {
        mean_intercept: -0.579899773,
        mean_coef_shallow: 0.1782827192,
        mean_coef_deep: -0.05679970097,
        std_intercept: 1.088674561,
        std_coef_shallow: -0.07719777542,
        std_coef_deep: 0.03698058014,
    },
    ProbcutParams {
        mean_intercept: -0.2928854679,
        mean_coef_shallow: 0.1222887368,
        mean_coef_deep: -0.04129830962,
        std_intercept: 1.107189641,
        std_coef_shallow: -0.07741379991,
        std_coef_deep: 0.03746782456,
    },
    ProbcutParams {
        mean_intercept: -1.155485744,
        mean_coef_shallow: 0.2054164323,
        mean_coef_deep: 0.0103039914,
        std_intercept: 1.130973144,
        std_coef_shallow: -0.08144509312,
        std_coef_deep: 0.03874310302,
    },
    ProbcutParams {
        mean_intercept: -0.03712873561,
        mean_coef_shallow: 0.09795121714,
        mean_coef_deep: -0.07851534389,
        std_intercept: 1.112832645,
        std_coef_shallow: -0.08738945875,
        std_coef_deep: 0.04256547269,
    },
    ProbcutParams {
        mean_intercept: -0.604139004,
        mean_coef_shallow: 0.1698580438,
        mean_coef_deep: -0.0404451086,
        std_intercept: 1.121087447,
        std_coef_shallow: -0.08563614497,
        std_coef_deep: 0.04136047841,
    },
    ProbcutParams {
        mean_intercept: -0.3188044648,
        mean_coef_shallow: 0.1225946218,
        mean_coef_deep: -0.05210858118,
        std_intercept: 1.110539885,
        std_coef_shallow: -0.0892550726,
        std_coef_deep: 0.04383619779,
    },
    ProbcutParams {
        mean_intercept: -0.2693254226,
        mean_coef_shallow: 0.1521933118,
        mean_coef_deep: -0.08508659096,
        std_intercept: 1.10485534,
        std_coef_shallow: -0.08569111073,
        std_coef_deep: 0.04339386781,
    },
    ProbcutParams {
        mean_intercept: -0.697037331,
        mean_coef_shallow: 0.1629754853,
        mean_coef_deep: -0.02175775844,
        std_intercept: 1.176446387,
        std_coef_shallow: -0.103221116,
        std_coef_deep: 0.03917377722,
    },
    ProbcutParams {
        mean_intercept: -0.2451364389,
        mean_coef_shallow: 0.1346062303,
        mean_coef_deep: -0.07753293203,
        std_intercept: 1.193488109,
        std_coef_shallow: -0.1286711242,
        std_coef_deep: 0.03787846154,
    },
    ProbcutParams {
        mean_intercept: -0.8308077896,
        mean_coef_shallow: 0.1674079561,
        mean_coef_deep: -0.001298731681,
        std_intercept: 1.29534317,
        std_coef_shallow: -0.1753633183,
        std_coef_deep: 0.02929607944,
    },
    ProbcutParams {
        mean_intercept: -0.364014443,
        mean_coef_shallow: 0.09611633765,
        mean_coef_deep: -0.03741995187,
        std_intercept: 1.691749933,
        std_coef_shallow: -0.4095623258,
        std_coef_deep: 0.02047673043,
    },
    ProbcutParams {
        mean_intercept: -0.008438237135,
        mean_coef_shallow: 0.01063904387,
        mean_coef_deep: -0.006404047956,
        std_intercept: 2.012216519,
        std_coef_shallow: -0.7787640401,
        std_coef_deep: 0.02386406881,
    },
    ProbcutParams {
        mean_intercept: -0.7560766726,
        mean_coef_shallow: 0.154127446,
        mean_coef_deep: -0.009403541422,
        std_intercept: 1.449750252,
        std_coef_shallow: -0.4890120451,
        std_coef_deep: 0.005055549058,
    },
    ProbcutParams {
        mean_intercept: -0.06348693182,
        mean_coef_shallow: 0.0101446733,
        mean_coef_deep: 0.000640719697,
        std_intercept: 1.63120596,
        std_coef_shallow: -0.9872745584,
        std_coef_deep: -0.002297646322,
    },
    ProbcutParams {
        mean_intercept: -0.1963791495,
        mean_coef_shallow: 0.03665686386,
        mean_coef_deep: 7.111921396e-18,
        std_intercept: 0.6690170131,
        std_coef_shallow: -0.6888530111,
        std_coef_deep: -2.805241374e-17,
    },
    ProbcutParams {
        mean_intercept: 0.087800352,
        mean_coef_shallow: -0.01794459749,
        mean_coef_deep: 3.037389789e-17,
        std_intercept: -0.5043334396,
        std_coef_shallow: -0.6754701688,
        std_coef_deep: -2.639004453e-16,
    },
    ProbcutParams {
        mean_intercept: 0.0,
        mean_coef_shallow: 0.0,
        mean_coef_deep: 0.0,
        std_intercept: -18.42068074,
        std_coef_shallow: 4.27686139e-31,
        std_coef_deep: -6.309727848e-31,
    },
];
