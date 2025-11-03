use std::sync::OnceLock;

use std::sync::Arc;

use crate::search::node_type::NonPV;
use crate::{
    board::Board,
    constants::{EVAL_SCORE_SCALE, EVAL_SCORE_SCALE_BITS, MID_SCORE_MAX, MID_SCORE_MIN},
    search::{midgame, search_context::SearchContext, threading::Thread},
    types::{Depth, Score},
};

/// Maximum selectivity level (disables ProbCut when `selectivity >= NO_SELECTIVITY`)
pub const NO_SELECTIVITY: u8 = 6;

/// Selectivity configuration table: (level, t_multiplier, probability_percent)
///
/// - `level`: Selectivity level (0-6)
/// - `t_multiplier`: Statistical confidence multiplier (higher = more conservative)
/// - `probability_percent`: Expected success probability percentage
const SELECTIVITY: [(u8, f64, i32); NO_SELECTIVITY as usize + 1] = [
    (0, 1.0, 68), // Most aggressive: 68% confidence
    (1, 1.1, 73),
    (2, 1.5, 87),
    (3, 2.0, 95),
    (4, 2.6, 98),
    (5, 3.3, 99),    // Most conservative: 99% confidence
    (6, 999.0, 100), // Effectively disabled
];

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

static MEAN_TABLE: OnceLock<Box<MeanTable>> = OnceLock::new();
static SIGMA_TABLE: OnceLock<Box<SigmaTable>> = OnceLock::new();
static MEAN_TABLE_END: OnceLock<Box<[[f64; MAX_DEPTH]; MAX_DEPTH]>> = OnceLock::new();
static SIGMA_TABLE_END: OnceLock<Box<[[f64; MAX_DEPTH]; MAX_DEPTH]>> = OnceLock::new();

/// Safely allocate a 3D table on the heap to avoid stack overflow
fn alloc_3d_table() -> Box<MeanTable> {
    let tbl = vec![[0.0f64; MAX_DEPTH]; MAX_PLY * MAX_DEPTH].into_boxed_slice();
    unsafe { Box::from_raw(Box::into_raw(tbl) as *mut MeanTable) }
}

/// Safely allocate a 2D table on the heap to avoid stack overflow
fn alloc_2d_table() -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    let tbl = vec![0.0f64; MAX_DEPTH * MAX_DEPTH].into_boxed_slice();
    unsafe { Box::from_raw(Box::into_raw(tbl) as *mut [[f64; MAX_DEPTH]; MAX_DEPTH]) }
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
    let mut tbl = unsafe { Box::from_raw(Box::into_raw(alloc_3d_table()) as *mut SigmaTable) };

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
            let v =
                PROBCUT_ENDGAME_PARAMS.mean(shallow as f64, deep as f64) * EVAL_SCORE_SCALE as f64;
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
            let v =
                PROBCUT_ENDGAME_PARAMS.sigma(shallow as f64, deep as f64) * EVAL_SCORE_SCALE as f64;
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
/// The depth of the shallow search.
fn determine_probcut_depth(depth: Depth) -> Depth {
    let mut probcut_depth = 2 * (depth as f64 * 0.2).floor() as Depth + (depth & 1);
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
/// * `thread` - Search thread used to run the shallow verification search
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
    thread: &Arc<Thread>,
) -> Option<Score> {
    if depth >= 3 && ctx.selectivity < NO_SELECTIVITY {
        let ply = ctx.ply();
        let pc_depth = determine_probcut_depth(depth);
        let mean = calc_mean(ply, pc_depth, depth);
        let sigma = calc_sigma(ply, pc_depth, depth);
        let t = get_t(ctx.selectivity);

        let eval_score = midgame::evaluate(ctx, board);
        let eval_mean = 0.5 * calc_mean(ply, 0, depth) + mean;
        let eval_sigma = t * 0.5 * calc_sigma(ply, 0, depth) + sigma;

        let eval_beta = (beta as f64 - eval_sigma - eval_mean).floor() as Score;
        let pc_beta = (beta as f64 + t * sigma - mean).ceil() as Score;
        if eval_score >= eval_beta && pc_beta < MID_SCORE_MAX {
            let score = midgame::search::<NonPV, false>(
                ctx,
                board,
                pc_depth,
                pc_beta - 1,
                pc_beta,
                thread,
                None,
            );
            if score >= pc_beta {
                return Some(beta);
            }
        }

        let eval_alpha = (alpha as f64 + eval_sigma - eval_mean).ceil() as Score;
        let pc_alpha = (alpha as f64 - t * sigma - mean).floor() as Score;
        if eval_score < eval_alpha && pc_alpha > MID_SCORE_MIN {
            let score = midgame::search::<NonPV, false>(
                ctx,
                board,
                pc_depth,
                pc_alpha,
                pc_alpha + 1,
                thread,
                None,
            );
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
/// * `thread` - Search thread used to run the shallow verification search
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
    thread: &Arc<Thread>,
) -> Option<Score> {
    if depth >= 10 && ctx.selectivity < NO_SELECTIVITY {
        let scaled_alpha = alpha << EVAL_SCORE_SCALE_BITS;
        let scaled_beta = beta << EVAL_SCORE_SCALE_BITS;
        if let Some(score) =
            probcut_endgame_internal(ctx, board, depth, scaled_alpha, scaled_beta, thread)
        {
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
    thread: &Arc<Thread>,
) -> Option<Score> {
    let pc_depth = determine_probcut_depth(depth);
    let mean: f64 = calc_mean_end(pc_depth, depth);
    let sigma: f64 = calc_sigma_end(pc_depth, depth);
    let t = get_t(ctx.selectivity);

    let eval_score = midgame::evaluate(ctx, board);

    let pc_beta = (beta as f64 + t * sigma - mean).ceil() as Score;
    if eval_score > alpha && pc_beta < MID_SCORE_MAX {
        let score = midgame::search::<NonPV, false>(
            ctx,
            board,
            pc_depth,
            pc_beta - 1,
            pc_beta,
            thread,
            None,
        );
        if score >= pc_beta {
            return Some(beta);
        }
    }

    let pc_alpha = (alpha as f64 - t * sigma - mean).floor() as Score;
    if eval_score < beta && pc_alpha > MID_SCORE_MIN {
        let score = midgame::search::<NonPV, false>(
            ctx,
            board,
            pc_depth,
            pc_alpha,
            pc_alpha + 1,
            thread,
            None,
        );
        if score <= pc_alpha {
            return Some(alpha);
        }
    }

    None
}

/// Statistical parameters for endgame ProbCut
#[rustfmt::skip]
const PROBCUT_ENDGAME_PARAMS: ProbcutParams = ProbcutParams {
    mean_intercept: -0.1781667645,
    mean_coef_shallow: 0.0147725713,
    mean_coef_deep: 0.0033718233,
    std_intercept: 0.8300938828,
    std_coef_shallow: -0.0612042521,
    std_coef_deep: 0.0359699791,
};

/// Statistical parameters for midgame ProbCut indexed by ply
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
        mean_intercept: -2.1313468777,
        mean_coef_shallow: 0.0341519591,
        mean_coef_deep: 0.1072443368,
        std_intercept: -1.1068617148,
        std_coef_shallow: 0.0353144936,
        std_coef_deep: 0.0067271156,
    },
    ProbcutParams {
        mean_intercept: 2.4245822295,
        mean_coef_shallow: -0.0621570039,
        mean_coef_deep: -0.1953904082,
        std_intercept: -1.1522100222,
        std_coef_shallow: 0.0141058237,
        std_coef_deep: 0.0857380528,
    },
    ProbcutParams {
        mean_intercept: -0.3009302998,
        mean_coef_shallow: 0.0125310969,
        mean_coef_deep: 0.0409582229,
        std_intercept: -0.4660790551,
        std_coef_shallow: -0.0006195042,
        std_coef_deep: 0.0463814221,
    },
    ProbcutParams {
        mean_intercept: -0.0793563697,
        mean_coef_shallow: -0.0563597897,
        mean_coef_deep: 0.0202502943,
        std_intercept: -0.2684121103,
        std_coef_shallow: -0.0004566834,
        std_coef_deep: 0.0455527594,
    },
    ProbcutParams {
        mean_intercept: 0.4848711599,
        mean_coef_shallow: -0.1100498840,
        mean_coef_deep: -0.0172015144,
        std_intercept: 0.1454689494,
        std_coef_shallow: -0.0080475021,
        std_coef_deep: 0.0226331993,
    },
    ProbcutParams {
        mean_intercept: -0.5031896387,
        mean_coef_shallow: 0.1041770936,
        mean_coef_deep: 0.0270372974,
        std_intercept: 0.3565112259,
        std_coef_shallow: 0.0092072540,
        std_coef_deep: 0.0000422342,
    },
    ProbcutParams {
        mean_intercept: -0.0949426567,
        mean_coef_shallow: -0.0011069226,
        mean_coef_deep: 0.0092929247,
        std_intercept: 0.4448424788,
        std_coef_shallow: 0.0100268108,
        std_coef_deep: -0.0049088232,
    },
    ProbcutParams {
        mean_intercept: -0.4433498596,
        mean_coef_shallow: 0.0092804478,
        mean_coef_deep: 0.0674505615,
        std_intercept: 0.6005036981,
        std_coef_shallow: -0.0192132264,
        std_coef_deep: -0.0021885507,
    },
    ProbcutParams {
        mean_intercept: 0.0527412722,
        mean_coef_shallow: -0.1305156085,
        mean_coef_deep: 0.0350978693,
        std_intercept: 0.6589973995,
        std_coef_shallow: -0.0309466493,
        std_coef_deep: 0.0026163745,
    },
    ProbcutParams {
        mean_intercept: -0.1067829932,
        mean_coef_shallow: -0.0846307666,
        mean_coef_deep: 0.0694129154,
        std_intercept: 0.6430469772,
        std_coef_shallow: -0.0353308180,
        std_coef_deep: 0.0106450080,
    },
    ProbcutParams {
        mean_intercept: -0.4860744484,
        mean_coef_shallow: 0.0279207155,
        mean_coef_deep: 0.0676375730,
        std_intercept: 0.7095842176,
        std_coef_shallow: -0.0389310864,
        std_coef_deep: 0.0136244348,
    },
    ProbcutParams {
        mean_intercept: -0.1751397600,
        mean_coef_shallow: -0.0348539432,
        mean_coef_deep: 0.0504614320,
        std_intercept: 0.6748298478,
        std_coef_shallow: -0.0318243509,
        std_coef_deep: 0.0147532260,
    },
    ProbcutParams {
        mean_intercept: -0.0854512485,
        mean_coef_shallow: -0.0222259941,
        mean_coef_deep: 0.0212183052,
        std_intercept: 0.7580838288,
        std_coef_shallow: -0.0298749935,
        std_coef_deep: 0.0091933702,
    },
    ProbcutParams {
        mean_intercept: -0.0785967628,
        mean_coef_shallow: -0.0121339869,
        mean_coef_deep: 0.0307314528,
        std_intercept: 0.8491082124,
        std_coef_shallow: -0.0474671362,
        std_coef_deep: 0.0067786592,
    },
    ProbcutParams {
        mean_intercept: -0.4141769455,
        mean_coef_shallow: 0.0292539150,
        mean_coef_deep: 0.0623489271,
        std_intercept: 0.8778002643,
        std_coef_shallow: -0.0556686103,
        std_coef_deep: 0.0085475277,
    },
    ProbcutParams {
        mean_intercept: -0.2593797826,
        mean_coef_shallow: 0.0227001822,
        mean_coef_deep: 0.0227782581,
        std_intercept: 0.8868351335,
        std_coef_shallow: -0.0621201638,
        std_coef_deep: 0.0108774581,
    },
    ProbcutParams {
        mean_intercept: -0.3924558889,
        mean_coef_shallow: 0.0162717159,
        mean_coef_deep: 0.0624599155,
        std_intercept: 0.9215301247,
        std_coef_shallow: -0.0643657002,
        std_coef_deep: 0.0100867014,
    },
    ProbcutParams {
        mean_intercept: -0.2995095922,
        mean_coef_shallow: 0.0357690455,
        mean_coef_deep: 0.0184001691,
        std_intercept: 0.9068128601,
        std_coef_shallow: -0.0622823051,
        std_coef_deep: 0.0116354143,
    },
    ProbcutParams {
        mean_intercept: -0.4214738079,
        mean_coef_shallow: 0.0368199917,
        mean_coef_deep: 0.0602632373,
        std_intercept: 0.8905333963,
        std_coef_shallow: -0.0499105626,
        std_coef_deep: 0.0109096420,
    },
    ProbcutParams {
        mean_intercept: -0.2614679363,
        mean_coef_shallow: 0.0292314784,
        mean_coef_deep: 0.0104484808,
        std_intercept: 0.8738545603,
        std_coef_shallow: -0.0474831267,
        std_coef_deep: 0.0126367287,
    },
    ProbcutParams {
        mean_intercept: -0.3196343251,
        mean_coef_shallow: 0.0366491118,
        mean_coef_deep: 0.0548628559,
        std_intercept: 0.9078237682,
        std_coef_shallow: -0.0544371665,
        std_coef_deep: 0.0117191207,
    },
    ProbcutParams {
        mean_intercept: -0.3528306622,
        mean_coef_shallow: 0.0543055155,
        mean_coef_deep: 0.0166223719,
        std_intercept: 0.9651997231,
        std_coef_shallow: -0.0667669126,
        std_coef_deep: 0.0114859664,
    },
    ProbcutParams {
        mean_intercept: -0.2327924719,
        mean_coef_shallow: 0.0398079219,
        mean_coef_deep: 0.0215399343,
        std_intercept: 0.8959020092,
        std_coef_shallow: -0.0538957916,
        std_coef_deep: 0.0148417767,
    },
    ProbcutParams {
        mean_intercept: -0.4877365519,
        mean_coef_shallow: 0.0798054443,
        mean_coef_deep: 0.0483340943,
        std_intercept: 0.9122935537,
        std_coef_shallow: -0.0589539100,
        std_coef_deep: 0.0161238400,
    },
    ProbcutParams {
        mean_intercept: -0.4357104199,
        mean_coef_shallow: 0.0658130455,
        mean_coef_deep: 0.0573278033,
        std_intercept: 0.8841087806,
        std_coef_shallow: -0.0591291450,
        std_coef_deep: 0.0205854111,
    },
    ProbcutParams {
        mean_intercept: -0.6308059957,
        mean_coef_shallow: 0.0877180972,
        mean_coef_deep: 0.0828635457,
        std_intercept: 0.8683305464,
        std_coef_shallow: -0.0612463445,
        std_coef_deep: 0.0268597881,
    },
    ProbcutParams {
        mean_intercept: -0.2764644035,
        mean_coef_shallow: 0.0359309767,
        mean_coef_deep: 0.0297889175,
        std_intercept: 0.9387319151,
        std_coef_shallow: -0.0652934057,
        std_coef_deep: 0.0224508842,
    },
    ProbcutParams {
        mean_intercept: -0.3041168015,
        mean_coef_shallow: 0.0373905863,
        mean_coef_deep: 0.0435192778,
        std_intercept: 0.9483959265,
        std_coef_shallow: -0.0688655570,
        std_coef_deep: 0.0216473590,
    },
    ProbcutParams {
        mean_intercept: -0.3786241821,
        mean_coef_shallow: 0.0357659523,
        mean_coef_deep: 0.0486305992,
        std_intercept: 0.9607922482,
        std_coef_shallow: -0.0695915566,
        std_coef_deep: 0.0213660565,
    },
    ProbcutParams {
        mean_intercept: -0.4033586053,
        mean_coef_shallow: 0.1007317569,
        mean_coef_deep: 0.0357515134,
        std_intercept: 0.9851099700,
        std_coef_shallow: -0.0705581325,
        std_coef_deep: 0.0191608349,
    },
    ProbcutParams {
        mean_intercept: -0.4471188218,
        mean_coef_shallow: 0.0245144260,
        mean_coef_deep: 0.0811342323,
        std_intercept: 0.9964868434,
        std_coef_shallow: -0.0785786707,
        std_coef_deep: 0.0196766205,
    },
    ProbcutParams {
        mean_intercept: -0.1953513350,
        mean_coef_shallow: 0.0607976450,
        mean_coef_deep: -0.0065608197,
        std_intercept: 1.0204893938,
        std_coef_shallow: -0.0767821532,
        std_coef_deep: 0.0175885985,
    },
    ProbcutParams {
        mean_intercept: -0.4451560546,
        mean_coef_shallow: 0.0044433244,
        mean_coef_deep: 0.0952448295,
        std_intercept: 1.0362448963,
        std_coef_shallow: -0.0781382346,
        std_coef_deep: 0.0180739950,
    },
    ProbcutParams {
        mean_intercept: -0.1649603141,
        mean_coef_shallow: 0.0233663170,
        mean_coef_deep: -0.0051751336,
        std_intercept: 1.0154256976,
        std_coef_shallow: -0.0740941140,
        std_coef_deep: 0.0208945456,
    },
    ProbcutParams {
        mean_intercept: -0.3834980658,
        mean_coef_shallow: 0.0616205183,
        mean_coef_deep: 0.0686791257,
        std_intercept: 1.0186208969,
        std_coef_shallow: -0.0720286320,
        std_coef_deep: 0.0199045300,
    },
    ProbcutParams {
        mean_intercept: -0.4385609904,
        mean_coef_shallow: 0.0001416742,
        mean_coef_deep: 0.0730584956,
        std_intercept: 1.0111964480,
        std_coef_shallow: -0.0708819726,
        std_coef_deep: 0.0218865564,
    },
    ProbcutParams {
        mean_intercept: -0.0011260966,
        mean_coef_shallow: 0.0377470743,
        mean_coef_deep: -0.0465389939,
        std_intercept: 1.0415616552,
        std_coef_shallow: -0.0723016892,
        std_coef_deep: 0.0219513541,
    },
    ProbcutParams {
        mean_intercept: -0.3957576447,
        mean_coef_shallow: 0.0202303579,
        mean_coef_deep: 0.0732592568,
        std_intercept: 1.1063508389,
        std_coef_shallow: -0.0752669164,
        std_coef_deep: 0.0186095123,
    },
    ProbcutParams {
        mean_intercept: -0.0772732052,
        mean_coef_shallow: 0.0548599540,
        mean_coef_deep: -0.0522614176,
        std_intercept: 1.1415672956,
        std_coef_shallow: -0.0819408983,
        std_coef_deep: 0.0203707522,
    },
    ProbcutParams {
        mean_intercept: -0.4784453166,
        mean_coef_shallow: 0.0553592571,
        mean_coef_deep: 0.1007978732,
        std_intercept: 1.1225751517,
        std_coef_shallow: -0.0790156250,
        std_coef_deep: 0.0215412930,
    },
    ProbcutParams {
        mean_intercept: -0.0765917076,
        mean_coef_shallow: 0.0446508127,
        mean_coef_deep: -0.0668158449,
        std_intercept: 1.1111631078,
        std_coef_shallow: -0.0736785385,
        std_coef_deep: 0.0224100643,
    },
    ProbcutParams {
        mean_intercept: -0.1502253916,
        mean_coef_shallow: 0.0775899796,
        mean_coef_deep: 0.0104472772,
        std_intercept: 1.0653196046,
        std_coef_shallow: -0.0631723732,
        std_coef_deep: 0.0237301391,
    },
    ProbcutParams {
        mean_intercept: -0.1993023410,
        mean_coef_shallow: 0.0501078329,
        mean_coef_deep: -0.0148756454,
        std_intercept: 1.0942805843,
        std_coef_shallow: -0.0663674315,
        std_coef_deep: 0.0241620359,
    },
    ProbcutParams {
        mean_intercept: -0.0304487090,
        mean_coef_shallow: 0.0428167812,
        mean_coef_deep: -0.0281267273,
        std_intercept: 1.1094914466,
        std_coef_shallow: -0.0613563117,
        std_coef_deep: 0.0249838112,
    },
    ProbcutParams {
        mean_intercept: -0.3285501912,
        mean_coef_shallow: 0.1027105131,
        mean_coef_deep: 0.0112373817,
        std_intercept: 1.1183228736,
        std_coef_shallow: -0.0652943767,
        std_coef_deep: 0.0264722766,
    },
    ProbcutParams {
        mean_intercept: 0.0284956072,
        mean_coef_shallow: 0.0485640973,
        mean_coef_deep: -0.0521169389,
        std_intercept: 1.1441195181,
        std_coef_shallow: -0.0696571863,
        std_coef_deep: 0.0252515941,
    },
    ProbcutParams {
        mean_intercept: -0.2661667024,
        mean_coef_shallow: 0.0898107230,
        mean_coef_deep: 0.0052474735,
        std_intercept: 1.1754526304,
        std_coef_shallow: -0.0748621696,
        std_coef_deep: 0.0227608029,
    },
    ProbcutParams {
        mean_intercept: 0.0004926980,
        mean_coef_shallow: 0.0038928261,
        mean_coef_deep: -0.0415715467,
        std_intercept: 1.1992754660,
        std_coef_shallow: -0.0804849704,
        std_coef_deep: 0.0207544216,
    },
    ProbcutParams {
        mean_intercept: -0.2017587822,
        mean_coef_shallow: 0.0857421600,
        mean_coef_deep: -0.0009230689,
        std_intercept: 1.2296994056,
        std_coef_shallow: -0.0888299739,
        std_coef_deep: 0.0180011974,
    },
    ProbcutParams {
        mean_intercept: -0.2208664352,
        mean_coef_shallow: 0.0774802652,
        mean_coef_deep: -0.0122029359,
        std_intercept: 1.2698423483,
        std_coef_shallow: -0.1059757697,
        std_coef_deep: 0.0149366199,
    },
    ProbcutParams {
        mean_intercept: -0.1476952475,
        mean_coef_shallow: 0.0306322345,
        mean_coef_deep: 0.0057395625,
        std_intercept: 1.2847873129,
        std_coef_shallow: -0.1284490880,
        std_coef_deep: 0.0125201138,
    },
    ProbcutParams {
        mean_intercept: -0.2159182303,
        mean_coef_shallow: 0.0420485280,
        mean_coef_deep: -0.0009107836,
        std_intercept: 1.3819349885,
        std_coef_shallow: -0.1968046130,
        std_coef_deep: 0.0090527615,
    },
    ProbcutParams {
        mean_intercept: -0.0512741670,
        mean_coef_shallow: 0.0105473789,
        mean_coef_deep: -0.0054611135,
        std_intercept: 1.7317519008,
        std_coef_shallow: -0.4139909242,
        std_coef_deep: -0.0000038648,
    },
    ProbcutParams {
        mean_intercept: -0.2367699262,
        mean_coef_shallow: -0.0033881554,
        mean_coef_deep: 0.0120923849,
        std_intercept: 1.3975603542,
        std_coef_shallow: -0.3391704205,
        std_coef_deep: 0.0032939479,
    },
    ProbcutParams {
        mean_intercept: -0.1440423234,
        mean_coef_shallow: -0.0016377396,
        mean_coef_deep: 0.0021133754,
        std_intercept: 1.4242398081,
        std_coef_shallow: -0.5777034532,
        std_coef_deep: 0.0018806391,
    },
    ProbcutParams {
        mean_intercept: 0.0197887130,
        mean_coef_shallow: 0.0000663472,
        mean_coef_deep: -0.0000576299,
        std_intercept: 1.1979868388,
        std_coef_shallow: -0.8339416234,
        std_coef_deep: -0.0001572716,
    },
    ProbcutParams {
        mean_intercept: -0.0595405476,
        mean_coef_shallow: -0.0001370766,
        mean_coef_deep: 0.0001201295,
        std_intercept: 0.4397657131,
        std_coef_shallow: -0.7768852121,
        std_coef_deep: -0.0004728749,
    },
    ProbcutParams {
        mean_intercept: 0.1266396458,
        mean_coef_shallow: 0.0001861600,
        mean_coef_deep: -0.0001634129,
        std_intercept: -0.7317989073,
        std_coef_shallow: -0.6733904722,
        std_coef_deep: -0.0008967669,
    },
    ProbcutParams {
        mean_intercept: 0.0000000000,
        mean_coef_shallow: 0.0000000000,
        mean_coef_deep: 0.0000000000,
        std_intercept: -18.4206807440,
        std_coef_shallow: -0.0000000000,
        std_coef_deep: 0.0000000000,
    },
];
