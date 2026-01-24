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
        mean_intercept: -0.0845736881,
        mean_coef_shallow: -0.0246248958,
        mean_coef_deep: 0.0033425205,
        std_intercept: -1.9322199184,
        std_coef_shallow: 0.0353416885,
        std_coef_deep: 0.0521757593,
    },
    ProbcutParams {
        mean_intercept: -1.0534929342,
        mean_coef_shallow: 0.0237317974,
        mean_coef_deep: 0.0864239660,
        std_intercept: -1.2188892346,
        std_coef_shallow: -0.0222358634,
        std_coef_deep: 0.1113808643,
    },
    ProbcutParams {
        mean_intercept: -0.3007066400,
        mean_coef_shallow: 0.0690096764,
        mean_coef_deep: -0.0166538176,
        std_intercept: -0.4283898576,
        std_coef_shallow: -0.0008861718,
        std_coef_deep: 0.0447292435,
    },
    ProbcutParams {
        mean_intercept: 0.4725234109,
        mean_coef_shallow: -0.0080591866,
        mean_coef_deep: -0.0544913420,
        std_intercept: -0.2234342452,
        std_coef_shallow: 0.0044493323,
        std_coef_deep: 0.0423991404,
    },
    ProbcutParams {
        mean_intercept: 0.3925963053,
        mean_coef_shallow: 0.0039801014,
        mean_coef_deep: -0.0581515329,
        std_intercept: 0.2381498060,
        std_coef_shallow: 0.0073669702,
        std_coef_deep: 0.0117942667,
    },
    ProbcutParams {
        mean_intercept: 0.0629975486,
        mean_coef_shallow: 0.0718756133,
        mean_coef_deep: -0.0340017721,
        std_intercept: 0.3212531302,
        std_coef_shallow: 0.0016128582,
        std_coef_deep: 0.0107579321,
    },
    ProbcutParams {
        mean_intercept: 0.4777680846,
        mean_coef_shallow: -0.0265918207,
        mean_coef_deep: -0.0742238609,
        std_intercept: 0.4270479955,
        std_coef_shallow: -0.0039068453,
        std_coef_deep: 0.0106006745,
    },
    ProbcutParams {
        mean_intercept: 0.3339672758,
        mean_coef_shallow: 0.0035428048,
        mean_coef_deep: -0.0538328207,
        std_intercept: 0.5371976392,
        std_coef_shallow: -0.0262979821,
        std_coef_deep: 0.0149992258,
    },
    ProbcutParams {
        mean_intercept: 0.3628652141,
        mean_coef_shallow: 0.0048098154,
        mean_coef_deep: -0.0666340340,
        std_intercept: 0.7166221677,
        std_coef_shallow: -0.0461437368,
        std_coef_deep: 0.0094949481,
    },
    ProbcutParams {
        mean_intercept: 0.1888423120,
        mean_coef_shallow: 0.0367728392,
        mean_coef_deep: -0.0341297327,
        std_intercept: 0.8035871325,
        std_coef_shallow: -0.0539093627,
        std_coef_deep: 0.0067498889,
    },
    ProbcutParams {
        mean_intercept: 0.2155904038,
        mean_coef_shallow: 0.0232361615,
        mean_coef_deep: -0.0441095709,
        std_intercept: 0.8623869490,
        std_coef_shallow: -0.0537789670,
        std_coef_deep: 0.0063713319,
    },
    ProbcutParams {
        mean_intercept: 0.2410192406,
        mean_coef_shallow: 0.0394636136,
        mean_coef_deep: -0.0551581043,
        std_intercept: 0.9104101696,
        std_coef_shallow: -0.0599571314,
        std_coef_deep: 0.0065420114,
    },
    ProbcutParams {
        mean_intercept: 0.1375402206,
        mean_coef_shallow: 0.0605708079,
        mean_coef_deep: -0.0389132365,
        std_intercept: 0.9457801433,
        std_coef_shallow: -0.0537068526,
        std_coef_deep: 0.0061434038,
    },
    ProbcutParams {
        mean_intercept: 0.1904708260,
        mean_coef_shallow: 0.0673058858,
        mean_coef_deep: -0.0580121100,
        std_intercept: 0.9208615461,
        std_coef_shallow: -0.0584068510,
        std_coef_deep: 0.0113415062,
    },
    ProbcutParams {
        mean_intercept: 0.1826265533,
        mean_coef_shallow: 0.0488677837,
        mean_coef_deep: -0.0482010058,
        std_intercept: 0.9568038687,
        std_coef_shallow: -0.0575570367,
        std_coef_deep: 0.0115641596,
    },
    ProbcutParams {
        mean_intercept: 0.1667092679,
        mean_coef_shallow: 0.0976037745,
        mean_coef_deep: -0.0779431909,
        std_intercept: 1.0000948447,
        std_coef_shallow: -0.0717855023,
        std_coef_deep: 0.0116355790,
    },
    ProbcutParams {
        mean_intercept: 0.1337636040,
        mean_coef_shallow: 0.0941287433,
        mean_coef_deep: -0.0580282776,
        std_intercept: 1.0828308273,
        std_coef_shallow: -0.0858747768,
        std_coef_deep: 0.0106156550,
    },
    ProbcutParams {
        mean_intercept: 0.1395528061,
        mean_coef_shallow: 0.1164897252,
        mean_coef_deep: -0.0843551342,
        std_intercept: 1.0917537649,
        std_coef_shallow: -0.0866948517,
        std_coef_deep: 0.0084885697,
    },
    ProbcutParams {
        mean_intercept: 0.1202349966,
        mean_coef_shallow: 0.1119889327,
        mean_coef_deep: -0.0615045352,
        std_intercept: 1.0889230204,
        std_coef_shallow: -0.0835667153,
        std_coef_deep: 0.0081537496,
    },
    ProbcutParams {
        mean_intercept: 0.1311088534,
        mean_coef_shallow: 0.1351723835,
        mean_coef_deep: -0.0939261992,
        std_intercept: 1.0573282674,
        std_coef_shallow: -0.0779752108,
        std_coef_deep: 0.0110502972,
    },
    ProbcutParams {
        mean_intercept: 0.1075156445,
        mean_coef_shallow: 0.0997364328,
        mean_coef_deep: -0.0562441107,
        std_intercept: 1.0409528149,
        std_coef_shallow: -0.0726007095,
        std_coef_deep: 0.0119322606,
    },
    ProbcutParams {
        mean_intercept: 0.0869240420,
        mean_coef_shallow: 0.1293821366,
        mean_coef_deep: -0.0802365202,
        std_intercept: 1.0857644066,
        std_coef_shallow: -0.0742529345,
        std_coef_deep: 0.0091113116,
    },
    ProbcutParams {
        mean_intercept: 0.1315101056,
        mean_coef_shallow: 0.0984600144,
        mean_coef_deep: -0.0648007188,
        std_intercept: 1.0498906176,
        std_coef_shallow: -0.0685206276,
        std_coef_deep: 0.0126212732,
    },
    ProbcutParams {
        mean_intercept: 0.0749973053,
        mean_coef_shallow: 0.1250448300,
        mean_coef_deep: -0.0732453991,
        std_intercept: 1.0501258047,
        std_coef_shallow: -0.0667771939,
        std_coef_deep: 0.0137272922,
    },
    ProbcutParams {
        mean_intercept: 0.0644414062,
        mean_coef_shallow: 0.1004153401,
        mean_coef_deep: -0.0503867020,
        std_intercept: 1.0791038044,
        std_coef_shallow: -0.0700139186,
        std_coef_deep: 0.0132375811,
    },
    ProbcutParams {
        mean_intercept: 0.0679532680,
        mean_coef_shallow: 0.0885281078,
        mean_coef_deep: -0.0489626750,
        std_intercept: 1.0667908426,
        std_coef_shallow: -0.0718138200,
        std_coef_deep: 0.0175299897,
    },
    ProbcutParams {
        mean_intercept: 0.0817153575,
        mean_coef_shallow: 0.0969339096,
        mean_coef_deep: -0.0594737051,
        std_intercept: 1.0861476373,
        std_coef_shallow: -0.0756106685,
        std_coef_deep: 0.0190870247,
    },
    ProbcutParams {
        mean_intercept: 0.0436416283,
        mean_coef_shallow: 0.1013480016,
        mean_coef_deep: -0.0499639754,
        std_intercept: 1.0848046637,
        std_coef_shallow: -0.0745250357,
        std_coef_deep: 0.0195058210,
    },
    ProbcutParams {
        mean_intercept: 0.1186013068,
        mean_coef_shallow: 0.0907245166,
        mean_coef_deep: -0.0711446562,
        std_intercept: 1.0910638535,
        std_coef_shallow: -0.0744637751,
        std_coef_deep: 0.0211501498,
    },
    ProbcutParams {
        mean_intercept: 0.0561794806,
        mean_coef_shallow: 0.0867385579,
        mean_coef_deep: -0.0424523531,
        std_intercept: 1.0991664501,
        std_coef_shallow: -0.0715297609,
        std_coef_deep: 0.0207325051,
    },
    ProbcutParams {
        mean_intercept: 0.1017772453,
        mean_coef_shallow: 0.0932370088,
        mean_coef_deep: -0.0807045512,
        std_intercept: 1.1178892308,
        std_coef_shallow: -0.0711448072,
        std_coef_deep: 0.0208236835,
    },
    ProbcutParams {
        mean_intercept: 0.0566584092,
        mean_coef_shallow: 0.0651000057,
        mean_coef_deep: -0.0309645814,
        std_intercept: 1.1669113861,
        std_coef_shallow: -0.0811608609,
        std_coef_deep: 0.0195842104,
    },
    ProbcutParams {
        mean_intercept: 0.0805666361,
        mean_coef_shallow: 0.0951858166,
        mean_coef_deep: -0.0749513698,
        std_intercept: 1.1991573227,
        std_coef_shallow: -0.0840012254,
        std_coef_deep: 0.0178074536,
    },
    ProbcutParams {
        mean_intercept: 0.0744432617,
        mean_coef_shallow: 0.0737855579,
        mean_coef_deep: -0.0396402708,
        std_intercept: 1.1920407006,
        std_coef_shallow: -0.0838891488,
        std_coef_deep: 0.0202063217,
    },
    ProbcutParams {
        mean_intercept: 0.0351166152,
        mean_coef_shallow: 0.0967659718,
        mean_coef_deep: -0.0629492439,
        std_intercept: 1.2201355828,
        std_coef_shallow: -0.0872791473,
        std_coef_deep: 0.0199932543,
    },
    ProbcutParams {
        mean_intercept: 0.0666860447,
        mean_coef_shallow: 0.0891744207,
        mean_coef_deep: -0.0573705653,
        std_intercept: 1.2095059976,
        std_coef_shallow: -0.0819098044,
        std_coef_deep: 0.0210095785,
    },
    ProbcutParams {
        mean_intercept: 0.0536999781,
        mean_coef_shallow: 0.1314590281,
        mean_coef_deep: -0.0954015012,
        std_intercept: 1.2455111342,
        std_coef_shallow: -0.0868657709,
        std_coef_deep: 0.0213219626,
    },
    ProbcutParams {
        mean_intercept: 0.0059342616,
        mean_coef_shallow: 0.0954975036,
        mean_coef_deep: -0.0423256931,
        std_intercept: 1.2481756710,
        std_coef_shallow: -0.0896889361,
        std_coef_deep: 0.0231531573,
    },
    ProbcutParams {
        mean_intercept: 0.0305471230,
        mean_coef_shallow: 0.1486323854,
        mean_coef_deep: -0.0944670044,
        std_intercept: 1.2795219952,
        std_coef_shallow: -0.1003947551,
        std_coef_deep: 0.0258452454,
    },
    ProbcutParams {
        mean_intercept: 0.0309361390,
        mean_coef_shallow: 0.0749813717,
        mean_coef_deep: -0.0387947883,
        std_intercept: 1.3003878350,
        std_coef_shallow: -0.1067917917,
        std_coef_deep: 0.0240741398,
    },
    ProbcutParams {
        mean_intercept: 0.0578642840,
        mean_coef_shallow: 0.1556919010,
        mean_coef_deep: -0.1125129521,
        std_intercept: 1.2837412259,
        std_coef_shallow: -0.0993770301,
        std_coef_deep: 0.0228534457,
    },
    ProbcutParams {
        mean_intercept: 0.0024805099,
        mean_coef_shallow: 0.0521855685,
        mean_coef_deep: -0.0083511649,
        std_intercept: 1.2261492005,
        std_coef_shallow: -0.0902034224,
        std_coef_deep: 0.0284292128,
    },
    ProbcutParams {
        mean_intercept: 0.0116503411,
        mean_coef_shallow: 0.0909130339,
        mean_coef_deep: -0.0762774796,
        std_intercept: 1.1956513209,
        std_coef_shallow: -0.0787563931,
        std_coef_deep: 0.0324488129,
    },
    ProbcutParams {
        mean_intercept: 0.1276437020,
        mean_coef_shallow: 0.0818198470,
        mean_coef_deep: -0.0963661366,
        std_intercept: 1.1814676511,
        std_coef_shallow: -0.0809839533,
        std_coef_deep: 0.0355751865,
    },
    ProbcutParams {
        mean_intercept: -0.0192924652,
        mean_coef_shallow: 0.0851649030,
        mean_coef_deep: -0.0561926976,
        std_intercept: 1.1355306284,
        std_coef_shallow: -0.0711119967,
        std_coef_deep: 0.0404994703,
    },
    ProbcutParams {
        mean_intercept: 0.1219806500,
        mean_coef_shallow: 0.0406381242,
        mean_coef_deep: -0.0970145901,
        std_intercept: 1.1635030646,
        std_coef_shallow: -0.0673198532,
        std_coef_deep: 0.0386337715,
    },
    ProbcutParams {
        mean_intercept: -0.0486068524,
        mean_coef_shallow: 0.0620469789,
        mean_coef_deep: -0.0460186507,
        std_intercept: 1.1812136332,
        std_coef_shallow: -0.0673812384,
        std_coef_deep: 0.0357092386,
    },
    ProbcutParams {
        mean_intercept: 0.1216034249,
        mean_coef_shallow: 0.0243575080,
        mean_coef_deep: -0.0848899183,
        std_intercept: 1.2258059543,
        std_coef_shallow: -0.0678800782,
        std_coef_deep: 0.0307972456,
    },
    ProbcutParams {
        mean_intercept: -0.1063893434,
        mean_coef_shallow: 0.0601811331,
        mean_coef_deep: -0.0166652449,
        std_intercept: 1.2930191086,
        std_coef_shallow: -0.0686207809,
        std_coef_deep: 0.0246269689,
    },
    ProbcutParams {
        mean_intercept: 0.0196125307,
        mean_coef_shallow: 0.0081573158,
        mean_coef_deep: -0.0534063752,
        std_intercept: 1.3192266533,
        std_coef_shallow: -0.0802483999,
        std_coef_deep: 0.0223976884,
    },
    ProbcutParams {
        mean_intercept: -0.0451696137,
        mean_coef_shallow: 0.0124736624,
        mean_coef_deep: -0.0090337187,
        std_intercept: 1.6650865230,
        std_coef_shallow: -0.3045255816,
        std_coef_deep: 0.0182887783,
    },
    ProbcutParams {
        mean_intercept: -0.1399857999,
        mean_coef_shallow: 0.0459624514,
        mean_coef_deep: -0.0193362187,
        std_intercept: 1.8057852369,
        std_coef_shallow: -0.4463361758,
        std_coef_deep: 0.0149378796,
    },
    ProbcutParams {
        mean_intercept: -0.0390740581,
        mean_coef_shallow: -0.0162246481,
        mean_coef_deep: 0.0107354686,
        std_intercept: 2.0648035468,
        std_coef_shallow: -0.7007819683,
        std_coef_deep: 0.0118336497,
    },
    ProbcutParams {
        mean_intercept: -0.2054976391,
        mean_coef_shallow: 0.0152209456,
        mean_coef_deep: -0.0051132079,
        std_intercept: 1.8073585127,
        std_coef_shallow: -0.6417698298,
        std_coef_deep: 0.0128399395,
    },
    ProbcutParams {
        mean_intercept: -0.0392584732,
        mean_coef_shallow: -0.0005420706,
        mean_coef_deep: 0.0003806570,
        std_intercept: 1.7202920524,
        std_coef_shallow: -0.9898553948,
        std_coef_deep: 0.0000970926,
    },
    ProbcutParams {
        mean_intercept: -0.1100170797,
        mean_coef_shallow: -0.0010008201,
        mean_coef_deep: 0.0008303257,
        std_intercept: 1.6178407691,
        std_coef_shallow: -0.9303770000,
        std_coef_deep: 0.0000557814,
    },
    ProbcutParams {
        mean_intercept: -0.0667264391,
        mean_coef_shallow: -0.0001644273,
        mean_coef_deep: 0.0001407421,
        std_intercept: -0.0024519485,
        std_coef_shallow: -0.8872527811,
        std_coef_deep: 0.0126018159,
    },
    ProbcutParams {
        mean_intercept: 0.0516369635,
        mean_coef_shallow: 0.0000807170,
        mean_coef_deep: -0.0000580063,
        std_intercept: -1.4594731871,
        std_coef_shallow: -0.8024882619,
        std_coef_deep: 0.0000762469,
    },
    ProbcutParams {
        mean_intercept: -0.1047514369,
        mean_coef_shallow: -0.0001979580,
        mean_coef_deep: 0.0001432684,
        std_intercept: -1.6816104370,
        std_coef_shallow: -0.7312201377,
        std_coef_deep: 0.0016577036,
    },
];
