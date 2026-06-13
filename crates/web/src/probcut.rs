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
                return Some(beta);
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
    mean_intercept: 0.3799414600,
    mean_coef_shallow: -0.4311058217,
    mean_coef_deep: -0.0277298037,
    std_intercept: 0.7386852901,
    std_coef_shallow: 0.0000000000,
    std_coef_deep: 0.0289630793,
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
        mean_intercept: -0.0644217738,
        mean_coef_shallow: -0.0839105649,
        mean_coef_deep: 0.0304419615,
        std_intercept: -0.8780667763,
        std_coef_shallow: 0.0406628308,
        std_coef_deep: -0.0970470670,
    },
    ProbcutParams {
        mean_intercept: 0.1153363659,
        mean_coef_shallow: 0.0691514678,
        mean_coef_deep: -0.0329509652,
        std_intercept: 0.1535802800,
        std_coef_shallow: 0.0633178477,
        std_coef_deep: -0.0256817791,
    },
    ProbcutParams {
        mean_intercept: -0.1681465770,
        mean_coef_shallow: 0.0054994923,
        mean_coef_deep: 0.0443014343,
        std_intercept: 0.5033716206,
        std_coef_shallow: 0.1247518574,
        std_coef_deep: -0.0888972702,
    },
    ProbcutParams {
        mean_intercept: -0.3293709245,
        mean_coef_shallow: 0.1844408894,
        mean_coef_deep: -0.0021166952,
        std_intercept: 0.3444528675,
        std_coef_shallow: 0.1171078685,
        std_coef_deep: -0.0586839302,
    },
    ProbcutParams {
        mean_intercept: -0.0475670575,
        mean_coef_shallow: -0.0402510959,
        mean_coef_deep: 0.0451741456,
        std_intercept: 0.3725462982,
        std_coef_shallow: 0.0671576403,
        std_coef_deep: -0.0365199751,
    },
    ProbcutParams {
        mean_intercept: 0.0712474529,
        mean_coef_shallow: -0.0360582851,
        mean_coef_deep: 0.0025966894,
        std_intercept: 0.5043865827,
        std_coef_shallow: 0.0026183603,
        std_coef_deep: -0.0248297157,
    },
    ProbcutParams {
        mean_intercept: -0.0943926679,
        mean_coef_shallow: 0.0466447476,
        mean_coef_deep: 0.0210214566,
        std_intercept: 0.5498225871,
        std_coef_shallow: -0.0433753180,
        std_coef_deep: -0.0071732601,
    },
    ProbcutParams {
        mean_intercept: -0.0576236746,
        mean_coef_shallow: 0.0370505048,
        mean_coef_deep: 0.0013407030,
        std_intercept: 0.5470486719,
        std_coef_shallow: -0.0698815147,
        std_coef_deep: 0.0059966816,
    },
    ProbcutParams {
        mean_intercept: 0.1736992873,
        mean_coef_shallow: -0.0370103793,
        mean_coef_deep: 0.0084045679,
        std_intercept: 0.5717634819,
        std_coef_shallow: -0.0908048296,
        std_coef_deep: 0.0170818704,
    },
    ProbcutParams {
        mean_intercept: 0.0873188918,
        mean_coef_shallow: -0.0196005756,
        mean_coef_deep: 0.0029767805,
        std_intercept: 0.6215047118,
        std_coef_shallow: -0.1083445341,
        std_coef_deep: 0.0226922893,
    },
    ProbcutParams {
        mean_intercept: 0.2750028358,
        mean_coef_shallow: -0.0283373449,
        mean_coef_deep: -0.0379505472,
        std_intercept: 0.7001219707,
        std_coef_shallow: -0.1218173877,
        std_coef_deep: 0.0227713275,
    },
    ProbcutParams {
        mean_intercept: 0.2971584544,
        mean_coef_shallow: -0.0327336780,
        mean_coef_deep: -0.0443285100,
        std_intercept: 0.7210410684,
        std_coef_shallow: -0.1190148309,
        std_coef_deep: 0.0238930065,
    },
    ProbcutParams {
        mean_intercept: 0.1462107139,
        mean_coef_shallow: -0.0575737039,
        mean_coef_deep: -0.0037112744,
        std_intercept: 0.7632373570,
        std_coef_shallow: -0.1129352219,
        std_coef_deep: 0.0206563079,
    },
    ProbcutParams {
        mean_intercept: 0.1657684886,
        mean_coef_shallow: -0.0148758744,
        mean_coef_deep: -0.0370953235,
        std_intercept: 0.7762576431,
        std_coef_shallow: -0.1173326292,
        std_coef_deep: 0.0222470903,
    },
    ProbcutParams {
        mean_intercept: 0.1625985202,
        mean_coef_shallow: 0.0136270504,
        mean_coef_deep: -0.0461498444,
        std_intercept: 0.7581832916,
        std_coef_shallow: -0.1118359691,
        std_coef_deep: 0.0252203545,
    },
    ProbcutParams {
        mean_intercept: 0.1089021328,
        mean_coef_shallow: -0.0163470614,
        mean_coef_deep: -0.0321643154,
        std_intercept: 0.8187085746,
        std_coef_shallow: -0.1197378661,
        std_coef_deep: 0.0230978140,
    },
    ProbcutParams {
        mean_intercept: 0.1863509169,
        mean_coef_shallow: 0.0020991338,
        mean_coef_deep: -0.0584267059,
        std_intercept: 0.8578213211,
        std_coef_shallow: -0.1214423291,
        std_coef_deep: 0.0217515644,
    },
    ProbcutParams {
        mean_intercept: 0.0534675060,
        mean_coef_shallow: 0.0375860492,
        mean_coef_deep: -0.0419961630,
        std_intercept: 0.8579477424,
        std_coef_shallow: -0.1171789351,
        std_coef_deep: 0.0208580780,
    },
    ProbcutParams {
        mean_intercept: 0.1355196781,
        mean_coef_shallow: 0.0178181418,
        mean_coef_deep: -0.0670845860,
        std_intercept: 0.8616728256,
        std_coef_shallow: -0.1097144269,
        std_coef_deep: 0.0200578361,
    },
    ProbcutParams {
        mean_intercept: 0.0349485385,
        mean_coef_shallow: 0.0232324421,
        mean_coef_deep: -0.0364689298,
        std_intercept: 0.8476598639,
        std_coef_shallow: -0.1033288981,
        std_coef_deep: 0.0206482572,
    },
    ProbcutParams {
        mean_intercept: 0.0885062601,
        mean_coef_shallow: 0.0386346715,
        mean_coef_deep: -0.0726722305,
        std_intercept: 0.8597389698,
        std_coef_shallow: -0.1012996269,
        std_coef_deep: 0.0200874476,
    },
    ProbcutParams {
        mean_intercept: 0.1819298451,
        mean_coef_shallow: 0.0368659754,
        mean_coef_deep: -0.0880496169,
        std_intercept: 0.8674574317,
        std_coef_shallow: -0.1012451855,
        std_coef_deep: 0.0219470434,
    },
    ProbcutParams {
        mean_intercept: 0.0700908442,
        mean_coef_shallow: 0.0153669167,
        mean_coef_deep: -0.0505137974,
        std_intercept: 0.8361647918,
        std_coef_shallow: -0.0925667327,
        std_coef_deep: 0.0258981132,
    },
    ProbcutParams {
        mean_intercept: 0.0998694576,
        mean_coef_shallow: 0.0251140217,
        mean_coef_deep: -0.0679613056,
        std_intercept: 0.8452162749,
        std_coef_shallow: -0.0917720418,
        std_coef_deep: 0.0265919035,
    },
    ProbcutParams {
        mean_intercept: 0.0621880539,
        mean_coef_shallow: 0.0214100616,
        mean_coef_deep: -0.0475885189,
        std_intercept: 0.8770256454,
        std_coef_shallow: -0.0907105061,
        std_coef_deep: 0.0244345453,
    },
    ProbcutParams {
        mean_intercept: 0.1053983545,
        mean_coef_shallow: -0.0002674857,
        mean_coef_deep: -0.0636183193,
        std_intercept: 0.8954526504,
        std_coef_shallow: -0.0962289057,
        std_coef_deep: 0.0271063850,
    },
    ProbcutParams {
        mean_intercept: 0.0818296621,
        mean_coef_shallow: -0.0077986559,
        mean_coef_deep: -0.0433388546,
        std_intercept: 0.9347623345,
        std_coef_shallow: -0.0969380703,
        std_coef_deep: 0.0244341287,
    },
    ProbcutParams {
        mean_intercept: 0.0646263876,
        mean_coef_shallow: 0.0087947798,
        mean_coef_deep: -0.0553427270,
        std_intercept: 0.9623714217,
        std_coef_shallow: -0.0887940421,
        std_coef_deep: 0.0201820030,
    },
    ProbcutParams {
        mean_intercept: 0.1504757495,
        mean_coef_shallow: 0.0064334816,
        mean_coef_deep: -0.0741664372,
        std_intercept: 0.9589574250,
        std_coef_shallow: -0.0952964792,
        std_coef_deep: 0.0242128570,
    },
    ProbcutParams {
        mean_intercept: 0.0481287511,
        mean_coef_shallow: 0.0123863241,
        mean_coef_deep: -0.0441900168,
        std_intercept: 0.9566821154,
        std_coef_shallow: -0.0967955149,
        std_coef_deep: 0.0268458709,
    },
    ProbcutParams {
        mean_intercept: 0.0780309137,
        mean_coef_shallow: 0.0079749223,
        mean_coef_deep: -0.0580808008,
        std_intercept: 0.9572268498,
        std_coef_shallow: -0.0987267691,
        std_coef_deep: 0.0295276626,
    },
    ProbcutParams {
        mean_intercept: 0.0678546263,
        mean_coef_shallow: 0.0004012738,
        mean_coef_deep: -0.0407287983,
        std_intercept: 0.9912900446,
        std_coef_shallow: -0.1019015947,
        std_coef_deep: 0.0278384485,
    },
    ProbcutParams {
        mean_intercept: 0.1093161438,
        mean_coef_shallow: -0.0017994448,
        mean_coef_deep: -0.0666476025,
        std_intercept: 1.0031511572,
        std_coef_shallow: -0.0996541158,
        std_coef_deep: 0.0274190821,
    },
    ProbcutParams {
        mean_intercept: 0.0942009539,
        mean_coef_shallow: 0.0073635254,
        mean_coef_deep: -0.0594668079,
        std_intercept: 0.9883165731,
        std_coef_shallow: -0.0978781596,
        std_coef_deep: 0.0304092167,
    },
    ProbcutParams {
        mean_intercept: 0.0612576479,
        mean_coef_shallow: 0.0245653011,
        mean_coef_deep: -0.0611901973,
        std_intercept: 1.0307632123,
        std_coef_shallow: -0.1029119261,
        std_coef_deep: 0.0290347922,
    },
    ProbcutParams {
        mean_intercept: 0.1233480274,
        mean_coef_shallow: 0.0247091939,
        mean_coef_deep: -0.0818955741,
        std_intercept: 1.0354016493,
        std_coef_shallow: -0.1032292124,
        std_coef_deep: 0.0309422011,
    },
    ProbcutParams {
        mean_intercept: 0.0876058161,
        mean_coef_shallow: 0.0511957015,
        mean_coef_deep: -0.0905777547,
        std_intercept: 1.0264799025,
        std_coef_shallow: -0.0972267862,
        std_coef_deep: 0.0318487135,
    },
    ProbcutParams {
        mean_intercept: 0.0719341353,
        mean_coef_shallow: 0.0272926729,
        mean_coef_deep: -0.0636337324,
        std_intercept: 1.0562762044,
        std_coef_shallow: -0.0957539845,
        std_coef_deep: 0.0309299821,
    },
    ProbcutParams {
        mean_intercept: 0.0691629923,
        mean_coef_shallow: 0.0364284294,
        mean_coef_deep: -0.0826499596,
        std_intercept: 1.0673684721,
        std_coef_shallow: -0.0954850853,
        std_coef_deep: 0.0316094461,
    },
    ProbcutParams {
        mean_intercept: 0.0643602384,
        mean_coef_shallow: 0.0065208817,
        mean_coef_deep: -0.0564088178,
        std_intercept: 1.0906063319,
        std_coef_shallow: -0.0955466359,
        std_coef_deep: 0.0302774919,
    },
    ProbcutParams {
        mean_intercept: 0.1142700194,
        mean_coef_shallow: 0.0394560146,
        mean_coef_deep: -0.0964946634,
        std_intercept: 1.1133727039,
        std_coef_shallow: -0.0999403960,
        std_coef_deep: 0.0301538071,
    },
    ProbcutParams {
        mean_intercept: 0.0238005135,
        mean_coef_shallow: -0.0058570796,
        mean_coef_deep: -0.0292893323,
        std_intercept: 1.1039175950,
        std_coef_shallow: -0.0998910035,
        std_coef_deep: 0.0334750509,
    },
    ProbcutParams {
        mean_intercept: 0.1398934464,
        mean_coef_shallow: 0.0324674165,
        mean_coef_deep: -0.1084404036,
        std_intercept: 1.0987417582,
        std_coef_shallow: -0.0954985610,
        std_coef_deep: 0.0342206248,
    },
    ProbcutParams {
        mean_intercept: 0.0153749695,
        mean_coef_shallow: 0.0069926404,
        mean_coef_deep: -0.0280790571,
        std_intercept: 1.1028075459,
        std_coef_shallow: -0.0901927863,
        std_coef_deep: 0.0358688733,
    },
    ProbcutParams {
        mean_intercept: 0.1027365598,
        mean_coef_shallow: 0.0243686468,
        mean_coef_deep: -0.0979983436,
        std_intercept: 1.0794992323,
        std_coef_shallow: -0.0905899755,
        std_coef_deep: 0.0405882411,
    },
    ProbcutParams {
        mean_intercept: 0.0318412271,
        mean_coef_shallow: -0.0136371383,
        mean_coef_deep: -0.0186159771,
        std_intercept: 1.0846989765,
        std_coef_shallow: -0.0923597104,
        std_coef_deep: 0.0428601598,
    },
    ProbcutParams {
        mean_intercept: 0.0576135804,
        mean_coef_shallow: 0.0245997540,
        mean_coef_deep: -0.0914296467,
        std_intercept: 1.1071928516,
        std_coef_shallow: -0.0964601845,
        std_coef_deep: 0.0436968290,
    },
    ProbcutParams {
        mean_intercept: 0.0596617156,
        mean_coef_shallow: -0.0290982276,
        mean_coef_deep: -0.0220009091,
        std_intercept: 1.1016819650,
        std_coef_shallow: -0.0911369321,
        std_coef_deep: 0.0419244000,
    },
    ProbcutParams {
        mean_intercept: -0.0285119414,
        mean_coef_shallow: 0.0253438459,
        mean_coef_deep: -0.0510961214,
        std_intercept: 1.1164099797,
        std_coef_shallow: -0.0893972936,
        std_coef_deep: 0.0412045395,
    },
    ProbcutParams {
        mean_intercept: 0.1464892731,
        mean_coef_shallow: 0.0033349841,
        mean_coef_deep: -0.0899637091,
        std_intercept: 1.1352459860,
        std_coef_shallow: -0.0908806337,
        std_coef_deep: 0.0381354966,
    },
    ProbcutParams {
        mean_intercept: -0.0737422073,
        mean_coef_shallow: 0.0289039270,
        mean_coef_deep: -0.0125679876,
        std_intercept: 1.1820173050,
        std_coef_shallow: -0.1021229148,
        std_coef_deep: 0.0327732109,
    },
    ProbcutParams {
        mean_intercept: 0.0330551146,
        mean_coef_shallow: 0.0118125850,
        mean_coef_deep: -0.0556301727,
        std_intercept: 1.2167796082,
        std_coef_shallow: -0.1213719571,
        std_coef_deep: 0.0254848350,
    },
    ProbcutParams {
        mean_intercept: -0.0828273571,
        mean_coef_shallow: 0.0471072476,
        mean_coef_deep: -0.0102271838,
        std_intercept: 1.2858229886,
        std_coef_shallow: -0.1647293544,
        std_coef_deep: 0.0157921947,
    },
    ProbcutParams {
        mean_intercept: -0.0645210226,
        mean_coef_shallow: 0.0507770397,
        mean_coef_deep: -0.0288942390,
        std_intercept: 1.3678997238,
        std_coef_shallow: -0.2712429629,
        std_coef_deep: 0.0086854478,
    },
    ProbcutParams {
        mean_intercept: -0.0477140722,
        mean_coef_shallow: 0.0113440610,
        mean_coef_deep: -0.0066939998,
        std_intercept: 1.6864132022,
        std_coef_shallow: -0.7157986574,
        std_coef_deep: 0.0086946772,
    },
    ProbcutParams {
        mean_intercept: -0.0936022252,
        mean_coef_shallow: -0.0045134773,
        mean_coef_deep: 0.0031396229,
        std_intercept: 1.6334857239,
        std_coef_shallow: -0.9457001115,
        std_coef_deep: -0.0015830356,
    },
    ProbcutParams {
        mean_intercept: -0.1062581627,
        mean_coef_shallow: -0.0000610273,
        mean_coef_deep: 0.0001128441,
        std_intercept: 1.3573434436,
        std_coef_shallow: -1.3100703572,
        std_coef_deep: -0.0003085781,
    },
    ProbcutParams {
        mean_intercept: 0.0261380465,
        mean_coef_shallow: -0.0004929573,
        mean_coef_deep: 0.0002302494,
        std_intercept: -0.2907401204,
        std_coef_shallow: -0.8629591777,
        std_coef_deep: 0.0003811085,
    },
    ProbcutParams {
        mean_intercept: -0.2853778387,
        mean_coef_shallow: -0.0023387537,
        mean_coef_deep: 0.0016854299,
        std_intercept: -1.4498057711,
        std_coef_shallow: -0.6334385117,
        std_coef_deep: -0.0000144028,
    },
];
