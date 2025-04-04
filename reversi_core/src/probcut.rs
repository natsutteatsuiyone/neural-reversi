use crate::{
    board::Board,
    constants::{EVAL_SCORE_SCALE, EVAL_SCORE_SCALE_BITS, MID_SCORE_MAX, MID_SCORE_MIN},
    search::{midgame, search_context::SearchContext},
    types::{Depth, NonPV, Score},
};

pub const NO_SELECTIVITY: u8 = 6;
const SELECTIVITY: [(u8, f64, i32); NO_SELECTIVITY as usize + 1] = [
    (0, 1.0, 68),
    (1, 1.1, 73),
    (2, 1.5, 87),
    (3, 2.0, 95),
    (4, 2.6, 98),
    (5, 3.3, 99),
    (6, 999.0, 100),
];

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

pub fn get_probability(selectivity: u8) -> i32 {
    SELECTIVITY[selectivity as usize].2
}

fn get_t(selectivity: u8) -> f64 {
    SELECTIVITY[selectivity as usize].1
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
    let mut probcut_depth = 2 * (depth as f64 * 0.20).floor() as Depth + (depth & 1);
    if probcut_depth == 0 {
        probcut_depth = depth - 2;
    }
    probcut_depth
}

fn calc_mean(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    PROBCUT_PARAMS[ply].mean(shallow as f64, deep as f64) * EVAL_SCORE_SCALE as f64
}

fn calc_sigma(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    PROBCUT_PARAMS[ply].sigma(shallow as f64, deep as f64) * EVAL_SCORE_SCALE as f64
}

pub fn probcut_midgame(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    alpha: Score,
    beta: Score,
) -> Option<Score> {
    if depth >= 3 && ctx.selectivity < NO_SELECTIVITY {
        return probcut(ctx, board, depth, alpha, beta);
    }
    None
}

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
        if let Some(score) = probcut(ctx, board, depth, scaled_alpha, scaled_beta) {
            return Some(score >> EVAL_SCORE_SCALE_BITS);
        }
    }
    None
}

fn probcut(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    alpha: Score,
    beta: Score,
) -> Option<Score> {
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
        let score = midgame::search::<NonPV, false>(ctx, board, pc_depth, pc_beta - 1, pc_beta);
        ctx.undo_probcut(current_selectivity);
        if score >= pc_beta {
            return Some(beta);
        }
    }

    let eval_alpha = (alpha as f64 + eval_sigma - eval_mean).ceil() as Score;
    let pc_alpha = (alpha as f64 - t * sigma - mean).floor() as Score;
    if eval_score < eval_alpha && pc_alpha > MID_SCORE_MIN {
        ctx.update_probcut();
        let score = midgame::search::<NonPV, false>(ctx, board, pc_depth, pc_alpha, pc_alpha + 1);
        ctx.undo_probcut(current_selectivity);
        if score <= pc_alpha {
            return Some(alpha);
        }
    }

    None
}

#[rustfmt::skip]
const PROBCUT_PARAMS: [ProbcutParams; 60] = [
    ProbcutParams {
        mean_intercept: 0.0,
        mean_coef_shallow: 0.0,
        mean_coef_deep: 0.0,
        std_intercept: -18.42068074,
        std_coef_shallow: 1.924173103e-31,
        std_coef_deep: -2.406236173e-47,
    },
    ProbcutParams {
        mean_intercept: -0.6548240824,
        mean_coef_shallow: -0.02506858299,
        mean_coef_deep: 0.07863596972,
        std_intercept: 0.2868614401,
        std_coef_shallow: 0.001542931611,
        std_coef_deep: 0.005046891843,
    },
    ProbcutParams {
        mean_intercept: -0.4049828803,
        mean_coef_shallow: 0.04453027699,
        mean_coef_deep: 0.06247147378,
        std_intercept: 0.6723895617,
        std_coef_shallow: 0.03956364735,
        std_coef_deep: -0.03746888057,
    },
    ProbcutParams {
        mean_intercept: -1.321644029,
        mean_coef_shallow: 0.1672977753,
        mean_coef_deep: 0.05473878957,
        std_intercept: 0.7539212175,
        std_coef_shallow: 0.04556004493,
        std_coef_deep: -0.04035736722,
    },
    ProbcutParams {
        mean_intercept: -0.8414263456,
        mean_coef_shallow: 0.1515483729,
        mean_coef_deep: 0.02633878829,
        std_intercept: 0.8355421288,
        std_coef_shallow: 0.04642820639,
        std_coef_deep: -0.04167088281,
    },
    ProbcutParams {
        mean_intercept: -1.859458798,
        mean_coef_shallow: 0.2321039983,
        mean_coef_deep: 0.09581674839,
        std_intercept: 0.8038864964,
        std_coef_shallow: 0.03594674072,
        std_coef_deep: -0.03046541898,
    },
    ProbcutParams {
        mean_intercept: -1.00240218,
        mean_coef_shallow: 0.1538238095,
        mean_coef_deep: 0.03490197838,
        std_intercept: 0.8131214752,
        std_coef_shallow: 0.02576681727,
        std_coef_deep: -0.0212924565,
    },
    ProbcutParams {
        mean_intercept: -1.663983677,
        mean_coef_shallow: 0.2030258966,
        mean_coef_deep: 0.1070772362,
        std_intercept: 0.8452408253,
        std_coef_shallow: 0.007953564157,
        std_coef_deep: -0.0109905511,
    },
    ProbcutParams {
        mean_intercept: -1.214045565,
        mean_coef_shallow: 0.1424684222,
        mean_coef_deep: 0.06398051251,
        std_intercept: 0.8950324177,
        std_coef_shallow: -0.01449122333,
        std_coef_deep: -0.0009906927965,
    },
    ProbcutParams {
        mean_intercept: -1.159215917,
        mean_coef_shallow: 0.1169613631,
        mean_coef_deep: 0.1041446313,
        std_intercept: 0.9093091067,
        std_coef_shallow: -0.02274078619,
        std_coef_deep: 0.006175007315,
    },
    ProbcutParams {
        mean_intercept: -1.408387615,
        mean_coef_shallow: 0.2372776254,
        mean_coef_deep: 0.03780616327,
        std_intercept: 0.9637229657,
        std_coef_shallow: -0.0296371539,
        std_coef_deep: 0.005037690569,
    },
    ProbcutParams {
        mean_intercept: -0.9947274102,
        mean_coef_shallow: 0.09620914662,
        mean_coef_deep: 0.09214472128,
        std_intercept: 0.9972431599,
        std_coef_shallow: -0.03110555763,
        std_coef_deep: 0.00638363718,
    },
    ProbcutParams {
        mean_intercept: -1.171206604,
        mean_coef_shallow: 0.1875223992,
        mean_coef_deep: 0.04122015014,
        std_intercept: 1.037569341,
        std_coef_shallow: -0.03289325355,
        std_coef_deep: 0.003714345663,
    },
    ProbcutParams {
        mean_intercept: -1.421589002,
        mean_coef_shallow: 0.1645054162,
        mean_coef_deep: 0.1025629912,
        std_intercept: 1.061601124,
        std_coef_shallow: -0.02691958265,
        std_coef_deep: 0.001578625708,
    },
    ProbcutParams {
        mean_intercept: -1.016286783,
        mean_coef_shallow: 0.1843714574,
        mean_coef_deep: 0.02033477226,
        std_intercept: 1.063340864,
        std_coef_shallow: -0.02438521048,
        std_coef_deep: 0.002251552766,
    },
    ProbcutParams {
        mean_intercept: -1.574954675,
        mean_coef_shallow: 0.1718054306,
        mean_coef_deep: 0.1137449974,
        std_intercept: 1.100273342,
        std_coef_shallow: -0.0331660285,
        std_coef_deep: 0.004075656665,
    },
    ProbcutParams {
        mean_intercept: -1.323470659,
        mean_coef_shallow: 0.198902939,
        mean_coef_deep: 0.03437523371,
        std_intercept: 1.106524319,
        std_coef_shallow: -0.03963176592,
        std_coef_deep: 0.0055510427,
    },
    ProbcutParams {
        mean_intercept: -1.461477458,
        mean_coef_shallow: 0.1938410993,
        mean_coef_deep: 0.08577044289,
        std_intercept: 1.147260546,
        std_coef_shallow: -0.0499423436,
        std_coef_deep: 0.006876475434,
    },
    ProbcutParams {
        mean_intercept: -1.61228532,
        mean_coef_shallow: 0.2581541963,
        mean_coef_deep: 0.02986680109,
        std_intercept: 1.143770426,
        std_coef_shallow: -0.05042686968,
        std_coef_deep: 0.006530164276,
    },
    ProbcutParams {
        mean_intercept: -1.401882176,
        mean_coef_shallow: 0.1977307387,
        mean_coef_deep: 0.06549393119,
        std_intercept: 1.093643241,
        std_coef_shallow: -0.05369837928,
        std_coef_deep: 0.01245778815,
    },
    ProbcutParams {
        mean_intercept: -1.911755163,
        mean_coef_shallow: 0.3128115781,
        mean_coef_deep: 0.0389805801,
        std_intercept: 1.044872393,
        std_coef_shallow: -0.05170832439,
        std_coef_deep: 0.0163371031,
    },
    ProbcutParams {
        mean_intercept: -1.446470991,
        mean_coef_shallow: 0.2520504653,
        mean_coef_deep: 0.03052364757,
        std_intercept: 1.055540534,
        std_coef_shallow: -0.04672412105,
        std_coef_deep: 0.01390991047,
    },
    ProbcutParams {
        mean_intercept: -1.933584403,
        mean_coef_shallow: 0.2651215872,
        mean_coef_deep: 0.07275220464,
        std_intercept: 1.043224443,
        std_coef_shallow: -0.04145288877,
        std_coef_deep: 0.01406053564,
    },
    ProbcutParams {
        mean_intercept: -1.008735803,
        mean_coef_shallow: 0.1621603983,
        mean_coef_deep: 0.02357825715,
        std_intercept: 1.005535267,
        std_coef_shallow: -0.0378919791,
        std_coef_deep: 0.01855477658,
    },
    ProbcutParams {
        mean_intercept: -1.434571512,
        mean_coef_shallow: 0.2356066422,
        mean_coef_deep: 0.03934569884,
        std_intercept: 1.038095987,
        std_coef_shallow: -0.03812202061,
        std_coef_deep: 0.01959123709,
    },
    ProbcutParams {
        mean_intercept: -1.70020894,
        mean_coef_shallow: 0.2247048192,
        mean_coef_deep: 0.06727686524,
        std_intercept: 1.072923737,
        std_coef_shallow: -0.04107190576,
        std_coef_deep: 0.01932254863,
    },
    ProbcutParams {
        mean_intercept: -0.9280672982,
        mean_coef_shallow: 0.2077912242,
        mean_coef_deep: -0.01067013577,
        std_intercept: 1.086108808,
        std_coef_shallow: -0.04720335924,
        std_coef_deep: 0.02316967962,
    },
    ProbcutParams {
        mean_intercept: -1.672106417,
        mean_coef_shallow: 0.2771202044,
        mean_coef_deep: 0.04565973841,
        std_intercept: 1.168734318,
        std_coef_shallow: -0.05351051053,
        std_coef_deep: 0.01777200588,
    },
    ProbcutParams {
        mean_intercept: -1.552876555,
        mean_coef_shallow: 0.2536255915,
        mean_coef_deep: 0.0259357839,
        std_intercept: 1.183363304,
        std_coef_shallow: -0.05573600559,
        std_coef_deep: 0.01635063838,
    },
    ProbcutParams {
        mean_intercept: -1.50408036,
        mean_coef_shallow: 0.2721373888,
        mean_coef_deep: 0.02629527236,
        std_intercept: 1.195813058,
        std_coef_shallow: -0.05805091516,
        std_coef_deep: 0.01475314989,
    },
    ProbcutParams {
        mean_intercept: -1.502582244,
        mean_coef_shallow: 0.2656528478,
        mean_coef_deep: 0.01561111398,
        std_intercept: 1.203319392,
        std_coef_shallow: -0.06592702048,
        std_coef_deep: 0.01584350364,
    },
    ProbcutParams {
        mean_intercept: -1.48681865,
        mean_coef_shallow: 0.2181503548,
        mean_coef_deep: 0.04965548112,
        std_intercept: 1.2016187,
        std_coef_shallow: -0.06863597666,
        std_coef_deep: 0.01730116854,
    },
    ProbcutParams {
        mean_intercept: -1.576276891,
        mean_coef_shallow: 0.2874184676,
        mean_coef_deep: 0.007854233927,
        std_intercept: 1.206448499,
        std_coef_shallow: -0.07409711833,
        std_coef_deep: 0.01743220466,
    },
    ProbcutParams {
        mean_intercept: -1.204856396,
        mean_coef_shallow: 0.187883336,
        mean_coef_deep: 0.03893531017,
        std_intercept: 1.196856016,
        std_coef_shallow: -0.07415448663,
        std_coef_deep: 0.018366148,
    },
    ProbcutParams {
        mean_intercept: -1.785099668,
        mean_coef_shallow: 0.2748744672,
        mean_coef_deep: 0.04222285091,
        std_intercept: 1.140465191,
        std_coef_shallow: -0.06999198494,
        std_coef_deep: 0.02492267724,
    },
    ProbcutParams {
        mean_intercept: -1.240394923,
        mean_coef_shallow: 0.2289071233,
        mean_coef_deep: 0.0105368452,
        std_intercept: 1.14702589,
        std_coef_shallow: -0.07813728732,
        std_coef_deep: 0.02747589995,
    },
    ProbcutParams {
        mean_intercept: -1.635756852,
        mean_coef_shallow: 0.2422761164,
        mean_coef_deep: 0.05159861275,
        std_intercept: 1.155120112,
        std_coef_shallow: -0.07519297508,
        std_coef_deep: 0.02609532314,
    },
    ProbcutParams {
        mean_intercept: -1.041870724,
        mean_coef_shallow: 0.2246083511,
        mean_coef_deep: -0.03245081406,
        std_intercept: 1.157401797,
        std_coef_shallow: -0.07196649767,
        std_coef_deep: 0.02631947755,
    },
    ProbcutParams {
        mean_intercept: -1.312909745,
        mean_coef_shallow: 0.210689035,
        mean_coef_deep: 0.03585971707,
        std_intercept: 1.209654557,
        std_coef_shallow: -0.07634128637,
        std_coef_deep: 0.02379288015,
    },
    ProbcutParams {
        mean_intercept: -0.5617576853,
        mean_coef_shallow: 0.1505378608,
        mean_coef_deep: -0.04494807632,
        std_intercept: 1.255308445,
        std_coef_shallow: -0.08085102138,
        std_coef_deep: 0.0224213629,
    },
    ProbcutParams {
        mean_intercept: -1.259644057,
        mean_coef_shallow: 0.1939838631,
        mean_coef_deep: 0.04050027171,
        std_intercept: 1.243264203,
        std_coef_shallow: -0.08051742677,
        std_coef_deep: 0.02373460893,
    },
    ProbcutParams {
        mean_intercept: -0.906824763,
        mean_coef_shallow: 0.2407860372,
        mean_coef_deep: -0.06048416029,
        std_intercept: 1.228253564,
        std_coef_shallow: -0.07433028985,
        std_coef_deep: 0.02259471143,
    },
    ProbcutParams {
        mean_intercept: -0.663549319,
        mean_coef_shallow: 0.1191284248,
        mean_coef_deep: 0.01325698665,
        std_intercept: 1.194439479,
        std_coef_shallow: -0.07033486895,
        std_coef_deep: 0.02574122369,
    },
    ProbcutParams {
        mean_intercept: -0.8945690761,
        mean_coef_shallow: 0.2330038189,
        mean_coef_deep: -0.05269239067,
        std_intercept: 1.21364459,
        std_coef_shallow: -0.07258113567,
        std_coef_deep: 0.02571520244,
    },
    ProbcutParams {
        mean_intercept: -0.757423969,
        mean_coef_shallow: 0.1533324737,
        mean_coef_deep: -0.003381214804,
        std_intercept: 1.215917914,
        std_coef_shallow: -0.0744761427,
        std_coef_deep: 0.028456807,
    },
    ProbcutParams {
        mean_intercept: -0.5956551233,
        mean_coef_shallow: 0.1897991411,
        mean_coef_deep: -0.06611914368,
        std_intercept: 1.153951037,
        std_coef_shallow: -0.07252213188,
        std_coef_deep: 0.0345414786,
    },
    ProbcutParams {
        mean_intercept: -0.7895215546,
        mean_coef_shallow: 0.1926630122,
        mean_coef_deep: -0.02633336888,
        std_intercept: 1.177877372,
        std_coef_shallow: -0.07891529391,
        std_coef_deep: 0.03515037181,
    },
    ProbcutParams {
        mean_intercept: -0.6016137004,
        mean_coef_shallow: 0.1840366578,
        mean_coef_deep: -0.06884673015,
        std_intercept: 1.184785091,
        std_coef_shallow: -0.07936793372,
        std_coef_deep: 0.03407264849,
    },
    ProbcutParams {
        mean_intercept: -0.4210380863,
        mean_coef_shallow: 0.1398816809,
        mean_coef_deep: -0.04166500557,
        std_intercept: 1.203047447,
        std_coef_shallow: -0.08300087497,
        std_coef_deep: 0.03331258562,
    },
    ProbcutParams {
        mean_intercept: -0.9443884006,
        mean_coef_shallow: 0.2227680792,
        mean_coef_deep: -0.05317021116,
        std_intercept: 1.211561692,
        std_coef_shallow: -0.08913094416,
        std_coef_deep: 0.03306239494,
    },
    ProbcutParams {
        mean_intercept: -0.238628459,
        mean_coef_shallow: 0.08861762316,
        mean_coef_deep: -0.04114609905,
        std_intercept: 1.261802685,
        std_coef_shallow: -0.1073459774,
        std_coef_deep: 0.02977658579,
    },
    ProbcutParams {
        mean_intercept: -0.8070015621,
        mean_coef_shallow: 0.1701145553,
        mean_coef_deep: -0.02350914179,
        std_intercept: 1.329597014,
        std_coef_shallow: -0.1416608547,
        std_coef_deep: 0.02519246911,
    },
    ProbcutParams {
        mean_intercept: -0.5167217826,
        mean_coef_shallow: 0.1202212077,
        mean_coef_deep: -0.02941236195,
        std_intercept: 1.596323618,
        std_coef_shallow: -0.2689274509,
        std_coef_deep: 0.01606742578,
    },
    ProbcutParams {
        mean_intercept: -0.6280440475,
        mean_coef_shallow: 0.1247952279,
        mean_coef_deep: -0.009637479379,
        std_intercept: 1.640722921,
        std_coef_shallow: -0.3279938328,
        std_coef_deep: 0.0112731466,
    },
    ProbcutParams {
        mean_intercept: -0.5945886378,
        mean_coef_shallow: 0.1157806312,
        mean_coef_deep: -0.01146347019,
        std_intercept: 1.668858137,
        std_coef_shallow: -0.5339924942,
        std_coef_deep: 0.02281250466,
    },
    ProbcutParams {
        mean_intercept: -0.410787376,
        mean_coef_shallow: 0.07760380691,
        mean_coef_deep: -0.00257934056,
        std_intercept: 1.624073456,
        std_coef_shallow: -0.5845196049,
        std_coef_deep: 0.002526596121,
    },
    ProbcutParams {
        mean_intercept: -0.2951245299,
        mean_coef_shallow: 0.05441250645,
        mean_coef_deep: -0.0005995780149,
        std_intercept: 1.248332304,
        std_coef_shallow: -0.6311093622,
        std_coef_deep: 0.001674796736,
    },
    ProbcutParams {
        mean_intercept: -0.1663344267,
        mean_coef_shallow: 0.03100385898,
        mean_coef_deep: -1.413758795e-17,
        std_intercept: 0.5734378427,
        std_coef_shallow: -0.6487613336,
        std_coef_deep: 7.611856954e-17,
    },
    ProbcutParams {
        mean_intercept: 0.0285866237,
        mean_coef_shallow: -0.005406476197,
        mean_coef_deep: 6.273349958e-18,
        std_intercept: -0.8245465375,
        std_coef_shallow: -0.7505342655,
        std_coef_deep: 4.365762338e-16,
    },
    ProbcutParams {
        mean_intercept: 0.0,
        mean_coef_shallow: 0.0,
        mean_coef_deep: 0.0,
        std_intercept: -18.42068074,
        std_coef_shallow: -6.849352976e-32,
        std_coef_deep: 9.577451968e-33,
    },
];
