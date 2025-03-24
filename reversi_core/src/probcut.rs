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
        std_coef_shallow: 7.06389871e-32,
        std_coef_deep: 4.628688511e-31,
    },
    ProbcutParams {
        mean_intercept: -0.6507852488,
        mean_coef_shallow: -0.02394386337,
        mean_coef_deep: 0.07815831023,
        std_intercept: 0.3086890622,
        std_coef_shallow: 0.00160085226,
        std_coef_deep: 0.00223466039,
    },
    ProbcutParams {
        mean_intercept: -0.3903892769,
        mean_coef_shallow: 0.04605197634,
        mean_coef_deep: 0.06070652314,
        std_intercept: 0.6833398493,
        std_coef_shallow: 0.03805747042,
        std_coef_deep: -0.0399341746,
    },
    ProbcutParams {
        mean_intercept: -1.319754362,
        mean_coef_shallow: 0.1690992482,
        mean_coef_deep: 0.05423266642,
        std_intercept: 0.762824287,
        std_coef_shallow: 0.04779553929,
        std_coef_deep: -0.04265251571,
    },
    ProbcutParams {
        mean_intercept: -0.8210325407,
        mean_coef_shallow: 0.1535277774,
        mean_coef_deep: 0.02412993568,
        std_intercept: 0.8526779328,
        std_coef_shallow: 0.03716419314,
        std_coef_deep: -0.04075647624,
    },
    ProbcutParams {
        mean_intercept: -1.843257518,
        mean_coef_shallow: 0.2329806304,
        mean_coef_deep: 0.09417516389,
        std_intercept: 0.8189065456,
        std_coef_shallow: 0.04596939685,
        std_coef_deep: -0.03631980994,
    },
    ProbcutParams {
        mean_intercept: -0.9963902483,
        mean_coef_shallow: 0.155599072,
        mean_coef_deep: 0.03406960657,
        std_intercept: 0.7335800316,
        std_coef_shallow: 0.02835119085,
        std_coef_deep: -0.01660865815,
    },
    ProbcutParams {
        mean_intercept: -1.659051925,
        mean_coef_shallow: 0.2052456263,
        mean_coef_deep: 0.1063633701,
        std_intercept: 0.7911285274,
        std_coef_shallow: 0.01821685437,
        std_coef_deep: -0.01052063488,
    },
    ProbcutParams {
        mean_intercept: -1.219103201,
        mean_coef_shallow: 0.1454317328,
        mean_coef_deep: 0.06371473198,
        std_intercept: 0.902212272,
        std_coef_shallow: -0.01502193759,
        std_coef_deep: -0.001208306273,
    },
    ProbcutParams {
        mean_intercept: -1.132098545,
        mean_coef_shallow: 0.118966724,
        mean_coef_deep: 0.1011968103,
        std_intercept: 0.9573270174,
        std_coef_shallow: -0.01786777693,
        std_coef_deep: -0.0007070436565,
    },
    ProbcutParams {
        mean_intercept: -1.424112622,
        mean_coef_shallow: 0.239691559,
        mean_coef_deep: 0.03832136843,
        std_intercept: 0.9789336498,
        std_coef_shallow: -0.03285607446,
        std_coef_deep: 0.008577819218,
    },
    ProbcutParams {
        mean_intercept: -0.9763677093,
        mean_coef_shallow: 0.09504597446,
        mean_coef_deep: 0.09091034273,
        std_intercept: 0.9265237464,
        std_coef_shallow: -0.03369504157,
        std_coef_deep: 0.01277793882,
    },
    ProbcutParams {
        mean_intercept: -1.173390767,
        mean_coef_shallow: 0.1910718646,
        mean_coef_deep: 0.0408255542,
        std_intercept: 1.06017822,
        std_coef_shallow: -0.04335231225,
        std_coef_deep: 0.006600674432,
    },
    ProbcutParams {
        mean_intercept: -1.415237849,
        mean_coef_shallow: 0.1651184996,
        mean_coef_deep: 0.1019702376,
        std_intercept: 1.042061965,
        std_coef_shallow: -0.02058736461,
        std_coef_deep: 0.0003654107036,
    },
    ProbcutParams {
        mean_intercept: -1.03739033,
        mean_coef_shallow: 0.1844551473,
        mean_coef_deep: 0.02280300385,
        std_intercept: 1.083276877,
        std_coef_shallow: -0.02322062442,
        std_coef_deep: -0.0009312402825,
    },
    ProbcutParams {
        mean_intercept: -1.536409683,
        mean_coef_shallow: 0.1736697536,
        mean_coef_deep: 0.1096712353,
        std_intercept: 1.169356586,
        std_coef_shallow: -0.03553793419,
        std_coef_deep: -0.003874466557,
    },
    ProbcutParams {
        mean_intercept: -1.332326526,
        mean_coef_shallow: 0.2024359305,
        mean_coef_deep: 0.03423032277,
        std_intercept: 1.121361055,
        std_coef_shallow: -0.03669418893,
        std_coef_deep: 0.00220712739,
    },
    ProbcutParams {
        mean_intercept: -1.463107257,
        mean_coef_shallow: 0.1925054078,
        mean_coef_deep: 0.08726724772,
        std_intercept: 1.198618658,
        std_coef_shallow: -0.05273369747,
        std_coef_deep: 0.004012673417,
    },
    ProbcutParams {
        mean_intercept: -1.624099777,
        mean_coef_shallow: 0.2614151657,
        mean_coef_deep: 0.03039175024,
        std_intercept: 1.173842631,
        std_coef_shallow: -0.0507076344,
        std_coef_deep: 0.002384913732,
    },
    ProbcutParams {
        mean_intercept: -1.392147302,
        mean_coef_shallow: 0.2016492588,
        mean_coef_deep: 0.06324152854,
        std_intercept: 1.138005247,
        std_coef_shallow: -0.04364345953,
        std_coef_deep: 0.004330516865,
    },
    ProbcutParams {
        mean_intercept: -1.943361114,
        mean_coef_shallow: 0.3128046433,
        mean_coef_deep: 0.04283670899,
        std_intercept: 1.044794769,
        std_coef_shallow: -0.04290714865,
        std_coef_deep: 0.0120070811,
    },
    ProbcutParams {
        mean_intercept: -1.425796308,
        mean_coef_shallow: 0.2548136464,
        mean_coef_deep: 0.02842279642,
        std_intercept: 1.085237605,
        std_coef_shallow: -0.05003370006,
        std_coef_deep: 0.009800326545,
    },
    ProbcutParams {
        mean_intercept: -1.969638247,
        mean_coef_shallow: 0.2670718079,
        mean_coef_deep: 0.07564052,
        std_intercept: 1.048212388,
        std_coef_shallow: -0.03945906973,
        std_coef_deep: 0.01342003115,
    },
    ProbcutParams {
        mean_intercept: -0.9788959968,
        mean_coef_shallow: 0.1635143912,
        mean_coef_deep: 0.02118316854,
        std_intercept: 1.107867382,
        std_coef_shallow: -0.04112691173,
        std_coef_deep: 0.007636377787,
    },
    ProbcutParams {
        mean_intercept: -1.487017444,
        mean_coef_shallow: 0.2344251563,
        mean_coef_deep: 0.04575927797,
        std_intercept: 1.050393127,
        std_coef_shallow: -0.02735142309,
        std_coef_deep: 0.01288943028,
    },
    ProbcutParams {
        mean_intercept: -1.662135659,
        mean_coef_shallow: 0.226462093,
        mean_coef_deep: 0.06316900055,
        std_intercept: 1.141169035,
        std_coef_shallow: -0.04287010999,
        std_coef_deep: 0.01181080115,
    },
    ProbcutParams {
        mean_intercept: -0.9536124396,
        mean_coef_shallow: 0.2142297076,
        mean_coef_deep: -0.00995250442,
        std_intercept: 1.078834673,
        std_coef_shallow: -0.04056425639,
        std_coef_deep: 0.02042568125,
    },
    ProbcutParams {
        mean_intercept: -1.661753043,
        mean_coef_shallow: 0.2768761982,
        mean_coef_deep: 0.044814918,
        std_intercept: 1.292155535,
        std_coef_shallow: -0.0425469108,
        std_coef_deep: 0.0004254658968,
    },
    ProbcutParams {
        mean_intercept: -1.562405271,
        mean_coef_shallow: 0.2593478129,
        mean_coef_deep: 0.02545715518,
        std_intercept: 1.212298241,
        std_coef_shallow: -0.05426455806,
        std_coef_deep: 0.01131973928,
    },
    ProbcutParams {
        mean_intercept: -1.500907327,
        mean_coef_shallow: 0.2726190374,
        mean_coef_deep: 0.02575736277,
        std_intercept: 1.175436243,
        std_coef_shallow: -0.05422892291,
        std_coef_deep: 0.01476241033,
    },
    ProbcutParams {
        mean_intercept: -1.494644757,
        mean_coef_shallow: 0.2670389245,
        mean_coef_deep: 0.01465385243,
        std_intercept: 1.143379274,
        std_coef_shallow: -0.07012106405,
        std_coef_deep: 0.02353571562,
    },
    ProbcutParams {
        mean_intercept: -1.486467627,
        mean_coef_shallow: 0.2199564819,
        mean_coef_deep: 0.04961899381,
        std_intercept: 1.231360997,
        std_coef_shallow: -0.06399170269,
        std_coef_deep: 0.01172621509,
    },
    ProbcutParams {
        mean_intercept: -1.589749281,
        mean_coef_shallow: 0.2890252498,
        mean_coef_deep: 0.00855638448,
        std_intercept: 1.164755909,
        std_coef_shallow: -0.08033457549,
        std_coef_deep: 0.02349862373,
    },
    ProbcutParams {
        mean_intercept: -1.225279468,
        mean_coef_shallow: 0.1908036398,
        mean_coef_deep: 0.04025644618,
        std_intercept: 1.304621631,
        std_coef_shallow: -0.07579756545,
        std_coef_deep: 0.008159358907,
    },
    ProbcutParams {
        mean_intercept: -1.783016701,
        mean_coef_shallow: 0.2750389369,
        mean_coef_deep: 0.04234254992,
        std_intercept: 1.125477742,
        std_coef_shallow: -0.07186613984,
        std_coef_deep: 0.03049249933,
    },
    ProbcutParams {
        mean_intercept: -1.220717842,
        mean_coef_shallow: 0.2314163782,
        mean_coef_deep: 0.007464134855,
        std_intercept: 1.13921436,
        std_coef_shallow: -0.0753214347,
        std_coef_deep: 0.02883421447,
    },
    ProbcutParams {
        mean_intercept: -1.660700775,
        mean_coef_shallow: 0.2445914132,
        mean_coef_deep: 0.05317841367,
        std_intercept: 1.171004585,
        std_coef_shallow: -0.08260505031,
        std_coef_deep: 0.02910737395,
    },
    ProbcutParams {
        mean_intercept: -1.018150209,
        mean_coef_shallow: 0.2212954985,
        mean_coef_deep: -0.03316377088,
        std_intercept: 1.06954187,
        std_coef_shallow: -0.06988142102,
        std_coef_deep: 0.03529661307,
    },
    ProbcutParams {
        mean_intercept: -1.306702184,
        mean_coef_shallow: 0.2169544532,
        mean_coef_deep: 0.03285521926,
        std_intercept: 1.096118021,
        std_coef_shallow: -0.08807548381,
        std_coef_deep: 0.03873401172,
    },
    ProbcutParams {
        mean_intercept: -0.5612680726,
        mean_coef_shallow: 0.1507815247,
        mean_coef_deep: -0.04597408712,
        std_intercept: 1.135281551,
        std_coef_shallow: -0.08441889701,
        std_coef_deep: 0.03650135818,
    },
    ProbcutParams {
        mean_intercept: -1.273656065,
        mean_coef_shallow: 0.1944820308,
        mean_coef_deep: 0.04283648533,
        std_intercept: 1.275860526,
        std_coef_shallow: -0.08487094022,
        std_coef_deep: 0.02223446914,
    },
    ProbcutParams {
        mean_intercept: -0.877257108,
        mean_coef_shallow: 0.2376714066,
        mean_coef_deep: -0.06232659606,
        std_intercept: 1.221121539,
        std_coef_shallow: -0.06360722735,
        std_coef_deep: 0.02148264586,
    },
    ProbcutParams {
        mean_intercept: -0.6777673502,
        mean_coef_shallow: 0.123603007,
        mean_coef_deep: 0.01251702221,
        std_intercept: 1.250193841,
        std_coef_shallow: -0.07279033962,
        std_coef_deep: 0.02161094445,
    },
    ProbcutParams {
        mean_intercept: -0.8797934812,
        mean_coef_shallow: 0.2272078137,
        mean_coef_deep: -0.05143537355,
        std_intercept: 1.190709469,
        std_coef_shallow: -0.0745872726,
        std_coef_deep: 0.03058725235,
    },
    ProbcutParams {
        mean_intercept: -0.7759439963,
        mean_coef_shallow: 0.1553482133,
        mean_coef_deep: -0.002034360231,
        std_intercept: 1.277230086,
        std_coef_shallow: -0.07612634435,
        std_coef_deep: 0.02212539264,
    },
    ProbcutParams {
        mean_intercept: -0.6245731398,
        mean_coef_shallow: 0.1944583321,
        mean_coef_deep: -0.06545588529,
        std_intercept: 1.295630841,
        std_coef_shallow: -0.06417723204,
        std_coef_deep: 0.01702678746,
    },
    ProbcutParams {
        mean_intercept: -0.8009973501,
        mean_coef_shallow: 0.191072991,
        mean_coef_deep: -0.02379165629,
        std_intercept: 1.242540245,
        std_coef_shallow: -0.06121973564,
        std_coef_deep: 0.02014606317,
    },
    ProbcutParams {
        mean_intercept: -0.5913749557,
        mean_coef_shallow: 0.1839776866,
        mean_coef_deep: -0.06960403639,
        std_intercept: 1.203874488,
        std_coef_shallow: -0.08731274279,
        std_coef_deep: 0.03307744408,
    },
    ProbcutParams {
        mean_intercept: -0.4311795641,
        mean_coef_shallow: 0.1390742492,
        mean_coef_deep: -0.04077467603,
        std_intercept: 1.17833211,
        std_coef_shallow: -0.08989667273,
        std_coef_deep: 0.03724801088,
    },
    ProbcutParams {
        mean_intercept: -0.9423580651,
        mean_coef_shallow: 0.2243990337,
        mean_coef_deep: -0.05395300602,
        std_intercept: 1.207855234,
        std_coef_shallow: -0.09760527127,
        std_coef_deep: 0.03743351327,
    },
    ProbcutParams {
        mean_intercept: -0.2368873867,
        mean_coef_shallow: 0.09095653403,
        mean_coef_deep: -0.042946769,
        std_intercept: 1.257942796,
        std_coef_shallow: -0.1210186052,
        std_coef_deep: 0.0328695225,
    },
    ProbcutParams {
        mean_intercept: -0.8276170681,
        mean_coef_shallow: 0.172180444,
        mean_coef_deep: -0.02231329076,
        std_intercept: 1.338579915,
        std_coef_shallow: -0.1399009342,
        std_coef_deep: 0.02394666498,
    },
    ProbcutParams {
        mean_intercept: -0.5569456605,
        mean_coef_shallow: 0.1199768665,
        mean_coef_deep: -0.02596202136,
        std_intercept: 1.559155799,
        std_coef_shallow: -0.267335556,
        std_coef_deep: 0.01945945865,
    },
    ProbcutParams {
        mean_intercept: -0.6178746292,
        mean_coef_shallow: 0.1265210266,
        mean_coef_deep: -0.01071458573,
        std_intercept: 1.581504231,
        std_coef_shallow: -0.3322257878,
        std_coef_deep: 0.02067334088,
    },
    ProbcutParams {
        mean_intercept: -0.5912793376,
        mean_coef_shallow: 0.1164670746,
        mean_coef_deep: -0.0122018622,
        std_intercept: 1.637780113,
        std_coef_shallow: -0.5352702458,
        std_coef_deep: 0.02379510854,
    },
    ProbcutParams {
        mean_intercept: -0.4374145776,
        mean_coef_shallow: 0.07656301635,
        mean_coef_deep: 0.0006315907975,
        std_intercept: 1.618687985,
        std_coef_shallow: -0.5829536148,
        std_coef_deep: 0.003386035901,
    },
    ProbcutParams {
        mean_intercept: -0.3080245851,
        mean_coef_shallow: 0.05652600113,
        mean_coef_deep: -0.0004151980981,
        std_intercept: 1.246006619,
        std_coef_shallow: -0.624405363,
        std_coef_deep: 0.001311078131,
    },
    ProbcutParams {
        mean_intercept: -0.1804398686,
        mean_coef_shallow: 0.03084867061,
        mean_coef_deep: 0.001415587158,
        std_intercept: 0.588034341,
        std_coef_shallow: -0.6468880267,
        std_coef_deep: -0.00133506787,
    },
    ProbcutParams {
        mean_intercept: 0.02968418528,
        mean_coef_shallow: -0.005770620998,
        mean_coef_deep: 9.213080378e-05,
        std_intercept: -0.9260895813,
        std_coef_shallow: -0.7422784819,
        std_coef_deep: 0.01172411695,
    },
    ProbcutParams {
        mean_intercept: 0.0,
        mean_coef_shallow: 0.0,
        mean_coef_deep: 0.0,
        std_intercept: -18.42068074,
        std_coef_shallow: -4.327408458e-32,
        std_coef_deep: -6.992601738e-32,
    },
];
