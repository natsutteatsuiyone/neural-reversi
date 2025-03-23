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
        std_coef_shallow: 0.0,
        std_coef_deep: 0.0,
    },
    ProbcutParams {
        mean_intercept: -0.6668459657,
        mean_coef_shallow: 0.02949387569,
        mean_coef_deep: 0.04447626657,
        std_intercept: -0.5136776097,
        std_coef_shallow: 0.05909184445,
        std_coef_deep: -0.02000218835,
    },
    ProbcutParams {
        mean_intercept: 0.4836481033,
        mean_coef_shallow: -0.03218431667,
        mean_coef_deep: -0.02319130027,
        std_intercept: -0.3612228586,
        std_coef_shallow: 0.05800459647,
        std_coef_deep: -0.0176294081,
    },
    ProbcutParams {
        mean_intercept: -0.3511712036,
        mean_coef_shallow: 0.03541916251,
        mean_coef_deep: 0.009724421039,
        std_intercept: -0.08816948425,
        std_coef_shallow: 0.01607891255,
        std_coef_deep: -0.003972589671,
    },
    ProbcutParams {
        mean_intercept: 0.1438826016,
        mean_coef_shallow: 0.002341486311,
        mean_coef_deep: -0.01428437491,
        std_intercept: 0.07841693992,
        std_coef_shallow: 0.03375297187,
        std_coef_deep: -0.02080853875,
    },
    ProbcutParams {
        mean_intercept: -0.5706823068,
        mean_coef_shallow: 0.09230908602,
        mean_coef_deep: 0.01372525178,
        std_intercept: 0.2520709125,
        std_coef_shallow: 0.01759197265,
        std_coef_deep: -0.02096290968,
    },
    ProbcutParams {
        mean_intercept: 0.02343050537,
        mean_coef_shallow: -0.01700040078,
        mean_coef_deep: 0.008859001746,
        std_intercept: 0.2813610223,
        std_coef_shallow: -0.01388136285,
        std_coef_deep: -0.004637932421,
    },
    ProbcutParams {
        mean_intercept: -0.1371673563,
        mean_coef_shallow: 0.0457769151,
        mean_coef_deep: 0.001846322255,
        std_intercept: 0.4104173507,
        std_coef_shallow: -0.03698420886,
        std_coef_deep: 0.000265992322,
    },
    ProbcutParams {
        mean_intercept: -0.8309836605,
        mean_coef_shallow: -0.0341134205,
        mean_coef_deep: 0.1234965311,
        std_intercept: 0.4481686602,
        std_coef_shallow: -0.06614823185,
        std_coef_deep: 0.01773995022,
    },
    ProbcutParams {
        mean_intercept: 0.6749879429,
        mean_coef_shallow: -0.1122871971,
        mean_coef_deep: -0.01768790365,
        std_intercept: 0.5271706958,
        std_coef_shallow: -0.07249167614,
        std_coef_deep: 0.02954657534,
    },
    ProbcutParams {
        mean_intercept: -0.5218518548,
        mean_coef_shallow: -0.008848757731,
        mean_coef_deep: 0.09123564165,
        std_intercept: 0.5016261732,
        std_coef_shallow: -0.06409545568,
        std_coef_deep: 0.03384924626,
    },
    ProbcutParams {
        mean_intercept: -0.195353603,
        mean_coef_shallow: -0.07792651488,
        mean_coef_deep: 0.05518612114,
        std_intercept: 0.5045271564,
        std_coef_shallow: -0.06032625021,
        std_coef_deep: 0.04345139227,
    },
    ProbcutParams {
        mean_intercept: 0.3655026034,
        mean_coef_shallow: -0.05534492201,
        mean_coef_deep: 0.01434721127,
        std_intercept: 0.7946022704,
        std_coef_shallow: -0.07682337542,
        std_coef_deep: 0.01892307047,
    },
    ProbcutParams {
        mean_intercept: -1.266733273,
        mean_coef_shallow: 0.07671176211,
        mean_coef_deep: 0.1185093193,
        std_intercept: 0.8189407862,
        std_coef_shallow: -0.05957163845,
        std_coef_deep: 0.01999356533,
    },
    ProbcutParams {
        mean_intercept: 0.1772992629,
        mean_coef_shallow: -0.03431048203,
        mean_coef_deep: 0.004196197892,
        std_intercept: 0.7377433583,
        std_coef_shallow: -0.03657752749,
        std_coef_deep: 0.02076596619,
    },
    ProbcutParams {
        mean_intercept: -1.120294845,
        mean_coef_shallow: 0.06696571826,
        mean_coef_deep: 0.1231066899,
        std_intercept: 0.7981032353,
        std_coef_shallow: -0.05026734235,
        std_coef_deep: 0.02362538234,
    },
    ProbcutParams {
        mean_intercept: -0.007260695048,
        mean_coef_shallow: -0.006861809848,
        mean_coef_deep: -0.01050053324,
        std_intercept: 0.8069535222,
        std_coef_shallow: -0.0673000309,
        std_coef_deep: 0.03158482279,
    },
    ProbcutParams {
        mean_intercept: -0.9413976381,
        mean_coef_shallow: 0.0797886215,
        mean_coef_deep: 0.09190027709,
        std_intercept: 0.9433929051,
        std_coef_shallow: -0.04754064519,
        std_coef_deep: 0.01774007866,
    },
    ProbcutParams {
        mean_intercept: -0.6449545324,
        mean_coef_shallow: 0.07649212452,
        mean_coef_deep: 0.01506257635,
        std_intercept: 0.9996693679,
        std_coef_shallow: -0.05191080115,
        std_coef_deep: 0.01083257022,
    },
    ProbcutParams {
        mean_intercept: -0.3151722537,
        mean_coef_shallow: 0.008594573495,
        mean_coef_deep: 0.05652142953,
        std_intercept: 1.007266109,
        std_coef_shallow: -0.03694822119,
        std_coef_deep: 0.00650032231,
    },
    ProbcutParams {
        mean_intercept: -2.04954205,
        mean_coef_shallow: 0.3263972422,
        mean_coef_deep: 0.03976104669,
        std_intercept: 0.9641125566,
        std_coef_shallow: -0.04679174057,
        std_coef_deep: 0.017015977,
    },
    ProbcutParams {
        mean_intercept: -0.9076060944,
        mean_coef_shallow: 0.1625602836,
        mean_coef_deep: 0.018896268,
        std_intercept: 1.015545266,
        std_coef_shallow: -0.04646168514,
        std_coef_deep: 0.01217093571,
    },
    ProbcutParams {
        mean_intercept: -1.765400238,
        mean_coef_shallow: 0.2491516205,
        mean_coef_deep: 0.06065109995,
        std_intercept: 0.9497283917,
        std_coef_shallow: -0.04657363638,
        std_coef_deep: 0.02223902354,
    },
    ProbcutParams {
        mean_intercept: -1.240393064,
        mean_coef_shallow: 0.2073544828,
        mean_coef_deep: 0.01953352516,
        std_intercept: 0.8905076143,
        std_coef_shallow: -0.04596042887,
        std_coef_deep: 0.02895965933,
    },
    ProbcutParams {
        mean_intercept: -1.094909733,
        mean_coef_shallow: 0.2042699937,
        mean_coef_deep: 0.01968955592,
        std_intercept: 0.9370187469,
        std_coef_shallow: -0.03237982048,
        std_coef_deep: 0.02108509566,
    },
    ProbcutParams {
        mean_intercept: -1.821030922,
        mean_coef_shallow: 0.2366400328,
        mean_coef_deep: 0.06994995379,
        std_intercept: 0.9145469555,
        std_coef_shallow: -0.03687769119,
        std_coef_deep: 0.02956981164,
    },
    ProbcutParams {
        mean_intercept: -1.192590591,
        mean_coef_shallow: 0.233994297,
        mean_coef_deep: 0.002539261998,
        std_intercept: 0.9744352457,
        std_coef_shallow: -0.04069505914,
        std_coef_deep: 0.03112143487,
    },
    ProbcutParams {
        mean_intercept: -1.67683668,
        mean_coef_shallow: 0.2575222737,
        mean_coef_deep: 0.05911447868,
        std_intercept: 1.148406049,
        std_coef_shallow: -0.0410819859,
        std_coef_deep: 0.01298012376,
    },
    ProbcutParams {
        mean_intercept: -1.329586087,
        mean_coef_shallow: 0.2202452363,
        mean_coef_deep: 0.01193448893,
        std_intercept: 1.082197759,
        std_coef_shallow: -0.04304501112,
        std_coef_deep: 0.01946352095,
    },
    ProbcutParams {
        mean_intercept: -1.367581465,
        mean_coef_shallow: 0.2380812306,
        mean_coef_deep: 0.0355053148,
        std_intercept: 1.133088112,
        std_coef_shallow: -0.06291165795,
        std_coef_deep: 0.01940591642,
    },
    ProbcutParams {
        mean_intercept: -1.664126411,
        mean_coef_shallow: 0.2103963482,
        mean_coef_deep: 0.06804747911,
        std_intercept: 1.121721164,
        std_coef_shallow: -0.05471380956,
        std_coef_deep: 0.02086967578,
    },
    ProbcutParams {
        mean_intercept: -0.9069755417,
        mean_coef_shallow: 0.229078322,
        mean_coef_deep: -0.03320286493,
        std_intercept: 1.112295,
        std_coef_shallow: -0.07183442089,
        std_coef_deep: 0.02837883093,
    },
    ProbcutParams {
        mean_intercept: -2.19636332,
        mean_coef_shallow: 0.2731388194,
        mean_coef_deep: 0.1096174754,
        std_intercept: 1.194777512,
        std_coef_shallow: -0.06733656026,
        std_coef_deep: 0.01930043623,
    },
    ProbcutParams {
        mean_intercept: -0.661968307,
        mean_coef_shallow: 0.1715069556,
        mean_coef_deep: -0.04291267468,
        std_intercept: 1.212334663,
        std_coef_shallow: -0.06777132501,
        std_coef_deep: 0.01952033399,
    },
    ProbcutParams {
        mean_intercept: -2.053427458,
        mean_coef_shallow: 0.2973297963,
        mean_coef_deep: 0.08974457606,
        std_intercept: 1.24048381,
        std_coef_shallow: -0.0769639472,
        std_coef_deep: 0.02337135366,
    },
    ProbcutParams {
        mean_intercept: -1.242210812,
        mean_coef_shallow: 0.2550951549,
        mean_coef_deep: -0.03952368139,
        std_intercept: 1.279810685,
        std_coef_shallow: -0.07310891433,
        std_coef_deep: 0.01526780862,
    },
    ProbcutParams {
        mean_intercept: -1.470735228,
        mean_coef_shallow: 0.196723987,
        mean_coef_deep: 0.08414700392,
        std_intercept: 1.266808093,
        std_coef_shallow: -0.07748364766,
        std_coef_deep: 0.01867234535,
    },
    ProbcutParams {
        mean_intercept: -1.353908648,
        mean_coef_shallow: 0.3181204964,
        mean_coef_deep: -0.07033263721,
        std_intercept: 1.241542034,
        std_coef_shallow: -0.07641729518,
        std_coef_deep: 0.02275946107,
    },
    ProbcutParams {
        mean_intercept: -1.174636839,
        mean_coef_shallow: 0.1532148497,
        mean_coef_deep: 0.06730455457,
        std_intercept: 1.281593153,
        std_coef_shallow: -0.07019145542,
        std_coef_deep: 0.01851578911,
    },
    ProbcutParams {
        mean_intercept: -1.158048548,
        mean_coef_shallow: 0.2936836919,
        mean_coef_deep: -0.07658811286,
        std_intercept: 1.223671427,
        std_coef_shallow: -0.07726896776,
        std_coef_deep: 0.02821722555,
    },
    ProbcutParams {
        mean_intercept: -0.9117628854,
        mean_coef_shallow: 0.09383875915,
        mean_coef_deep: 0.07351533643,
        std_intercept: 1.286443773,
        std_coef_shallow: -0.06999328167,
        std_coef_deep: 0.02050766021,
    },
    ProbcutParams {
        mean_intercept: -1.283841718,
        mean_coef_shallow: 0.2821548359,
        mean_coef_deep: -0.05148040358,
        std_intercept: 1.133404078,
        std_coef_shallow: -0.07047499073,
        std_coef_deep: 0.03756916528,
    },
    ProbcutParams {
        mean_intercept: -0.4867276008,
        mean_coef_shallow: 0.1433835358,
        mean_coef_deep: -0.01695174437,
        std_intercept: 1.20135579,
        std_coef_shallow: -0.07071111001,
        std_coef_deep: 0.0300480579,
    },
    ProbcutParams {
        mean_intercept: -1.420002526,
        mean_coef_shallow: 0.2697587312,
        mean_coef_deep: -0.005429102463,
        std_intercept: 1.346238843,
        std_coef_shallow: -0.06676482552,
        std_coef_deep: 0.01717513241,
    },
    ProbcutParams {
        mean_intercept: -0.5145967352,
        mean_coef_shallow: 0.1646343151,
        mean_coef_deep: -0.05234290531,
        std_intercept: 1.246597469,
        std_coef_shallow: -0.08235470245,
        std_coef_deep: 0.03079688219,
    },
    ProbcutParams {
        mean_intercept: -1.164247136,
        mean_coef_shallow: 0.220799765,
        mean_coef_deep: 0.0008763436237,
        std_intercept: 1.208768533,
        std_coef_shallow: -0.0878507473,
        std_coef_deep: 0.04062609226,
    },
    ProbcutParams {
        mean_intercept: -0.3027589769,
        mean_coef_shallow: 0.1498038586,
        mean_coef_deep: -0.08352846658,
        std_intercept: 1.2264972,
        std_coef_shallow: -0.08207341921,
        std_coef_deep: 0.03787209259,
    },
    ProbcutParams {
        mean_intercept: -0.9055030701,
        mean_coef_shallow: 0.192772869,
        mean_coef_deep: -0.01408716451,
        std_intercept: 1.240718394,
        std_coef_shallow: -0.07156804414,
        std_coef_deep: 0.02990701904,
    },
    ProbcutParams {
        mean_intercept: -0.4646178923,
        mean_coef_shallow: 0.1906541917,
        mean_coef_deep: -0.08834743534,
        std_intercept: 1.233335371,
        std_coef_shallow: -0.07862164825,
        std_coef_deep: 0.03449329722,
    },
    ProbcutParams {
        mean_intercept: -0.8062587075,
        mean_coef_shallow: 0.1589211717,
        mean_coef_deep: -0.009244026978,
        std_intercept: 1.193411737,
        std_coef_shallow: -0.08886814152,
        std_coef_deep: 0.04045107408,
    },
    ProbcutParams {
        mean_intercept: -0.8018017465,
        mean_coef_shallow: 0.2389957146,
        mean_coef_deep: -0.09332426878,
        std_intercept: 1.318508295,
        std_coef_shallow: -0.09766781194,
        std_coef_deep: 0.02351155078,
    },
    ProbcutParams {
        mean_intercept: -0.674719614,
        mean_coef_shallow: 0.1261118474,
        mean_coef_deep: 0.001933201985,
        std_intercept: 1.394366683,
        std_coef_shallow: -0.141228431,
        std_coef_deep: 0.02324176721,
    },
    ProbcutParams {
        mean_intercept: -0.8787601799,
        mean_coef_shallow: 0.190798057,
        mean_coef_deep: -0.04649355238,
        std_intercept: 1.649473866,
        std_coef_shallow: -0.2726065338,
        std_coef_deep: 0.01787080845,
    },
    ProbcutParams {
        mean_intercept: -0.5374896298,
        mean_coef_shallow: 0.0962396391,
        mean_coef_deep: 0.006713485639,
        std_intercept: 1.650072243,
        std_coef_shallow: -0.3188090209,
        std_coef_deep: 0.02065004002,
    },
    ProbcutParams {
        mean_intercept: -1.128586558,
        mean_coef_shallow: 0.2144925179,
        mean_coef_deep: -0.01965795872,
        std_intercept: 1.680925777,
        std_coef_shallow: -0.448247365,
        std_coef_deep: 0.01587633125,
    },
    ProbcutParams {
        mean_intercept: -0.2663952412,
        mean_coef_shallow: 0.05553728375,
        mean_coef_deep: -0.003112034866,
        std_intercept: 1.697762016,
        std_coef_shallow: -0.6403054618,
        std_coef_deep: 0.009129210487,
    },
    ProbcutParams {
        mean_intercept: -0.5277843122,
        mean_coef_shallow: 0.1036531157,
        mean_coef_deep: -0.003408256833,
        std_intercept: 1.188303405,
        std_coef_shallow: -0.5238654107,
        std_coef_deep: 0.002673654515,
    },
    ProbcutParams {
        mean_intercept: -0.2906782467,
        mean_coef_shallow: 0.05109927665,
        mean_coef_deep: 0.001454713682,
        std_intercept: 0.7474139833,
        std_coef_shallow: -0.5627283518,
        std_coef_deep: -0.01468667559,
    },
    ProbcutParams {
        mean_intercept: 0.06282127637,
        mean_coef_shallow: -0.0108291298,
        mean_coef_deep: -0.0005802241436,
        std_intercept: -0.7200149416,
        std_coef_shallow: -0.6379412379,
        std_coef_deep: -0.001982840805,
    },
    ProbcutParams {
        mean_intercept: 0.0,
        mean_coef_shallow: 0.0,
        mean_coef_deep: 0.0,
        std_intercept: -18.42068074,
        std_coef_shallow: 0.0,
        std_coef_deep: 0.0,
    },
];
