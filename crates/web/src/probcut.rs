use std::sync::OnceLock;

use reversi_core::{
    board::Board,
    probcut::Selectivity,
    search::node_type::NonPV,
    types::{Depth, ScaledScore},
};

use crate::search::{self, search_context::SearchContext};

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

const SCORE_SCALE_F64: f64 = ScaledScore::SCALE as f64;

static MEAN_TABLE: OnceLock<Box<MeanTable>> = OnceLock::new();
static SIGMA_TABLE: OnceLock<Box<SigmaTable>> = OnceLock::new();

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

/// Initializes probcut lookup tables.
pub fn init() {
    MEAN_TABLE.set(build_mean_table()).ok();
    SIGMA_TABLE.set(build_sigma_table()).ok();
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
/// Returns [`Some(score)`] if a beta cutoff is predicted, or [`None`] if the
/// deep search should proceed.
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
        mean_intercept: 2.3657274715,
        mean_coef_shallow: -0.2062258321,
        mean_coef_deep: -0.1860090700,
        std_intercept: -0.5548543061,
        std_coef_shallow: 0.0335795412,
        std_coef_deep: -0.0018915959,
    },
    ProbcutParams {
        mean_intercept: -0.1065684202,
        mean_coef_shallow: 0.1009317802,
        mean_coef_deep: 0.0281197368,
        std_intercept: 0.4769835858,
        std_coef_shallow: 0.0389339992,
        std_coef_deep: -0.0937092361,
    },
    ProbcutParams {
        mean_intercept: 0.5521136420,
        mean_coef_shallow: -0.3597909864,
        mean_coef_deep: -0.0318666974,
        std_intercept: 0.7213086466,
        std_coef_shallow: 0.2049355508,
        std_coef_deep: -0.1001259732,
    },
    ProbcutParams {
        mean_intercept: -0.0629208784,
        mean_coef_shallow: 0.0960764436,
        mean_coef_deep: 0.0417402526,
        std_intercept: 0.6069898742,
        std_coef_shallow: 0.1169781206,
        std_coef_deep: -0.0527290022,
    },
    ProbcutParams {
        mean_intercept: -0.2499992094,
        mean_coef_shallow: 0.0203899552,
        mean_coef_deep: 0.0321296765,
        std_intercept: 0.9709726220,
        std_coef_shallow: -0.0924412791,
        std_coef_deep: -0.0085002675,
    },
    ProbcutParams {
        mean_intercept: 0.3899947648,
        mean_coef_shallow: -0.0835504458,
        mean_coef_deep: -0.0379334882,
        std_intercept: 1.0580459641,
        std_coef_shallow: -0.1665397534,
        std_coef_deep: -0.0036140370,
    },
    ProbcutParams {
        mean_intercept: -0.2542623065,
        mean_coef_shallow: 0.0350889507,
        mean_coef_deep: 0.0476920408,
        std_intercept: 1.0811652191,
        std_coef_shallow: -0.1981277933,
        std_coef_deep: 0.0070690604,
    },
    ProbcutParams {
        mean_intercept: 0.0820776994,
        mean_coef_shallow: -0.0010119156,
        mean_coef_deep: -0.0168501460,
        std_intercept: 0.7215857714,
        std_coef_shallow: -0.1137471023,
        std_coef_deep: 0.0133332312,
    },
    ProbcutParams {
        mean_intercept: 0.2338540833,
        mean_coef_shallow: -0.0422476063,
        mean_coef_deep: -0.0057719556,
        std_intercept: 0.7061343270,
        std_coef_shallow: -0.0972503697,
        std_coef_deep: 0.0143656447,
    },
    ProbcutParams {
        mean_intercept: 0.1664674565,
        mean_coef_shallow: -0.0280823762,
        mean_coef_deep: -0.0153211628,
        std_intercept: 0.7242992274,
        std_coef_shallow: -0.1078759919,
        std_coef_deep: 0.0212174782,
    },
    ProbcutParams {
        mean_intercept: 0.3099273645,
        mean_coef_shallow: -0.0441447588,
        mean_coef_deep: -0.0458588108,
        std_intercept: 0.8071896553,
        std_coef_shallow: -0.1232396857,
        std_coef_deep: 0.0227877911,
    },
    ProbcutParams {
        mean_intercept: 0.1484798434,
        mean_coef_shallow: -0.0202079940,
        mean_coef_deep: -0.0228167688,
        std_intercept: 0.8227946645,
        std_coef_shallow: -0.1230942357,
        std_coef_deep: 0.0223929995,
    },
    ProbcutParams {
        mean_intercept: 0.2150084846,
        mean_coef_shallow: -0.0372648622,
        mean_coef_deep: -0.0396459369,
        std_intercept: 0.8799356736,
        std_coef_shallow: -0.1209812026,
        std_coef_deep: 0.0186144422,
    },
    ProbcutParams {
        mean_intercept: 0.1420298338,
        mean_coef_shallow: -0.0188727029,
        mean_coef_deep: -0.0324613812,
        std_intercept: 0.8507108344,
        std_coef_shallow: -0.1164639497,
        std_coef_deep: 0.0220291285,
    },
    ProbcutParams {
        mean_intercept: 0.1298033668,
        mean_coef_shallow: 0.0165584578,
        mean_coef_deep: -0.0523242650,
        std_intercept: 0.8351227348,
        std_coef_shallow: -0.1096179316,
        std_coef_deep: 0.0240085532,
    },
    ProbcutParams {
        mean_intercept: 0.1512300075,
        mean_coef_shallow: 0.0047826345,
        mean_coef_deep: -0.0500001813,
        std_intercept: 0.8749337496,
        std_coef_shallow: -0.1170416835,
        std_coef_deep: 0.0242372630,
    },
    ProbcutParams {
        mean_intercept: 0.1229011662,
        mean_coef_shallow: 0.0087990583,
        mean_coef_deep: -0.0528696698,
        std_intercept: 0.9001428059,
        std_coef_shallow: -0.1161848640,
        std_coef_deep: 0.0231468391,
    },
    ProbcutParams {
        mean_intercept: 0.1329970550,
        mean_coef_shallow: 0.0302821986,
        mean_coef_deep: -0.0698473327,
        std_intercept: 0.9369409939,
        std_coef_shallow: -0.1160131564,
        std_coef_deep: 0.0192504878,
    },
    ProbcutParams {
        mean_intercept: 0.1001466977,
        mean_coef_shallow: 0.0397999873,
        mean_coef_deep: -0.0676942529,
        std_intercept: 0.9435825940,
        std_coef_shallow: -0.1072527516,
        std_coef_deep: 0.0175817394,
    },
    ProbcutParams {
        mean_intercept: 0.1504150363,
        mean_coef_shallow: 0.0406291912,
        mean_coef_deep: -0.0926054690,
        std_intercept: 0.9101101045,
        std_coef_shallow: -0.0978752177,
        std_coef_deep: 0.0192627254,
    },
    ProbcutParams {
        mean_intercept: 0.1192403984,
        mean_coef_shallow: 0.0324359745,
        mean_coef_deep: -0.0731165906,
        std_intercept: 0.9046701551,
        std_coef_shallow: -0.0921373561,
        std_coef_deep: 0.0200651300,
    },
    ProbcutParams {
        mean_intercept: 0.1316038211,
        mean_coef_shallow: 0.0425896754,
        mean_coef_deep: -0.0944022727,
        std_intercept: 0.9325325980,
        std_coef_shallow: -0.0930320212,
        std_coef_deep: 0.0200910454,
    },
    ProbcutParams {
        mean_intercept: 0.1588684981,
        mean_coef_shallow: 0.0185110407,
        mean_coef_deep: -0.0797322720,
        std_intercept: 0.9404441990,
        std_coef_shallow: -0.0901357206,
        std_coef_deep: 0.0207257386,
    },
    ProbcutParams {
        mean_intercept: 0.1093490991,
        mean_coef_shallow: 0.0464591458,
        mean_coef_deep: -0.0926336419,
        std_intercept: 0.9482829011,
        std_coef_shallow: -0.0882606227,
        std_coef_deep: 0.0217427501,
    },
    ProbcutParams {
        mean_intercept: 0.1951624308,
        mean_coef_shallow: 0.0195093336,
        mean_coef_deep: -0.1013347674,
        std_intercept: 0.9693129096,
        std_coef_shallow: -0.0896226758,
        std_coef_deep: 0.0234384005,
    },
    ProbcutParams {
        mean_intercept: 0.1055005339,
        mean_coef_shallow: 0.0165158044,
        mean_coef_deep: -0.0736388294,
        std_intercept: 0.9740976281,
        std_coef_shallow: -0.0893047811,
        std_coef_deep: 0.0257093613,
    },
    ProbcutParams {
        mean_intercept: 0.1620185818,
        mean_coef_shallow: 0.0086938186,
        mean_coef_deep: -0.0935782308,
        std_intercept: 1.0211375328,
        std_coef_shallow: -0.0920665475,
        std_coef_deep: 0.0230233693,
    },
    ProbcutParams {
        mean_intercept: 0.1389589832,
        mean_coef_shallow: 0.0168747544,
        mean_coef_deep: -0.0846369723,
        std_intercept: 1.0314008269,
        std_coef_shallow: -0.0901396284,
        std_coef_deep: 0.0220336675,
    },
    ProbcutParams {
        mean_intercept: 0.1635539253,
        mean_coef_shallow: 0.0065917117,
        mean_coef_deep: -0.0961402324,
        std_intercept: 1.0357538164,
        std_coef_shallow: -0.0903389886,
        std_coef_deep: 0.0249139213,
    },
    ProbcutParams {
        mean_intercept: 0.1627496986,
        mean_coef_shallow: -0.0102711294,
        mean_coef_deep: -0.0822308153,
        std_intercept: 1.0274074858,
        std_coef_shallow: -0.0839426265,
        std_coef_deep: 0.0248583804,
    },
    ProbcutParams {
        mean_intercept: 0.1588390321,
        mean_coef_shallow: 0.0089891069,
        mean_coef_deep: -0.0979990538,
        std_intercept: 1.0411540859,
        std_coef_shallow: -0.0844730757,
        std_coef_deep: 0.0254954106,
    },
    ProbcutParams {
        mean_intercept: 0.1450978241,
        mean_coef_shallow: -0.0033109587,
        mean_coef_deep: -0.0784219088,
        std_intercept: 1.0758127401,
        std_coef_shallow: -0.0876265760,
        std_coef_deep: 0.0243985971,
    },
    ProbcutParams {
        mean_intercept: 0.1766619874,
        mean_coef_shallow: 0.0072056800,
        mean_coef_deep: -0.1056286519,
        std_intercept: 1.0961626513,
        std_coef_shallow: -0.0856595744,
        std_coef_deep: 0.0237337465,
    },
    ProbcutParams {
        mean_intercept: 0.1622066798,
        mean_coef_shallow: -0.0032887004,
        mean_coef_deep: -0.0915955441,
        std_intercept: 1.1230373380,
        std_coef_shallow: -0.0887077849,
        std_coef_deep: 0.0232993044,
    },
    ProbcutParams {
        mean_intercept: 0.1465430087,
        mean_coef_shallow: -0.0055802345,
        mean_coef_deep: -0.0887182413,
        std_intercept: 1.1324201562,
        std_coef_shallow: -0.0938069177,
        std_coef_deep: 0.0241278778,
    },
    ProbcutParams {
        mean_intercept: 0.1662115684,
        mean_coef_shallow: 0.0040887931,
        mean_coef_deep: -0.1016784199,
        std_intercept: 1.1287440098,
        std_coef_shallow: -0.0927359372,
        std_coef_deep: 0.0270847301,
    },
    ProbcutParams {
        mean_intercept: 0.1912669948,
        mean_coef_shallow: 0.0276846205,
        mean_coef_deep: -0.1310779897,
        std_intercept: 1.1279050185,
        std_coef_shallow: -0.0911196737,
        std_coef_deep: 0.0287893948,
    },
    ProbcutParams {
        mean_intercept: 0.1288069328,
        mean_coef_shallow: 0.0002807740,
        mean_coef_deep: -0.0825369746,
        std_intercept: 1.1532760101,
        std_coef_shallow: -0.0913491404,
        std_coef_deep: 0.0270171103,
    },
    ProbcutParams {
        mean_intercept: 0.1648810391,
        mean_coef_shallow: 0.0243801094,
        mean_coef_deep: -0.1251622987,
        std_intercept: 1.1574589769,
        std_coef_shallow: -0.0899467168,
        std_coef_deep: 0.0280231201,
    },
    ProbcutParams {
        mean_intercept: 0.1142334412,
        mean_coef_shallow: 0.0014787097,
        mean_coef_deep: -0.0746495046,
        std_intercept: 1.1832324641,
        std_coef_shallow: -0.0963151707,
        std_coef_deep: 0.0285434189,
    },
    ProbcutParams {
        mean_intercept: 0.1716312834,
        mean_coef_shallow: 0.0315109852,
        mean_coef_deep: -0.1316486633,
        std_intercept: 1.2010933787,
        std_coef_shallow: -0.0967413041,
        std_coef_deep: 0.0274162038,
    },
    ProbcutParams {
        mean_intercept: 0.1003644685,
        mean_coef_shallow: -0.0258477037,
        mean_coef_deep: -0.0537622151,
        std_intercept: 1.2142490348,
        std_coef_shallow: -0.0955490316,
        std_coef_deep: 0.0277114547,
    },
    ProbcutParams {
        mean_intercept: 0.1691467262,
        mean_coef_shallow: 0.0226796046,
        mean_coef_deep: -0.1302507821,
        std_intercept: 1.2027103805,
        std_coef_shallow: -0.0912863320,
        std_coef_deep: 0.0286380971,
    },
    ProbcutParams {
        mean_intercept: 0.1147675148,
        mean_coef_shallow: -0.0275363878,
        mean_coef_deep: -0.0566145781,
        std_intercept: 1.2016051650,
        std_coef_shallow: -0.0885751899,
        std_coef_deep: 0.0312890890,
    },
    ProbcutParams {
        mean_intercept: 0.1293381834,
        mean_coef_shallow: 0.0223657209,
        mean_coef_deep: -0.1187955437,
        std_intercept: 1.1780206612,
        std_coef_shallow: -0.0904349521,
        std_coef_deep: 0.0363123223,
    },
    ProbcutParams {
        mean_intercept: 0.1091736626,
        mean_coef_shallow: -0.0301050036,
        mean_coef_deep: -0.0473954980,
        std_intercept: 1.1722910133,
        std_coef_shallow: -0.0912841643,
        std_coef_deep: 0.0405108987,
    },
    ProbcutParams {
        mean_intercept: 0.1007254053,
        mean_coef_shallow: 0.0162515298,
        mean_coef_deep: -0.1134138966,
        std_intercept: 1.2020125263,
        std_coef_shallow: -0.0924685564,
        std_coef_deep: 0.0387657346,
    },
    ProbcutParams {
        mean_intercept: 0.1304208025,
        mean_coef_shallow: -0.0405532058,
        mean_coef_deep: -0.0568705370,
        std_intercept: 1.1853285774,
        std_coef_shallow: -0.0856977797,
        std_coef_deep: 0.0395856381,
    },
    ProbcutParams {
        mean_intercept: 0.0042720627,
        mean_coef_shallow: 0.0035816781,
        mean_coef_deep: -0.0564344203,
        std_intercept: 1.2207797687,
        std_coef_shallow: -0.0922023905,
        std_coef_deep: 0.0382069697,
    },
    ProbcutParams {
        mean_intercept: 0.1647545781,
        mean_coef_shallow: -0.0157549564,
        mean_coef_deep: -0.1029516463,
        std_intercept: 1.2331864516,
        std_coef_shallow: -0.0894420872,
        std_coef_deep: 0.0349170751,
    },
    ProbcutParams {
        mean_intercept: -0.0329711140,
        mean_coef_shallow: 0.0065208686,
        mean_coef_deep: -0.0203074315,
        std_intercept: 1.2692694457,
        std_coef_shallow: -0.0969471317,
        std_coef_deep: 0.0308792476,
    },
    ProbcutParams {
        mean_intercept: 0.0652679017,
        mean_coef_shallow: -0.0050945121,
        mean_coef_deep: -0.0701358716,
        std_intercept: 1.2988403751,
        std_coef_shallow: -0.1107125597,
        std_coef_deep: 0.0244191827,
    },
    ProbcutParams {
        mean_intercept: -0.0566460137,
        mean_coef_shallow: 0.0299947476,
        mean_coef_deep: -0.0128169812,
        std_intercept: 1.3336266901,
        std_coef_shallow: -0.1352454315,
        std_coef_deep: 0.0167035832,
    },
    ProbcutParams {
        mean_intercept: -0.0451652244,
        mean_coef_shallow: 0.0409861564,
        mean_coef_deep: -0.0339141722,
        std_intercept: 1.3778449943,
        std_coef_shallow: -0.2101632731,
        std_coef_deep: 0.0103014352,
    },
    ProbcutParams {
        mean_intercept: -0.0541936123,
        mean_coef_shallow: 0.0217660708,
        mean_coef_deep: -0.0100570477,
        std_intercept: 1.6327110657,
        std_coef_shallow: -0.5304877157,
        std_coef_deep: 0.0073974623,
    },
    ProbcutParams {
        mean_intercept: -0.0909811108,
        mean_coef_shallow: 0.0015022419,
        mean_coef_deep: 0.0000985046,
        std_intercept: 1.7242548048,
        std_coef_shallow: -0.8613512149,
        std_coef_deep: 0.0040049250,
    },
    ProbcutParams {
        mean_intercept: -0.0860279674,
        mean_coef_shallow: -0.0010062658,
        mean_coef_deep: 0.0008352868,
        std_intercept: 1.3845730855,
        std_coef_shallow: -1.0305052498,
        std_coef_deep: 0.0000527647,
    },
    ProbcutParams {
        mean_intercept: 0.0301541251,
        mean_coef_shallow: 0.0001450670,
        mean_coef_deep: -0.0000979177,
        std_intercept: 0.3107399376,
        std_coef_shallow: -1.1553590909,
        std_coef_deep: 0.0001280388,
    },
    ProbcutParams {
        mean_intercept: 0.1031033156,
        mean_coef_shallow: 0.0003409591,
        mean_coef_deep: -0.0002279656,
        std_intercept: -1.5425002325,
        std_coef_shallow: -0.8698830350,
        std_coef_deep: 0.0002552290,
    },
];
