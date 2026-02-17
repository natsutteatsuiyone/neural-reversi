//! ProbCut forward pruning implementation for search optimization.

use std::sync::OnceLock;

use crate::types::Depth;
use crate::types::ScaledScore;

/// Selectivity level for search pruning (ProbCut confidence levels).
///
/// Lower levels are more aggressive (prune more), higher levels are more conservative.
/// `None` disables ProbCut entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Selectivity {
    /// Most aggressive: 73% confidence (t=1.1)
    #[default]
    Level1 = 0,
    /// 87% confidence (t=1.5)
    Level2 = 1,
    /// 95% confidence (t=2.0)
    Level3 = 2,
    /// 98% confidence (t=2.6)
    Level4 = 3,
    /// Most conservative: 99% confidence (t=3.3)
    Level5 = 4,
    /// ProbCut disabled.
    None = 5,
}

impl Selectivity {
    /// Selectivity configuration: (t_multiplier, probability_percent)
    const CONFIG: [(f64, i32); 6] = [
        (1.1, 73),    // Level1: Most aggressive
        (1.5, 87),    // Level2
        (2.0, 95),    // Level3
        (2.6, 98),    // Level4
        (3.3, 99),    // Level5: Most conservative
        (999.0, 100), // None: Effectively disabled
    ];

    /// Gets the statistical confidence multiplier (t-value).
    #[inline]
    pub fn t_value(self) -> f64 {
        Self::CONFIG[self as usize].0
    }

    /// Gets the expected success probability percentage.
    #[inline]
    pub fn probability(self) -> i32 {
        Self::CONFIG[self as usize].1
    }

    /// Converts to u8.
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Creates a Selectivity from a u8 value, clamping to valid range.
    ///
    /// Values > 5 are clamped to `Selectivity::None` (5).
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Selectivity::Level1,
            1 => Selectivity::Level2,
            2 => Selectivity::Level3,
            3 => Selectivity::Level4,
            4 => Selectivity::Level5,
            _ => Selectivity::None,
        }
    }

    /// Checks if ProbCut is enabled for this selectivity level.
    #[inline]
    pub fn is_enabled(self) -> bool {
        self != Selectivity::None
    }
}

/// Statistical parameters for ProbCut prediction models.
///
/// - `mean = mean_intercept + mean_coef_shallow * shallow + mean_coef_deep * deep`
/// - `sigma = exp(std_intercept + std_coef_shallow * shallow + std_coef_deep * deep)`
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
static MEAN_TABLE_END: OnceLock<Box<[[f64; MAX_DEPTH]; MAX_DEPTH]>> = OnceLock::new();
static SIGMA_TABLE_END: OnceLock<Box<[[f64; MAX_DEPTH]; MAX_DEPTH]>> = OnceLock::new();

fn alloc_3d_table() -> Box<MeanTable> {
    vec![[[0.0f64; MAX_DEPTH]; MAX_DEPTH]; MAX_PLY]
        .into_boxed_slice()
        .try_into()
        .unwrap()
}

fn alloc_2d_table() -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    vec![[0.0f64; MAX_DEPTH]; MAX_DEPTH]
        .into_boxed_slice()
        .try_into()
        .unwrap()
}

/// Builds a symmetric 3D [ply][shallow][deep] table from midgame ProbCut parameters.
fn build_mid_table(f: impl Fn(&ProbcutParams, f64, f64) -> f64) -> Box<MeanTable> {
    let mut tbl = alloc_3d_table();
    for ply in 0..MAX_PLY {
        let params = &PROBCUT_PARAMS[ply];
        for shallow in 0..MAX_DEPTH {
            for deep in shallow..MAX_DEPTH {
                let v = f(params, shallow as f64, deep as f64) * SCORE_SCALE_F64;
                tbl[ply][shallow][deep] = v;
                tbl[ply][deep][shallow] = v;
            }
        }
    }
    tbl
}

/// Builds a symmetric 2D [shallow][deep] table from endgame ProbCut parameters.
fn build_end_table(
    f: impl Fn(&ProbcutParams, f64, f64) -> f64,
) -> Box<[[f64; MAX_DEPTH]; MAX_DEPTH]> {
    let mut tbl = alloc_2d_table();
    for shallow in 0..MAX_DEPTH {
        for deep in shallow..MAX_DEPTH {
            let v = f(&PROBCUT_ENDGAME_PARAMS, shallow as f64, deep as f64) * SCORE_SCALE_F64;
            tbl[shallow][deep] = v;
            tbl[deep][shallow] = v;
        }
    }
    tbl
}

/// Initializes probcut tables. Called from Search::new().
pub fn init() {
    MEAN_TABLE.get_or_init(|| build_mid_table(ProbcutParams::mean));
    SIGMA_TABLE.get_or_init(|| build_mid_table(ProbcutParams::sigma));
    MEAN_TABLE_END.get_or_init(|| build_end_table(ProbcutParams::mean));
    SIGMA_TABLE_END.get_or_init(|| build_end_table(ProbcutParams::sigma));
}

/// Returns the pre-computed mean value for midgame positions.
#[inline]
pub fn get_mean(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    debug_assert!(ply < MAX_PLY);
    debug_assert!((shallow as usize) < MAX_DEPTH);
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = MEAN_TABLE.get().expect("probcut not initialized");
    tbl[ply][shallow as usize][deep as usize]
}

/// Returns the pre-computed sigma value for midgame positions.
#[inline]
pub fn get_sigma(ply: usize, shallow: Depth, deep: Depth) -> f64 {
    debug_assert!(ply < MAX_PLY);
    debug_assert!((shallow as usize) < MAX_DEPTH);
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = SIGMA_TABLE.get().expect("probcut not initialized");
    tbl[ply][shallow as usize][deep as usize]
}

/// Returns the pre-computed mean value for endgame positions.
#[inline]
pub fn get_mean_end(shallow: Depth, deep: Depth) -> f64 {
    debug_assert!((shallow as usize) < MAX_DEPTH);
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = MEAN_TABLE_END.get().expect("probcut not initialized");
    tbl[shallow as usize][deep as usize]
}

/// Returns the pre-computed sigma value for endgame positions.
#[inline]
pub fn get_sigma_end(shallow: Depth, deep: Depth) -> f64 {
    debug_assert!((shallow as usize) < MAX_DEPTH);
    debug_assert!((deep as usize) < MAX_DEPTH);
    let tbl = SIGMA_TABLE_END.get().expect("probcut not initialized");
    tbl[shallow as usize][deep as usize]
}

/// Computes the ProbCut beta threshold for verification search.
#[inline]
pub fn compute_probcut_beta(beta: ScaledScore, t: f64, mean: f64, sigma: f64) -> ScaledScore {
    ScaledScore::from_raw((beta.value() as f64 + t * sigma - mean).ceil() as i32)
}

/// Computes the evaluation threshold for ProbCut pre-screening.
#[inline]
pub fn compute_eval_beta(
    beta: ScaledScore,
    t: f64,
    mean: f64,
    sigma: f64,
    mean0: f64,
    sigma0: f64,
) -> ScaledScore {
    let eval_mean = 0.5 * mean0 + mean;
    let eval_sigma = t * 0.5 * sigma0 + sigma;
    ScaledScore::from_raw((beta.value() as f64 - eval_sigma - eval_mean).floor() as i32)
}

/// Statistical parameters for endgame ProbCut.
#[rustfmt::skip]
const PROBCUT_ENDGAME_PARAMS: ProbcutParams = ProbcutParams {
    mean_intercept: -0.2069276080,
    mean_coef_shallow: 0.0261871778,
    mean_coef_deep: 0.0067046108,
    std_intercept: 0.9052427490,
    std_coef_shallow: -0.0609488670,
    std_coef_deep: 0.0312314258,
};

/// Statistical parameters for midgame ProbCut indexed by ply.
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
        mean_intercept: -1.6241500826,
        mean_coef_shallow: 0.0896272953,
        mean_coef_deep: 0.0399202487,
        std_intercept: -1.0911935438,
        std_coef_shallow: 0.0500167023,
        std_coef_deep: 0.0077950977,
    },
    ProbcutParams {
        mean_intercept: 0.3260215825,
        mean_coef_shallow: 0.0049247806,
        mean_coef_deep: -0.0162435675,
        std_intercept: -0.0289131956,
        std_coef_shallow: 0.0697754731,
        std_coef_deep: -0.0380200871,
    },
    ProbcutParams {
        mean_intercept: -0.6125993516,
        mean_coef_shallow: 0.1480851881,
        mean_coef_deep: 0.0289136238,
        std_intercept: -0.0248256602,
        std_coef_shallow: 0.0831218138,
        std_coef_deep: -0.0267589604,
    },
    ProbcutParams {
        mean_intercept: -0.0976775585,
        mean_coef_shallow: 0.1172016162,
        mean_coef_deep: -0.0470486261,
        std_intercept: 0.1694195289,
        std_coef_shallow: 0.0598178683,
        std_coef_deep: -0.0217729561,
    },
    ProbcutParams {
        mean_intercept: -0.1907128028,
        mean_coef_shallow: -0.0247026754,
        mean_coef_deep: 0.0787244667,
        std_intercept: 0.3277816830,
        std_coef_shallow: 0.0126836238,
        std_coef_deep: -0.0113898962,
    },
    ProbcutParams {
        mean_intercept: 0.1789283097,
        mean_coef_shallow: -0.0354901693,
        mean_coef_deep: -0.0166627667,
        std_intercept: 0.4270933361,
        std_coef_shallow: -0.0158006945,
        std_coef_deep: -0.0053412976,
    },
    ProbcutParams {
        mean_intercept: 0.3210064900,
        mean_coef_shallow: -0.0917749986,
        mean_coef_deep: 0.0418147976,
        std_intercept: 0.4110921248,
        std_coef_shallow: -0.0170365775,
        std_coef_deep: -0.0041102875,
    },
    ProbcutParams {
        mean_intercept: -0.4017531595,
        mean_coef_shallow: -0.0680990829,
        mean_coef_deep: 0.0850559271,
        std_intercept: 0.4624712957,
        std_coef_shallow: -0.0615643296,
        std_coef_deep: 0.0172424837,
    },
    ProbcutParams {
        mean_intercept: 0.9020867546,
        mean_coef_shallow: -0.0372977504,
        mean_coef_deep: -0.1005678037,
        std_intercept: 0.5782427262,
        std_coef_shallow: -0.0721396602,
        std_coef_deep: 0.0149015376,
    },
    ProbcutParams {
        mean_intercept: -0.1756714356,
        mean_coef_shallow: -0.1331834630,
        mean_coef_deep: 0.1040748538,
        std_intercept: 0.6078911140,
        std_coef_shallow: -0.0811087621,
        std_coef_deep: 0.0155938761,
    },
    ProbcutParams {
        mean_intercept: 0.2959318738,
        mean_coef_shallow: -0.0139447165,
        mean_coef_deep: -0.0469662516,
        std_intercept: 0.6503640672,
        std_coef_shallow: -0.0758994460,
        std_coef_deep: 0.0144442083,
    },
    ProbcutParams {
        mean_intercept: -0.0884287027,
        mean_coef_shallow: 0.0342618205,
        mean_coef_deep: 0.0277946244,
        std_intercept: 0.7010827931,
        std_coef_shallow: -0.0765548709,
        std_coef_deep: 0.0123675907,
    },
    ProbcutParams {
        mean_intercept: -0.1271039808,
        mean_coef_shallow: -0.1683577301,
        mean_coef_deep: 0.0721942641,
        std_intercept: 0.7168449532,
        std_coef_shallow: -0.0715644268,
        std_coef_deep: 0.0128842988,
    },
    ProbcutParams {
        mean_intercept: 0.1927493694,
        mean_coef_shallow: 0.0733545647,
        mean_coef_deep: -0.0462139319,
        std_intercept: 0.7721304184,
        std_coef_shallow: -0.0763263172,
        std_coef_deep: 0.0129946933,
    },
    ProbcutParams {
        mean_intercept: -0.4821328504,
        mean_coef_shallow: -0.1129912397,
        mean_coef_deep: 0.1203593885,
        std_intercept: 0.8094429981,
        std_coef_shallow: -0.0827219347,
        std_coef_deep: 0.0117617649,
    },
    ProbcutParams {
        mean_intercept: 0.2837508314,
        mean_coef_shallow: -0.0265026596,
        mean_coef_deep: -0.0689210157,
        std_intercept: 0.8483713399,
        std_coef_shallow: -0.0670197742,
        std_coef_deep: 0.0085257545,
    },
    ProbcutParams {
        mean_intercept: -0.6136847406,
        mean_coef_shallow: 0.0854193667,
        mean_coef_deep: 0.1087030968,
        std_intercept: 0.8545353785,
        std_coef_shallow: -0.0797517477,
        std_coef_deep: 0.0108134655,
    },
    ProbcutParams {
        mean_intercept: 0.0857318668,
        mean_coef_shallow: -0.1311851567,
        mean_coef_deep: -0.0077733175,
        std_intercept: 0.8027475282,
        std_coef_shallow: -0.0802211287,
        std_coef_deep: 0.0187620451,
    },
    ProbcutParams {
        mean_intercept: -0.3539162302,
        mean_coef_shallow: 0.0840598084,
        mean_coef_deep: 0.0626087446,
        std_intercept: 0.8015752635,
        std_coef_shallow: -0.0671960764,
        std_coef_deep: 0.0183563883,
    },
    ProbcutParams {
        mean_intercept: -0.5127698120,
        mean_coef_shallow: -0.0213173407,
        mean_coef_deep: 0.0599224428,
        std_intercept: 0.8135731515,
        std_coef_shallow: -0.0785555392,
        std_coef_deep: 0.0194186551,
    },
    ProbcutParams {
        mean_intercept: 0.1082171828,
        mean_coef_shallow: -0.0434877111,
        mean_coef_deep: 0.0004388606,
        std_intercept: 0.7996306138,
        std_coef_shallow: -0.0733575993,
        std_coef_deep: 0.0222479249,
    },
    ProbcutParams {
        mean_intercept: -1.0207969380,
        mean_coef_shallow: 0.1683103463,
        mean_coef_deep: 0.0963896446,
        std_intercept: 0.7821350776,
        std_coef_shallow: -0.0677173109,
        std_coef_deep: 0.0216405249,
    },
    ProbcutParams {
        mean_intercept: 0.2191784348,
        mean_coef_shallow: -0.1097749271,
        mean_coef_deep: -0.0129049484,
        std_intercept: 0.8036010138,
        std_coef_shallow: -0.0690151173,
        std_coef_deep: 0.0213601397,
    },
    ProbcutParams {
        mean_intercept: -0.5511842073,
        mean_coef_shallow: 0.0691997290,
        mean_coef_deep: 0.0890085124,
        std_intercept: 0.8274308946,
        std_coef_shallow: -0.0649121112,
        std_coef_deep: 0.0207492936,
    },
    ProbcutParams {
        mean_intercept: -0.1706840997,
        mean_coef_shallow: -0.0152865366,
        mean_coef_deep: 0.0042667053,
        std_intercept: 0.8898403164,
        std_coef_shallow: -0.0706485955,
        std_coef_deep: 0.0190608184,
    },
    ProbcutParams {
        mean_intercept: -0.1414984236,
        mean_coef_shallow: -0.0359355216,
        mean_coef_deep: 0.0468859363,
        std_intercept: 0.8906660315,
        std_coef_shallow: -0.0652735187,
        std_coef_deep: 0.0200731346,
    },
    ProbcutParams {
        mean_intercept: -0.4825967955,
        mean_coef_shallow: 0.0712549553,
        mean_coef_deep: 0.0470552149,
        std_intercept: 0.9436736519,
        std_coef_shallow: -0.0688438256,
        std_coef_deep: 0.0176567578,
    },
    ProbcutParams {
        mean_intercept: -0.2345712350,
        mean_coef_shallow: -0.0378550947,
        mean_coef_deep: 0.0584031529,
        std_intercept: 0.9530817201,
        std_coef_shallow: -0.0690198030,
        std_coef_deep: 0.0187932057,
    },
    ProbcutParams {
        mean_intercept: -0.4524467460,
        mean_coef_shallow: 0.0872435511,
        mean_coef_deep: 0.0360173807,
        std_intercept: 0.9424380778,
        std_coef_shallow: -0.0591270791,
        std_coef_deep: 0.0185625812,
    },
    ProbcutParams {
        mean_intercept: -0.2581430537,
        mean_coef_shallow: 0.0052096056,
        mean_coef_deep: 0.0508915010,
        std_intercept: 0.9870132634,
        std_coef_shallow: -0.0695731984,
        std_coef_deep: 0.0175473205,
    },
    ProbcutParams {
        mean_intercept: -0.2498579279,
        mean_coef_shallow: 0.0282029893,
        mean_coef_deep: 0.0202855065,
        std_intercept: 1.0095676982,
        std_coef_shallow: -0.0746588050,
        std_coef_deep: 0.0178589585,
    },
    ProbcutParams {
        mean_intercept: -0.4121454269,
        mean_coef_shallow: 0.0310542311,
        mean_coef_deep: 0.0711216283,
        std_intercept: 1.0157307464,
        std_coef_shallow: -0.0721688340,
        std_coef_deep: 0.0178371380,
    },
    ProbcutParams {
        mean_intercept: -0.3573005332,
        mean_coef_shallow: 0.0635741568,
        mean_coef_deep: 0.0289601444,
        std_intercept: 1.0438554723,
        std_coef_shallow: -0.0730514267,
        std_coef_deep: 0.0168058048,
    },
    ProbcutParams {
        mean_intercept: -0.3549334540,
        mean_coef_shallow: 0.0594853079,
        mean_coef_deep: 0.0460298846,
        std_intercept: 1.0441977532,
        std_coef_shallow: -0.0709975520,
        std_coef_deep: 0.0170912566,
    },
    ProbcutParams {
        mean_intercept: -0.3734817790,
        mean_coef_shallow: 0.0483464373,
        mean_coef_deep: 0.0382379845,
        std_intercept: 1.0628026864,
        std_coef_shallow: -0.0730813252,
        std_coef_deep: 0.0166221020,
    },
    ProbcutParams {
        mean_intercept: -0.1927179841,
        mean_coef_shallow: 0.0236171262,
        mean_coef_deep: 0.0207946362,
        std_intercept: 1.0575408791,
        std_coef_shallow: -0.0695476051,
        std_coef_deep: 0.0179835281,
    },
    ProbcutParams {
        mean_intercept: -0.2506045952,
        mean_coef_shallow: 0.0406046850,
        mean_coef_deep: 0.0051572206,
        std_intercept: 1.0960948932,
        std_coef_shallow: -0.0724076358,
        std_coef_deep: 0.0175106038,
    },
    ProbcutParams {
        mean_intercept: -0.2728326421,
        mean_coef_shallow: 0.0570940188,
        mean_coef_deep: 0.0334547952,
        std_intercept: 1.1066285411,
        std_coef_shallow: -0.0721886083,
        std_coef_deep: 0.0181546651,
    },
    ProbcutParams {
        mean_intercept: -0.3912003165,
        mean_coef_shallow: 0.0534188132,
        mean_coef_deep: 0.0376633655,
        std_intercept: 1.1045842739,
        std_coef_shallow: -0.0757783281,
        std_coef_deep: 0.0208622515,
    },
    ProbcutParams {
        mean_intercept: -0.1065669794,
        mean_coef_shallow: 0.0099677091,
        mean_coef_deep: -0.0047029455,
        std_intercept: 1.1098897697,
        std_coef_shallow: -0.0753250088,
        std_coef_deep: 0.0225018872,
    },
    ProbcutParams {
        mean_intercept: -0.2880578180,
        mean_coef_shallow: 0.1012772502,
        mean_coef_deep: 0.0029199144,
        std_intercept: 1.0951982156,
        std_coef_shallow: -0.0725919745,
        std_coef_deep: 0.0233397918,
    },
    ProbcutParams {
        mean_intercept: -0.2210621941,
        mean_coef_shallow: 0.0732765189,
        mean_coef_deep: 0.0019877656,
        std_intercept: 1.0883305130,
        std_coef_shallow: -0.0680485556,
        std_coef_deep: 0.0244438167,
    },
    ProbcutParams {
        mean_intercept: -0.1627381666,
        mean_coef_shallow: 0.1129692524,
        mean_coef_deep: -0.0297898302,
        std_intercept: 1.0948019957,
        std_coef_shallow: -0.0647457420,
        std_coef_deep: 0.0233444838,
    },
    ProbcutParams {
        mean_intercept: -0.2242207408,
        mean_coef_shallow: 0.0510721365,
        mean_coef_deep: 0.0187459747,
        std_intercept: 1.1211379188,
        std_coef_shallow: -0.0670495160,
        std_coef_deep: 0.0240474611,
    },
    ProbcutParams {
        mean_intercept: -0.0392735960,
        mean_coef_shallow: 0.0688504118,
        mean_coef_deep: -0.0516939461,
        std_intercept: 1.1196749421,
        std_coef_shallow: -0.0682786720,
        std_coef_deep: 0.0261358270,
    },
    ProbcutParams {
        mean_intercept: -0.2136604195,
        mean_coef_shallow: 0.0733903471,
        mean_coef_deep: 0.0049621173,
        std_intercept: 1.1596344475,
        std_coef_shallow: -0.0735394544,
        std_coef_deep: 0.0247665485,
    },
    ProbcutParams {
        mean_intercept: -0.1065746828,
        mean_coef_shallow: 0.0400861017,
        mean_coef_deep: -0.0342537203,
        std_intercept: 1.1893568806,
        std_coef_shallow: -0.0707571594,
        std_coef_deep: 0.0213563859,
    },
    ProbcutParams {
        mean_intercept: -0.1053356347,
        mean_coef_shallow: 0.0776837218,
        mean_coef_deep: -0.0194605780,
        std_intercept: 1.2134570151,
        std_coef_shallow: -0.0752337837,
        std_coef_deep: 0.0194611415,
    },
    ProbcutParams {
        mean_intercept: -0.1701592163,
        mean_coef_shallow: 0.0400189575,
        mean_coef_deep: -0.0092878503,
        std_intercept: 1.2608785464,
        std_coef_shallow: -0.0900043192,
        std_coef_deep: 0.0176686696,
    },
    ProbcutParams {
        mean_intercept: -0.0474917204,
        mean_coef_shallow: 0.0408550862,
        mean_coef_deep: -0.0218318346,
        std_intercept: 1.2851448218,
        std_coef_shallow: -0.1053644224,
        std_coef_deep: 0.0150233812,
    },
    ProbcutParams {
        mean_intercept: -0.2101363000,
        mean_coef_shallow: 0.0621827818,
        mean_coef_deep: 0.0051317880,
        std_intercept: 1.3315555385,
        std_coef_shallow: -0.1350851675,
        std_coef_deep: 0.0120401167,
    },
    ProbcutParams {
        mean_intercept: -0.0938631626,
        mean_coef_shallow: 0.0013996507,
        mean_coef_deep: -0.0053146214,
        std_intercept: 1.4019982763,
        std_coef_shallow: -0.1976761444,
        std_coef_deep: 0.0089897738,
    },
    ProbcutParams {
        mean_intercept: -0.2740765823,
        mean_coef_shallow: 0.0140331307,
        mean_coef_deep: 0.0207697314,
        std_intercept: 1.4121076751,
        std_coef_shallow: -0.2477095461,
        std_coef_deep: 0.0058147486,
    },
    ProbcutParams {
        mean_intercept: -0.2171730622,
        mean_coef_shallow: 0.0121699138,
        mean_coef_deep: -0.0051931055,
        std_intercept: 1.6662768327,
        std_coef_shallow: -0.5804872540,
        std_coef_deep: 0.0147064097,
    },
    ProbcutParams {
        mean_intercept: -0.0193913535,
        mean_coef_shallow: -0.0073053272,
        mean_coef_deep: 0.0045033259,
        std_intercept: 1.3229515242,
        std_coef_shallow: -0.5155756685,
        std_coef_deep: 0.0074412334,
    },
    ProbcutParams {
        mean_intercept: -0.1749986694,
        mean_coef_shallow: -0.0001200673,
        mean_coef_deep: 0.0003462770,
        std_intercept: 1.4329094722,
        std_coef_shallow: -0.8076616640,
        std_coef_deep: 0.0009475232,
    },
    ProbcutParams {
        mean_intercept: 0.1968980462,
        mean_coef_shallow: 0.0011064460,
        mean_coef_deep: -0.0010713861,
        std_intercept: 0.4329888678,
        std_coef_shallow: -0.6152019372,
        std_coef_deep: 0.0000472670,
    },
    ProbcutParams {
        mean_intercept: -0.2320099253,
        mean_coef_shallow: -0.0006375949,
        mean_coef_deep: 0.0005847591,
        std_intercept: -0.6559805529,
        std_coef_shallow: -0.5806823655,
        std_coef_deep: -0.0001678472,
    },
    ProbcutParams {
        mean_intercept: 0.0000000000,
        mean_coef_shallow: 0.0000000000,
        mean_coef_deep: 0.0000000000,
        std_intercept: -18.4206807440,
        std_coef_shallow: 0.0000000000,
        std_coef_deep: 0.0000000000,
    },
];

const _: () = assert!(PROBCUT_PARAMS.len() == MAX_PLY);
