const BASE_EMPTIES: i32 = 30;
const CURVES: [(f64, f64); 4] = [
    (0.254, 1.340),
    (0.777, 1.296),
    (2.333, 1.295),
    (11.70, 1.156),
];

pub(super) fn factors(n_empties: u32) -> [f64; 5] {
    let exponent = n_empties as i32 - BASE_EMPTIES;
    let [level1, level2, level3, level4] =
        CURVES.map(|(scale, growth)| scale.mul_add(growth.powi(exponent), 1.0));
    [level1, level2, level3, level4, f64::INFINITY]
}

pub(super) fn projected_time(elapsed_ms: u64, factor: f64) -> f64 {
    elapsed_ms as f64 * factor
}

pub(super) fn permits(elapsed_ms: u64, budget_ms: u64, factor: f64) -> bool {
    projected_time(elapsed_ms, factor) < budget_ms as f64
}
