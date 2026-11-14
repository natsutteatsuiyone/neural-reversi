pub(crate) fn weighted_least_squares<const N: usize>(
    rows: impl Iterator<Item = ([f64; N], f64, f64)>,
) -> Option<[f64; N]> {
    let mut matrix = [[0.0f64; N]; N];
    let mut vector = [0.0f64; N];
    for (features, response, weight) in rows {
        for row in 0..N {
            for column in 0..N {
                matrix[row][column] += weight * features[row] * features[column];
            }
            vector[row] += weight * features[row] * response;
        }
    }
    solve(matrix, vector)
}

fn solve<const N: usize>(mut matrix: [[f64; N]; N], mut vector: [f64; N]) -> Option<[f64; N]> {
    let norm = matrix
        .iter()
        .flatten()
        .fold(0.0f64, |acc, &value| acc.max(value.abs()));
    let threshold = norm * 1e-12;

    for column in 0..N {
        let pivot = (column..N).max_by(|&left, &right| {
            matrix[left][column]
                .abs()
                .total_cmp(&matrix[right][column].abs())
        })?;
        if matrix[pivot][column].abs() <= threshold {
            return None;
        }
        matrix.swap(column, pivot);
        vector.swap(column, pivot);

        let (pivot_rows, remaining) = matrix.split_at_mut(column + 1);
        let pivot_row = &pivot_rows[column];
        for (offset, row) in remaining.iter_mut().enumerate() {
            let factor = row[column] / pivot_row[column];
            for (destination, source) in row[column..].iter_mut().zip(&pivot_row[column..]) {
                *destination -= factor * source;
            }
            vector[column + 1 + offset] -= factor * vector[column];
        }
    }

    let mut solution = [0.0f64; N];
    for row in (0..N).rev() {
        let mut value = vector[row];
        for (coefficient, solved) in matrix[row][row + 1..].iter().zip(&solution[row + 1..]) {
            value -= coefficient * solved;
        }
        solution[row] = value / matrix[row][row];
    }
    Some(solution)
}
