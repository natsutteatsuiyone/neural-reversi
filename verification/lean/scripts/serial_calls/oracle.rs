// Independent of reversi-core. Input rows are player, opponent, alpha, beta and returned raw score.
use std::collections::HashMap;
use std::io::{self, BufRead};

fn flips(player: u64, opponent: u64, square: u32) -> u64 {
    let mut captured = 0;
    for (dx, dy) in [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ] {
        let mut x = (square % 8) as i32 + dx;
        let mut y = (square / 8) as i32 + dy;
        let mut line = 0;
        while (0..8).contains(&x) && (0..8).contains(&y) {
            let bit = 1_u64 << (8 * y + x);
            if opponent & bit == 0 {
                if player & bit != 0 {
                    captured |= line;
                }
                break;
            }
            line |= bit;
            x += dx;
            y += dy;
        }
    }
    captured
}

fn negamax(player: u64, opponent: u64, mut alpha: i32, beta: i32, passed: bool) -> i32 {
    let empty = !(player | opponent);
    let mut candidates = empty;
    let mut best = -65;
    while candidates != 0 {
        let square = candidates.trailing_zeros();
        candidates &= candidates - 1;
        let taken = flips(player, opponent, square);
        if taken != 0 {
            let child = -negamax(
                opponent ^ taken,
                player | taken | (1 << square),
                -beta,
                -alpha,
                false,
            );
            best = best.max(child);
            if best >= beta {
                return best;
            }
            alpha = alpha.max(best);
        }
    }
    if best != -65 {
        return best;
    }
    if !passed {
        return -negamax(opponent, player, -beta, -alpha, true);
    }
    let discs = player.count_ones() as i32 - opponent.count_ones() as i32;
    discs + discs.signum() * empty.count_ones() as i32
}

fn main() {
    let values_only = std::env::args().nth(1).as_deref() == Some("--values");
    let mut values = HashMap::new();
    let mut count = [0usize; 3];
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let mut fields = line.split_whitespace();
        let player: u64 = fields.next().unwrap().parse().unwrap();
        let opponent: u64 = fields.next().unwrap().parse().unwrap();
        let alpha: i32 = fields.next().unwrap().parse().unwrap();
        let beta: i32 = fields.next().unwrap().parse().unwrap();
        let word: i32 = fields.next().unwrap().parse().unwrap();
        assert!(fields.next().is_none() && player & opponent == 0 && alpha < beta);
        let exact = *values
            .entry((player, opponent))
            .or_insert_with(|| 256 * negamax(player, opponent, -65, 65, false));
        if values_only {
            println!("{player} {opponent} {exact}");
            continue;
        }
        let (case, valid) = if word <= alpha {
            (0, exact <= word)
        } else if word >= beta {
            (2, word <= exact)
        } else {
            (1, word == exact)
        };
        assert!(
            valid,
            "board {player:016x}/{opponent:016x}, window ({alpha}, {beta}), returned {word}, minimax {exact}"
        );
        count[case] += 1;
    }
    if values_only {
        return;
    }
    assert!(count.iter().all(|n| *n > 0));
    println!(
        "Independent minimax on {} boards: {} fail-low, {} interior and {} fail-high returns.",
        values.len(),
        count[0],
        count[1],
        count[2]
    );
}
