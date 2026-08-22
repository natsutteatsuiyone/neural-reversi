//! Parsing helpers for the board editor.

use reversi_core::square::Square;

/// Parses a concatenated move string like "f5d6c3" into a list of squares.
///
/// Reads two characters at a time, interpreting each pair as a square in algebraic notation.
pub fn parse_move_string(input: &str) -> Result<Vec<Square>, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("Empty input".to_string());
    }
    Square::parse_sequence(input).map_err(|e| e.to_string())
}

/// Parses a hex string (with optional "0x" prefix) into a u64 bitboard value.
pub fn parse_hex_u64(input: &str) -> Result<u64, String> {
    let input = input.trim();
    if input.is_empty() {
        return Err("Empty input".to_string());
    }
    let hex_str = input
        .strip_prefix("0x")
        .or_else(|| input.strip_prefix("0X"))
        .unwrap_or(input);
    u64::from_str_radix(hex_str, 16).map_err(|e| format!("Invalid hex value: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_trimmed_move_sequence() {
        let moves = parse_move_string(" f5d6c3 ").unwrap();
        let moves: Vec<_> = moves.iter().map(ToString::to_string).collect();
        assert_eq!(moves, ["f5", "d6", "c3"]);
    }

    #[test]
    fn parses_trimmed_prefixed_hex() {
        assert_eq!(parse_hex_u64(" 0X1008000000 ").unwrap(), 0x1008000000);
    }
}
