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
    if !input.len().is_multiple_of(2) {
        return Err("Input length must be even (each move is 2 characters)".to_string());
    }

    let mut moves = Vec::new();
    for i in (0..input.len()).step_by(2) {
        let move_str = &input[i..i + 2];
        match move_str.parse::<Square>() {
            Ok(sq) => moves.push(sq),
            Err(_) => {
                return Err(format!(
                    "Invalid square at position {}: '{move_str}'",
                    (i / 2) + 1
                ));
            }
        }
    }
    Ok(moves)
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
    fn test_parse_move_string_valid() {
        let moves = parse_move_string("f5d6c3").unwrap();
        assert_eq!(moves.len(), 3);
        assert_eq!(moves[0].to_string(), "f5");
        assert_eq!(moves[1].to_string(), "d6");
        assert_eq!(moves[2].to_string(), "c3");
    }

    #[test]
    fn test_parse_move_string_empty() {
        assert!(parse_move_string("").is_err());
    }

    #[test]
    fn test_parse_move_string_odd_length() {
        assert!(parse_move_string("f5d").is_err());
    }

    #[test]
    fn test_parse_move_string_invalid_square() {
        assert!(parse_move_string("f5z9").is_err());
    }

    #[test]
    fn test_parse_hex_u64_with_prefix() {
        assert_eq!(
            parse_hex_u64("0x0000001008000000").unwrap(),
            0x0000001008000000
        );
    }

    #[test]
    fn test_parse_hex_u64_without_prefix() {
        assert_eq!(
            parse_hex_u64("0000001008000000").unwrap(),
            0x0000001008000000
        );
    }

    #[test]
    fn test_parse_hex_u64_uppercase_prefix() {
        assert_eq!(parse_hex_u64("0X1008000000").unwrap(), 0x1008000000);
    }

    #[test]
    fn test_parse_hex_u64_empty() {
        assert!(parse_hex_u64("").is_err());
    }

    #[test]
    fn test_parse_hex_u64_invalid() {
        assert!(parse_hex_u64("0xGGGG").is_err());
    }
}
