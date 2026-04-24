//! OBF (Othello Board Format) line parser.
//!
//! Parses lines of the form
//! `<board64> <side>; <move1>:<score1>; <move2>:<score2>; ...`
//! into a neutral [`ObfPosition`].
//!
//! The parser is intentionally lenient so the same routine can serve both
//! the evaluation test suite (full move:score data) and the endgame solver
//! CLI (board + side only). Specifically it accepts:
//!
//! - both `<board> <side>` (whitespace-separated, standard OBF) and
//!   `<board><side>` (compact, no separator);
//! - trailing `%` comments, which are stripped before parsing;
//! - lines with zero or more `;`-separated `move:score` segments;
//! - sentinel `--:<score>` segments, which are filtered out.
//!
//! Blank lines and lines whose only content is a `%` comment yield
//! `Ok(None)`.

use std::str::FromStr;

use crate::board::Board;
use crate::disc::Disc;
use crate::square::Square;

/// A parsed OBF line.
#[derive(Debug, Clone)]
pub struct ObfPosition {
    /// Board parsed from the 64-character field.
    pub board: Board,
    /// Side to move (`X` → Black, `O` → White).
    pub side_to_move: Disc,
    /// Move-score pairs in file order. OBF convention is descending score,
    /// and [`rank_of`](Self::rank_of) / [`best_moves`](Self::best_moves)
    /// rely on that ordering. Empty when no move segments were present.
    /// Sentinel entries (`--:<score>`) are not included.
    move_scores: Vec<(Square, i32)>,
    /// Score from a `PS:<score>` segment for pass positions.
    /// `None` when no such segment was present.
    pass_score: Option<i32>,
}

impl ObfPosition {
    /// Parse a single OBF line. Returns `Ok(None)` for blank/comment-only input.
    pub fn parse(line: &str) -> Result<Option<Self>, String> {
        let stripped = line.split('%').next().unwrap_or("").trim();
        if stripped.is_empty() {
            return Ok(None);
        }

        let mut segments = stripped.split(';');
        let header = segments
            .next()
            .ok_or_else(|| "Missing board header".to_string())?
            .trim();
        let (board, side_to_move) = parse_board_header(header)?;

        let mut move_scores = Vec::new();
        let mut pass_score = None;

        for segment in segments {
            let segment = segment.trim();
            if segment.is_empty() {
                continue;
            }
            let (key, value) = segment
                .split_once(':')
                .ok_or_else(|| format!("Invalid move:score format: '{segment}'"))?;
            let key = key.trim();

            if key.eq_ignore_ascii_case("PS") {
                pass_score = Some(parse_score(value)?);
                continue;
            }

            if key == "--" {
                continue;
            }

            let square = Square::from_str(key).map_err(|e| format!("Invalid move '{key}': {e}"))?;
            let score = parse_score(value)?;
            move_scores.push((square, score));
        }

        debug_assert!(
            move_scores.windows(2).all(|w| w[0].1 >= w[1].1),
            "OBF move_scores must be in non-ascending score order"
        );

        Ok(Some(Self {
            board,
            side_to_move,
            move_scores,
            pass_score,
        }))
    }

    /// True when no scored moves are listed (typically a `PS:`-only line).
    pub fn is_pass(&self) -> bool {
        self.move_scores.is_empty()
    }

    /// Expected outcome score: the best-listed move's score when moves are
    /// present, otherwise the `PS:` value. `None` when neither is available.
    pub fn expected_score(&self) -> Option<i32> {
        self.move_scores
            .first()
            .map(|(_, s)| *s)
            .or(self.pass_score)
    }

    /// Score of the given move, or `None` if it is not listed.
    pub fn score_of(&self, sq: Square) -> Option<i32> {
        self.move_scores
            .iter()
            .find(|(s, _)| *s == sq)
            .map(|(_, score)| *score)
    }

    /// Returns 0/1/2 for best/second-best/third-best, or `None` for
    /// lower-ranked or unlisted moves. Ties share a rank (descending order).
    pub fn rank_of(&self, sq: Square) -> Option<usize> {
        self.move_scores
            .chunk_by(|a, b| a.1 == b.1)
            .take(3)
            .enumerate()
            .find_map(|(rank, chunk)| chunk.iter().any(|(s, _)| *s == sq).then_some(rank))
    }

    /// Iterator over the best-scoring moves (those tied with the highest score).
    pub fn best_moves(&self) -> impl Iterator<Item = Square> + '_ {
        let best_score = self.move_scores.first().map(|(_, s)| *s);
        self.move_scores
            .iter()
            .take_while(move |(_, s)| best_score == Some(*s))
            .map(|(sq, _)| *sq)
    }
}

fn parse_board_header(header: &str) -> Result<(Board, Disc), String> {
    if !header.is_ascii() {
        return Err(format!(
            "Board header contains non-ASCII characters: '{header}'"
        ));
    }
    if header.len() < 65 {
        return Err(format!(
            "Board header too short (need 64 board chars + side): '{header}'"
        ));
    }

    let board_str = &header[..64];
    let side_char = header[64..]
        .trim_start()
        .chars()
        .next()
        .ok_or_else(|| "Missing side-to-move marker".to_string())?;

    let side_to_move = match side_char {
        'X' => Disc::Black,
        'O' => Disc::White,
        other => return Err(format!("Invalid side to move: '{other}'")),
    };

    let board = Board::from_string(board_str, side_to_move)
        .map_err(|e| format!("Invalid board string: {e}"))?;

    Ok((board, side_to_move))
}

fn parse_score(s: &str) -> Result<i32, String> {
    s.trim()
        .trim_start_matches('+')
        .parse()
        .map_err(|e| format!("Invalid score '{s}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const INITIAL_BOARD: &str = "---------------------------OX------XO---------------------------";

    /// Helper: parse a line known to produce a valid position.
    fn parse(line: &str) -> ObfPosition {
        ObfPosition::parse(line).unwrap().unwrap()
    }

    #[test]
    fn blank_line_returns_none() {
        assert!(ObfPosition::parse("").unwrap().is_none());
        assert!(ObfPosition::parse("   ").unwrap().is_none());
        assert!(ObfPosition::parse("\t").unwrap().is_none());
    }

    #[test]
    fn comment_only_line_returns_none() {
        assert!(ObfPosition::parse("% just a comment").unwrap().is_none());
        assert!(
            ObfPosition::parse("   % indented comment")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn parses_standard_obf_with_moves() {
        let pos = parse(&format!("{INITIAL_BOARD} X; e6:+10; d3:+8; --:-127"));
        assert_eq!(pos.side_to_move, Disc::Black);
        assert_eq!(pos.move_scores.len(), 2);
        assert_eq!(pos.move_scores[0], (Square::E6, 10));
        assert_eq!(pos.move_scores[1], (Square::D3, 8));
        assert!(pos.pass_score.is_none());
    }

    #[test]
    fn parses_compact_header_without_separator() {
        let pos = parse(&format!("{INITIAL_BOARD}X"));
        assert_eq!(pos.side_to_move, Disc::Black);
        assert!(pos.move_scores.is_empty());
        assert!(pos.pass_score.is_none());
    }

    #[test]
    fn parses_white_to_move() {
        let pos = parse(&format!("{INITIAL_BOARD} O"));
        assert_eq!(pos.side_to_move, Disc::White);
    }

    #[test]
    fn parses_pass_position() {
        let pos = parse(&format!("{INITIAL_BOARD} X; PS:-4"));
        assert!(pos.move_scores.is_empty());
        assert_eq!(pos.pass_score, Some(-4));
    }

    #[test]
    fn strips_trailing_comment() {
        let pos = parse(&format!("{INITIAL_BOARD} X; e6:+10 % trailing comment"));
        assert_eq!(pos.move_scores, vec![(Square::E6, 10)]);
    }

    #[test]
    fn rejects_invalid_side_marker() {
        let line = format!("{INITIAL_BOARD} Z");
        let err = ObfPosition::parse(&line).unwrap_err();
        assert!(err.contains("Invalid side to move"), "{err}");
    }

    #[test]
    fn rejects_short_header() {
        let err = ObfPosition::parse("XO").unwrap_err();
        assert!(err.contains("too short"), "{err}");
    }

    #[test]
    fn rejects_malformed_segment() {
        let line = format!("{INITIAL_BOARD} X; not_a_pair");
        let err = ObfPosition::parse(&line).unwrap_err();
        assert!(err.contains("move:score"), "{err}");
    }

    #[test]
    fn rejects_invalid_move_name() {
        let line = format!("{INITIAL_BOARD} X; z9:+10");
        let err = ObfPosition::parse(&line).unwrap_err();
        assert!(err.contains("Invalid move"), "{err}");
    }

    #[test]
    fn rejects_invalid_score() {
        let line = format!("{INITIAL_BOARD} X; e6:not_a_number");
        let err = ObfPosition::parse(&line).unwrap_err();
        assert!(err.contains("Invalid score"), "{err}");
    }

    #[test]
    fn rejects_non_ascii_header() {
        let line = format!("{}_X", "あ".repeat(64));
        let err = ObfPosition::parse(&line).unwrap_err();
        assert!(err.contains("non-ASCII"), "{err}");
    }

    #[test]
    fn allows_extra_whitespace_around_segments() {
        let pos = parse(&format!("{INITIAL_BOARD} X ;   e6 : +10  ;  d3:+8  "));
        assert_eq!(pos.move_scores, vec![(Square::E6, 10), (Square::D3, 8)]);
    }

    #[test]
    fn is_pass_and_expected_score() {
        let pos = parse(&format!("{INITIAL_BOARD} X; e6:+10; d3:+8"));
        assert!(!pos.is_pass());
        assert_eq!(pos.expected_score(), Some(10));

        let pass = parse(&format!("{INITIAL_BOARD} X; PS:-4"));
        assert!(pass.is_pass());
        assert_eq!(pass.expected_score(), Some(-4));

        let bare = parse(&format!("{INITIAL_BOARD}X"));
        assert!(bare.is_pass());
        assert_eq!(bare.expected_score(), None);
    }

    #[test]
    fn score_of_lookup() {
        let pos = parse(&format!("{INITIAL_BOARD} X; e6:+10; d3:+8"));
        assert_eq!(pos.score_of(Square::E6), Some(10));
        assert_eq!(pos.score_of(Square::D3), Some(8));
        assert_eq!(pos.score_of(Square::F5), None);
    }

    #[test]
    fn rank_of_groups_ties() {
        let pos = parse(&format!(
            "{INITIAL_BOARD} X; e6:+10; d3:+10; c4:+8; f5:+6; b3:+4"
        ));
        assert_eq!(pos.rank_of(Square::E6), Some(0));
        assert_eq!(pos.rank_of(Square::D3), Some(0));
        assert_eq!(pos.rank_of(Square::C4), Some(1));
        assert_eq!(pos.rank_of(Square::F5), Some(2));
        assert_eq!(pos.rank_of(Square::B3), None);
        assert_eq!(pos.rank_of(Square::A1), None);
    }

    #[test]
    fn best_moves_includes_ties() {
        let pos = parse(&format!("{INITIAL_BOARD} X; e6:+10; d3:+10; c4:+8"));
        let bests: Vec<Square> = pos.best_moves().collect();
        assert_eq!(bests, vec![Square::E6, Square::D3]);

        let pass = parse(&format!("{INITIAL_BOARD} X; PS:-4"));
        assert_eq!(pass.best_moves().count(), 0);
    }
}
