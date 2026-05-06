//! Search result types.

use crate::{
    probcut::Selectivity,
    search::root_move::{RootMove, RootMoves},
    search::search_counters::SearchCounters,
    square::Square,
    types::{Depth, ScaledScore, Scoref},
};

/// Represents a single move with its evaluation score for Multi-PV results.
#[derive(Clone, Debug)]
pub struct PvMove {
    pub sq: Square,
    pub score: Scoref,
    pub pv_line: Vec<Square>,
}

/// Result of a search operation.
pub enum SearchResult {
    /// Search completed with a playable move.
    BestMove {
        sq: Square,
        score: Scoref,
        n_nodes: u64,
        pv_line: Vec<Square>,
        depth: Depth,
        selectivity: Selectivity,
        is_endgame: bool,
        /// All evaluated moves with scores (populated in Multi-PV mode).
        pv_moves: Vec<PvMove>,
        /// Diagnostic counters accumulated during search.
        counters: SearchCounters,
    },
    /// No legal root move is available.
    NoLegalMove,
}

impl SearchResult {
    /// Creates a result for a random move in the opening position.
    pub fn new_random_move(mv: Square) -> Self {
        Self::BestMove {
            sq: mv,
            score: 0.0,
            n_nodes: 0,
            pv_line: vec![],
            depth: 0,
            selectivity: Selectivity::None,
            is_endgame: false,
            pv_moves: vec![],
            counters: SearchCounters::default(),
        }
    }

    /// Creates a result when no legal moves are available.
    pub fn new_no_moves(_is_endgame: bool) -> Self {
        Self::NoLegalMove
    }

    /// Creates a search result from the root move state.
    pub fn from_root_move(
        root_moves: &RootMoves,
        best_move: &RootMove,
        depth: Depth,
        selectivity: Selectivity,
        is_endgame: bool,
        counters: SearchCounters,
    ) -> Self {
        let pv_moves: Vec<PvMove> = root_moves.map(|rm| PvMove {
            sq: rm.sq,
            score: rm.score.to_disc_diff_f32(),
            pv_line: rm.pv.clone(),
        });

        Self::BestMove {
            sq: best_move.sq,
            score: best_move.score.to_disc_diff_f32(),
            n_nodes: counters.n_nodes,
            pv_line: best_move.pv.clone(),
            depth,
            selectivity,
            is_endgame,
            pv_moves,
            counters,
        }
    }

    /// Returns the best move square, if the search produced one.
    #[inline]
    pub fn best_move(&self) -> Option<Square> {
        match self {
            SearchResult::BestMove { sq, .. } => Some(*sq),
            SearchResult::NoLegalMove => None,
        }
    }

    /// Returns the best move score, if the search produced one.
    #[inline]
    pub fn score(&self) -> Option<Scoref> {
        match self {
            SearchResult::BestMove { score, .. } => Some(*score),
            SearchResult::NoLegalMove => None,
        }
    }

    /// Returns the searched node count.
    #[inline]
    pub fn n_nodes(&self) -> u64 {
        match self {
            SearchResult::BestMove { n_nodes, .. } => *n_nodes,
            SearchResult::NoLegalMove => 0,
        }
    }

    /// Returns the principal variation line.
    #[inline]
    pub fn pv_line(&self) -> &[Square] {
        match self {
            SearchResult::BestMove { pv_line, .. } => pv_line,
            SearchResult::NoLegalMove => &[],
        }
    }

    /// Returns the completed search depth.
    #[inline]
    pub fn depth(&self) -> Depth {
        match self {
            SearchResult::BestMove { depth, .. } => *depth,
            SearchResult::NoLegalMove => 0,
        }
    }

    /// Returns the selectivity used by the result.
    #[inline]
    pub fn selectivity(&self) -> Selectivity {
        match self {
            SearchResult::BestMove { selectivity, .. } => *selectivity,
            SearchResult::NoLegalMove => Selectivity::None,
        }
    }

    /// Returns whether this result came from endgame search.
    #[inline]
    pub fn is_endgame(&self) -> bool {
        match self {
            SearchResult::BestMove { is_endgame, .. } => *is_endgame,
            SearchResult::NoLegalMove => false,
        }
    }

    /// Returns all evaluated Multi-PV moves.
    #[inline]
    pub fn pv_moves(&self) -> &[PvMove] {
        match self {
            SearchResult::BestMove { pv_moves, .. } => pv_moves,
            SearchResult::NoLegalMove => &[],
        }
    }

    /// Returns diagnostic counters accumulated during search.
    #[inline]
    pub fn counters(&self) -> SearchCounters {
        match self {
            SearchResult::BestMove { counters, .. } => counters.clone(),
            SearchResult::NoLegalMove => SearchCounters::default(),
        }
    }

    /// Returns the probability percentage based on selectivity.
    #[inline]
    pub fn get_probability(&self) -> i32 {
        self.selectivity().probability()
    }

    /// Returns `true` if the search was aborted before completing any iteration.
    ///
    /// Before the first iteration finishes, the score still carries the
    /// `-ScaledScore::INF` sentinel propagated from the initial root move;
    /// any completed iteration overwrites it with a real evaluation.
    #[inline]
    pub fn is_invalid_sentinel(&self) -> bool {
        matches!(
            self,
            SearchResult::BestMove { score, .. } if *score == (-ScaledScore::INF).to_disc_diff_f32()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn random_move_result_is_best_move() {
        let result = SearchResult::new_random_move(Square::D3);

        assert_eq!(result.best_move(), Some(Square::D3));
        assert_eq!(result.score(), Some(0.0));
        assert_eq!(result.n_nodes(), 0);
        assert!(result.pv_line().is_empty());
        assert_eq!(result.depth(), 0);
        assert_eq!(result.selectivity(), Selectivity::None);
        assert!(!result.is_endgame());
        assert!(result.pv_moves().is_empty());
        assert_eq!(result.counters().n_nodes, 0);
        assert!(!result.is_invalid_sentinel());
    }

    #[test]
    fn no_legal_move_has_no_best_move_data() {
        let result = SearchResult::new_no_moves(true);

        assert_eq!(result.best_move(), None);
        assert_eq!(result.score(), None);
        assert_eq!(result.n_nodes(), 0);
        assert!(result.pv_line().is_empty());
        assert_eq!(result.depth(), 0);
        assert_eq!(result.selectivity(), Selectivity::None);
        assert!(!result.is_endgame());
        assert!(result.pv_moves().is_empty());
        assert_eq!(result.counters().n_nodes, 0);
        assert!(!result.is_invalid_sentinel());
    }

    #[test]
    fn root_move_result_preserves_search_data() {
        let root_moves = RootMoves::new(&Board::new());
        let best_move = RootMove {
            sq: Square::D3,
            score: ScaledScore::from_disc_diff(4),
            previous_score: -ScaledScore::INF,
            average_score: -ScaledScore::INF,
            pv: vec![Square::D3, Square::C3],
        };
        #[cfg(feature = "search-stats")]
        let counters = SearchCounters {
            n_nodes: 42,
            ..Default::default()
        };
        #[cfg(not(feature = "search-stats"))]
        let counters = SearchCounters { n_nodes: 42 };

        let result = SearchResult::from_root_move(
            &root_moves,
            &best_move,
            7,
            Selectivity::Level1,
            false,
            counters,
        );

        assert_eq!(result.best_move(), Some(Square::D3));
        assert_eq!(result.score(), Some(4.0));
        assert_eq!(result.n_nodes(), 42);
        assert_eq!(result.pv_line(), &[Square::D3, Square::C3]);
        assert_eq!(result.depth(), 7);
        assert_eq!(result.selectivity(), Selectivity::Level1);
        assert!(!result.is_endgame());
        assert_eq!(result.pv_moves().len(), 4);
        assert_eq!(result.counters().n_nodes, 42);
        assert!(!result.is_invalid_sentinel());
    }
}
