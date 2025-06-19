use std::sync::Arc;

use crate::board::Board;
use crate::constants::MAX_PLY;
use crate::empty_list::EmptyList;
use crate::eval::pattern_feature::{PatternFeature, PatternFeatures};
use crate::eval::Eval;
use crate::move_list::{Move, MoveList};
use crate::probcut::{self};
use crate::search::root_move::RootMove;
use crate::search::threading::{SplitPoint, ThreadPool};
use crate::search::SearchProgressCallback;
use crate::search::{SearchProgress, SCORE_INF};
use crate::square::Square;
use crate::transposition_table::TranspositionTable;
use crate::types::{Depth, Score, Scoref};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GamePhase {
    MidGame,
    EndGame,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SideToMove {
    Player = 0,
    Opponent = 1,
}

impl SideToMove {
    pub fn switch(self) -> Self {
        match self {
            SideToMove::Player => SideToMove::Opponent,
            SideToMove::Opponent => SideToMove::Player,
        }
    }

    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy)]
pub struct StackRecord {
    pub pv: [Square; MAX_PLY],
}

pub struct SearchContext {
    pub n_nodes: u64,
    pub side_to_move: SideToMove,
    pub generation: u8,
    pub selectivity: u8,
    pub empty_list: EmptyList,
    pub tt: Arc<TranspositionTable>,
    pub root_moves: Arc<std::sync::Mutex<Vec<RootMove>>>,
    pub pool: Arc<ThreadPool>,
    pub eval: Arc<Eval>,
    pub pattern_features: PatternFeatures,
    pub callback: Option<Arc<SearchProgressCallback>>,
    stack: [StackRecord; MAX_PLY],
    pub game_phase: GamePhase,
}

impl SearchContext {
    pub fn new(
        board: &Board,
        generation: u8,
        selectivity: u8,
        tt: Arc<TranspositionTable>,
        pool: Arc<ThreadPool>,
        eval: Arc<Eval>,
    ) -> SearchContext {
        let empty_list = EmptyList::new(board);
        let ply = empty_list.ply();
        SearchContext {
            n_nodes: 0,
            side_to_move: SideToMove::Player,
            generation,
            selectivity,
            empty_list,
            tt,
            root_moves: Arc::new(std::sync::Mutex::new(Self::create_root_moves(board))),
            pool,
            eval,
            pattern_features: PatternFeatures::new(board, ply),
            callback: None,
            stack: [StackRecord {
                pv: [Square::None; MAX_PLY],
            }; MAX_PLY],
            game_phase: GamePhase::MidGame,
        }
    }

    #[inline]
    pub fn from_split_point(sp: &Arc<SplitPoint>) -> SearchContext {
        let state = sp.state();
        let task = state.task.as_ref().unwrap();
        let empty_list = task.empty_list.clone();
        let ply = empty_list.ply();
        let pattern_features = if task.side_to_move == SideToMove::Player {
            PatternFeatures::new(&task.board, ply)
        } else {
            PatternFeatures::new(&task.board.switch_players(), ply)
        };
        SearchContext {
            n_nodes: 0,
            side_to_move: task.side_to_move,
            empty_list,
            generation: task.generation,
            selectivity: task.selectivity,
            tt: task.tt.clone(),
            root_moves: task.root_moves.clone(),
            pool: task.pool.clone(),
            eval: task.eval.clone(),
            pattern_features,
            callback: None,
            stack: [StackRecord {
                pv: [Square::None; MAX_PLY],
            }; MAX_PLY],
            game_phase: task.game_phase,
        }
    }

    #[inline]
    fn switch_players(&mut self) {
        self.side_to_move = self.side_to_move.switch();
    }

    #[inline]
    pub fn update(&mut self, mv: &Move) {
        self.increment_nodes();
        self.pattern_features.update(mv, self.ply(), self.side_to_move);
        self.switch_players();
        self.empty_list.remove(mv.sq);
    }

    #[inline]
    pub fn undo(&mut self, mv: &Move) {
        self.empty_list.restore(mv.sq);
        self.switch_players();
    }

    #[inline]
    pub fn update_endgame(&mut self, sq: Square) {
        self.increment_nodes();
        self.empty_list.remove(sq);
    }

    #[inline]
    pub fn undo_endgame(&mut self, sq: Square) {
        self.empty_list.restore(sq);
    }

    #[inline]
    pub fn update_pass(&mut self) {
        self.increment_nodes();
        self.switch_players();
    }

    #[inline]
    pub fn undo_pass(&mut self) {
        self.switch_players();
    }

    #[inline]
    pub fn update_probcut(&mut self) {
        self.selectivity = probcut::NO_SELECTIVITY;
    }

    #[inline]
    pub fn undo_probcut(&mut self, selectivity: u8) {
        self.selectivity = selectivity;
    }

    #[inline]
    pub fn ply(&self) -> usize {
        self.empty_list.ply()
    }

    #[inline]
    pub fn increment_nodes(&mut self) {
        self.n_nodes += 1;
    }

    #[inline]
    pub fn get_pattern_feature(&self) -> &PatternFeature {
        let ply = self.ply();
        if self.side_to_move == SideToMove::Player {
            &self.pattern_features.p_features[ply]
        } else {
            &self.pattern_features.o_features[ply]
        }
    }

    pub fn update_root_move(&mut self, sq: Square, score: Score, move_count: usize, alpha: Score) {
        let is_pv = move_count == 1 || score > alpha;
        if is_pv {
            self.update_pv(sq);
        }

        let mut root_moves = self.root_moves.lock().unwrap();
        let rm = root_moves.iter_mut().find(|rm| rm.sq == sq).unwrap();
        rm.average_score = if rm.average_score == -SCORE_INF {
            score
        } else {
            (rm.average_score + score) / 2
        };

        if is_pv {
            let ply = self.ply();
            rm.score = score;
            rm.pv.clear();
            for sq in self.stack[ply].pv.iter() {
                if *sq == Square::None {
                    break;
                }
                rm.pv.push(*sq);
            }
        } else {
            rm.score = -(SCORE_INF << 6);
        }
    }

    pub fn get_best_root_move(&self, skip_seached_move: bool) -> Option<RootMove> {
        let root_moves = self
            .root_moves
            .lock()
            .expect("Failed to acquire lock on root_moves");
        if skip_seached_move {
            root_moves
                .iter()
                .filter(|rm| !rm.searched)
                .max_by_key(|rm| rm.score)
                .cloned()
        } else {
            root_moves.iter().max_by_key(|rm| rm.score).cloned()
        }
    }

    pub fn mark_root_move_searched(&mut self, sq: Square) {
        let mut root_moves = self
            .root_moves
            .lock()
            .unwrap();
        if let Some(rm) = root_moves.iter_mut().find(|rm| rm.sq == sq) {
            rm.searched = true;
        }
    }

    pub fn reset_root_move_searched(&mut self) {
        let mut root_moves = self
            .root_moves
            .lock()
            .unwrap();
        for rm in root_moves.iter_mut() {
            rm.searched = false;
        }
    }

    fn create_root_moves(board: &Board) -> Vec<RootMove> {
        let move_list = MoveList::new(board);
        let mut root_moves = Vec::<RootMove>::with_capacity(move_list.count());
        for m in move_list.iter() {
            root_moves.push(RootMove {
                sq: m.sq,
                score: -SCORE_INF,
                average_score: -SCORE_INF,
                pv: Vec::new(),
                searched: false,
            });
        }
        root_moves
    }

    pub fn is_move_searched(&self, sq: Square) -> bool {
        let root_moves = self
            .root_moves
            .lock()
            .unwrap();
        root_moves.iter().any(|rm| rm.sq == sq && rm.searched)
    }

    pub fn update_pv(&mut self, sq: Square) {
        let ply = self.ply();
        self.stack[ply].pv[0] = sq;
        if ply == 0 {
            return;
        }
        let mut idx = 0;
        while idx < self.stack[ply + 1].pv.len() && self.stack[ply + 1].pv[idx] != Square::None {
            self.stack[ply].pv[idx + 1] = self.stack[ply + 1].pv[idx];
            idx += 1;
        }
        self.stack[ply].pv[idx + 1] = Square::None;
    }

    pub fn clear_pv(&mut self) {
        self.stack[self.ply()].pv.fill(Square::None);
    }

    pub fn set_callback(&mut self, callback: Arc<SearchProgressCallback>) {
        self.callback = Some(callback);
    }

    pub fn notify_progress(&self, depth: Depth, score: Scoref, best_move: Square, selectivity: u8) {
        if let Some(ref callback) = self.callback {
            callback(SearchProgress {
                depth,
                score,
                best_move,
                probability: probcut::get_probability(selectivity),
            });
        }
    }

    pub fn is_search_aborted(&self) -> bool {
        self.pool.is_aborted()
    }

    pub fn root_moves_count(&self) -> usize {
        self.root_moves.lock().unwrap().len()
    }
}
