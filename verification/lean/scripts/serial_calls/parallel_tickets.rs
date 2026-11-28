use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::level::Level;
use crate::obf::ObfPosition;
use std::path::Path;

pub fn run(root: &Path) {
    let eval = Arc::new(
        Eval::with_weight_files(
            Some(&root.join(EVAL_FILE_NAME)),
            Some(&root.join(EVAL_SM_FILE_NAME)),
        )
        .unwrap(),
    );
    let resources = crate::engine::Engine::from_eval(eval, 8, Some(4));
    let search = resources.open();
    let problems = std::fs::read_to_string(root.join("problem/hard-20.obf")).unwrap();
    let problem = ObfPosition::parse(problems.lines().nth(165).unwrap())
        .unwrap()
        .unwrap();
    let result = resources.search(search, &problem.board,
        &SearchRunOptions::with_level(Level::perfect()).multi_pv(true),
    );
    assert!(result.is_endgame());
    assert_eq!(result.selectivity(), Selectivity::None);
    assert_eq!(result.score().unwrap().value(), 6 * 256);
    assert_eq!(
        result.pv_moves().len(),
        problem.board.legal_moves().count() as usize
    );
    for sq in problem.board.legal_moves().iter() {
        let row = result.pv_moves().iter().find(|row| row.sq == sq).unwrap();
        assert_eq!(row.score.value(), problem.score_of(sq).unwrap() * 256);
    }
}
