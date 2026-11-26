use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::level::Level;
use crate::obf::ObfPosition;
use std::path::Path;

pub fn run(root: &Path) {
    let eval = Eval::with_weight_files(
        Some(&root.join(EVAL_FILE_NAME)),
        Some(&root.join(EVAL_SM_FILE_NAME)),
    )
    .unwrap();
    let resources = SearchSharedResources::from_eval(Arc::new(eval), 1, Some(4));
    assert!(resources.n_threads() > 1, "at least two workers required");
    let mut search = Search::from_shared_resources(&resources);
    let problems = std::fs::read_to_string(root.join("problem/hard-20.obf")).unwrap();
    let obf = ObfPosition::parse(problems.lines().nth(165).unwrap())
        .unwrap()
        .unwrap();
    let opts = SearchRunOptions::with_level(Level::perfect()).multi_pv(true);
    for session in 0..2 {
        search.tt().warm_dump(2 * session);
        let res = search.run(&obf.board, &opts);
        search.tt().warm_dump(2 * session + 1);
        assert!(res.is_endgame());
        assert_eq!(res.selectivity(), Selectivity::None);
        assert_eq!(res.score().unwrap().value(), 6 * 256);
        let rows = res.pv_moves();
        assert_eq!(rows.len(), obf.board.legal_moves().count() as usize);
        for sq in obf.board.legal_moves().iter() {
            let row = rows.iter().find(|row| row.sq == sq).unwrap();
            assert_eq!(row.score.value(), obf.score_of(sq).unwrap() * 256);
        }
        println!(
            "result {} {} {}",
            session,
            res.score().unwrap().value(),
            rows.len()
        );
    }
}
