use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::Context;
use reversi_core::{
    board::Board, game_state::GameState, move_list::MoveList, square::Square, types::Depth,
};

pub(crate) fn generate(max_depth: Depth) -> anyhow::Result<()> {
    let mut path: Vec<Square> = Vec::with_capacity(60);
    let mut board = Board::new();
    let mut visited_boards = HashSet::new();

    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    let first_move_sq = Square::F5;
    board = board.make_move(first_move_sq);
    path.push(first_move_sq);
    generate_openings(
        &board,
        max_depth - 1,
        &mut path,
        &mut visited_boards,
        &mut writer,
    )
    .context("failed to write opening sequence to stdout")?;

    writer.flush().context("failed to flush stdout")
}

fn generate_openings(
    board: &Board,
    depth: Depth,
    path: &mut Vec<Square>,
    visited_boards: &mut HashSet<Board>,
    writer: &mut impl Write,
) -> io::Result<()> {
    if !visited_boards.insert(*board) {
        return Ok(());
    }

    if depth == 0 {
        for sq in path.iter() {
            write!(writer, "{sq}")?;
        }
        writeln!(writer)?;
        return Ok(());
    }

    let move_list = MoveList::new(board);
    if move_list.count() > 0 {
        for m in move_list.iter() {
            let next = board.make_move_with_flipped(m.flipped, m.sq);
            path.push(m.sq);
            generate_openings(&next, depth - 1, path, visited_boards, writer)?;
            path.pop();
        }
    } else {
        let next = board.switch_players();
        if next.has_legal_moves() {
            generate_openings(&next, depth, path, visited_boards, writer)?;
        }
    }
    Ok(())
}

pub(crate) fn load_openings(path: &Path) -> anyhow::Result<Vec<Vec<Square>>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open openings file '{}'", path.display()))?;
    let reader = BufReader::new(file);
    let mut openings = Vec::new();

    for (line_index, line) in reader.lines().enumerate() {
        let line_no = line_index + 1;
        let line =
            line.with_context(|| format!("failed to read line {line_no} of '{}'", path.display()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let sequence =
            Square::parse_sequence(line).map_err(|e| anyhow::anyhow!("Line {line_no}: {e}"))?;
        GameState::from_moves(&sequence).map_err(|e| anyhow::anyhow!("Line {line_no}: {e}"))?;
        openings.push(sequence);
    }

    Ok(openings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        fn new(name: &str) -> Self {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "neural-reversi-opening-{name}-{}-{id}.txt",
                std::process::id()
            ));
            Self { path }
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    #[test]
    fn load_openings_reports_one_based_line_number_on_illegal_move() -> anyhow::Result<()> {
        let file = TempFile::new("illegal");
        fs::write(&file.path, "\nf5d6\nf5f5\n")?;

        let err = load_openings(&file.path).unwrap_err();
        assert!(err.to_string().contains("Line 3:"), "{err}");
        Ok(())
    }
}
