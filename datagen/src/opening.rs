use std::{
    collections::HashSet,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use reversi_core::{board::Board, move_list::MoveList, square::Square, types::Depth};

/// Generates all possible Reversi opening sequences up to the specified depth.
///
/// This function initiates the generation process starting with a standard opening move (F5).
/// It outputs all unique sequences to stdout, with each line representing one opening sequence.
///
/// # Parameters
///
/// * `max_depth` - The maximum number of moves to explore in the opening sequences
pub fn generate(max_depth: Depth) {
    let mut path: Vec<Square> = Vec::with_capacity(60);
    let mut board = Board::new();
    let mut visited_boards = HashSet::new();

    let stdout = io::stdout();
    let locked_stdout = stdout.lock();
    let mut writer = BufWriter::new(locked_stdout);

    let first_move_sq = Square::F5;
    board = board.make_move(first_move_sq);
    path.push(first_move_sq);
    generate_openings(
        &board,
        max_depth - 1,
        &mut path,
        &mut visited_boards,
        &mut writer,
    );

    writer.flush().unwrap();
}

/// Recursively generates all possible opening sequences from the given board position.
///
/// # Parameters
///
/// * `board` - The current board state to explore from
/// * `depth` - Remaining depth to explore
/// * `path` - Sequence of moves made so far in the current path
/// * `visited_boards` - Set of board positions already visited to avoid duplicates
/// * `writer` - Buffer for writing output sequences
fn generate_openings(
    board: &Board,
    depth: Depth,
    path: &mut Vec<Square>,
    visited_boards: &mut HashSet<Board>,
    writer: &mut BufWriter<io::StdoutLock>,
) {
    if !visited_boards.insert(*board) {
        return;
    }

    if depth == 0 {
        for sq in path.iter() {
            write!(writer, "{sq}").unwrap();
        }
        writeln!(writer).unwrap();
        return;
    }

    let move_list = MoveList::new(board);
    if move_list.count() > 0 {
        for m in move_list.iter() {
            let next = board.make_move_with_flipped(m.flipped, m.sq);
            path.push(m.sq);
            generate_openings(&next, depth - 1, path, visited_boards, writer);
            path.pop();
        }
    } else {
        let next = board.switch_players();
        if next.has_legal_moves() {
            generate_openings(&next, depth, path, visited_boards, writer);
        }
    }
}

/// Loads opening sequences from a file.
///
/// Each line in the file should represent one opening sequence,
/// with squares specified in algebraic notation (e.g., "F5D6C3").
///
/// # Parameters
///
/// * `path` - Path to the file containing opening sequences
///
/// # Returns
///
/// A vector of sequences, where each sequence is a vector of squares
pub fn load_openings(path: &str) -> io::Result<Vec<Vec<Square>>> {
    let file = File::open(Path::new(path))?;
    let reader = BufReader::new(file);
    let mut openings = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let mut sequence = Vec::new();
        let mut i = 0;

        while i + 1 < line.len() {
            // Parse two characters at a time into a square
            let square_str = &line[i..i + 2];
            match square_str.parse::<Square>() {
                Ok(sq) => sequence.push(sq),
                Err(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Invalid square notation: {square_str}"),
                    ));
                }
            }
            i += 2;
        }

        openings.push(sequence);
    }

    Ok(openings)
}
