//! Binary record format for training data I/O.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::square::Square;
use reversi_core::types::Scoref;
use std::fs;
use std::fs::OpenOptions;
use std::io::{self, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// Size of each record in bytes
pub const RECORD_SIZE: u64 = 27;

/// Byte offsets of individual fields inside a serialized `GameRecord`.
/// Must stay in sync with the write order in `write_records`.
pub const SCORE_OFFSET: usize = 16;
pub const GAME_SCORE_OFFSET: usize = 20;
pub const PLY_OFFSET: usize = 21;
pub const IS_RANDOM_OFFSET: usize = 22;

/// Sentinel value for `game_score` when the true game outcome is unavailable
/// (e.g. positions produced by `score-openings` rather than a full self-play game).
pub const GAME_SCORE_UNAVAILABLE: i8 = i8::MIN;

/// Represents a single position record from a self-play game.
#[derive(Clone)]
pub struct GameRecord {
    pub game_id: u16,
    pub ply: u8,
    pub board: Board,
    pub score: Scoref,
    pub game_score: i8,
    pub side_to_move: Disc,
    pub is_random: bool,
    pub sq: Square,
}

/// Writes game records to a binary file (append mode).
pub fn write_records_to_file(path: &Path, records: &[GameRecord]) -> io::Result<()> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    write_records(&mut writer, records)?;
    writer.flush()
}

/// Writes game records to the given writer.
pub fn write_records(writer: &mut impl Write, records: &[GameRecord]) -> io::Result<()> {
    for record in records {
        writer.write_u64::<LittleEndian>(record.board.player.bits())?;
        writer.write_u64::<LittleEndian>(record.board.opponent.bits())?;
        writer.write_f32::<LittleEndian>(record.score)?;
        writer.write_i8(record.game_score)?;
        writer.write_u8(record.ply)?;
        writer.write_u8(if record.is_random { 1 } else { 0 })?;
        writer.write_u8(record.sq as u8)?;
        writer.write_u8(if record.side_to_move == Disc::Black {
            0
        } else {
            1
        })?;
        writer.write_u16::<LittleEndian>(record.game_id)?;
    }
    Ok(())
}

/// Counts the number of complete records in a binary file.
pub fn count_records_in_file(path: &Path) -> io::Result<u32> {
    match fs::metadata(path) {
        Ok(metadata) => Ok((metadata.len() / RECORD_SIZE) as u32),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e),
    }
}

/// Truncates any trailing incomplete record from a binary file.
///
/// If the file size is not a multiple of `RECORD_SIZE`, the trailing
/// bytes are removed so that only complete records remain.
pub fn truncate_incomplete_record(path: &Path) -> io::Result<()> {
    let file = match OpenOptions::new().write(true).open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    let file_size = file.metadata()?.len();
    let remainder = file_size % RECORD_SIZE;
    if remainder != 0 {
        let aligned = file_size - remainder;
        file.set_len(aligned)?;
        eprintln!(
            "Warning: Truncated {} trailing bytes from {} (file size {} was not aligned to record size {})",
            remainder,
            path.display(),
            file_size,
            RECORD_SIZE,
        );
    }
    Ok(())
}

/// Reads the `game_id` of the last complete record in a binary file.
pub fn read_last_game_id(path: &Path) -> io::Result<Option<u16>> {
    let mut file = fs::File::open(path)?;
    let file_size = file.metadata()?.len();
    let aligned = file_size - file_size % RECORD_SIZE;
    if aligned < RECORD_SIZE {
        return Ok(None);
    }
    // game_id is the last 2 bytes of each complete record
    file.seek(SeekFrom::Start(aligned - 2))?;
    Ok(Some(file.read_u16::<LittleEndian>()?))
}

/// Reads all game records from a binary file.
pub fn read_records_from_file(path: &Path) -> io::Result<Vec<GameRecord>> {
    let metadata = fs::metadata(path)?;
    let file_size = metadata.len();

    if file_size % RECORD_SIZE != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "File size {} is not a multiple of RECORD_SIZE {} for file {}",
                file_size,
                RECORD_SIZE,
                path.display()
            ),
        ));
    }

    let num_records = (file_size / RECORD_SIZE) as usize;
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut records = Vec::with_capacity(num_records);

    for _ in 0..num_records {
        let player = reader.read_u64::<LittleEndian>()?;
        let opponent = reader.read_u64::<LittleEndian>()?;
        let score = reader.read_f32::<LittleEndian>()?;
        let game_score = reader.read_i8()?;
        let ply = reader.read_u8()?;
        let is_random_byte = reader.read_u8()?;
        let sq_byte = reader.read_u8()?;
        let side_to_move_byte = reader.read_u8()?;
        let game_id = reader.read_u16::<LittleEndian>()?;

        let board = Board::from_bitboards(Bitboard::new(player), Bitboard::new(opponent));
        let side_to_move = if side_to_move_byte == 0 {
            Disc::Black
        } else {
            Disc::White
        };
        let sq = Square::from_u8(sq_byte).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid square: {sq_byte}"),
            )
        })?;

        records.push(GameRecord {
            game_id,
            ply,
            board,
            score,
            game_score,
            side_to_move,
            is_random: is_random_byte != 0,
            sq,
        });
    }

    Ok(records)
}
