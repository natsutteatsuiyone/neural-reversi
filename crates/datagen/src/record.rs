//! Binary record format for training data I/O.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::square::Square;
use reversi_core::types::Scoref;
use std::fs;
use std::fs::OpenOptions;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;

/// Size of each record in bytes
pub const RECORD_SIZE: u64 = 24;

/// Represents a single position record from a self-play game.
#[derive(Clone)]
pub struct GameRecord {
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
    }
    Ok(())
}

/// Counts the number of records in a binary file.
pub fn count_records_in_file(path: &Path) -> io::Result<u32> {
    if !path.exists() {
        return Ok(0);
    }
    let metadata = fs::metadata(path)?;
    let file_size = metadata.len();

    if file_size % RECORD_SIZE != 0 {
        eprintln!(
            "Warning: File size {} is not a multiple of RECORD_SIZE {} for file {}. File might be corrupted.",
            file_size,
            RECORD_SIZE,
            path.display()
        );
    }

    Ok((file_size / RECORD_SIZE) as u32)
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

        let board = Board::from_bitboards(Bitboard::new(player), Bitboard::new(opponent));
        // Note: side_to_move cannot be faithfully reconstructed from the binary format
        // (passes are not recorded). This approximation is sufficient for rescore
        // since side_to_move is not serialized back to the output.
        let side_to_move = if ply % 2 == 0 {
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
