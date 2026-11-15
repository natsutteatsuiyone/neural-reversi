//! Binary 27-byte record format and file I/O for training data.
//!
//! Wire layout (little-endian, packed, 27 bytes total):
//! - `0..8`: `player` bitboard (`u64` LE)
//! - `8..16`: `opponent` bitboard (`u64` LE)
//! - `16..20`: `score` (`f32` LE, in discs)
//! - `20`: `game_score` (`i8`, `i8::MIN` sentinel when unavailable)
//! - `21`: `ply` (`u8`)
//! - `22`: `is_random` (`u8`: encoded as `0` or `1`, decoded as `!= 0`)
//! - `23`: `sq` (`u8`: square index `0..64`)
//! - `24`: `side_to_move` (`u8`: encoded as `0` Black or `1` White, decoded as `0` => Black, non-zero => White)
//! - `25..27`: `game_id` (`u16` LE)

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::Context;
use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::player_color::PlayerColor;
use reversi_core::square::Square;

pub(crate) const RECORD_SIZE: usize = 27;

const PLAYER_OFFSET: usize = 0;
const OPPONENT_OFFSET: usize = 8;
const SCORE_OFFSET: usize = 16;
const GAME_SCORE_OFFSET: usize = 20;
const PLY_OFFSET: usize = 21;
const IS_RANDOM_OFFSET: usize = 22;
const SQ_OFFSET: usize = 23;
const SIDE_TO_MOVE_OFFSET: usize = 24;
const GAME_ID_OFFSET: usize = 25;

const _: () = assert!(PLAYER_OFFSET + 8 == OPPONENT_OFFSET);
const _: () = assert!(OPPONENT_OFFSET + 8 == SCORE_OFFSET);
const _: () = assert!(SCORE_OFFSET + 4 == GAME_SCORE_OFFSET);
const _: () = assert!(GAME_SCORE_OFFSET + 1 == PLY_OFFSET);
const _: () = assert!(PLY_OFFSET + 1 == IS_RANDOM_OFFSET);
const _: () = assert!(IS_RANDOM_OFFSET + 1 == SQ_OFFSET);
const _: () = assert!(SQ_OFFSET + 1 == SIDE_TO_MOVE_OFFSET);
const _: () = assert!(SIDE_TO_MOVE_OFFSET + 1 == GAME_ID_OFFSET);
const _: () = assert!(GAME_ID_OFFSET + 2 == RECORD_SIZE);

const GAME_SCORE_UNAVAILABLE: i8 = i8::MIN;

const READ_BUFFER_RECORDS: usize = 4096;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GameRecord {
    pub game_id: u16,
    pub ply: u8,
    pub board: Board,
    pub score: f32,
    pub game_score: Option<i8>,
    pub side_to_move: PlayerColor,
    pub is_random: bool,
    pub sq: Square,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RewriteStats {
    pub scanned: u64,
    pub updated: u64,
}

/// Unvalidated 27-byte wire record; [`Self::board`] and [`Self::decode`] check
/// bitboard disjointness and square bounds on demand.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct RawRecord([u8; RECORD_SIZE]);

impl RawRecord {
    #[must_use]
    pub(crate) fn from_record(record: &GameRecord) -> Self {
        debug_assert_ne!(record.game_score, Some(GAME_SCORE_UNAVAILABLE));

        let mut bytes = [0u8; RECORD_SIZE];
        bytes[PLAYER_OFFSET..PLAYER_OFFSET + 8]
            .copy_from_slice(&record.board.player().bits().to_le_bytes());
        bytes[OPPONENT_OFFSET..OPPONENT_OFFSET + 8]
            .copy_from_slice(&record.board.opponent().bits().to_le_bytes());
        bytes[SCORE_OFFSET..SCORE_OFFSET + 4].copy_from_slice(&record.score.to_le_bytes());
        bytes[GAME_SCORE_OFFSET] = record.game_score.unwrap_or(GAME_SCORE_UNAVAILABLE) as u8;
        bytes[PLY_OFFSET] = record.ply;
        bytes[IS_RANDOM_OFFSET] = u8::from(record.is_random);
        bytes[SQ_OFFSET] = record.sq as u8;
        bytes[SIDE_TO_MOVE_OFFSET] = u8::from(record.side_to_move != PlayerColor::Black);
        bytes[GAME_ID_OFFSET..GAME_ID_OFFSET + 2].copy_from_slice(&record.game_id.to_le_bytes());
        Self(bytes)
    }

    pub(crate) fn decode(&self) -> anyhow::Result<GameRecord> {
        let board = self.board();
        let side_to_move = if self.0[SIDE_TO_MOVE_OFFSET] == 0 {
            PlayerColor::Black
        } else {
            PlayerColor::White
        };
        let sq_byte = self.0[SQ_OFFSET];
        let sq =
            Square::from_u8(sq_byte).ok_or_else(|| anyhow::anyhow!("Invalid square: {sq_byte}"))?;

        Ok(GameRecord {
            game_id: self.game_id(),
            ply: self.ply(),
            board,
            score: self.score(),
            game_score: self.game_score(),
            side_to_move,
            is_random: self.is_random(),
            sq,
        })
    }

    #[inline]
    #[must_use]
    pub(crate) fn ply(&self) -> u8 {
        self.0[PLY_OFFSET]
    }

    #[inline]
    #[must_use]
    pub(crate) fn is_random(&self) -> bool {
        self.0[IS_RANDOM_OFFSET] != 0
    }

    #[inline]
    #[must_use]
    pub(crate) fn score(&self) -> f32 {
        let bytes: [u8; 4] = self.0[SCORE_OFFSET..SCORE_OFFSET + 4]
            .try_into()
            .expect("4-byte score slice");
        f32::from_le_bytes(bytes)
    }

    #[inline]
    #[must_use]
    pub(crate) fn game_score(&self) -> Option<i8> {
        let raw = self.0[GAME_SCORE_OFFSET] as i8;
        (raw != GAME_SCORE_UNAVAILABLE).then_some(raw)
    }

    #[inline]
    #[must_use]
    pub(crate) fn game_id(&self) -> u16 {
        let bytes: [u8; 2] = self.0[GAME_ID_OFFSET..GAME_ID_OFFSET + 2]
            .try_into()
            .expect("2-byte game_id slice");
        u16::from_le_bytes(bytes)
    }

    #[inline]
    #[must_use]
    pub(crate) fn board(&self) -> Board {
        let player = u64::from_le_bytes(
            self.0[PLAYER_OFFSET..PLAYER_OFFSET + 8]
                .try_into()
                .expect("8-byte player slice"),
        );
        let opponent = u64::from_le_bytes(
            self.0[OPPONENT_OFFSET..OPPONENT_OFFSET + 8]
                .try_into()
                .expect("8-byte opponent slice"),
        );
        Board::from_bitboards(Bitboard::new(player), Bitboard::new(opponent))
    }

    #[inline]
    pub(crate) fn set_scores(&mut self, score: f32, game_score: Option<i8>, is_random: bool) {
        debug_assert_ne!(game_score, Some(GAME_SCORE_UNAVAILABLE));

        self.0[SCORE_OFFSET..SCORE_OFFSET + 4].copy_from_slice(&score.to_le_bytes());
        self.0[GAME_SCORE_OFFSET] = game_score.unwrap_or(GAME_SCORE_UNAVAILABLE) as u8;
        self.0[IS_RANDOM_OFFSET] = u8::from(is_random);
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn from_bytes(bytes: [u8; RECORD_SIZE]) -> Self {
        Self(bytes)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn as_bytes(&self) -> &[u8; RECORD_SIZE] {
        &self.0
    }
}

fn write_records(writer: &mut impl Write, records: &[GameRecord]) -> io::Result<()> {
    for record in records {
        writer.write_all(&RawRecord::from_record(record).0)?;
    }
    Ok(())
}

pub(crate) fn append_records(path: &Path, records: &[GameRecord]) -> anyhow::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open '{}' for append", path.display()))?;
    let mut writer = BufWriter::new(file);
    write_records(&mut writer, records)
        .with_context(|| format!("failed to write records to '{}'", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("failed to flush '{}'", path.display()))
}

pub(crate) fn append_raw_records(path: &Path, records: &[RawRecord]) -> anyhow::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open '{}' for append", path.display()))?;
    let mut writer = BufWriter::new(file);
    for record in records {
        writer
            .write_all(&record.0)
            .with_context(|| format!("failed to write raw record to '{}'", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("failed to flush '{}'", path.display()))
}

pub(crate) fn count_aligned_records(path: &Path) -> anyhow::Result<u64> {
    let file_size = fs::metadata(path)
        .with_context(|| format!("failed to inspect '{}'", path.display()))?
        .len();
    anyhow::ensure!(
        file_size.is_multiple_of(RECORD_SIZE as u64),
        "'{}' has size {file_size}, which is not a multiple of record size {RECORD_SIZE}",
        path.display()
    );
    Ok(file_size / RECORD_SIZE as u64)
}

pub(crate) fn truncate_incomplete_record(path: &Path) -> anyhow::Result<()> {
    let file = match OpenOptions::new().write(true).open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(e).with_context(|| format!("failed to open '{}'", path.display()));
        }
    };
    let file_size = file.metadata()?.len();
    let remainder = file_size % RECORD_SIZE as u64;
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

pub(crate) fn read_last_game_id(path: &Path) -> anyhow::Result<Option<u16>> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(e).with_context(|| format!("failed to open '{}'", path.display()));
        }
    };
    let file_size = file.metadata()?.len();
    let aligned = file_size - file_size % RECORD_SIZE as u64;
    if aligned < RECORD_SIZE as u64 {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(aligned - RECORD_SIZE as u64))?;
    let mut raw = RawRecord([0u8; RECORD_SIZE]);
    file.read_exact(&mut raw.0)?;
    Ok(Some(raw.game_id()))
}

pub(crate) fn read_records_from_file(path: &Path) -> anyhow::Result<Vec<GameRecord>> {
    let file = File::open(path).with_context(|| format!("failed to open '{}'", path.display()))?;
    let file_size = file.metadata()?.len();
    anyhow::ensure!(
        file_size.is_multiple_of(RECORD_SIZE as u64),
        "File size {} is not a multiple of record size {} for file {}",
        file_size,
        RECORD_SIZE,
        path.display()
    );

    let num_records = usize::try_from(file_size / RECORD_SIZE as u64)?;
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0u8; RECORD_SIZE * READ_BUFFER_RECORDS.min(num_records)];
    let mut records = Vec::with_capacity(num_records);
    let mut remaining = num_records;

    while remaining > 0 {
        let batch = remaining.min(READ_BUFFER_RECORDS);
        let batch_bytes = batch * RECORD_SIZE;
        reader.read_exact(&mut buffer[..batch_bytes])?;
        for chunk in buffer[..batch_bytes].as_chunks::<RECORD_SIZE>().0 {
            records.push(RawRecord(*chunk).decode()?);
        }
        remaining -= batch;
    }

    Ok(records)
}

pub(crate) fn visit_aligned_raw_records_or_warn(
    path: &Path,
    mut visit: impl FnMut(RawRecord),
) -> anyhow::Result<()> {
    let file_size = fs::metadata(path)
        .with_context(|| format!("failed to inspect '{}'", path.display()))?
        .len();
    if file_size == 0 || !file_size.is_multiple_of(RECORD_SIZE as u64) {
        eprintln!(
            "Warning: {} skipped (size {} is not a positive multiple of record size {})",
            path.display(),
            file_size,
            RECORD_SIZE,
        );
        return Ok(());
    }

    let file = File::open(path).with_context(|| format!("failed to open '{}'", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut remaining = usize::try_from(file_size / RECORD_SIZE as u64)?;
    let mut buffer = vec![0u8; RECORD_SIZE * READ_BUFFER_RECORDS.min(remaining)];

    while remaining > 0 {
        let batch = remaining.min(READ_BUFFER_RECORDS);
        let batch_bytes = batch * RECORD_SIZE;
        reader.read_exact(&mut buffer[..batch_bytes])?;
        for chunk in buffer[..batch_bytes].as_chunks::<RECORD_SIZE>().0 {
            visit(RawRecord(*chunk));
        }
        remaining -= batch;
    }
    Ok(())
}

pub(crate) fn rewrite_raw_records(
    path: &Path,
    mut patch: impl FnMut(&mut RawRecord) -> bool,
) -> anyhow::Result<RewriteStats> {
    let file_size = fs::metadata(path)
        .with_context(|| format!("failed to inspect '{}'", path.display()))?
        .len();
    if file_size == 0 || !file_size.is_multiple_of(RECORD_SIZE as u64) {
        eprintln!(
            "Warning: {} skipped (size {} is not a positive multiple of record size {})",
            path.display(),
            file_size,
            RECORD_SIZE,
        );
        return Ok(RewriteStats::default());
    }

    let num_records = usize::try_from(file_size / RECORD_SIZE as u64)?;
    let mut bytes = Vec::with_capacity(usize::try_from(file_size)?);
    File::open(path)
        .with_context(|| format!("failed to open '{}'", path.display()))?
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() == num_records * RECORD_SIZE,
        "'{}' changed size while reading",
        path.display()
    );

    let mut updated = 0u64;
    for chunk in bytes.as_chunks_mut::<RECORD_SIZE>().0 {
        let mut raw = RawRecord(*chunk);
        if patch(&mut raw) {
            *chunk = raw.0;
            updated += 1;
        }
    }

    if updated > 0 {
        let temp_path = path.with_extension("bin.tmp");
        {
            let mut file = File::create(&temp_path)
                .with_context(|| format!("failed to create '{}'", temp_path.display()))?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        fs::rename(&temp_path, path).with_context(|| {
            format!(
                "failed to rename '{}' to '{}'",
                temp_path.display(),
                path.display()
            )
        })?;
    }

    Ok(RewriteStats {
        scanned: num_records as u64,
        updated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
                "neural-reversi-datagen-{name}-{}-{id}.bin",
                std::process::id()
            ));
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn make_record(index: u8, game_id: u16) -> GameRecord {
        GameRecord {
            game_id,
            ply: index + 10,
            board: Board::from_bitboards(
                Bitboard::new(1 << index),
                Bitboard::new(1 << (index + 8)),
            ),
            score: f32::from(index) + 0.5,
            game_score: Some(index as i8 - 2),
            side_to_move: if index.is_multiple_of(2) {
                PlayerColor::Black
            } else {
                PlayerColor::White
            },
            is_random: index.is_multiple_of(2),
            sq: Square::from_u8(index).expect("test square is valid"),
        }
    }

    #[test]
    fn record_layout_matches_offsets() -> anyhow::Result<()> {
        let record = GameRecord {
            game_id: 0xBEEF,
            ply: 42,
            board: Board::from_bitboards(
                Bitboard::new(0x0123_4567_89AB_CDEF),
                Bitboard::new(0xFEDC_BA98_7654_3210),
            ),
            score: 1.5,
            game_score: Some(-3),
            side_to_move: PlayerColor::White,
            is_random: true,
            sq: Square::D3,
        };
        let mut bytes = Vec::new();

        write_records(&mut bytes, std::slice::from_ref(&record))?;

        assert_eq!(bytes.len(), RECORD_SIZE);
        assert_eq!(
            &bytes[PLAYER_OFFSET..PLAYER_OFFSET + 8],
            &0x0123_4567_89AB_CDEFu64.to_le_bytes()
        );
        assert_eq!(
            &bytes[OPPONENT_OFFSET..OPPONENT_OFFSET + 8],
            &0xFEDC_BA98_7654_3210u64.to_le_bytes()
        );
        assert_eq!(
            &bytes[SCORE_OFFSET..SCORE_OFFSET + 4],
            &1.5f32.to_le_bytes()
        );
        assert_eq!(bytes[GAME_SCORE_OFFSET] as i8, -3);
        assert_eq!(bytes[PLY_OFFSET], 42);
        assert_eq!(bytes[IS_RANDOM_OFFSET], 1);
        assert_eq!(bytes[SQ_OFFSET], Square::D3 as u8);
        assert_eq!(bytes[SIDE_TO_MOVE_OFFSET], 1);
        assert_eq!(
            &bytes[GAME_ID_OFFSET..GAME_ID_OFFSET + 2],
            &0xBEEFu16.to_le_bytes()
        );

        let unavailable = GameRecord {
            game_score: None,
            ..record
        };
        let raw_unavailable = RawRecord::from_record(&unavailable);
        assert_eq!(
            raw_unavailable.as_bytes()[GAME_SCORE_OFFSET] as i8,
            GAME_SCORE_UNAVAILABLE
        );
        assert_eq!(raw_unavailable.decode()?, unavailable);
        Ok(())
    }

    #[test]
    fn record_round_trips_through_file() -> anyhow::Result<()> {
        let file = TempFile::new("round-trip");
        let empty = TempFile::new("round-trip-empty");
        let unaligned = TempFile::new("round-trip-unaligned");
        let mut unavailable = make_record(4, 9);
        unavailable.game_score = None;
        let records = [
            make_record(1, 1),
            make_record(2, 7),
            make_record(3, 0xBEEF),
            unavailable,
        ];

        append_records(file.path(), &records)?;
        let actual = read_records_from_file(file.path())?;
        assert_eq!(actual, records);

        File::create(empty.path())?;
        assert!(read_records_from_file(empty.path())?.is_empty());

        fs::write(unaligned.path(), [1u8, 2, 3])?;
        assert!(
            read_records_from_file(unaligned.path())
                .unwrap_err()
                .to_string()
                .contains("not a multiple of record size")
        );
        Ok(())
    }

    #[test]
    fn truncate_incomplete_record_removes_partial_tail() -> anyhow::Result<()> {
        let file = TempFile::new("partial-tail");
        let missing = TempFile::new("missing");
        let record = make_record(4, 11);
        append_records(file.path(), std::slice::from_ref(&record))?;
        OpenOptions::new()
            .append(true)
            .open(file.path())?
            .write_all(&[1, 2, 3, 4, 5])?;

        truncate_incomplete_record(file.path())?;

        assert_eq!(fs::metadata(file.path())?.len(), RECORD_SIZE as u64);
        let actual = read_records_from_file(file.path())?;
        assert_eq!(actual, [record]);
        assert!(!missing.path().exists());
        truncate_incomplete_record(missing.path())?;
        assert!(!missing.path().exists());
        Ok(())
    }

    #[test]
    fn read_last_game_id_reads_trailing_u16() -> anyhow::Result<()> {
        let populated = TempFile::new("last-game-id");
        let empty = TempFile::new("empty");
        let missing = TempFile::new("missing-last-game-id");
        let records = [make_record(5, 1), make_record(6, 7)];
        append_records(populated.path(), &records)?;
        File::create(empty.path())?;

        assert_eq!(read_last_game_id(populated.path())?, Some(7));
        assert_eq!(read_last_game_id(empty.path())?, None);
        assert_eq!(read_last_game_id(missing.path())?, None);
        Ok(())
    }
}
