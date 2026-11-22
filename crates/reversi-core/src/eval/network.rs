//! Main neural network for position evaluation.

use std::cell::UnsafeCell;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use crate::board::Board;
use crate::eval::network::input_layer::{
    BASE_OUTPUT_DIMS, BaseInput, PA_OUTPUT_DIMS, PhaseAdaptiveInput,
};
use crate::eval::network::layer_stack::{LayerStack, NUM_LAYER_STACKS, load_layer_stacks};
use crate::eval::pattern_feature::PatternFeature;
use crate::eval::util::{ceil_to_multiple, expect_end_of_weights};
use crate::score::Score;
use crate::util::align::Align64;

mod activations;
mod input_layer;
mod layer_stack;
mod linear_layer;
mod output_layer;

const L1_INPUT_DIMS: usize = BASE_OUTPUT_DIMS + PA_OUTPUT_DIMS + 1;
const L1_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L1_INPUT_DIMS, 32);
const L1_OUTPUT_DIMS: usize = 16;
const L1_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L1_OUTPUT_DIMS, 32);

const L2_INPUT_DIMS: usize = L1_OUTPUT_DIMS * 2;
const L2_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L2_INPUT_DIMS, 32);
const L2_OUTPUT_DIMS: usize = 64;
const L2_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L2_OUTPUT_DIMS, 32);

const LO_INPUT_DIMS: usize = L2_OUTPUT_DIMS + BASE_OUTPUT_DIMS + PA_OUTPUT_DIMS;
const LO_PADDED_INPUT_DIMS: usize = ceil_to_multiple(LO_INPUT_DIMS, 32);

const PA_INPUT_START: usize = BASE_OUTPUT_DIMS;
const PA_INPUT_END: usize = PA_INPUT_START + PA_OUTPUT_DIMS;
const MOBILITY_INPUT_INDEX: usize = L1_INPUT_DIMS - 1;
const MOBILITY_SCALE: u8 = 7;

/// Working buffers for one network forward pass.
struct NetworkBuffers {
    l1_input: Align64<[u8; L1_PADDED_INPUT_DIMS]>,
    l1_li_out: Align64<[i32; L1_PADDED_OUTPUT_DIMS]>,
    l1_out: Align64<[u8; L2_PADDED_INPUT_DIMS]>,
    l2_li_out: Align64<[i32; L2_PADDED_OUTPUT_DIMS]>,
    l2_out: Align64<[u8; L2_PADDED_OUTPUT_DIMS]>,
}

impl NetworkBuffers {
    #[cfg(target_os = "windows")]
    const ZEROED: Self = Self {
        l1_input: Align64([0; L1_PADDED_INPUT_DIMS]),
        l1_li_out: Align64([0; L1_PADDED_OUTPUT_DIMS]),
        l1_out: Align64([0; L2_PADDED_INPUT_DIMS]),
        l2_li_out: Align64([0; L2_PADDED_OUTPUT_DIMS]),
        l2_out: Align64([0; L2_PADDED_OUTPUT_DIMS]),
    };

    #[inline(always)]
    fn base_input_mut(&mut self) -> &mut Align64<[u8; BASE_OUTPUT_DIMS]> {
        self.l1_input.sub_array_mut::<0, BASE_OUTPUT_DIMS>()
    }

    #[inline(always)]
    fn pa_input_mut(&mut self) -> &mut Align64<[u8; PA_OUTPUT_DIMS]> {
        self.l1_input
            .sub_array_mut::<PA_INPUT_START, PA_OUTPUT_DIMS>()
    }

    #[inline(always)]
    fn output_segments(&self) -> (&Align64<[u8; L2_OUTPUT_DIMS]>, &Align64<[u8; PA_INPUT_END]>) {
        (&self.l2_out, self.l1_input.sub_array::<0, PA_INPUT_END>())
    }
}

/// Length of the raw per-thread scratch storage, with `align_of` bytes of slack
/// so a 64-aligned `NetworkBuffers` can always be carved out of it.
#[cfg(not(target_os = "windows"))]
const SCRATCH_LEN: usize =
    std::mem::size_of::<NetworkBuffers>() + std::mem::align_of::<NetworkBuffers>();

#[cfg(target_os = "windows")]
thread_local! {
    /// Windows TLS preserves the type's 64-byte alignment, so store the
    /// buffers directly and avoid realigning raw storage on every evaluation.
    static NETWORK_BUFFERS: UnsafeCell<NetworkBuffers> =
        const { UnsafeCell::new(NetworkBuffers::ZEROED) };
}

#[cfg(not(target_os = "windows"))]
thread_local! {
    /// Per-thread scratch reused across evaluations: the 832-byte buffer is
    /// zeroed once per thread, not per call. `const` init keeps the access on
    /// the guard-free thread-local path.
    ///
    /// Stored as raw bytes rather than a `NetworkBuffers` because macOS TLV
    /// storage only guarantees 16-byte alignment for the per-thread block and
    /// ignores the 64-byte alignment the `Align64` fields require for their SIMD
    /// loads. We over-allocate and realign at the access site (see [`scratch_ptr`]).
    /// The all-zero byte pattern is a valid zeroed `NetworkBuffers`; only the
    /// `l1_input` padding tail is read before being written, and it stays zero.
    static NETWORK_BUFFERS: UnsafeCell<[u8; SCRATCH_LEN]> =
        const { UnsafeCell::new([0; SCRATCH_LEN]) };
}

/// Returns a 64-aligned pointer into the over-allocated raw thread-local
/// scratch storage, suitable to reinterpret as a zeroed `NetworkBuffers`.
///
/// Computing the pointer is safe; dereferencing it requires that no other
/// borrow of `NETWORK_BUFFERS` is live (`evaluate` is not reentrant).
#[cfg(not(target_os = "windows"))]
#[inline(always)]
fn scratch_ptr(cell: &UnsafeCell<[u8; SCRATCH_LEN]>) -> *mut NetworkBuffers {
    let base = cell.get().cast::<u8>();
    let offset = base.align_offset(std::mem::align_of::<NetworkBuffers>());
    // SAFETY: the storage has `align_of::<NetworkBuffers>()` bytes of slack, so
    // `offset` (in `0..align`) keeps the `size_of::<NetworkBuffers>()`-byte
    // window in bounds.
    unsafe { base.add(offset).cast::<NetworkBuffers>() }
}

#[inline(always)]
fn with_network_buffers<R>(f: impl FnOnce(&mut NetworkBuffers) -> R) -> R {
    #[cfg(target_os = "windows")]
    {
        NETWORK_BUFFERS.with(|cell| {
            // SAFETY: `evaluate` is not reentrant (no forward pass calls back
            // into it), so this is the only live borrow of the scratch.
            f(unsafe { &mut *cell.get() })
        })
    }

    #[cfg(not(target_os = "windows"))]
    {
        NETWORK_BUFFERS.with(|cell| {
            // SAFETY: `evaluate` is not reentrant (no forward pass calls back
            // into it), so this is the only live borrow of the scratch.
            f(unsafe { &mut *scratch_ptr(cell) })
        })
    }
}

/// Main neural network structure for position evaluation.
pub struct Network {
    base_input: BaseInput,
    pa_input: PhaseAdaptiveInput,
    layer_stacks: Box<[LayerStack; NUM_LAYER_STACKS]>,
}

impl Network {
    /// Creates a new network by loading weights from a compressed file.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the file cannot be opened or the weights are malformed.
    pub fn new(file_path: &Path) -> io::Result<Self> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    /// Creates a new network by loading weights from an in-memory blob.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the weights are malformed.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let cursor = io::Cursor::new(bytes);
        Self::from_reader(cursor)
    }

    fn from_reader<R: Read>(reader: R) -> io::Result<Self> {
        let mut decoder = zstd::stream::read::Decoder::new(reader)?;
        let base_input = BaseInput::load(&mut decoder)?;
        let pa_input = PhaseAdaptiveInput::load(&mut decoder)?;
        let layer_stacks = load_layer_stacks(&mut decoder)?;
        expect_end_of_weights(&mut decoder)?;
        Ok(Network {
            base_input,
            pa_input,
            layer_stacks,
        })
    }

    /// Evaluates a board position using the neural network.
    #[inline(always)]
    pub fn evaluate(&self, board: &Board, pattern_feature: &PatternFeature, ply: usize) -> Score {
        let mobility = board.legal_moves().count() as u8;

        with_network_buffers(|buffers| {
            self.base_input
                .forward(pattern_feature, buffers.base_input_mut());
            self.pa_input
                .forward(pattern_feature, ply, buffers.pa_input_mut());
            buffers.l1_input[MOBILITY_INPUT_INDEX] = mobility * MOBILITY_SCALE;
            let score = self.layer_stacks[ply].forward(buffers);
            score.clamp(Score::MIN + 1, Score::MAX - 1)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn compressed_main_weights_produce_the_expected_score_for_every_ply() {
        let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 1).unwrap();
        encoder.include_checksum(true).unwrap();
        let mut position = 0;
        let mut write_at = |offset, bytes: &[u8]| {
            static ZEROS: [u8; 64 * 1024] = [0; 64 * 1024];
            let mut remaining = usize::try_from(offset - position).unwrap();
            while remaining > 0 {
                let chunk = remaining.min(ZEROS.len());
                encoder.write_all(&ZEROS[..chunk]).unwrap();
                remaining -= chunk;
            }
            encoder.write_all(bytes).unwrap();
            position = offset + bytes.len() as u64;
        };

        let rows = 297_432;
        let base_size = 512 * (rows + 1);
        let phase_size = 256 * (rows + 1);
        write_at(0, &256i16.to_le_bytes());
        write_at(2 * 128, &256i16.to_le_bytes());
        write_at(512 + 512, &256i16.to_le_bytes());
        for phase in 0..6 {
            let bias = 256 + 32 * phase as i16;
            write_at(base_size + phase * phase_size, &bias.to_le_bytes());
        }

        let stacks_start = base_size + 6 * phase_size;
        for ply in 0..60 {
            let l1 = stacks_start + 7_620 * ply;
            let l2 = l1 + 16 * (4 + 288);
            let output = l2 + 64 * (4 + 32);
            let l1_bias = -256 + 24 * ply as i32;
            let l2_bias = 3_456 + 113 * ply as i32;
            let output_bias = 64 * ply as i32 - 513;
            write_at(l1, &l1_bias.to_le_bytes());
            write_at(l1 + 16 * 4, &[3]);
            write_at(l1 + 16 * 4 + 128, &[(-2i8) as u8]);
            write_at(l1 + 16 * 4 + 256, &[36, 127]);
            write_at(l2 + 37 * 4, &l2_bias.to_le_bytes());
            write_at(l2 + 64 * 4 + 37 * 32, &[7]);
            write_at(l2 + 64 * 4 + 37 * 32 + 16, &[23]);
            write_at(output, &output_bias.to_le_bytes());
            write_at(output + 4 + 2 * 37, &(-61i16).to_le_bytes());
            write_at(output + 4 + 2 * 64, &32i16.to_le_bytes());
            write_at(output + 4 + 2 * 192, &(-16i16).to_le_bytes());
        }
        write_at(609_599_984, &[]);

        let bytes = encoder.finish().unwrap();
        let network = Network::from_bytes(&bytes).unwrap();
        let feature = PatternFeature::new();
        let mut different_feature = feature;
        different_feature.set(0, 1);
        let initial = Board::new();
        let single_move = Board::from_bitboards(1u64, 2u64);
        for ply in 0..60 {
            let phase_bias = 256 + 32 * i32::try_from(ply / 10).unwrap();
            let phase = phase_bias.pow(2) / 1_024;
            for (board, feature, mobility, base) in [
                (&initial, &feature, 4, 64),
                (&single_move, &different_feature, 1, 127),
            ] {
                let ply_i32 = i32::try_from(ply).unwrap();
                let l1 = -256 + 24 * ply_i32 + 3 * base - 2 * phase + 36 * 7 * mobility;
                let square = (l1.pow(2) / 1_048_576).min(255);
                let relu = (l1 / 64).clamp(0, 255);
                let l2 = 3_456 + 113 * ply_i32 + 7 * square + 23 * relu;
                let activation = l2.clamp(0, 255 * 64).pow(2) / 1_048_576;
                let raw = 64 * ply_i32 - 513 - 61 * activation + 32 * base - 16 * phase;
                assert_eq!(
                    network.evaluate(board, feature, ply).value(),
                    raw.div_euclid(64),
                    "ply {ply}, mobility {mobility}"
                );
            }
        }
        drop(network);
        assert_eq!(
            Network::from_bytes(&bytes[..bytes.len() - 1])
                .err()
                .unwrap()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    /// macOS TLV storage only 16-byte-aligns the per-thread block, so freshly
    /// spawned worker threads land the raw scratch at 16/32/48 mod 64. Assert
    /// the carved `NetworkBuffers` is realigned to 64 bytes on many threads;
    /// without that realignment, `evaluate`'s `&mut *cell.get()` aborted under
    /// the debug `misaligned_pointer_dereference` check.
    #[test]
    fn scratch_is_64_aligned_on_worker_threads() {
        assert_eq!(std::mem::size_of::<NetworkBuffers>(), 832);
        assert_eq!(std::mem::align_of::<NetworkBuffers>(), 64);
        let handles: Vec<_> = (0..256)
            .map(|_| {
                std::thread::spawn(|| {
                    with_network_buffers(|buffers| {
                        let addr = std::ptr::from_ref(buffers).addr();
                        assert_eq!(addr % 64, 0, "scratch not 64-aligned: {addr:#x}");
                        for field in [
                            buffers.l1_input.as_ptr().addr(),
                            buffers.l1_li_out.as_ptr().addr(),
                            buffers.l1_out.as_ptr().addr(),
                            buffers.l2_li_out.as_ptr().addr(),
                            buffers.l2_out.as_ptr().addr(),
                        ] {
                            assert!((addr..addr + 832).contains(&field));
                            assert_eq!(field % 64, 0);
                        }
                    })
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
