//! Neural network for midgame evaluation.

use std::cell::UnsafeCell;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use crate::board::Board;
use crate::eval::network::input_layer::{
    BASE_OUTPUT_DIMS, BaseInput, PA_OUTPUT_DIMS, PhaseAdaptiveInput,
};
use crate::eval::network::layer_stack::{LayerStack, load_layer_stacks};
use crate::eval::pattern_feature::PatternFeature;
use crate::eval::util::ceil_to_multiple;
use crate::types::ScaledScore;
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
    #[inline(always)]
    fn base_input_mut(&mut self) -> &mut [u8] {
        &mut self.l1_input[..BASE_OUTPUT_DIMS]
    }

    #[inline(always)]
    fn pa_input_mut(&mut self) -> &mut [u8] {
        &mut self.l1_input[PA_INPUT_START..PA_INPUT_END]
    }

    #[inline(always)]
    fn output_segments(&self) -> [&[u8]; 2] {
        [
            &self.l2_out[..L2_OUTPUT_DIMS],
            &self.l1_input[..PA_INPUT_END],
        ]
    }
}

/// Length of the raw per-thread scratch storage, with `align_of` bytes of slack
/// so a 64-aligned `NetworkBuffers` can always be carved out of it.
const SCRATCH_LEN: usize =
    std::mem::size_of::<NetworkBuffers>() + std::mem::align_of::<NetworkBuffers>();

thread_local! {
    /// Per-thread scratch reused across evaluations: the ~768-byte buffer is
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
#[inline(always)]
fn scratch_ptr(cell: &UnsafeCell<[u8; SCRATCH_LEN]>) -> *mut NetworkBuffers {
    let base = cell.get().cast::<u8>();
    let offset = base.align_offset(std::mem::align_of::<NetworkBuffers>());
    // SAFETY: the storage has `align_of::<NetworkBuffers>()` bytes of slack, so
    // `offset` (in `0..align`) keeps the `size_of::<NetworkBuffers>()`-byte
    // window in bounds.
    unsafe { base.add(offset).cast::<NetworkBuffers>() }
}

/// Main neural network structure for position evaluation.
pub struct Network {
    base_input: BaseInput,
    pa_input: PhaseAdaptiveInput,
    layer_stacks: Box<[LayerStack]>,
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
        Ok(Network {
            base_input,
            pa_input,
            layer_stacks,
        })
    }

    /// Evaluates a board position using the neural network.
    #[inline(always)]
    pub fn evaluate(
        &self,
        board: &Board,
        pattern_feature: &PatternFeature,
        ply: usize,
    ) -> ScaledScore {
        let mobility = board.get_moves().count() as u8;

        NETWORK_BUFFERS.with(|cell| {
            // SAFETY: `evaluate` is not reentrant (no forward pass calls back
            // into it), so this is the only live borrow of the scratch.
            let buffers = unsafe { &mut *scratch_ptr(cell) };
            self.base_input
                .forward(pattern_feature, buffers.base_input_mut());
            self.pa_input
                .forward(pattern_feature, ply, buffers.pa_input_mut());
            buffers.l1_input[MOBILITY_INPUT_INDEX] = mobility * MOBILITY_SCALE;
            let score = self.layer_stacks[ply].forward(buffers);
            score.clamp(ScaledScore::MIN + 1, ScaledScore::MAX - 1)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// macOS TLV storage only 16-byte-aligns the per-thread block, so freshly
    /// spawned worker threads land the raw scratch at 16/32/48 mod 64. Assert
    /// the carved `NetworkBuffers` is realigned to 64 bytes on many threads;
    /// without that realignment, `evaluate`'s `&mut *cell.get()` aborted under
    /// the debug `misaligned_pointer_dereference` check.
    #[test]
    fn scratch_is_64_aligned_on_worker_threads() {
        let handles: Vec<_> = (0..256)
            .map(|_| {
                std::thread::spawn(|| {
                    NETWORK_BUFFERS.with(|cell| {
                        let addr = scratch_ptr(cell).addr();
                        assert_eq!(addr % 64, 0, "scratch not 64-aligned: {addr:#x}");
                    });
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }
}
