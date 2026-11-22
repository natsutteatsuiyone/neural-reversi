use crate::score::Score;

pub(super) const KEY_MASK: u64 = 0xFFFFFFFFFFFF;
pub(super) const SCORE_BITS: u32 = 16;
const EMPTY_SCORE_BITS: u16 = i16::MIN as u16;
pub(super) const EMPTY_ENTRY: u64 = EMPTY_SCORE_BITS as u64;
pub(super) const WAYS: usize = 4;
pub(super) const WAY_BITS: u32 = WAYS.trailing_zeros();

pub(super) fn bucket_count(size_log2: u32) -> usize {
    assert!(size_log2 < usize::BITS, "EvalCache size does not fit usize");
    1usize << size_log2.saturating_sub(WAY_BITS)
}

#[inline(always)]
pub(super) fn location(key: u64, mask: u64) -> (usize, usize) {
    let rotated = key.rotate_left(SCORE_BITS);
    let index = (rotated & mask) as usize;
    let victim_way = (rotated >> (u64::BITS - WAY_BITS)) as usize;
    (index, victim_way)
}

#[inline(always)]
pub(super) fn pack(key: u64, score: Score) -> u64 {
    let score = score.value();
    assert!(
        (-Score::INF.value()..=Score::INF.value()).contains(&score),
        "cache score must be within the Score sentinel range"
    );
    ((key & KEY_MASK) << SCORE_BITS) | score as u16 as u64
}

#[inline(always)]
pub(super) fn select_way(
    key: u64,
    entries: &[u64; WAYS],
    victim_way: usize,
) -> (Option<Score>, usize) {
    let key_masked = key & KEY_MASK;
    let mut hit_ways = 0u32;
    let mut empty_ways = 0u32;
    for (way, &entry) in entries.iter().enumerate() {
        let matches = (entry >> SCORE_BITS == key_masked) & (entry as u16 != EMPTY_SCORE_BITS);
        hit_ways |= (matches as u32) << way;
        empty_ways |= ((entry == EMPTY_ENTRY) as u32) << way;
    }

    if hit_ways != 0 {
        let way = hit_ways.trailing_zeros() as usize;
        // SAFETY: a nonzero hit mask has bits set only for indices in entries.
        let entry = unsafe { *entries.get_unchecked(way) };
        return (Some(Score::from_raw(entry as i16 as i32)), way);
    }

    let way = if empty_ways != 0 {
        empty_ways.trailing_zeros() as usize
    } else {
        victim_way
    };
    (None, way)
}
