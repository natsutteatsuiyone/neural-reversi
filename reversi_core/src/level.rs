use crate::{probcut::NO_SELECTIVITY, types::Depth};

#[derive(Copy, Clone)]
pub struct Level {
    pub mid_depth: Depth,
    pub end_depth: [Depth; NO_SELECTIVITY as usize + 1],
}

impl Level {
    pub fn get_end_depth(&self, selectivity: u8) -> Depth {
        self.end_depth[selectivity as usize]
    }
}

pub fn get_level(lv: usize) -> Level {
    LEVELS[lv]
}

#[rustfmt::skip]
const LEVELS: [Level; 22] = [
    Level { mid_depth:  1, end_depth: [ 1, 1, 1, 1, 1, 1, 1] },
    Level { mid_depth:  1, end_depth: [ 2, 2, 2, 2, 2, 2, 2] },
    Level { mid_depth:  2, end_depth: [ 4, 4, 4, 4, 4, 4, 4] },
    Level { mid_depth:  3, end_depth: [ 6, 6, 6, 6, 6, 6, 6] },
    Level { mid_depth:  4, end_depth: [ 8, 8, 8, 8, 8, 8, 8] },
    Level { mid_depth:  5, end_depth: [10,10,10,10,10,10,10] },
    Level { mid_depth:  6, end_depth: [12,12,12,12,12,12,12] },
    Level { mid_depth:  7, end_depth: [14,14,14,14,14,14,14] },
    Level { mid_depth:  8, end_depth: [16,16,16,16,16,16,16] },
    Level { mid_depth:  9, end_depth: [18,18,18,18,18,18,18] },
    Level { mid_depth: 10, end_depth: [20,20,20,20,20,20,20] },
    Level { mid_depth: 11, end_depth: [21,21,21,20,20,20,20] },
    Level { mid_depth: 12, end_depth: [21,21,21,21,21,20,20] },
    Level { mid_depth: 13, end_depth: [22,22,22,22,21,21,21] },
    Level { mid_depth: 14, end_depth: [22,22,22,22,22,22,22] },
    Level { mid_depth: 15, end_depth: [23,23,23,22,22,22,22] },
    Level { mid_depth: 16, end_depth: [23,23,23,23,23,22,22] },
    Level { mid_depth: 17, end_depth: [23,23,23,23,23,23,23] },
    Level { mid_depth: 18, end_depth: [24,24,24,24,23,23,23] },
    Level { mid_depth: 19, end_depth: [24,24,24,24,24,23,23] },
    Level { mid_depth: 20, end_depth: [25,25,25,25,24,24,24] },
    Level { mid_depth: 21, end_depth: [26,26,26,26,25,25,25] },
];
