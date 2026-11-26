use std::io::{self, Write};

use crate::time_control::{TimeControlMode, TimeSettings};

fn format_mode(mode: TimeControlMode) -> String {
    match mode {
        TimeControlMode::Infinite => "infinite".to_owned(),
        TimeControlMode::Fixed { .. } => unreachable!("GTP time settings do not use fixed mode"),
        TimeControlMode::Byoyomi { time_per_move_ms } => format!("byo {time_per_move_ms}"),
        TimeControlMode::Fischer {
            main_time_ms,
            increment_ms,
        } => format!("fischer {main_time_ms} {increment_ms}"),
        TimeControlMode::JapaneseByo {
            main_time_ms,
            time_per_move_ms,
        } => format!("japanese {main_time_ms} {time_per_move_ms}"),
    }
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    const MAX_WHOLE_SECONDS: u64 = u64::MAX / 1000;
    for main in [0, 1, 5, MAX_WHOLE_SECONDS, MAX_WHOLE_SECONDS + 1, u64::MAX] {
        for extra in [0, 1, 3, MAX_WHOLE_SECONDS + 1] {
            for stones in [0, 1, 3, u32::MAX] {
                let settings = TimeSettings::from_gtp_secs(main, extra, stones);
                let initial = format_mode(settings.initial_time_control());
                for remaining in [0, 1, 1234, u64::MAX] {
                    for left in [0, 1, 2, u32::MAX] {
                        let per_move = format_mode(settings.move_time_control(remaining, left));
                        writeln!(
                            out,
                            "{main} {extra} {stones} {remaining} {left}|{} {} {} {}|{initial}|{per_move}",
                            u8::from(settings.is_enabled()),
                            settings.main_time_secs(),
                            settings.byoyomi_time_secs(),
                            settings.byoyomi_stones(),
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}
