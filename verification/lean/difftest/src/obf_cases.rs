use std::error::Error;
use std::io::{self, Write};

use flip_scalar_difftest::opt_or_dash;

use crate::engine::obf::{ObfPosition, ParseObfError};
use crate::player_color::PlayerColor;
use crate::square::Square;

const OPENING: &str = "---------------------------OX------XO---------------------------";

fn write_hex(out: &mut impl Write, text: &str) -> io::Result<()> {
    if text.is_empty() {
        write!(out, "-")?;
    }
    for byte in text.bytes() {
        write!(out, "{byte:02x}")?;
    }
    Ok(())
}

fn write_case(out: &mut impl Write, input: &str) -> io::Result<()> {
    write_hex(out, input)?;
    match ObfPosition::parse(input) {
        Ok(None) => writeln!(out, "|blank"),
        Ok(Some(pos)) => {
            let best = pos
                .best_moves()
                .map(|square| square.index().to_string())
                .collect::<Vec<_>>()
                .join(",");
            write!(
                out,
                "|ok {} {} {} {} {}|{best}|",
                pos.board.player().bits(),
                pos.board.opponent().bits(),
                u8::from(pos.side_to_move == PlayerColor::White),
                u8::from(pos.is_pass()),
                opt_or_dash(pos.expected_score())
            )?;
            for index in 0..64 {
                let square = Square::from_u8(index).unwrap();
                if index > 0 {
                    write!(out, ",")?;
                }
                write!(
                    out,
                    "{}:{}",
                    opt_or_dash(pos.score_of(square)),
                    opt_or_dash(pos.rank_of(square))
                )?;
            }
            writeln!(out)
        }
        Err(err) => {
            let category = match err {
                ParseObfError::NonAsciiHeader { .. } => "non-ascii",
                ParseObfError::HeaderTooShort { .. } => "short",
                ParseObfError::InvalidSideToMove(_) => "side",
                ParseObfError::InvalidBoard(_) => "board",
                ParseObfError::InvalidSegment { .. } => "segment",
                ParseObfError::InvalidMove { .. } => "move",
                ParseObfError::InvalidScore { .. } => "score",
            };
            write!(out, "|error {category}|")?;
            write_hex(out, &err.to_string())?;
            write!(out, "|")?;
            if let Some(cause) = err.source() {
                write_hex(out, &cause.to_string())?;
            } else {
                write!(out, "-")?;
            }
            writeln!(out)
        }
    }
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    for input in ["", " \t\r\n", " \u{2000}%é", "XO", "A1", "あ", OPENING] {
        write_case(out, input)?;
    }
    for side in ["X", " O", "   X ", "\tO ", " O2", " x", "\u{00a0}X"] {
        for body in [
            "",
            ";",
            ";d3:1;% note",
            "; d3:+3; e6:+3; a1:1; h8:-6; a7:-8",
            "; ps:-1; PS:+2; pS:3",
            "; --:bogus; PS:+4; d3:+18; D3:-2",
            "; d3",
            "; No:12",
            "; a9:12",
            "; a|:12",
            "; d3:1:2",
            "; é1:9",
            "; \u{2000}d3\u{2001} : -2 ",
        ] {
            write_case(out, &format!(" {OPENING}{side}{body}"))?;
        }
    }
    for score in [
        "2147483647",
        "2147483648",
        "-2147483648",
        "-2147483649",
        "++0",
        "++-27",
        "+-0",
        "1_000",
        "0x42",
        "99999999999",
        "999999999x",
        "9999999999x",
        "-999999999x",
        "-9999999999x",
        "-+1",
        "2147483647x",
        "2147483648x",
        "-2147483649x",
        "000000000000000000000000012",
        "",
        "+",
        "-",
        " \u{2007}10 \t",
        "１２",
    ] {
        write_case(out, &format!("{OPENING} X; d3:{score}"))?;
        write_case(out, &format!("{OPENING} O; Ps:{score}"))?;
    }
    for (square, ch) in [(0, 'O'), (63, 'X'), (12, '?'), (60, 'é')] {
        let board = OPENING
            .chars()
            .enumerate()
            .map(|(index, cell)| if index == square { ch } else { cell })
            .collect::<String>();
        for side in ['X', 'O'] {
            write_case(out, &format!("{board} {side}"))?;
        }
    }
    Ok(())
}
