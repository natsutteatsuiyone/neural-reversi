//! Integration test for the GGS play mode.
//!
//! Runs the `cli` binary as a subprocess against a mock TCP server that
//! replays a one-move scripted conversation. Gated with `#[ignore]` because
//! it requires the Neural Reversi weight files present at the workspace
//! root — without them `Search::new` panics.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::thread;

const SCRIPT_PATH: &str = "tests/fixtures/ggs/short-script.txt";
const SCRIPT_LINES: usize = 3;

fn crlf(lines: &[&str]) -> String {
    let mut s = lines.join("\r\n");
    s.push_str("\r\n");
    s
}

#[test]
#[ignore = "requires weight files at project root; run with: cargo test -p cli --test ggs_session -- --ignored"]
fn ggs_plays_opening_move_in_mock_session() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let addr = listener.local_addr().expect("local addr");

    // NOTE: the server thread does a blocking accept + blocking line reads.
    // If the child misbehaves and never sends a `tell /os play` line, this
    // test will deadlock on `server.join()`. Acceptable trade-off given the
    // test is `#[ignore]` and only run manually.
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let mut writer = stream.try_clone().expect("clone stream");
        let mut reader = BufReader::new(stream).lines();

        for _ in 0..SCRIPT_LINES {
            let _ = reader.next();
        }

        writer
            .write_all(crlf(&["/os: + match .5 1720 alice 1720 bob 8b U", "READY"]).as_bytes())
            .unwrap();

        let join_chunk = crlf(&[
            "/os: join .5 8b K?",
            "|0 move(s)",
            "|  0: PASS",
            "|alice    (1720.0 *) 00:05,0:0//00:02,0:0",
            "|bob      (1720.0 O) 00:05,0:0//00:02,0:0",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
            "READY",
        ]);
        writer.write_all(join_chunk.as_bytes()).unwrap();

        // Block until the engine responds with a `tell /os play ...` line.
        let reply = reader
            .find(|line| {
                line.as_ref()
                    .map(|l| l.starts_with("tell /os play "))
                    .unwrap_or(false)
            })
            .expect("expected a play line from the engine")
            .expect("line read error");

        // Send match-end so the client can finish gracefully.
        let end = crlf(&["/os: - match .5 1720 alice 1720 bob 8b U 0.00", "READY"]);
        let _ = writer.write_all(end.as_bytes());

        reply
    });

    let bin = env!("CARGO_BIN_EXE_cli");
    let mut child = Command::new(bin)
        .args([
            "ggs",
            "--user",
            "alice",
            "--script",
            SCRIPT_PATH,
            "--host",
            &addr.ip().to_string(),
            "--port",
            &addr.port().to_string(),
            "--level",
            "1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn cli");

    let reply = server.join().expect("server thread panicked");
    let _ = child.kill();
    let _ = child.wait();

    let mut parts = reply
        .strip_prefix("tell /os play ")
        .unwrap_or("")
        .split_whitespace();
    let id = parts.next().unwrap_or("");
    let mv = parts.next().unwrap_or("");
    assert_eq!(id, ".5", "unexpected match id in reply: {reply:?}");
    assert!(
        matches!(mv, "d3" | "c4" | "e6" | "f5"),
        "unexpected opening move: {mv:?} (full reply: {reply:?})"
    );
}
