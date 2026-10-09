//! Integration test for the GGS play mode.
//!
//! Runs the `cli` binary as a subprocess against a mock TCP server that
//! replays a scripted conversation.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::thread;

const EXPECTED_STARTUP: &[&str] = &[
    "alice",
    "testpass",
    "tell /os open 0",
    "vt100 -",
    "var _ready",
    "var _alert",
    "notify + /os",
    "tell /os client -",
    "tell /os trust +",
    "tell /os vt100 -",
    "tell /os aform",
    "tell /os dform size != 8 | anti | mt1 < 60 | mm1 != 0 | !ml1 | mm2 != 1 | !ma2",
    "tell /os request +",
    "tell /os open 1",
];
fn crlf(lines: &[&str]) -> String {
    let mut s = lines.join("\r\n");
    s.push_str("\r\n");
    s
}

fn join_chunk(to_move: &str) -> String {
    crlf(&[
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
        to_move,
        "READY",
    ])
}

#[test]
fn ggs_exits_when_os_restarts_during_search() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
    let addr = listener.local_addr().expect("local addr");

    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        let mut writer = stream.try_clone().expect("clone stream");
        let mut reader = BufReader::new(stream).lines();

        for expected in EXPECTED_STARTUP {
            assert_eq!(reader.next().unwrap().unwrap(), *expected);
        }

        writer
            .write_all(
                crlf(&[
                    ": + /os 1",
                    "READY",
                    "/os: + .5 1720 bob 00:05//00:02 8b U 1720 alice",
                    "READY",
                ])
                .as_bytes(),
            )
            .unwrap();
        assert_eq!(reader.next().unwrap().unwrap(), "tell /os accept .5");

        writer
            .write_all(crlf(&["/os: + match .5 1720 alice 1720 bob 8b U", "READY"]).as_bytes())
            .unwrap();
        writer
            .write_all(join_chunk("|* to move").as_bytes())
            .unwrap();
        writer
            .write_all(crlf(&["\x07: - /os 1", "READY"]).as_bytes())
            .unwrap();
    });

    let bin = env!("CARGO_BIN_EXE_cli");
    let child = Command::new(bin)
        .args([
            "ggs",
            "--user",
            "alice",
            "--password",
            "testpass",
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

    server.join().expect("server thread panicked");
    let output = child.wait_with_output().expect("wait for cli");
    assert!(
        !output.status.success(),
        "cli should exit after /os restart"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("/os went offline; reconnect required"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn plain_and_redirected_modes_preserve_telnet_lines() {
    for explicit_plain in [false, true] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(15)))
                .unwrap();
            let mut writer = stream.try_clone().unwrap();
            let mut reader = BufReader::new(stream).lines();
            for expected in EXPECTED_STARTUP {
                assert_eq!(reader.next().unwrap().unwrap(), *expected);
            }
            for expected in [" help  ", "", "tell bob 日本語 | @ \\ {", "quit"] {
                assert_eq!(reader.next().unwrap().unwrap(), expected);
            }
            writer
                .write_all(
                    crlf(&["MOTD", "", "unknown service output\x1b[31m", "READY"]).as_bytes(),
                )
                .unwrap();
        });
        let mut command = Command::new(env!("CARGO_BIN_EXE_cli"));
        command.args([
            "ggs",
            "--user",
            "alice",
            "--password",
            "testpass",
            "--host",
            &addr.ip().to_string(),
            "--port",
            &addr.port().to_string(),
            "--hash-size",
            "2",
            "--threads",
            "1",
        ]);
        if explicit_plain {
            command.arg("--plain");
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(" help  \n\ntell bob 日本語 | @ \\ {\nquit\n".as_bytes())
            .unwrap();
        server.join().unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "MOTD\n\nunknown service output\x1b[31m\nREADY\n"
        );
        assert!(output.stderr.is_empty());
    }
}
