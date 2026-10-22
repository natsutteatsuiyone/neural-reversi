use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::process::Command;
use std::sync::mpsc;

use super::*;

#[test]
fn os_restart_returns_while_search_is_in_progress() {
    const CHILD: &str = "REVERSI_GGS_RESTART_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ggs::lifecycle_tests::os_restart_returns_while_search_is_in_progress",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(40);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "GGS restart child failed: {status}");
                return;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("GGS restart hung while a search was active");
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (started_tx, started_rx) = mpsc::channel();
    let (returned_tx, returned_rx) = mpsc::channel();
    let released = Arc::new((Mutex::new(false), Condvar::new()));
    let observed = Arc::new(AtomicBool::new(false));
    let observer: Arc<search::SearchProgressCallback> = Arc::new({
        let released = Arc::clone(&released);
        let observed = Arc::clone(&observed);
        move |progress| {
            if progress.depth < 4 || progress.nodes == 0 || observed.swap(true, Ordering::AcqRel) {
                return;
            }
            started_tx.send(()).unwrap();
            // Keep the real search inside its progress callback until the event
            // loop has returned, so an early/completed search cannot satisfy this test.
            let (lock, changed) = &*released;
            let done = lock.lock().unwrap();
            drop(changed.wait_while(done, |done| !*done).unwrap());
        }
    });
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        for expected in startup_lines("alice", "testpass", 1, &[]).unwrap() {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            assert_eq!(line.trim_end(), expected);
        }
        stream
            .write_all(
                concat!(
                    ": + /os 1\nREADY\n",
                    "/os: + match .5 1720 bob 1720 alice 8b U\nREADY\n",
                    "/os: join .5 8b K?\n|1 move(s)\n|  1: D3\n",
                    "|alice    (1720.0 O) 01:00:00,0:0//00:02,0:0\n",
                    "|bob      (1720.0 *) 01:00:00,0:0//00:02,0:0\n|\n",
                    "|   A B C D E F G H\n",
                    "| 1 - - - - - - - - 1\n| 2 - - - - - - - - 2\n",
                    "| 3 - - - * - - - - 3\n| 4 - - - * * - - - 4\n",
                    "| 5 - - - * O - - - 5\n| 6 - - - - - - - - 6\n",
                    "| 7 - - - - - - - - 7\n| 8 - - - - - - - - 8\n",
                    "|   A B C D E F G H\n|\n|O to move\nREADY\n",
                )
                .as_bytes(),
            )
            .unwrap();
        started_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("real search progress");
        stream.write_all(b"\x07: - /os 1\nREADY\n").unwrap();
        // Keep the socket open: EOF must not be the reason the event loop exits.
        returned_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("GGS loop returned");
    });
    let config = EngineConfig {
        hash_size: 2,
        level: 21,
        threads: Some(1),
        eval_file: None,
        eval_sm_file: None,
    };
    let result = run_ggs_inner(
        GgsOptions {
            host: "127.0.0.1",
            port,
            user: "alice",
            password: "testpass",
            max_games: 1,
            ponder: false,
            plain: true,
            join_tournaments: false,
            startup_commands: &[],
            log_file: None,
        },
        &config,
        Some(observer),
    );
    returned_tx.send(()).unwrap();
    assert!(
        observed.load(Ordering::Acquire),
        "no search progress observed"
    );
    assert_eq!(
        result.unwrap_err().to_string(),
        "/os went offline; reconnect required"
    );
    let (lock, changed) = &*released;
    *lock.lock().unwrap() = true;
    changed.notify_all();
    server.join().unwrap();
}
