use std::{
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
};

pub struct GtpEngine {
    process: Child,
}

impl GtpEngine {
    pub fn new(executable: &str, args: &[String], working_dir: Option<PathBuf>) -> io::Result<Self> {
        let exec_path = Path::new(executable);
        let default_working_dir = if let Some(parent) = exec_path.parent() {
            parent.to_path_buf()
        } else {
            PathBuf::from(".")
        };

        let working_dir = working_dir.unwrap_or(default_working_dir);

        let process = Command::new(executable)
            .args(args)
            .current_dir(&working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        Ok(GtpEngine {
            process,
        })
    }

    pub fn send_command(&mut self, command: &str) -> io::Result<String> {
        let stdin = self
            .process
            .stdin
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "Failed to open stdin"))?;

        writeln!(stdin, "{}", command)?;
        stdin.flush()?;

        let stdout =
            self.process.stdout.as_mut().ok_or_else(|| {
                io::Error::new(io::ErrorKind::BrokenPipe, "Failed to open stdout")
            })?;

        let mut reader = BufReader::new(stdout);
        let mut response = String::new();
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Process closed stdout",
                ));
            }

            if line.trim().is_empty() {
                break;
            }

            response.push_str(&line);
        }

        Ok(response)
    }

    pub fn name(&mut self) -> io::Result<String> {
        let name_response = self.send_command("name")?;
        let version_response = self.send_command("version")?;

        let name = if name_response.starts_with("= ") {
            name_response.strip_prefix("= ").unwrap().trim()
        } else {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Invalid name response: {}", name_response),
            ));
        };

        let version = if version_response.starts_with("= ") {
            let v = version_response.strip_prefix("= ").unwrap().trim();
            if !v.is_empty() && !v.starts_with('v') && !v.starts_with('V') {
                format!("v{}", v)
            } else {
                v.to_string()
            }
        } else if version_response.starts_with("? ") {
            "".to_string()
        } else {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Invalid version response: {}", version_response),
            ));
        };

        if !version.is_empty() {
            Ok(format!("{} {}", name, version))
        } else {
            Ok(name.to_string())
        }
    }

    pub fn clear_board(&mut self) -> io::Result<()> {
        let response = self.send_command("clear_board")?;
        if response.starts_with("=") {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to clear board: {}", response),
            ))
        }
    }

    pub fn play(&mut self, color: &str, mv: &str) -> io::Result<()> {
        let response = self.send_command(&format!("play {} {}", color, mv))?;
        if response.starts_with("=") {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to play move: {}", response),
            ))
        }
    }

    pub fn genmove(&mut self, color: &str) -> io::Result<String> {
        let response = self.send_command(&format!("genmove {}", color))?;
        if response.starts_with("=") {
            Ok(response.strip_prefix("=").unwrap().trim().to_string())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to generate move: {}", response),
            ))
        }
    }

    pub fn quit(&mut self) -> io::Result<()> {
        let _ = self.send_command("quit");
        Ok(())
    }
}

impl Drop for GtpEngine {
    fn drop(&mut self) {
        let _ = self.quit();
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}
