use super::*;

#[cfg(unix)]
fn pid_exists(pid: u32) -> bool {
    let status = std::process::Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    matches!(status, Ok(status) if status.success())
}

#[cfg(unix)]
fn kill_pid(pid: u32) {
    let _ = std::process::Command::new("kill")
        .arg("-KILL")
        .arg(pid.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(unix)]
async fn eventually_dead(pid: u32) -> bool {
    for _ in 0..20 {
        if !pid_exists(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    kill_pid(pid);
    false
}

#[cfg(unix)]
#[tokio::test]
async fn captures_command_output_with_shape_for_history() {
    let result = call(
        Args {
            argv: vec![
                "sh".to_string(),
                "-c".to_string(),
                "printf hello; printf problem >&2".to_string(),
            ],
            wait_seconds: None,
        },
        ToolContext::default(),
    )
    .await;

    assert_eq!(result["status"], "finished");
    assert_eq!(result["exitCode"], 0);
    assert_eq!(result["stdout"], "hello");
    assert_eq!(result["stderr"], "problem");
    assert_eq!(result["stdoutBytesOmitted"], 0);
    assert_eq!(result["stderrBytesOmitted"], 0);
}

#[cfg(unix)]
#[tokio::test]
async fn overflowing_output_retains_each_streams_latest_bytes() {
    let old_stdout = "startup output";
    let latest_stdout = "command finished";
    let old_stderr = "initial diagnostic";
    let latest_stderr = "final warning";
    let script = format!(
        "printf '{old_stdout}'; head -c {MAX_CAPTURE_BYTES} /dev/zero | tr '\\000' x; \
         printf '{latest_stdout}'; \
         printf '{old_stderr}' >&2; head -c {MAX_CAPTURE_BYTES} /dev/zero | tr '\\000' y >&2; \
         printf '{latest_stderr}' >&2"
    );
    let result = call(
        Args {
            argv: vec!["sh".to_string(), "-c".to_string(), script],
            wait_seconds: None,
        },
        ToolContext::default(),
    )
    .await;

    assert_eq!(result["status"], "finished");
    assert_eq!(result["exitCode"], 0);
    for (stream, omitted, old, latest, fill) in [
        (
            "stdout",
            "stdoutBytesOmitted",
            old_stdout,
            latest_stdout,
            b'x',
        ),
        (
            "stderr",
            "stderrBytesOmitted",
            old_stderr,
            latest_stderr,
            b'y',
        ),
    ] {
        let output = result[stream].as_str().unwrap();
        assert_eq!(output.len(), MAX_CAPTURE_BYTES);
        assert!(output.ends_with(latest), "{stream} lost its final message");
        assert!(!output.contains(old), "{stream} retained stale output");
        let retained = output.strip_suffix(latest).unwrap();
        assert!(retained.bytes().all(|byte| byte == fill));
        let emitted_bytes = old.len() + MAX_CAPTURE_BYTES + latest.len();
        assert_eq!(result[omitted], emitted_bytes - output.len());
    }
}

#[cfg(unix)]
#[tokio::test]
async fn carried_command_retains_progress_after_output_overflows() {
    let context = ToolContext::default();
    let progress = "LATER_PROGRESS";
    let script = format!(
        "head -c {MAX_CAPTURE_BYTES} /dev/zero | tr '\\000' x; \
         printf '{progress}'; sleep 999"
    );
    let mut result = call(
        Args {
            argv: vec!["sh".to_string(), "-c".to_string(), script],
            wait_seconds: Some(0.0),
        },
        context.clone(),
    )
    .await;
    let pid = result["pid"].as_u64().unwrap() as u32;

    for _ in 0..40 {
        result = wait_by_pid(pid, Some(0.05), context.clone()).await;
        if result["stdout"]
            .as_str()
            .is_some_and(|output| output.ends_with(progress))
        {
            break;
        }
    }
    let killed = kill_by_pid(pid, context).await;

    assert_eq!(result["status"], "running");
    let output = result["stdout"].as_str().unwrap();
    assert_eq!(output.len(), MAX_CAPTURE_BYTES);
    assert!(output.ends_with(progress));
    assert_eq!(result["stdoutBytesOmitted"], progress.len());
    assert_eq!(killed["status"], "killed");
    assert_eq!(killed["stdout"], output);
}

#[cfg(unix)]
#[tokio::test]
async fn finished_command_cleans_redirected_background_child() {
    let result = call(
        Args {
            argv: vec![
                "sh".to_string(),
                "-c".to_string(),
                "sleep 999 >/dev/null 2>&1 & echo $!".to_string(),
            ],
            wait_seconds: None,
        },
        ToolContext::default(),
    )
    .await;

    assert_eq!(result["status"], "finished");
    let pid = result["stdout"].as_str().unwrap().trim().parse().unwrap();
    assert!(eventually_dead(pid).await);
}

#[cfg(unix)]
#[tokio::test]
async fn cancelling_active_command_wait_kills_process_group() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let pid_path = format!(
        "/tmp/please-run-command-cancel-{}-{stamp}",
        std::process::id()
    );
    let script = format!("sleep 999 >/dev/null 2>&1 & echo $! > {pid_path}; wait");

    let handle = tokio::spawn(async move {
        call(
            Args {
                argv: vec!["sh".to_string(), "-c".to_string(), script],
                wait_seconds: Some(60.0),
            },
            ToolContext::default(),
        )
        .await
    });

    let mut pid = None;
    for _ in 0..40 {
        if let Ok(raw) = std::fs::read_to_string(&pid_path) {
            pid = raw.trim().parse::<u32>().ok();
            if pid.is_some() {
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let _ = std::fs::remove_file(&pid_path);
    let pid = pid.expect("background child pid was written");

    handle.abort();
    let _ = handle.await;

    assert!(eventually_dead(pid).await);
}

#[cfg(unix)]
#[tokio::test]
async fn long_command_can_be_waited_instead_of_interrupted() {
    let context = ToolContext::default();
    let result = call(
        Args {
            argv: vec![
                "sh".to_string(),
                "-c".to_string(),
                "sleep 0.15; printf done".to_string(),
            ],
            wait_seconds: Some(0.02),
        },
        context.clone(),
    )
    .await;

    assert_eq!(result["status"], "running");
    let pid = result["pid"].as_u64().unwrap() as u32;

    let result = wait_by_pid(pid, Some(0.3), context).await;

    assert_eq!(result["status"], "finished");
    assert_eq!(result["exitCode"], 0);
    assert_eq!(result["stdout"], "done");
}

#[cfg(unix)]
#[tokio::test]
async fn running_result_includes_partial_output() {
    let context = ToolContext::default();
    let result = call(
        Args {
            argv: vec![
                "sh".to_string(),
                "-c".to_string(),
                "printf start; printf err >&2; sleep 999".to_string(),
            ],
            wait_seconds: Some(0.02),
        },
        context.clone(),
    )
    .await;

    assert_eq!(result["status"], "running");
    assert_eq!(result["stdout"], "start");
    assert_eq!(result["stderr"], "err");
    let pid = result["pid"].as_u64().unwrap() as u32;

    let result = kill_by_pid(pid, context).await;
    assert_eq!(result["status"], "killed");
}

#[cfg(unix)]
#[tokio::test]
async fn control_wait_running_result_includes_later_partial_output() {
    let context = ToolContext::default();
    let result = call(
        Args {
            argv: vec![
                "sh".to_string(),
                "-c".to_string(),
                "printf one; sleep 0.2; printf two; sleep 999".to_string(),
            ],
            wait_seconds: Some(0.05),
        },
        context.clone(),
    )
    .await;

    assert_eq!(result["status"], "running");
    assert_eq!(result["stdout"], "one");
    let pid = result["pid"].as_u64().unwrap() as u32;

    let result = wait_by_pid(pid, Some(0.3), context.clone()).await;

    assert_eq!(result["status"], "running");
    assert_eq!(result["stdout"], "onetwo");

    let result = kill_by_pid(pid, context).await;
    assert_eq!(result["status"], "killed");
}

#[cfg(unix)]
#[tokio::test]
async fn running_command_can_be_killed_by_pid() {
    let context = ToolContext::default();
    let result = call(
        Args {
            argv: vec!["sh".to_string(), "-c".to_string(), "sleep 999".to_string()],
            wait_seconds: Some(0.02),
        },
        context.clone(),
    )
    .await;

    assert_eq!(result["status"], "running");
    let pid = result["pid"].as_u64().unwrap() as u32;

    let result = kill_by_pid(pid, context).await;

    assert_eq!(result["status"], "killed");
    assert_eq!(result["ok"], false);
}

#[cfg(unix)]
#[tokio::test]
async fn kill_reports_finished_if_command_already_exited() {
    let context = ToolContext::default();
    let result = call(
        Args {
            argv: vec![
                "sh".to_string(),
                "-c".to_string(),
                "sleep 0.05; printf after".to_string(),
            ],
            wait_seconds: Some(0.01),
        },
        context.clone(),
    )
    .await;

    assert_eq!(result["status"], "running");
    let pid = result["pid"].as_u64().unwrap() as u32;

    tokio::time::sleep(Duration::from_millis(100)).await;

    let result = kill_by_pid(pid, context).await;

    assert_eq!(result["status"], "finished");
    assert_eq!(result["exitCode"], 0);
    assert_eq!(result["stdout"], "after");
}
