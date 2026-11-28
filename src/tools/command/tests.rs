use super::job::MAX_CAPTURE_BYTES;
use super::*;
use crate::test_support::TempDir;
use nix::sys::signal::kill;
use nix::unistd::Pid;
use std::path::Path;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

async fn shell(
    context: ToolContext<'_>,
    script: &str,
    args: &[&Path],
    wait_seconds: Option<f64>,
) -> Value {
    let mut argv = vec!["sh".into(), "-c".into(), script.into(), "sh".into()];
    argv.extend(args.iter().map(|path| path.to_str().unwrap().to_string()));
    run(Run { argv, wait_seconds }, context).await
}

async fn control(
    id: u64,
    action: Action,
    wait_seconds: Option<f64>,
    context: ToolContext<'_>,
) -> Value {
    super::control(
        Control {
            command_id: id,
            action,
            wait_seconds,
        },
        context,
    )
    .await
}

fn command_id(result: &Value) -> u64 {
    result["command_id"].as_u64().unwrap()
}

async fn wait_for_stdout(context: ToolContext<'_>, id: u64, text: &str) -> Value {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            let result = control(id, Action::Wait, Some(0.02), context).await;
            if result["status"] != "running" || result["stdout"] == text {
                return result;
            }
            assert_eq!(result["stdout"], "");
        }
    })
    .await;
    result.expect("command did not emit its next stdout chunk")
}

async fn assert_process_exited(pid: i32) {
    let result = tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            let exists = kill(Pid::from_raw(pid), None);
            if exists == Err(nix::errno::Errno::ESRCH) {
                return;
            }
            #[cfg(target_os = "linux")]
            {
                // Some container init processes leave orphaned grandchildren as zombies.
                let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"));
                if stat.is_ok_and(|stat| {
                    stat.rsplit_once(") ")
                        .is_some_and(|(_, tail)| tail.starts_with("Z "))
                }) {
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    result.expect("process survived group cleanup");
}

#[tokio::test]
async fn captures_command_output_with_shape_for_history() {
    let result = shell(
        ToolContext::new(&Jobs::default(), None),
        "printf hello; printf problem >&2",
        &[],
        None,
    )
    .await;

    assert_eq!(result["status"], "finished");
    assert_eq!(result["exit_code"], 0);
    assert_eq!(result["stdout"], "hello");
    assert_eq!(result["stderr"], "problem");
    assert_eq!(result["stdout_bytes_omitted"], 0);
    assert_eq!(result["stderr_bytes_omitted"], 0);
}

#[tokio::test]
async fn command_ids_remain_distinct_across_turns() {
    let first = shell(
        ToolContext::new(&Jobs::default(), None),
        "exit 0",
        &[],
        None,
    )
    .await;
    let second = shell(
        ToolContext::new(&Jobs::default(), None),
        "exit 0",
        &[],
        None,
    )
    .await;
    assert_ne!(command_id(&first), command_id(&second));
}

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
    let result = shell(ToolContext::new(&Jobs::default(), None), &script, &[], None).await;

    assert_eq!(result["status"], "finished");
    assert_eq!(result["exit_code"], 0);
    for (stream, omitted, old, latest, fill) in [
        (
            "stdout",
            "stdout_bytes_omitted",
            old_stdout,
            latest_stdout,
            b'x',
        ),
        (
            "stderr",
            "stderr_bytes_omitted",
            old_stderr,
            latest_stderr,
            b'y',
        ),
    ] {
        let output = result[stream].as_str().unwrap();
        assert_eq!(output.len(), MAX_CAPTURE_BYTES);
        let retained = output.strip_suffix(latest).unwrap();
        assert!(retained.bytes().all(|byte| byte == fill));
        assert_eq!(result[omitted], old.len() + latest.len());
    }
}

#[tokio::test]
async fn finished_command_cleans_redirected_background_child() {
    let result = shell(
        ToolContext::new(&Jobs::default(), None),
        "sleep 999 >/dev/null 2>&1 & echo $!",
        &[],
        None,
    )
    .await;

    assert_eq!(result["status"], "finished");
    let pid = result["stdout"].as_str().unwrap().trim().parse().unwrap();
    assert_process_exited(pid).await;
}

#[tokio::test]
async fn dropping_turn_state_stops_process_groups() {
    let directory = TempDir::new().unwrap();
    let pid_path = directory.0.join("child-pid");
    let script = "sleep 999 >/dev/null 2>&1 & echo $! > \"$1\"; wait";
    let jobs = Jobs::default();
    let context = ToolContext::new(&jobs, None);
    let started = shell(context, script, &[&pid_path], Some(0.0)).await;
    let descendant = tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            let text = std::fs::read_to_string(&pid_path);
            if let Ok(text) = text
                && let Ok(pid) = text.trim().parse::<i32>()
            {
                return pid;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();

    drop(jobs);

    assert_process_exited(started["pid"].as_i64().unwrap() as i32).await;
    assert_process_exited(descendant).await;
}

#[tokio::test]
async fn running_commands_and_waits_deliver_each_chunk_once() {
    let jobs = Jobs::default();
    let context = ToolContext::new(&jobs, None);
    let directory = TempDir::new().unwrap();
    let release = directory.0.join("write-two");
    let script =
        "printf one; while [ ! -f \"$1\" ]; do sleep 0.01; done; printf two; exec sleep 999";
    let started = shell(context, script, &[&release], Some(0.0)).await;
    let id = command_id(&started);
    assert_eq!(started["status"], "running");

    let first = wait_for_stdout(context, id, "one").await;
    assert_eq!(first["stdout"], "one");
    let empty = control(id, Action::Wait, Some(0.0), context).await;
    assert_eq!(empty["stdout"], "");
    assert_eq!(empty["stderr"], "");

    std::fs::write(release, []).unwrap();
    let later = wait_for_stdout(context, id, "two").await;
    assert_eq!(later["stdout"], "two");
    let killed = control(id, Action::Kill, None, context).await;
    assert_eq!(killed["status"], "killed");
    assert_eq!(killed["stdout"], "");
    assert_eq!(killed["stderr"], "");
}

#[tokio::test]
async fn control_stops_one_command_during_a_wait_and_preserves_the_other() {
    let jobs = Jobs::default();
    let context = ToolContext::new(&jobs, None);
    let first = shell(context, "printf first; exec sleep 999", &[], Some(0.0)).await;
    let second = shell(context, "printf second; exec sleep 999", &[], Some(0.0)).await;
    let first_id = command_id(&first);
    let second_id = command_id(&second);
    assert_ne!(first_id, second_id);
    assert_eq!(
        wait_for_stdout(context, first_id, "first").await["stdout"],
        "first"
    );
    assert_eq!(
        wait_for_stdout(context, second_id, "second").await["stdout"],
        "second"
    );

    let wait = control(first_id, Action::Wait, Some(60.0), context);
    let stop = control(first_id, Action::Kill, None, context);
    let (waited, stopped) =
        tokio::time::timeout(TEST_TIMEOUT, async { tokio::join!(biased; wait, stop) })
            .await
            .unwrap();
    assert_eq!(waited["status"], "killed");
    assert_eq!(stopped["status"], "killed");

    let remaining = control(second_id, Action::Wait, Some(0.0), context).await;
    assert_eq!(remaining["status"], "running");
    assert_eq!(jobs.pending_ids(), [second_id]);
    let stopped = control(second_id, Action::Kill, None, context).await;
    assert_eq!(stopped["status"], "killed");
}

#[tokio::test]
async fn kill_reports_finished_if_command_was_already_reaped() {
    let jobs = Jobs::default();
    let context = ToolContext::new(&jobs, None);
    let started = shell(context, "printf after", &[], Some(0.0)).await;
    let pid = started["pid"].as_i64().unwrap() as i32;
    assert_process_exited(pid).await;

    let result = control(command_id(&started), Action::Kill, None, context).await;
    assert_eq!(result["status"], "finished");
    assert_eq!(result["exit_code"], 0);
    assert_eq!(result["stdout"], "after");
}

#[tokio::test]
async fn a_poll_keeps_a_partial_utf8_character_for_the_next_result() {
    let jobs = Jobs::default();
    let context = ToolContext::new(&jobs, None);
    let directory = TempDir::new().unwrap();
    let release = directory.0.join("finish-character");
    let script =
        "printf 'first\\342\\202'; while [ ! -f \"$1\" ]; do sleep 0.01; done; printf '\\254'";
    let started = shell(context, script, &[&release], Some(0.0)).await;
    let id = command_id(&started);
    assert_eq!(
        wait_for_stdout(context, id, "first").await["stdout"],
        "first"
    );

    std::fs::write(release, []).unwrap();
    let result = control(id, Action::Wait, Some(TEST_TIMEOUT.as_secs_f64()), context).await;
    assert_eq!(result["status"], "finished");
    assert_eq!(result["stdout"], "€");
}
