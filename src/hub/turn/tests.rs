use super::*;
use crate::test_support::TempDir;

fn some_call(id: u64, name: &str) -> Message {
    Message::Call(ToolCall {
        id,
        name: name.into(),
        arguments: "{}".into(),
    })
}

#[test]
fn calls_left_without_results_get_an_honest_one() {
    let kept = vec![
        Message::User("go".into()),
        Message::Thinking(String::new()),
        some_call(3, "apply_patch"),
        some_call(4, "run_command"),
    ];
    let answers = unanswered(&kept);
    let answered: Vec<_> = answers
        .iter()
        .map(|message| match message {
            Message::Result(result) => (result.id, result.name.as_str()),
            _ => (0, ""),
        })
        .collect();
    assert_eq!(answered, [(3, "apply_patch"), (4, "run_command")]);
    let Message::Result(first) = &answers[0] else {
        unreachable!()
    };
    assert!(first.content.contains("may or may not have run"));
    assert!(unanswered(&kept[..2]).is_empty());
}

#[test]
fn a_delivery_is_kept_once_however_often_it_is_retried() -> Result<()> {
    let dir = TempDir::new()?;
    let room = Room::at(dir.0.join("room"))?;
    room.append(&[
        Message::User("first".into()),
        Message::Thinking(String::new()),
        some_call(3, "read_file"),
    ])?;
    // Results answer the dangling call.
    let results = vec![result(3, "read_file", "x".into())];
    keep_delivery(&room, 7, results.clone())?;
    keep_delivery(&room, 7, results)?;
    assert_eq!(room.conversation()?.len(), 4);
    // A new request after a step whose calls never got results answers them first.
    room.append(&[
        Message::Thinking(String::new()),
        some_call(6, "run_command"),
    ])?;
    keep_delivery(&room, 8, vec![Message::User("next".into())])?;
    let conversation = room.conversation()?;
    let answer = Message::Result(ToolResult {
        id: 6,
        name: "run_command".into(),
        content: wording::unrecorded(),
    });
    assert_eq!(conversation[6], answer);
    assert_eq!(conversation[7], Message::User("next".into()));
    Ok(())
}

#[test]
fn a_restart_keeps_as_much_of_the_turn_in_progress_as_fits() -> Result<()> {
    let dir = TempDir::new()?;
    let room = Room::at(dir.0.join("room"))?;
    room.append(&[Message::User("old".into()), Message::Answer("done".into())])?;
    room.restart(&[Message::Answer("You asked for the old thing.".into())])?;
    room.append(&[Message::Input("data".into()), Message::User("now".into())])?;
    for step in 0..3 {
        let content = format!("step {step}");
        let id = room.next()? + 1;
        room.append(&[
            Message::Thinking(content.clone()),
            some_call(id, "read_file"),
        ])?;
        room.append(&[result(id, "read_file", content)])?;
    }
    // Entries 1 and 2 are set aside, 3 is the restart message, and the turn is 4 to 14.
    let split = |most: usize| {
        let fits = |kept: &[Message]| Ok(kept.len() <= most);
        let (folded, kept) = cutover(room.entries()?, fits)?;
        let last = folded.last().map(|entry| entry.number);
        Ok::<_, eyre::Report>((last, kept))
    };
    let request = [Message::Input("data".into()), Message::User("now".into())];
    // The whole turn fits: only what came before it is folded.
    let (last, kept) = split(11)?;
    assert_eq!((last, kept.len()), (Some(3), 11));
    // The request and the newest step.
    let (last, kept) = split(5)?;
    assert_eq!(last, Some(11));
    assert_eq!(kept[..2], request);
    assert_eq!(kept[2], Message::Thinking("step 2".into()));
    assert_eq!(kept.len(), 5);
    // The request alone, then nothing; the whole turn is folded either way.
    let (last, kept) = split(2)?;
    assert_eq!((last, kept), (Some(14), request.to_vec()));
    let (last, kept) = split(1)?;
    assert_eq!((last, kept), (Some(14), vec![]));
    Ok(())
}

#[test]
fn a_third_identical_call_with_an_unchanged_result_is_refused() {
    let call = |name: &str, arguments| FunctionCall {
        name: name.into(),
        arguments,
    };
    let grep = call(
        "run_command",
        serde_json::json!({"argv": ["grep", "x", "f"]}),
    );
    let ids = std::cell::Cell::new(0);
    let step = |made: &FunctionCall, content: &str| {
        let thinking = Message::Thinking(String::new());
        ids.set(ids.get() + 1);
        let id = ids.get();
        vec![
            thinking,
            Message::Call(made.with_id(id)),
            result(id, &made.name, content.into()),
        ]
    };
    let twice = [step(&grep, "none"), step(&grep, "none")].concat();
    assert!(repeats(&twice, &grep));
    // A changed result, another call between, or other arguments let it run.
    assert!(!repeats(
        &[step(&grep, "none"), step(&grep, "one")].concat(),
        &grep
    ));
    let edit = call("apply_patch", serde_json::json!({"patch": "p"}));
    let between = [step(&grep, "none"), step(&edit, "ok"), step(&grep, "none")].concat();
    assert!(!repeats(&between, &grep));
    assert!(!repeats(&step(&grep, "none"), &grep));
    let other = call(
        "run_command",
        serde_json::json!({"argv": ["grep", "y", "f"]}),
    );
    assert!(!repeats(&twice, &other));
    // A command's handle, process, and running time differ every time; a running one is
    // being waited on.
    let ran = |pid: u32, status: &str| {
        let fields = serde_json::json!({"command_id": pid, "pid": pid, "status": status});
        fields.to_string()
    };
    let finished = [
        step(&grep, &ran(1, "finished")),
        step(&grep, &ran(2, "finished")),
    ];
    assert!(repeats(&finished.concat(), &grep));
    let running = [
        step(&grep, &ran(1, "running")),
        step(&grep, &ran(2, "running")),
    ];
    assert!(!repeats(&running.concat(), &grep));
}

#[test]
fn a_room_keeps_its_owner_until_it_restarts() {
    let place = Place {
        machine: "ssh-ed25519 AAAA".into(),
        cwd: "/".into(),
    };
    assert_eq!(owner(&place, 3), owner(&place, 3));
    assert_ne!(owner(&place, 3), owner(&place, 9));
}
