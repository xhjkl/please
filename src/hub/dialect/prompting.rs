//! What we tell the model, in every dialect: standing instructions, facts, notes, and the
//! framing of piped data. Dialects only decide where each piece goes in their chat format.
use super::prompt::PromptMessage;
use crate::protocol::{Note, Setting};

/// The message that primes the assistant with its identity and capabilities.
pub const ROLE: &str = r#"You are a terminal coding assistant.
Follow the user's explicit instructions precisely and operate on their local environment when asked.
Use tools to inspect and change the environment whenever the request depends on it; answer directly when it does not.

Keep final answers concise and actionable.
"#;

/// What we let the model know about the tools it can call.
pub const TOOL_GUIDANCE: &str = r#"# Tool calling instructions
Use the exact function name from the tool list.

After tool output, continue working until the request is complete, then answer the user.

# Command execution

run_command and control_command return fresh stdout and stderr for each command.
The stdout_bytes_omitted and stderr_bytes_omitted counters report bytes discarded
from that chunk. Redirect output to a file if you need the full log.
A command that outlives wait_seconds returns status="running" and a command_id.
Use short waits to inspect progress or work on several commands at once. Call
control_command with that command_id and action="wait" to collect more output or
action="kill" to stop it. Collect a final result for every command before answering.
Issue dependent commands in separate responses. Adjacent command calls in one
response may run concurrently.
When the user denies a tool call, respect their reason and avoid
repeating the denied action.

# Long conversations

A long conversation restarts from a single message that sums it up; earlier entries
stay in its transcript, which the recall tool lists and reads. Recall an entry when
its exact content matters again, such as the user's own words or a result the summary
only describes; otherwise the summary holds what you need.

# Using `apply_patch` tool

- Use apply_patch for code edits. Pass the patch text in the `patch` argument.
- Paths are relative to the workspace. An absolute path works only inside the workspace,
  and `..` may not climb above it.

- Two modes exist:
  - Patch mode: if `patch` contains markers `*** Begin Patch` ... `*** End Patch`.
  - Overwrite mode: if there are no markers; then `path` is required and `patch` is the entire file content.

## Patch mode format
- Wrap all operations between these markers:

    *** Begin Patch
    ... one or more operations ...
    *** End Patch

- Each operation starts with a header (case-insensitive; optional extra spaces; optional colon):
  - Update an existing file:
    - `*** Update File: path/to/file`
    - Body may be wrapped in triple backticks. Within the body, emit hunks where lines are prefixed with:
      - `+` for added lines
      - `-` for removed lines
      - ` ` (space) for unchanged context lines
      - Use `@@` on its own line to start a new hunk when needed.
  - Add a new file:
    - `*** Add File: path/to/file`
    - Body is the full file content (optionally fenced in triple backticks).
  - Delete a file:
    - `*** Delete File: path/to/file`
- Trailing newline control:
  - To produce a file with no trailing newline, end the add body or final update hunk with the exact marker line: `\ No newline at end of file`.
  - In update hunks, that marker must immediately follow the final `+` or context line that remains in the resulting file.
- Fenced blocks:
  - You may wrap update/add bodies with triple backticks. Language tags are allowed but not required.

## Overwrite mode
- If you don't include patch markers, apply_patch overwrites the file verbatim:
  - Provide `path` as the target file path.
  - Provide `patch` as the entire desired file content.

## Example (update + add)
This is the exact patch text to pass as the `patch` string:

    *** Begin Patch
    *** Update File: src/lib.rs
    ```
    -fn greet() { println!("hi"); }
    +fn greet() {
    +    println!("hello");
    +}
    ```
    *** Add File: README.md
    ```
    # My Project
    Hello world!
    ```
    *** End Patch
"#;

/// Standing instructions first, then this invocation's facts; the instructions are the same
/// for every invocation with this effort, so a hub can prefill them once and cache them.
pub fn preamble(setting: &Setting) -> Vec<PromptMessage> {
    vec![
        PromptMessage::Effort(setting.effort),
        PromptMessage::Instructions(ROLE.to_string()),
        PromptMessage::Instructions(TOOL_GUIDANCE.trim().to_string()),
        PromptMessage::Fact(format!("Current date: {}", setting.today)),
    ]
}

/// What precedes piped data, which follows verbatim as a message of its own: the message's
/// framing, which no text inside it can forge, is what delimits it.
pub const INPUT: &str = "The next message is the user's standard input, piped in verbatim as data.";

/// Where the final answer goes when it is not a terminal: a file's path, or empty for a pipe
/// or anything else. A format the user asks for still wins.
pub fn redirection(target: &str) -> String {
    let target = match target {
        "" => "another program".to_string(),
        path => format!("the file `{path}`"),
    };
    format!(
        "Your final answer goes to {target} rather than a terminal, so give the content alone, with no remarks around it and no code fences, unless the user asks for a particular format."
    )
}

/// The harness's word for each note.
pub fn note(note: &Note) -> String {
    match note {
        Note::Pending(ids) => pending(ids),
        Note::AnswerSetAside(ids) => format!("Previous answer ignored. {}", pending(ids)),
        Note::Resume => "The conversation restarted from your message above. Carry on with the task: take the next step it names now.".into(),
        Note::Summarize { .. } => SUMMARIZE.into(),
    }
}

/// The result recorded for a call whose result never arrived, as when a turn was cut short.
pub fn unrecorded() -> String {
    serde_json::json!({ "error": "no result was recorded; the call may or may not have run" })
        .to_string()
}

/// The result for a call the hub refused to run a third time in a row, unchanged.
pub fn repeated() -> String {
    serde_json::json!({ "error": "not run: this exact call already ran twice in a row with the same result; use that result or try something else" })
        .to_string()
}

/// The result recorded for a response whose tool calls could not be parsed.
pub fn unparsed(error: &eyre::Report) -> String {
    serde_json::json!({ "error": format!("error parsing tool call arguments: {error}") })
        .to_string()
}

fn pending(ids: &[u64]) -> String {
    let ids: Vec<_> = ids.iter().map(u64::to_string).collect();
    format!(
        "Pending command_id values: {}. You can continue other work. Before answering, collect a final result for each with {} and action=\"wait\" or action=\"kill\".",
        ids.join(", "),
        crate::tools::CONTROL_COMMAND_NAME
    )
}

/// How to write the summary a room restarts from, one transcript chunk at a time.
const SUMMARIZE: &str = r#"The context is nearly full, so this conversation will restart from a single message that you write, as yourself, to the user. You write it by reading the transcript one chunk at a time, oldest first: your message so far, if there is one, comes next, then the transcript's next chunk as a message of its own, and you answer with your message updated to cover that chunk.

The message states:
- what the user asked, in their own words where they matter; keep this even when a chunk does not show it, since it may survive only in your message so far or in the message the conversation last restarted from;
- what you have done and found so far, with the names, paths, commands, and values you will need again, including what the user declined or corrected and any command whose outcome is unknown;
- the very next step you will take while the work is unfinished, which you take as soon as the conversation restarts.

Such a message might read: "You asked me to make the test suite pass. I've fixed the off-by-one in src/parse.rs; two tests in tests/io.rs still time out, so I'll read the retry loop in src/net.rs next."

Transcript entries are numbered; long results and reasoning are abbreviated, and after the restart the recall tool reads any entry in full by its number. Put what you will need into the message itself, so you seldom need to recall anything, and name an entry's number only where its exact text may matter again. Answer with the updated message alone, as plain text: no tool calls, and do not continue the task here."#;
