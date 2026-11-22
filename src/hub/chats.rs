//! Conversations the hub keeps on disk, one directory each under `~/.please/chats`.
//!
//! A conversation is a directory of plain-text entries named by eight-digit sequence numbers,
//! so they list in order. An entry is its kind on the first line (`user`, `input`, `thinking`,
//! `answer`, `call TOOL`, or `result TOOL`), a blank line, and the body verbatim. A body
//! longer than 4 KiB lives in the `...` subdirectory under the name of its bytes, and the
//! entry holds `[name]` instead. Restarting sets the entries so far aside as backups named
//! with a trailing `~` (`00000012~`), which `rm *~` erases; until then recall still reads them.
//! The harness's notes are not kept.
//!
//! Each place a probe runs in, a host, working directory, and terminal session, has its own
//! conversation: the directory is named after the place, which its `where` file spells out.
//! Everything is private to its owner, and one hub at a time works on a conversation.
use crate::protocol::{Call, Message, Place, ToolResult};
use eyre::{Result, WrapErr, bail, eyre};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

/// Where large bodies live, inside a conversation's directory.
const ATTACHMENTS: &str = "...";
/// Bodies up to this many bytes stay inside their entry.
const INLINE: usize = 4096;
/// The last request kept and where its messages begin, so a retried one is kept once.
const APPLIED: &str = ".applied";

/// One conversation's directory.
#[derive(Debug, Clone)]
pub struct Chat {
    dir: PathBuf,
}

/// An entry as it lies on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub number: u64,
    /// Set aside by a restart.
    pub backup: bool,
    pub message: Message,
}

impl Chat {
    /// The conversation of `place`, recording where it is from when it starts.
    pub fn open(place: &Place) -> Result<Self> {
        let Place { host, cwd, session } = place;
        let key = format!("please chat\0{host}\0{cwd}\0{session}");
        let home = std::env::var_os("HOME").unwrap_or_else(|| ".".into());
        let dir = PathBuf::from(home).join(".please/chats");
        let chat = Self::at(dir.join(super::digest::name_of(key.as_bytes())))?;
        if !chat.dir.join("where").exists() {
            write(
                &chat.dir,
                "where",
                format!("{host}\n{cwd}\n{session}\n").as_bytes(),
            )?;
        }
        Ok(chat)
    }

    /// The conversation stored at `dir`, created if missing.
    pub fn at(dir: PathBuf) -> Result<Self> {
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(dir.join(ATTACHMENTS))
            .wrap_err_with(|| format!("cannot create {}", dir.display()))?;
        Ok(Self { dir })
    }

    #[cfg(test)]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Wait until no other hub works on this conversation; holding the file keeps it so.
    pub fn lock(&self) -> Result<File> {
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .mode(0o600)
            .open(self.dir.join(".lock"))?;
        lock.lock()?;
        Ok(lock)
    }

    /// Every readable entry, backups included, oldest first. Unreadable ones are left out with
    /// a warning rather than costing the conversation.
    pub fn entries(&self) -> Result<Vec<Entry>> {
        self.read_entries(|_| true)
    }

    /// The conversation since the last restart.
    pub fn messages(&self) -> Result<Vec<Message>> {
        let entries = self.read_entries(|backup| !backup)?;
        Ok(entries.into_iter().map(|entry| entry.message).collect())
    }

    /// The number of the conversation's first entry since the last restart, which changes with
    /// every restart; zero before anything is said.
    pub fn generation(&self) -> Result<u64> {
        let current = self.names()?.into_iter().find(|&(_, backup)| !backup);
        Ok(current.map_or(0, |(number, _)| number))
    }

    /// Record messages after the last entry; notes are the harness's and are not kept.
    pub fn append(&self, messages: &[Message]) -> Result<()> {
        let mut number = self.next()?;
        for message in messages {
            let Some(text) = self.encode(message)? else {
                continue;
            };
            write(&self.dir, &file_name(number, false), text.as_bytes())?;
            number += 1;
        }
        Ok(())
    }

    /// Continue from `messages` alone, setting every entry so far aside as a backup. The new
    /// entries come first, so a failure midway never leaves the conversation empty.
    pub fn restart(&self, messages: &[Message]) -> Result<()> {
        let before = self.names()?;
        self.append(messages)?;
        for (number, backup) in before {
            if !backup {
                let from = self.dir.join(file_name(number, false));
                std::fs::rename(&from, self.dir.join(file_name(number, true)))?;
            }
        }
        Ok(())
    }

    /// An entry's kind and body, attachments resolved, by its number; or an attachment.
    pub fn read(&self, name: &str) -> Result<String> {
        let name = name.trim().trim_start_matches('[').trim_end_matches(']');
        if super::digest::is_name(name) {
            let path = self.dir.join(ATTACHMENTS).join(name);
            if path.exists() {
                return Ok(std::fs::read_to_string(path)?);
            }
        }
        let number: u64 = name
            .trim_end_matches('~')
            .parse()
            .map_err(|_| eyre!("no entry or attachment named {name}"))?;
        for backup in [false, true] {
            let path = self.dir.join(file_name(number, backup));
            if path.exists() {
                let message = self.decode(&std::fs::read_to_string(path)?)?;
                let (header, body) = header_and_body(&message);
                return Ok(format!("{header}\n\n{body}"));
            }
        }
        bail!("no entry {number}")
    }

    /// The last request whose messages were kept, and the entry they begin at.
    pub fn applied(&self) -> Option<(u64, u64)> {
        let text = std::fs::read_to_string(self.dir.join(APPLIED)).ok()?;
        let (id, start) = text.trim().split_once(' ')?;
        Some((id.parse().ok()?, start.parse().ok()?))
    }

    /// Remember that request `id`'s messages are kept from entry `start` on.
    pub fn set_applied(&self, id: u64, start: u64) -> Result<()> {
        write(&self.dir, APPLIED, format!("{id} {start}").as_bytes())
    }

    /// The number the next entry gets.
    pub fn next(&self) -> Result<u64> {
        Ok(self.names()?.last().map_or(0, |&(number, _)| number) + 1)
    }

    /// Entry numbers in order, with whether each is a backup.
    fn names(&self) -> Result<Vec<(u64, bool)>> {
        let mut names = Vec::new();
        for item in std::fs::read_dir(&self.dir)? {
            if let Some(name) = item?.file_name().to_str().and_then(parse_name) {
                names.push(name);
            }
        }
        names.sort();
        Ok(names)
    }

    fn read_entries(&self, wanted: impl Fn(bool) -> bool) -> Result<Vec<Entry>> {
        let mut entries = Vec::new();
        for (number, backup) in self.names()? {
            if !wanted(backup) {
                continue;
            }
            let path = self.dir.join(file_name(number, backup));
            let read = std::fs::read_to_string(&path).map_err(eyre::Report::from);
            match read.and_then(|text| self.decode(&text)) {
                Ok(message) => entries.push(Entry {
                    number,
                    backup,
                    message,
                }),
                Err(error) => tracing::warn!("chat: skipping {}: {error:#}", path.display()),
            }
        }
        Ok(entries)
    }

    /// An entry's text, with a long body moved to an attachment; `None` for notes.
    fn encode(&self, message: &Message) -> Result<Option<String>> {
        if matches!(message, Message::Note(_)) {
            return Ok(None);
        }
        let (header, body) = header_and_body(message);
        if header.ends_with(' ') {
            bail!("a tool call or result needs a name to be kept");
        }
        // A short body spelled like a reference is attached too, so it reads back as written.
        if body.len() <= INLINE && reference(body).is_none() {
            return Ok(Some(format!("{header}\n\n{body}")));
        }
        let name = super::digest::name_of(body.as_bytes());
        let attachments = self.dir.join(ATTACHMENTS);
        if !attachments.join(&name).exists() {
            write(&attachments, &name, body.as_bytes())?;
        }
        Ok(Some(format!("{header}\n\n[{name}]")))
    }

    /// Parse an entry, reading its body back from the attachment it names.
    fn decode(&self, text: &str) -> Result<Message> {
        let (header, body) = text.split_once("\n\n").unwrap_or((text.trim_end(), ""));
        let body = match reference(body) {
            Some(name) => std::fs::read_to_string(self.dir.join(ATTACHMENTS).join(name))
                .wrap_err_with(|| format!("missing attachment {name}"))?,
            None => body.to_string(),
        };
        let (kind, name) = header.split_once(' ').unwrap_or((header, ""));
        let name = name.to_string();
        Ok(match kind {
            "user" => Message::User(body),
            "input" => Message::Input(body),
            "thinking" => Message::Thinking(body),
            "answer" => Message::Answer(body),
            "call" if !name.is_empty() => Message::Call(Call {
                name,
                arguments: body,
            }),
            "result" if !name.is_empty() => Message::Result(ToolResult {
                name,
                content: body,
            }),
            _ => bail!("unknown entry kind {header:?}"),
        })
    }
}

/// An entry's first line and body.
fn header_and_body(message: &Message) -> (String, &str) {
    match message {
        Message::User(text) => ("user".into(), text),
        Message::Input(data) => ("input".into(), data),
        Message::Thinking(text) => ("thinking".into(), text),
        Message::Answer(text) => ("answer".into(), text),
        Message::Call(Call { name, arguments }) => (format!("call {name}"), arguments),
        Message::Result(ToolResult { name, content }) => (format!("result {name}"), content),
        Message::Note(_) => ("note".into(), ""),
    }
}

/// The attachment a body names, if it is a reference.
fn reference(body: &str) -> Option<&str> {
    body.strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .filter(|name| super::digest::is_name(name))
}

/// An entry's file name: its number, zero-padded to sort, and `~` once set aside.
fn file_name(number: u64, backup: bool) -> String {
    format!("{number:08}{}", if backup { "~" } else { "" })
}

/// The number of an entry's file name and whether it is a backup.
fn parse_name(name: &str) -> Option<(u64, bool)> {
    let (digits, backup) = match name.strip_suffix('~') {
        Some(digits) => (digits, true),
        None => (name, false),
    };
    let digits_only = digits.len() >= 8 && digits.bytes().all(|byte| byte.is_ascii_digit());
    digits_only
        .then(|| digits.parse().ok())
        .flatten()
        .map(|n| (n, backup))
}

/// Write a private file whole: a reader never sees it half written.
fn write(dir: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    let partial = dir.join(format!(".{name}.partial"));
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .mode(0o600)
        .open(&partial)?;
    file.write_all(bytes)?;
    file.sync_data()?;
    std::fs::rename(&partial, dir.join(name))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Note;
    use crate::test_support::TempDir;

    fn sample() -> Vec<Message> {
        vec![
            Message::Input("diff --git a/x b/x".into()),
            Message::User("find me a gc implementation".into()),
            Message::Note(Note::Pending(vec![3])),
            Message::Thinking("I'll list the files.\n\nThen read.".into()),
            Message::Call(Call {
                name: "list_files".into(),
                arguments: r#"{"path":"."}"#.into(),
            }),
            Message::Result(ToolResult {
                name: "list_files".into(),
                content: "core\nlang\ngc\n".repeat(1000),
            }),
            Message::Answer("Nothing in ./core.\n".into()),
        ]
    }

    #[test]
    fn messages_round_trip_through_plain_files() -> Result<()> {
        let dir = TempDir::new()?;
        let chat = Chat::at(dir.0.join("chat"))?;
        chat.append(&sample())?;
        let kept: Vec<_> = sample()
            .into_iter()
            .filter(|message| !matches!(message, Message::Note(_)))
            .collect();
        assert_eq!(chat.messages()?, kept);
        let user = std::fs::read_to_string(chat.dir().join("00000002"))?;
        assert_eq!(user, "user\n\nfind me a gc implementation");
        let call = std::fs::read_to_string(chat.dir().join("00000004"))?;
        assert_eq!(call, "call list_files\n\n{\"path\":\".\"}");
        // The long result went to an attachment the entry names.
        let result = std::fs::read_to_string(chat.dir().join("00000005"))?;
        let name = result.strip_prefix("result list_files\n\n[").unwrap();
        let name = name.strip_suffix(']').unwrap();
        assert!(chat.dir().join("...").join(name).exists());
        assert_eq!(
            chat.read(&format!("[{name}]"))?,
            "core\nlang\ngc\n".repeat(1000)
        );
        Ok(())
    }

    #[test]
    fn a_restart_sets_entries_aside_and_numbering_goes_on() -> Result<()> {
        let dir = TempDir::new()?;
        let chat = Chat::at(dir.0.join("chat"))?;
        chat.append(&sample())?;
        let summary = Message::Answer("You asked me to find a GC; next I'll read ./gc.".into());
        chat.restart(std::slice::from_ref(&summary))?;
        assert_eq!(chat.messages()?, std::slice::from_ref(&summary));
        assert!(chat.dir().join("00000002~").exists());
        assert!(chat.dir().join("00000007").exists());
        assert_eq!(chat.read("2")?, "user\n\nfind me a gc implementation");
        chat.append(&[Message::User("thanks".into())])?;
        let entries = chat.entries()?;
        assert_eq!(entries.len(), 8);
        assert_eq!(entries.iter().filter(|entry| entry.backup).count(), 6);
        assert_eq!(entries.last().map(|entry| entry.number), Some(8));
        Ok(())
    }

    #[test]
    fn foreign_files_and_unreadable_entries_cost_nothing_else() -> Result<()> {
        let dir = TempDir::new()?;
        let chat = Chat::at(dir.0.join("chat"))?;
        std::fs::write(chat.dir().join("notes.txt"), "mine")?;
        std::fs::write(chat.dir().join(".00000001.partial"), "user\n\nhalf")?;
        assert!(chat.messages()?.is_empty());
        std::fs::write(chat.dir().join("00000001"), "letter\n\nhi")?;
        std::fs::write(chat.dir().join("00000002~"), "")?;
        chat.append(&[Message::User("still here".into())])?;
        assert_eq!(chat.messages()?, [Message::User("still here".into())]);
        // A restart sets entries aside without reading them.
        chat.restart(&[])?;
        assert!(chat.messages()?.is_empty() && chat.dir().join("00000001~").exists());
        Ok(())
    }

    #[test]
    fn literal_references_names_and_privacy_survive() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new()?;
        let chat = Chat::at(dir.0.join("chat"))?;
        let literal = format!("[{}]", crate::hub::digest::name_of(b"x"));
        chat.append(&[Message::User(literal.clone())])?;
        chat.append(&[Message::Answer("x".into())])?;
        assert_eq!(chat.messages()?[0], Message::User(literal));
        let nameless = Message::Call(Call {
            name: String::new(),
            arguments: "{}".into(),
        });
        assert!(chat.append(&[nameless]).is_err());
        let mode = |path: &Path| std::fs::metadata(path).map(|m| m.permissions().mode() & 0o777);
        assert_eq!(mode(chat.dir())?, 0o700);
        assert_eq!(mode(&chat.dir().join("00000002"))?, 0o600);
        chat.set_applied(42, 2)?;
        assert_eq!(chat.applied(), Some((42, 2)));
        Ok(())
    }
}
