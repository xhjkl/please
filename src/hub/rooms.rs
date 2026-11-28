//! Rooms the hub keeps on disk, one directory each under `~/.please/rooms`.
//!
//! Each place a probe runs in, a machine and working directory, has its own room, which keeps
//! what is said there from one run of please to the next: the directory is named after the
//! place, which its `.where` file spells out. `/new` starts the room over.
//!
//! A room is a directory of plain-text entries named by eight-digit sequence numbers, so they
//! list in order. An entry is its kind on the first line (`user`, `input`, `thinking`,
//! `answer`, `call TOOL ID`, or `result TOOL ID`), a blank line, and the body verbatim, however
//! long. A call's ID is the number of the entry that first kept it, so a call carried across a
//! restart still names its original, and the result answering it repeats the ID; the result
//! recorded for calls the model wrote but could not be parsed has its own entry's number.
//! The `...` subdirectory is for what an entry cannot hold as text, such as pictures; nothing
//! is kept there yet. Restarting sets the entries so far aside in the `old` subdirectory, which
//! `rm -r old` erases; until then recall still reads them. The harness's notes are not kept.
//!
//! Everything besides the entries and `old` is a dotfile, so the directory lists as what was
//! said alone. Everything is private to its owner, and one hub at a time works in a room.
use crate::protocol::{Message, Place, ToolCall, ToolResult};
use eyre::{Result, WrapErr, bail, eyre};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

/// Where what an entry cannot hold as text will live, inside a room's directory.
const ATTACHMENTS: &str = "...";
/// Where a restart sets the entries so far aside, inside a room's directory.
const OLD: &str = "old";
/// The place a room belongs to: its working directory, then its machine.
const WHERE: &str = ".where";
/// The ID of the last delivery kept, so a retried one is kept once.
const DELIVERY: &str = ".delivery";

/// One room's directory.
#[derive(Debug, Clone)]
pub struct Room {
    dir: PathBuf,
}

/// An entry as it lies on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub number: u64,
    /// Set aside in `old` by a restart.
    pub old: bool,
    pub message: Message,
}

impl Room {
    /// The room of `place`, recording where it belongs when it is first opened.
    pub fn open(place: &Place) -> Result<Self> {
        let Place { machine, cwd } = place;
        let key = format!("please room\0{machine}\0{cwd}");
        let dir = super::directory().join("rooms");
        let room = Self::at(dir.join(super::digest::name_of(key.as_bytes())))?;
        if !room.dir.join(WHERE).exists() {
            write(&room.dir, WHERE, format!("{cwd}\n{machine}\n").as_bytes())?;
        }
        Ok(room)
    }

    /// The room stored at `dir`, created if missing.
    pub fn at(dir: PathBuf) -> Result<Self> {
        make_dir(&dir.join(ATTACHMENTS))?;
        Ok(Self { dir })
    }

    #[cfg(test)]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Wait until no other hub works in this room; holding the file keeps it so.
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

    /// Every readable entry, old ones included, oldest first. Unreadable ones are left out with
    /// a warning rather than costing the room.
    pub fn entries(&self) -> Result<Vec<Entry>> {
        self.read_entries(|_| true)
    }

    /// What was said since the last restart.
    pub fn messages(&self) -> Result<Vec<Message>> {
        let entries = self.read_entries(|old| !old)?;
        Ok(entries.into_iter().map(|entry| entry.message).collect())
    }

    /// The number of the room's first entry since the last restart, which changes with every
    /// restart; zero before anything is said.
    pub fn epoch(&self) -> Result<u64> {
        let current = self.names()?.into_iter().find(|&(_, old)| !old);
        Ok(current.map_or(0, |(number, _)| number))
    }

    /// Record messages after the last entry; notes are the harness's and are not kept.
    pub fn append(&self, messages: &[Message]) -> Result<()> {
        let mut number = self.next()?;
        for message in messages {
            let Some(text) = encode(message)? else {
                continue;
            };
            write(&self.dir, &file_name(number), text.as_bytes())?;
            number += 1;
        }
        Ok(())
    }

    /// Continue from `messages` alone, setting every entry so far aside in `old`. The new
    /// entries come first, so a failure midway never leaves the room empty.
    pub fn restart(&self, messages: &[Message]) -> Result<()> {
        make_dir(&self.dir.join(OLD))?;
        let before = self.names()?;
        self.append(messages)?;
        for (number, old) in before {
            if !old {
                std::fs::rename(self.path(number, false), self.path(number, true))?;
            }
        }
        Ok(())
    }

    /// An entry's text as it lies on disk, by its number, whether or not it is set aside.
    pub fn read(&self, name: &str) -> Result<String> {
        let name = name.trim().trim_start_matches('[').trim_end_matches(']');
        let number: u64 = name.parse().map_err(|_| eyre!("no entry named {name}"))?;
        for old in [false, true] {
            let path = self.path(number, old);
            if path.exists() {
                return Ok(std::fs::read_to_string(path)?);
            }
        }
        bail!("no entry {number}")
    }

    /// The ID of the last delivery whose messages were kept.
    pub fn last_delivery(&self) -> Option<u64> {
        let text = std::fs::read_to_string(self.dir.join(DELIVERY)).ok()?;
        text.trim().parse().ok()
    }

    /// Remember that delivery `id`'s messages are kept.
    pub fn set_last_delivery(&self, id: u64) -> Result<()> {
        write(&self.dir, DELIVERY, id.to_string().as_bytes())
    }

    /// The number the next entry gets.
    pub fn next(&self) -> Result<u64> {
        Ok(self.names()?.last().map_or(0, |&(number, _)| number) + 1)
    }

    /// Where entry `number` lies: in the conversation, or set aside in `old`.
    fn path(&self, number: u64, old: bool) -> PathBuf {
        match old {
            true => self.dir.join(OLD).join(file_name(number)),
            false => self.dir.join(file_name(number)),
        }
    }

    /// Entry numbers in order, with whether each is set aside; `old` may have been erased.
    fn names(&self) -> Result<Vec<(u64, bool)>> {
        let mut names = Vec::new();
        for (dir, old) in [(self.dir.join(OLD), true), (self.dir.clone(), false)] {
            let items = std::fs::read_dir(&dir);
            let items = match items {
                Err(error) if old && error.kind() == std::io::ErrorKind::NotFound => continue,
                items => items?,
            };
            for item in items {
                if let Some(number) = item?.file_name().to_str().and_then(parse_name) {
                    names.push((number, old));
                }
            }
        }
        names.sort();
        Ok(names)
    }

    fn read_entries(&self, wanted: impl Fn(bool) -> bool) -> Result<Vec<Entry>> {
        let mut entries = Vec::new();
        for (number, old) in self.names()? {
            if !wanted(old) {
                continue;
            }
            let path = self.path(number, old);
            let read = std::fs::read_to_string(&path).map_err(eyre::Report::from);
            match read.and_then(|text| decode(&text)) {
                Err(error) => tracing::warn!("room: skipping {}: {error:#}", path.display()),
                Ok(message) => entries.push(Entry {
                    number,
                    old,
                    message,
                }),
            }
        }
        Ok(entries)
    }
}

/// An entry's text; `None` for notes.
fn encode(message: &Message) -> Result<Option<String>> {
    let tool = |name: &str| match name.is_empty() || name.contains('\n') {
        true => bail!("a tool call or result needs a one-line name to be kept"),
        false => Ok(()),
    };
    let (header, body) = match message {
        Message::Note(_) => return Ok(None),
        Message::User(text) => ("user".into(), text),
        Message::Input(data) => ("input".into(), data),
        Message::Thinking(text) => ("thinking".into(), text),
        Message::Answer(text) => ("answer".into(), text),
        Message::Call(ToolCall {
            id,
            name,
            arguments,
        }) => {
            tool(name)?;
            (format!("call {name} {id}"), arguments)
        }
        Message::Result(ToolResult { id, name, content }) => {
            tool(name)?;
            (format!("result {name} {id}"), content)
        }
    };
    Ok(Some(format!("{header}\n\n{body}")))
}

/// Parse an entry. A tool's name runs from the kind to the ID, so it may hold spaces.
fn decode(text: &str) -> Result<Message> {
    let (header, body) = text.split_once("\n\n").unwrap_or((text.trim_end(), ""));
    let body = body.to_string();
    let (kind, rest) = header.split_once(' ').unwrap_or((header, ""));
    let tool = || -> Result<(u64, String)> {
        let (name, id) = rest
            .rsplit_once(' ')
            .filter(|(name, _)| !name.is_empty())
            .ok_or_else(|| eyre!("a {kind} entry needs a tool name and an ID"))?;
        let id = id.parse().map_err(|_| eyre!("{id:?} is not a call ID"))?;
        Ok((id, name.to_string()))
    };
    Ok(match kind {
        "user" => Message::User(body),
        "input" => Message::Input(body),
        "thinking" => Message::Thinking(body),
        "answer" => Message::Answer(body),
        "call" => {
            let (id, name) = tool()?;
            Message::Call(ToolCall {
                id,
                name,
                arguments: body,
            })
        }
        "result" => {
            let (id, name) = tool()?;
            Message::Result(ToolResult {
                id,
                name,
                content: body,
            })
        }
        _ => bail!("unknown entry kind {header:?}"),
    })
}

/// An entry's file name: its number, zero-padded to sort.
fn file_name(number: u64) -> String {
    format!("{number:08}")
}

/// The number an entry's file name holds.
fn parse_name(name: &str) -> Option<u64> {
    let digits_only = name.len() >= 8 && name.bytes().all(|byte| byte.is_ascii_digit());
    digits_only.then(|| name.parse().ok()).flatten()
}

/// Create a private directory, with any missing above it.
fn make_dir(dir: &Path) -> Result<()> {
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
        .wrap_err_with(|| format!("cannot create {}", dir.display()))
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

    fn some_messages() -> Vec<Message> {
        vec![
            Message::Input("diff --git a/x b/x".into()),
            Message::User("find me a gc implementation".into()),
            Message::Note(Note::Pending(vec![3])),
            Message::Thinking("I'll list the files.\n\nThen read.".into()),
            Message::Call(ToolCall {
                id: 4,
                name: "list_files".into(),
                arguments: r#"{"path":"."}"#.into(),
            }),
            Message::Result(ToolResult {
                id: 4,
                name: "list_files".into(),
                content: "core\nlang\ngc\n".repeat(1000),
            }),
            Message::Answer("Nothing in ./core.\n".into()),
        ]
    }

    #[test]
    fn messages_round_trip_through_plain_files() -> Result<()> {
        let dir = TempDir::new()?;
        let room = Room::at(dir.0.join("room"))?;
        room.append(&some_messages())?;
        let kept: Vec<_> = some_messages()
            .into_iter()
            .filter(|message| !matches!(message, Message::Note(_)))
            .collect();
        assert_eq!(room.messages()?, kept);
        let user = std::fs::read_to_string(room.dir().join("00000002"))?;
        assert_eq!(user, "user\n\nfind me a gc implementation");
        let call = std::fs::read_to_string(room.dir().join("00000004"))?;
        assert_eq!(call, "call list_files 4\n\n{\"path\":\".\"}");
        // However long, the result lies inside its entry, under its call's ID.
        let result = format!("result list_files 4\n\n{}", "core\nlang\ngc\n".repeat(1000));
        assert_eq!(
            std::fs::read_to_string(room.dir().join("00000005"))?,
            result
        );
        assert_eq!(room.read("[5]")?, result);
        // Text bodies never go to the attachments.
        let listed = |dir: &Path| {
            let names = std::fs::read_dir(dir)?
                .map(|item| item.map(|item| item.file_name().to_string_lossy().into_owned()));
            let mut names = names.collect::<std::io::Result<Vec<_>>>()?;
            names.sort();
            Ok::<_, std::io::Error>(names)
        };
        let entries = [
            "...", "00000001", "00000002", "00000003", "00000004", "00000005", "00000006",
        ];
        assert_eq!(listed(room.dir())?, entries);
        assert!(listed(&room.dir().join(ATTACHMENTS))?.is_empty());
        Ok(())
    }

    #[test]
    fn a_restart_sets_entries_aside_and_numbering_goes_on() -> Result<()> {
        let dir = TempDir::new()?;
        let room = Room::at(dir.0.join("room"))?;
        room.append(&some_messages())?;
        let summary = Message::Answer("You asked me to find a GC; next I'll read ./gc.".into());
        room.restart(std::slice::from_ref(&summary))?;
        assert_eq!(room.messages()?, std::slice::from_ref(&summary));
        assert!(room.dir().join("old/00000002").exists());
        assert!(room.dir().join("00000007").exists());
        assert_eq!(room.read("2")?, "user\n\nfind me a gc implementation");
        room.append(&[Message::User("thanks".into())])?;
        let entries = room.entries()?;
        assert_eq!(entries.len(), 8);
        assert_eq!(entries.iter().filter(|entry| entry.old).count(), 6);
        assert_eq!(entries.last().map(|entry| entry.number), Some(8));
        // Erasing what was set aside leaves the conversation and its numbering be.
        std::fs::remove_dir_all(room.dir().join("old"))?;
        assert_eq!(room.entries()?.len(), 2);
        assert_eq!(room.next()?, 9);
        Ok(())
    }

    #[test]
    fn foreign_files_and_unreadable_entries_cost_nothing_else() -> Result<()> {
        let dir = TempDir::new()?;
        let room = Room::at(dir.0.join("room"))?;
        std::fs::write(room.dir().join("notes.txt"), "mine")?;
        std::fs::write(room.dir().join(".00000001.partial"), "user\n\nhalf")?;
        assert!(room.messages()?.is_empty());
        std::fs::write(room.dir().join("00000001"), "letter\n\nhi")?;
        std::fs::create_dir(room.dir().join("old"))?;
        std::fs::write(room.dir().join("old/00000002"), "")?;
        std::fs::write(room.dir().join("00000003"), "call list_files\n\n{}")?;
        room.append(&[Message::User("still here".into())])?;
        assert_eq!(room.messages()?, [Message::User("still here".into())]);
        // A restart sets entries aside without reading them.
        room.restart(&[])?;
        assert!(room.messages()?.is_empty() && room.dir().join("old/00000001").exists());
        Ok(())
    }

    #[test]
    fn odd_names_and_privacy_survive() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new()?;
        let room = Room::at(dir.0.join("room"))?;
        let spaced = Message::Call(ToolCall {
            id: 1,
            name: "list files 2".into(),
            arguments: "{}".into(),
        });
        room.append(std::slice::from_ref(&spaced))?;
        room.append(&[Message::Answer("x".into())])?;
        assert_eq!(room.messages()?[0], spaced);
        let nameless = Message::Result(ToolResult {
            id: 1,
            name: String::new(),
            content: "{}".into(),
        });
        assert!(room.append(&[nameless]).is_err());
        let mode = |path: &Path| std::fs::metadata(path).map(|m| m.permissions().mode() & 0o777);
        assert_eq!(mode(room.dir())?, 0o700);
        assert_eq!(mode(&room.dir().join("00000002"))?, 0o600);
        room.set_last_delivery(42)?;
        assert_eq!(room.last_delivery(), Some(42));
        Ok(())
    }
}
