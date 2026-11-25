//! Where this probe runs, which names its conversation on the hub.
use crate::protocol::Place;
use eyre::Result;

/// Host keys in order of preference, as sshd keeps them on Linux and macOS.
const HOST_KEYS: [&str; 3] = [
    "/etc/ssh/ssh_host_ed25519_key.pub",
    "/etc/ssh/ssh_host_ecdsa_key.pub",
    "/etc/ssh/ssh_host_rsa_key.pub",
];

/// This machine and working directory.
pub fn here() -> Result<Place> {
    let cwd = std::env::current_dir()?.canonicalize()?;
    Ok(Place {
        machine: machine(),
        cwd: cwd.to_string_lossy().into_owned(),
    })
}

/// The machine's SSH host key, the identity `known_hosts` pins: the same from any terminal
/// or client, unchanged by a new hostname, new after a re-image. Empty without one.
fn machine() -> String {
    for path in HOST_KEYS {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        if let Some(key) = key(&text) {
            return key;
        }
    }
    String::new()
}

/// A public key file's type and key; its comment, often `root@hostname`, is left out.
fn key(text: &str) -> Option<String> {
    let mut fields = text.split_whitespace();
    let (kind, key) = (fields.next()?, fields.next()?);
    let known = kind.starts_with("ssh-") || kind.starts_with("ecdsa-");
    known.then(|| format!("{kind} {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_machine_is_its_host_key_without_the_comment() -> Result<()> {
        let text = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB root@box-17\n";
        assert_eq!(
            key(text).as_deref(),
            Some("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB")
        );
        assert_eq!(key("not a key"), None);
        assert_eq!(key(""), None);
        assert_eq!(machine(), machine());
        assert!(here()?.cwd.starts_with('/'));
        Ok(())
    }
}
