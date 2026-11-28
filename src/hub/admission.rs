//! Which probe the hub serves. Its one session follows one conversation, so a second probe is
//! turned away instead of waiting behind the first or evicting its context.
use std::sync::Mutex;

/// The probe being served, if any.
#[derive(Default)]
pub struct Admission(Mutex<Option<Served>>);

struct Served {
    probe: u64,
    /// Open connections from it; one reconnecting after Ctrl-C briefly holds two.
    connections: usize,
}

/// One admitted connection; the hub is free again once its probe's last one is dropped.
pub struct Admitted<'a>(&'a Admission);

impl Admission {
    /// Admit a connection from `probe` if the hub is free or already serving it.
    pub fn admit(&self, probe: u64) -> Option<Admitted<'_>> {
        let mut served = self.0.lock().unwrap();
        match served.as_mut() {
            Some(served) if served.probe != probe => return None,
            Some(served) => served.connections += 1,
            None => {
                *served = Some(Served {
                    probe,
                    connections: 1,
                })
            }
        }
        Some(Admitted(self))
    }
}

impl Drop for Admitted<'_> {
    fn drop(&mut self) {
        let mut served = self.0.0.lock().unwrap();
        let admitted = served
            .as_mut()
            .expect("an admitted connection keeps its probe served");
        admitted.connections -= 1;
        if admitted.connections == 0 {
            *served = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn another_probe_is_turned_away_until_the_served_one_leaves() {
        let admission = Admission::default();
        let first = admission.admit(1).expect("a free hub admits");
        assert!(admission.admit(2).is_none());
        // Reconnecting after Ctrl-C overlaps the connection being torn down.
        let reconnected = admission.admit(1).expect("the served probe comes back");
        drop(first);
        assert!(admission.admit(2).is_none());
        drop(reconnected);
        assert!(admission.admit(2).is_some());
    }
}
