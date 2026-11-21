//! Which probe the hub serves. Its one session follows one conversation, so a second probe is
//! turned away instead of waiting behind the first or evicting its context.
use std::sync::Mutex;

/// The probe being served, if any.
#[derive(Default)]
pub struct Occupancy(Mutex<Option<Occupant>>);

struct Occupant {
    probe: u64,
    /// Open connections from it; one reconnecting after Ctrl-C briefly holds two.
    connections: usize,
}

/// One admitted connection; the hub is free again once its probe's last one is dropped.
pub struct Admission<'a>(&'a Occupancy);

impl Occupancy {
    /// Admit a connection from `probe` if the hub is free or already serving it.
    pub fn admit(&self, probe: u64) -> Option<Admission<'_>> {
        let mut occupant = self.0.lock().unwrap();
        match occupant.as_mut() {
            Some(occupant) if occupant.probe != probe => return None,
            Some(occupant) => occupant.connections += 1,
            None => {
                *occupant = Some(Occupant {
                    probe,
                    connections: 1,
                })
            }
        }
        Some(Admission(self))
    }
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        let mut occupant = self.0.0.lock().unwrap();
        if let Some(admitted) = occupant.as_mut() {
            admitted.connections -= 1;
            if admitted.connections == 0 {
                *occupant = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn another_probe_is_turned_away_until_the_served_one_leaves() {
        let occupancy = Occupancy::default();
        let first = occupancy.admit(1).expect("a free hub admits");
        assert!(occupancy.admit(2).is_none());
        // Reconnecting after Ctrl-C overlaps the connection being torn down.
        let reconnected = occupancy.admit(1).expect("the served probe comes back");
        drop(first);
        assert!(occupancy.admit(2).is_none());
        drop(reconnected);
        assert!(occupancy.admit(2).is_some());
    }
}
