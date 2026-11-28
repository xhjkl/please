//! This invocation's setting; the hub turns it into model instructions.
use crate::protocol::{Effort, Setting};

/// Capture the date, the requested effort, and where the answer goes.
pub fn current(redirection: Option<String>) -> Setting {
    let now = time::OffsetDateTime::now_local().unwrap_or_else(|_| time::OffsetDateTime::now_utc());
    let requested = std::env::var("PLEASE_TRY").ok();
    Setting {
        today: now.date().to_string(),
        effort: effort(requested.as_deref()),
        redirection,
    }
}

/// `PLEASE_TRY` as a level on Inkling's scale from 0 to 1, or as `hard`, `medium`, or `light`
/// by first letter (`easy` means light), each standing for a level; medium otherwise.
fn effort(requested: Option<&str>) -> Effort {
    let requested = requested.map(|value| value.trim().to_lowercase());
    let requested = requested.as_deref().unwrap_or_default();
    let level = requested.parse().ok().and_then(Effort::new);
    if let Some(level) = level {
        return level;
    }
    match requested.chars().next() {
        Some('h') => Effort::HIGH,
        Some('l' | 'e') => Effort::LOW,
        _ => Effort::MEDIUM,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_is_a_level_or_a_word_by_its_first_letter() {
        assert_eq!(effort(Some("Hard")), Effort::HIGH);
        assert_eq!(effort(Some(" light")), Effort::LOW);
        assert_eq!(effort(Some("easy")), Effort::LOW);
        assert_eq!(effort(Some("medium")), Effort::MEDIUM);
        assert_eq!(effort(Some("")), Effort::MEDIUM);
        assert_eq!(effort(None), Effort::MEDIUM);
        assert_eq!(Some(effort(Some("0.35"))), Effort::new(0.35));
        assert_eq!(Some(effort(Some(" 1 "))), Effort::new(1.0));
        // A number off the scale is no level.
        assert_eq!(effort(Some("7")), Effort::MEDIUM);
        assert_eq!(effort(Some("nan")), Effort::MEDIUM);
    }
}
