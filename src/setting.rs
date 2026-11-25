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

/// `PLEASE_TRY=hard|medium|light` by first letter (`easy` means light); medium otherwise.
fn effort(requested: Option<&str>) -> Effort {
    let requested = requested.map(|value| value.trim().to_lowercase());
    match requested.as_deref().and_then(|value| value.chars().next()) {
        Some('h') => Effort::High,
        Some('l' | 'e') => Effort::Low,
        _ => Effort::Medium,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_follows_the_first_letter() {
        assert_eq!(effort(Some("Hard")), Effort::High);
        assert_eq!(effort(Some(" light")), Effort::Low);
        assert_eq!(effort(Some("easy")), Effort::Low);
        assert_eq!(effort(Some("medium")), Effort::Medium);
        assert_eq!(effort(Some("")), Effort::Medium);
        assert_eq!(effort(None), Effort::Medium);
    }
}
