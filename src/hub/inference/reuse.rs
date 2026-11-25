//! How much of the session's context a request can keep.
//!
//! Generic over the session so these decisions are tested without a GPU.

/// What reuse decisions read from a session.
pub trait Context {
    /// Consumed input IDs, oldest first.
    fn history(&self) -> &[u32];
}

impl Context for oco::Session {
    fn history(&self) -> &[u32] {
        oco::Session::history(self)
    }
}

/// Count how many leading tokens two sequences share.
pub fn common_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter().zip(right).take_while(|(a, b)| a == b).count()
}

/// Find where to rewind for a prompt: past everything shared, except that at least one token
/// is fed, because only a prefill yields the next prediction.
pub fn rewind_point(history: &[u32], prompt: &[u32]) -> usize {
    common_prefix(history, prompt).min(prompt.len().saturating_sub(1))
}

/// Trust a session restored from the cache only if it holds exactly the expected head;
/// the file name is a hash, and a collision or a stale file must cost a prefill, not a wrong prompt.
pub fn adopt_head<S: Context>(opened: Option<S>, head: &[u32]) -> Option<S> {
    opened.filter(|session| session.history() == head)
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Context for Vec<u32> {
        fn history(&self) -> &[u32] {
            self
        }
    }

    #[test]
    fn rewind_keeps_the_shared_prefix_and_feeds_the_rest() {
        let history = [1, 2, 3, 4];
        let prompt = [1, 2, 3, 4, 5, 6];
        let point = rewind_point(&history, &prompt);
        assert_eq!((point, &prompt[point..]), (4, &[5, 6][..]));
        // A diverging reply rewinds to the divergence.
        assert_eq!(rewind_point(&[1, 2, 9, 9], &prompt), 2);
        // An identical prompt still feeds its last token.
        assert_eq!(rewind_point(&prompt, &prompt), 5);
        assert_eq!(rewind_point(&[], &[7]), 0);
    }

    #[test]
    fn cached_head_is_trusted_only_when_it_matches() {
        assert_eq!(adopt_head(Some(vec![1, 2]), &[1, 2]), Some(vec![1, 2]));
        assert_eq!(adopt_head(Some(vec![1, 3]), &[1, 2]), None);
        assert_eq!(adopt_head(Some(vec![1]), &[1, 2]), None);
        assert_eq!(adopt_head::<Vec<u32>>(None, &[1, 2]), None);
    }
}
