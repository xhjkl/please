//! How much of the session's context a request can keep.

/// Find where to rewind for a prompt: past every leading token it shares with the session's
/// history, except that at least one token is fed, because only a prefill yields the next
/// prediction.
pub fn rewind_point(history: &[u32], prompt: &[u32]) -> usize {
    let shared = history.iter().zip(prompt).take_while(|(a, b)| a == b);
    shared.count().min(prompt.len().saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
