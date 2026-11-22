//! Short, filename-safe names for content: the first 128 bits of a SHA-256 in z-base-32.
use sha2::{Digest, Sha256};

/// z-base-32 digit alphabet
const ALPHABET: &[u8; 32] = b"ybndrfg8ejkmcpqxot1uwisza345h769";

/// Name a finished hash by its first 128 bits.
pub fn name(hash: Sha256) -> String {
    let mut bits = [0; 16];
    bits.copy_from_slice(&hash.finalize()[..16]);
    z_base_32(u128::from_be_bytes(bits))
}

/// Name some bytes by their hash.
pub fn name_of(bytes: &[u8]) -> String {
    name(Sha256::new_with_prefix(bytes))
}

/// Whether `name` could be a name this module made.
pub fn is_name(name: &str) -> bool {
    (1..=26).contains(&name.len()) && name.bytes().all(|byte| ALPHABET.contains(&byte))
}

/// Encode `n` as a z-base-32 integer without leading zero digits.
pub fn z_base_32(mut n: u128) -> String {
    let mut digits = Vec::new();
    loop {
        digits.push(ALPHABET[(n % 32) as usize]);
        n /= 32;
        if n == 0 {
            break;
        }
    }
    digits
        .iter()
        .rev()
        .map(|&digit| char::from(digit))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_short_stable_and_recognized() {
        let name = name_of(b"hello");
        assert_eq!(name, name_of(b"hello"));
        assert_ne!(name, name_of(b"hello!"));
        assert!(is_name(&name) && name.len() <= 26);
        assert!(!is_name("00000001") && !is_name(""));
        assert_eq!(z_base_32(0), "y");
        assert_eq!(z_base_32(32), "by");
    }
}
