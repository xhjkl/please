//! Template output shared by the dialects, which insert control tokens by ID so their
//! spellings inside user, stdin, or tool content stay ordinary text.
use super::prompt::Rendered;
use eyre::{Result, eyre};
use oco::Tokenizer;

/// Template output before encoding: literal text, or a control token named by its spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    Text(String),
    Marker(&'static str),
}

/// Encode pieces in order, each text run on its own, inserting control tokens by ID.
pub fn encode_pieces(tokenizer: &Tokenizer, pieces: &[Piece]) -> Result<Vec<u32>> {
    let mut tokens = Vec::new();
    for piece in pieces {
        match piece {
            Piece::Text(text) => tokens.extend(tokenizer.encode(text)?),
            Piece::Marker(spelling) => tokens.push(special(tokenizer, spelling)?),
        }
    }
    Ok(tokens)
}

/// Encode a prompt whose first `head` pieces are its static head; pieces encode on their own,
/// so the head's tokens are a prefix of the whole.
pub fn encode_prompt(tokenizer: &Tokenizer, pieces: &[Piece], head: usize) -> Result<Rendered> {
    let (head, tail) = pieces.split_at(head);
    let mut tokens = encode_pieces(tokenizer, head)?;
    let head = tokens.len();
    tokens.extend(encode_pieces(tokenizer, tail)?);
    Ok(Rendered { tokens, head })
}

/// Look up a control token a template needs, failing for checkpoints without it.
pub fn special(tokenizer: &Tokenizer, spelling: &str) -> Result<u32> {
    tokenizer
        .special_token(spelling)
        .ok_or_else(|| eyre!("checkpoint is missing {spelling}"))
}

/// Spell pieces out, control tokens included, as a template's string output reads.
#[cfg(test)]
pub fn spelled(pieces: &[Piece]) -> String {
    pieces
        .iter()
        .map(|piece| match piece {
            Piece::Text(text) => text.as_str(),
            Piece::Marker(spelling) => spelling,
        })
        .collect()
}
