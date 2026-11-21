//! Template output shared by the dialects, which insert control tokens by ID so their
//! spellings inside user, stdin, or tool content stay ordinary text.
use eyre::{Result, eyre};
use oco::Tokenizer;

/// Template output before encoding: literal text, or a control token named by its spelling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    Text(String),
    Marker(&'static str),
}

/// Encode one text run; control-token spellings inside it stay ordinary text.
pub fn encode_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer.encode(text).map_err(|error| eyre!(error))
}

/// Encode pieces in order, each text run on its own, inserting control tokens by ID.
pub fn encode_pieces(tokenizer: &Tokenizer, pieces: &[Piece]) -> Result<Vec<u32>> {
    let mut tokens = Vec::new();
    for piece in pieces {
        match piece {
            Piece::Text(text) => tokens.extend(encode_text(tokenizer, text)?),
            Piece::Marker(spelling) => tokens.push(special(tokenizer, spelling)?),
        }
    }
    Ok(tokens)
}

/// Look up a control token a template needs, failing for checkpoints without it.
pub fn special(tokenizer: &Tokenizer, spelling: &str) -> Result<u32> {
    tokenizer
        .special_token(spelling)
        .ok_or_else(|| eyre!("checkpoint is missing {spelling}"))
}
