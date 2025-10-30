#[derive(Debug)]
pub enum PatchOp {
    Update {
        path: String,
        hunks: Vec<Hunk>,
        no_newline: bool,
    },
    Add {
        path: String,
        content: String,
        no_newline: bool,
    },
    Delete {
        path: String,
    },
}

#[derive(Debug, Default)]
pub struct Hunk {
    pub old_lines: Vec<String>,
    pub new_lines: Vec<String>,
}
