//! What the model is told about a tool. Only a build that carries a hub reads it.
#![cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]

/// What the model is told about a tool: its name, what it does, and its parameters.
#[derive(Clone, Copy)]
pub struct Spec {
    pub name: &'static str,
    pub description: &'static str,
    pub params: &'static [Param],
}

/// The JSON type a parameter's value has.
#[derive(Debug, Clone)]
pub enum ParamKind {
    String,
    StringArray,
    Choice(&'static [&'static str]),
    Number,
}

/// One parameter of a tool, as its schema declares it.
#[derive(Clone)]
pub struct Param {
    pub name: &'static str,
    pub description: &'static str,
    pub kind: ParamKind,
    pub required: bool,
}
