# 🙏🙏🙏

A command-line tool that lets you interact with your terminal in natural language.
All inference stays local.

*The name is chosen so that the invocation reads like a natural-language phrase.*

<!-- If you use ollama, they will share the weights. -->

```
$ git diff --cached | please summarize to a concise commit message
$ please take the creds from Justfile and make a fuzzer one-off script > fuzz.ts
$ please format all the git-staged files
$ please tar the current folder but without git-ignored files
$ please resolve all the merge conflicts in the current folder
$ please fix all clippy diagnostics
```

# Installation

```
$ cargo install please
```
