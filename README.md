# 🙏🙏🙏

A command-line tool that lets you interact with your terminal in natural language.
All inference stays local.

*The name is chosen so that the invocation reads like a natural-language phrase.*

<!-- If you use ollama, they will share the weights. -->

# Installation

First, install the tool itself.
You can get it from Homebrew:
```
$ brew install xhjkl/made/please
```

Alternatively, you can install it from source.
It needs to be pulled from git, crates-io do not have it:
```
# either the above, or this:
$ cargo install --git https://github.com/xhjkl/please
```

Then, download the weights just once:
```
$ please load
```

And, optionally, verify them:
```
$ sha256sum ~/.please/weights/gpt-oss-20b-mxfp4.gguf
be37a636aca0fc1aae0d32325f82f6b4d21495f06823b5fbc1898ae0303e9935
```

# Usage

```
$ git diff --cached | please summarize to a concise commit message
$ please take the creds from Justfile and make a fuzzer one-off script > fuzz.ts
$ please format all the git-staged files
$ please tar the current folder but without git-ignored files
$ please resolve all the merge conflicts in the current folder
$ please fix all clippy diagnostics
```

# Bridging

You can run `please` on a remote machine while using your local inference engine and downloaded weights.
See [BRIDGING.md](BRIDGING.md) for SSH forwarding instructions.

# Docker

You can also run `please` in a Docker container while inferring on the host machine.
See [DOCKER.md](DOCKER.md) for the instructions.
