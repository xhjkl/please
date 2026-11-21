# 🙏🙏🙏

A command-line tool that lets you interact with your terminal in natural language.
All inference stays local.

*The name is chosen so that the invocation reads like a natural-language phrase.*

https://github.com/user-attachments/assets/87775557-1197-474e-a987-b5661299e709

# Installation

First, install the tool itself.
You can get it from Homebrew:
```
$ brew install xhjkl/made/please
```

Or with the installer script:
```
# either the above, or this:
$ curl -fsSL xhjkl.xyz/please | sh
```

Then, download the weights just once:
```
$ please load
```

And, optionally, verify them:
```
$ (cd ~/.please/weights && sha256sum Qwen3.8-27B-Q4_0.gguf)
ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d  Qwen3.8-27B-Q4_0.gguf
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

You can run `please` in a different environment, such as a remote shell or a container, while keeping inference and weights on your machine.
This avoids copying weights around and lets you keep using your local compute power. For a remote machine, set up SSH socket forwarding as described in [BRIDGING.md](BRIDGING.md); for a Docker container, mount the host socket into the container as described in [DOCKER.md](DOCKER.md).
