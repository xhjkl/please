# 🙏🙏🙏

A command-line tool that lets you interact with your terminal in natural language.
All inference stays local.

*The name is chosen so that the invocation reads like a natural-language phrase.*

<picture>
  <source media="(max-width: 900px)" srcset="./assets/please-demo-narrow.webp 3x">
  <source srcset="./assets/please-demo.webp 2x">
  <img src="./assets/please-demo.webp" alt="Please writes a commit message from a staged Git diff, creates a Bun script, and finds a server setting.">
</picture>

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
$ cd ~/.please/weights
$ shasum -a 256 Inkling-2-Mini-IQ4_NL.gguf
cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4  Inkling-2-Mini-IQ4_NL.gguf
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
