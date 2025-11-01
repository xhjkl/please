# Bridging

You can run `please` on a remote machine while using your local inference engine and downloaded weights.

```
[ local machine with loaded weights running `please run` ] ← ssh → [ remote machine with please CLI, no weights ]
```

To achieve this, configure SSH to forward the socket to the remote host, then start the local hub as shown below.

## Configure SSH

For *Linux* remotes:

```
Host *
  ExitOnForwardFailure yes
  StreamLocalBindMask 0177
  StreamLocalBindUnlink yes
  RemoteForward /home/%r/.please/socket %d/.please/socket
```

For *macOS* remotes (if needed):

```
Host my.macos.box
  ExitOnForwardFailure yes
  StreamLocalBindMask 0177
  StreamLocalBindUnlink yes
  RemoteForward /Users/%r/.please/socket %d/.please/socket
```

Use absolute paths in `RemoteForward`: `~` is not reliably expanded by `sshd`.
Ensure the directory exists on both sides: `mkdir -p ~/.please`.

## Start the Local Hub

In a separate terminal, start the local hub:

```
$ please run
```

Keep it running, then SSH into the remote machine.
Ensure the `please` CLI is installed on the remote host — it will automatically connect to your local hub through the forwarded socket.
