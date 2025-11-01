# Docker

You can run `please` inside a container while using the host's inference hub and weights.

Ensure the host has a hub running:

```
$ please run
```

Then just prefix any docker command with `please docker ...`:

```
$ please docker run --rm -it bash:latest
```

Now `please` inside the container will connect to the host hub automatically.

## How it works
`please docker ...` mounts your host `~/.please/socket` to `/root/.please/socket` inside the container.
This assumes the container user's home is `/root`.
