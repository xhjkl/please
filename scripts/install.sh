#!/usr/bin/env sh
set -eu

# Detect downloader
have_curl=0
have_wget=0
if command -v curl >/dev/null 2>&1; then
  have_curl=1
fi
if command -v wget >/dev/null 2>&1; then
  have_wget=1
fi
if [ "$have_curl" -ne 1 ] && [ "$have_wget" -ne 1 ]; then
  echo "error: need either curl or wget to proceed" >&2
  exit 1
fi

# Detect OS and ARCH
OS=$(uname -s)
ARCH=$(uname -m)
ASSET=""
case "$OS" in
  Linux)
    case "$ARCH" in
      x86_64|amd64)
        ASSET="please-linux-x86_64"
        ;;
      aarch64|arm64)
        ASSET="please-linux-arm64"
        ;;
      *)
        echo "error: unsupported Linux architecture: $ARCH" >&2
        exit 1
        ;;
    esac
    ;;
  Darwin)
    case "$ARCH" in
      arm64|aarch64)
        ASSET="please-darwin-arm64"
        ;;
      *)
        echo "error: unsupported macOS architecture: $ARCH (only arm64 supported)" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "error: unsupported OS: $OS" >&2
    exit 1
    ;;
esac

URL="https://github.com/xhjkl/please/releases/latest/download/$ASSET"
INSTALL_DIR="/usr/local/bin"
INSTALL_PATH="$INSTALL_DIR/please"

cat <<MSG
This installer will:
  - download:     $URL
  - place it at:  $INSTALL_PATH (using sudo if required)

Proceed? [y/N]
MSG

# Read confirmation from the controlling terminal; ignore empty input
ans=""
if [ -r /dev/tty ]; then
  while :; do
    IFS='' read -r ans </dev/tty || exit 1
    [ -n "$ans" ] && break
  done
else
  # No TTY available; default to abort
  echo "no TTY available to confirm; aborted" >&2
  exit 1
fi
case "$ans" in
  y|Y|yes|YES)
    ;;
  *)
    echo "aborted by user"
    exit 0
    ;;
esac

TMP=$(mktemp "please.XXXXXXXXXX")
cleanup() { rm -f "$TMP" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Download
if [ "$have_curl" -eq 1 ]; then
  curl -fL "$URL" -o "$TMP"
else
  wget -O "$TMP" "$URL"
fi
chmod +x "$TMP"

# Install
mkdir -p "$INSTALL_DIR"
if [ -w "$INSTALL_DIR" ] && [ ! -e "$INSTALL_PATH" -o -w "$INSTALL_PATH" ]; then
  install -m 0755 "$TMP" "$INSTALL_PATH"
else
  if command -v sudo >/dev/null 2>&1; then
    echo "elevating with sudo to write $INSTALL_PATH"
    sudo install -m 0755 "$TMP" "$INSTALL_PATH"
  else
    echo "error: $INSTALL_DIR not writable and sudo not available" >&2
    exit 1
  fi
fi

# Post-check
FOUND=$(command -v please || true)
if [ "$FOUND" != "$INSTALL_PATH" ]; then
  echo "warning: 'please' on PATH is '$FOUND', expected '$INSTALL_PATH'" >&2
  echo "          ensure $INSTALL_DIR is in your PATH (export PATH=\"$INSTALL_DIR:\$PATH\")" >&2
fi

echo "installed: $INSTALL_PATH"
echo "enjoy! 🙏✨"
