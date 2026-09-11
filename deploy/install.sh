#!/usr/bin/env bash
# Provision a Freecher host (Ubuntu 24.04, ARM64 or x86_64).
#
# Idempotent: safe to re-run after a code change or a config edit. It installs
# system packages, creates the service account and state directories, builds the
# virtualenv, installs the units, and refuses to enable anything until
# `freecher-worker preflight` passes -- a host that cannot burn subtitles or
# reach R2 should not be accepting jobs.
#
#   sudo ./deploy/install.sh            # from a checkout at /opt/freecher/app
#   sudo ./deploy/install.sh --no-start # provision only
set -euo pipefail

APP_DIR=${FREECHER_APP_DIR:-/opt/freecher/app}
VENV_DIR=${FREECHER_VENV_DIR:-/opt/freecher/venv}
STATE_DIR=/var/lib/freecher
LOG_DIR=/var/log/freecher
ENV_DIR=/etc/freecher
ENV_FILE="$ENV_DIR/freecher.env"
SERVICE_USER=freecher
START=1

for arg in "$@"; do
  case "$arg" in
    --no-start) START=0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# ffmpeg from Ubuntu is built with libass; libgl1/libglib2.0-0 are what
# opencv-python-headless still dlopens for the YuNet detector. The font
# packages matter because libass silently substitutes a default face for a
# family it cannot find, which changes how every clip looks -- DejaVu Sans is
# the second name in the configured subtitle stack, so having it makes the
# fallback the one that was actually designed for.
apt-get install -y -qq \
  python3-venv python3-dev build-essential pkg-config \
  ffmpeg libgl1 libglib2.0-0t64 curl ca-certificates git rsync \
  fonts-dejavu-core fonts-liberation2

echo "==> service account and directories"
id -u "$SERVICE_USER" >/dev/null 2>&1 || useradd --system --home "$STATE_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0755 "$STATE_DIR" "$STATE_DIR/jobs" "$STATE_DIR/runs" "$LOG_DIR"
install -d -m 0750 "$ENV_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  install -m 0640 -o root -g "$SERVICE_USER" "$HERE/freecher.env.example" "$ENV_FILE"
  echo "    wrote $ENV_FILE from the template — FILL IN THE R2 CREDENTIALS before starting"
else
  chown root:"$SERVICE_USER" "$ENV_FILE"; chmod 0640 "$ENV_FILE"
  echo "    $ENV_FILE already exists; left untouched"
fi

echo "==> application at $APP_DIR"
if [[ "$(cd "$HERE/.." && pwd)" != "$APP_DIR" ]]; then
  install -d "$(dirname "$APP_DIR")"
  # .env is excluded on purpose: a deployed host takes its configuration from
  # $ENV_FILE via systemd, and a stray .env in the app directory would silently
  # win over it wherever the process happens to be started from.
  rsync -a --delete --exclude .git --exclude .venv --exclude runs \
    --exclude research --exclude .env --exclude __pycache__ \
    "$HERE/../" "$APP_DIR/"
fi
rm -f "$APP_DIR/.env"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$APP_DIR"

echo "==> virtualenv at $VENV_DIR"
[[ -x "$VENV_DIR/bin/python" ]] || python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip wheel
# -e keeps the console scripts pointing at $APP_DIR, so a `git pull` is a
# restart rather than a reinstall.
"$VENV_DIR/bin/pip" install --quiet -e "$APP_DIR"
"$VENV_DIR/bin/pip" install --quiet --upgrade yt-dlp
chown -R "$SERVICE_USER":"$SERVICE_USER" "$VENV_DIR"

echo "==> operator helper"
install -m 0755 "$HERE/freecher-run" "$HERE/freecher-run-override" /usr/local/bin/

echo "==> systemd units"
install -m 0644 "$HERE/freecher-api.service" "$HERE/freecher-worker.service" /etc/systemd/system/
systemctl daemon-reload

echo "==> preflight"
# Run it exactly as the units will: as the service user, from the service
# working directory, with only $ENV_FILE for configuration.
if ! ( cd "$STATE_DIR" && runuser -u "$SERVICE_USER" -- \
       env -i HOME="$STATE_DIR" PATH=/usr/local/bin:/usr/bin:/bin \
       $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs) \
       "$VENV_DIR/bin/freecher-worker" preflight --jobs-dir "$STATE_DIR/jobs" ); then
  echo
  echo "preflight FAILED — fix the checks above, then re-run this script." >&2
  echo "Nothing was enabled or started." >&2
  exit 1
fi

if [[ $START -eq 1 ]]; then
  echo "==> enable + (re)start"
  systemctl enable freecher-api.service freecher-worker.service
  # restart, not `enable --now`: --now only *starts* a unit that is stopped, so
  # on an upgrade it silently left the old code running. The venv install is
  # editable, so new code takes effect only when the process is replaced.
  systemctl restart freecher-api.service freecher-worker.service
  systemctl --no-pager --lines=0 status freecher-api.service freecher-worker.service || true
else
  echo "==> --no-start: units installed but not enabled"
fi

echo
echo "done."
echo "  logs:    journalctl -u freecher-worker -f"
echo "  health:  curl -s localhost:8000/health | python3 -m json.tool"
