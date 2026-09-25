#!/usr/bin/env bash
# Installa (o aggiorna) il timer systemd --user del job notturno.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dst="$HOME/.config/systemd/user"
mkdir -p "$dst"
cp "$here/config/systemd/polimi-notes-nightly.service" "$here/config/systemd/polimi-notes-nightly.timer" "$dst/"
systemctl --user daemon-reload
systemctl --user enable --now polimi-notes-nightly.timer
echo "Timer installato:"
systemctl --user list-timers polimi-notes-nightly.timer --no-pager
echo
echo "Lancio manuale:   systemctl --user start polimi-notes-nightly.service"
echo "Log:              journalctl --user -u polimi-notes-nightly.service -f   (o output/auto/nightly.log)"
