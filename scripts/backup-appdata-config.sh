#!/usr/bin/env bash
# Backup frigate-buffer runtime config on Unraid/Tower (run on the host, not in container).
set -euo pipefail

APPDATA="${APPDATA:-/mnt/user/appdata/frigate_buffer}"
STAMP="$(date +%Y%m%d-%H%M%S)"

cp -a "${APPDATA}/config.yaml" "${APPDATA}/config.yaml.backup"
echo "Wrote ${APPDATA}/config.yaml.backup"

if [[ -f "${APPDATA}/.env" ]]; then
  cp -a "${APPDATA}/.env" "${APPDATA}/.env.backup"
  echo "Wrote ${APPDATA}/.env.backup"
fi

ls -la "${APPDATA}/config.yaml" "${APPDATA}/config.yaml.backup" 2>/dev/null || true
