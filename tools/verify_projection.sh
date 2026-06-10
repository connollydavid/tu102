#!/usr/bin/env bash
# thin wrapper; the gate lives in verify_projection.py
exec python3 "$(dirname "$0")/verify_projection.py" "$@"
