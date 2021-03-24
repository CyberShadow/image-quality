#!/bin/bash
set -eEuo pipefail

exec ./docker-run.sh python ./quality.py "$@"
