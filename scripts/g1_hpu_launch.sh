#!/bin/sh
# Launch from the existing G1 1.10 HPU release environment.
set -eu
root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
sdk_lib=${GALBOT_G1_SDK_LIB:-/userdata/pak/output/hpu_release/lib}
export PYTHONPATH="$sdk_lib${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$sdk_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /usr/bin/python3 "$root/worker.py" --profile "$1"
