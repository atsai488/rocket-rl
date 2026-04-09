#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

docker run -it \
	--entrypoint /bin/bash \
	--privileged \
	--network host \
	--device /dev/i2c-1 \
	-v "${REPO_ROOT}/jetson:/rocket/jetson" \
	-v /proc//device-tree:/proc/device-tree:ro \
	-v /sys:/sys:ro \
	rocket

cd jetson/
python3 rocket_rl.py rocket_direct.onnx
