#/bin/bash
sudo docker run -it \
	--entrypoint /bin/bash \
	--privileged \
	--network host \
	--device /dev/i2c-1 \
	-v /proc//device-tree:/proc/device-tree:ro \
	-v /sys:/sys:ro \
	rocket
