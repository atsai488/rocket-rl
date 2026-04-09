FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /rocket

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python packages that rarely change
RUN pip3 install --no-cache-dir \
    pygame \
    pyPS4Controller \
    spatialmath-python \
    onnxruntime \
    smbus2 \
    adafruit-blinka

RUN pip3 install adafruit-circuitpython-bno055 \
		Jetson.GPIO

# Do not bake source code into the image; mount it from the host at runtime.
RUN mkdir -p /rocket/jetson
ENV PYTHONPATH=/rocket/jetson
VOLUME ["/rocket/jetson"]
