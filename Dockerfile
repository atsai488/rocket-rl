FROM ubuntu:22.04

# Install required apt packages
RUN apt-get update && apt-get install -y \
    git \
    python3-pip

# Clone the repositories
COPY . /rocket
WORKDIR /rocket

# Put spot-private-sdk wheels in spot-rl/external/spot-python-sdk/prebuilt
# ^ This can be done with a `gitman update` or by manual intervention

RUN pip3 install pygame \
                pyPS4Controller \
                spatialmath-python \
                onnxruntime

# Copy the entrypoint script to the container
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
