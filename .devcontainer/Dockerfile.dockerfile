# Use NVIDIA CUDA 12.1 base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch and DGL for CUDA 12.1
COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt

# Default to bash shell
CMD ["/bin/bash"]
