# Setting up Docker + GPU for PyTorch

This markdown file provides detailed instructions on how to setup Tensorflow with the GPU and Docker though you can treat both of these as separate problems, there is a crucial installation step which joins your local GPU to your Docker instance.

| **Feature**               | **Value**                                               |
|---------------------------|---------------------------------------------------------|
| **Platform**                  | Windows 10 Enterprise Edition PC with **admin rights**  |
| **Date**                      | 24th February, 2025                                 |
| **NVIDIA GPU** 🟩             | Quadro P2000                                        |
| **PyTorch version**           | 2.1.0+cu121                                         |
| **CUDA Toolkit**              | 12.1.0                                              |
| **Python Version**            | 3.9.15                                              |
| **Docker Version**  🐳        | 4.38.0                                             |
| **Works with Anaconda** 🐍    | ✔️                                                 |
| **Works with Miniconda** 🐍   | ✔️                                                |


## PyTorch 🔥 + GPU ⚡

Before you attempt to install this please make sure your device has a GPU. There are a few simple ways to check if you are not sure:

1) Use the taskbar: ![NVIDIA Icon on the Taskbar](markdown_imgs/nvidia_icon_taskbar.PNG)
2) Check your task manager: ![Windows Task Manager](markdown_imgs/task_manager_gpu.PNG)

The remaining steps for installing `PyTorch` on your machine will use a local `conda` environment. If you are interested in using a `devcontainer` then install `Docker` and skip ahead to the end of this guide which brings all of this together.

### Create a blank conda 🐍 environment

This step is very precarious. If at any point your install is corrupted (either due to incorrect installation methods or other issues) you will have to delete the environment and start from scratch.

Assuming you have `Anaconda3` or `miniconda3` on your system, in your Windows Anaconda Prompt type the following:

```bash
conda env create --name torch_temporal python=3.9
```

If it worked you will see something pop up asking for a whole bunch of default installs just hit `y` and press Enter ⏎

After it is done you should see something like:

```bash
(base) C:\
```

To activate your environment type:

```bash
conda activate torch_temporal
```

Then you should see:

```bash
(gnn_temporal) C:\
```

If you made it this far, well done! You now have a tucked away virtual environment ready for a lot of software.

## Docker 🐳

Dockerizing your project is a great way to freeze the development environment. It has the added benefit of being a more reproducible mechanism to ensure people working on your project do not have to struggle with packages/versioning etc. This is not your typical run of the mill Docker installation guide and actually includes a little bit of how-tos so you don't get lost in the installation.

### Step 1: Install Docker Desktop

Go to the [Docker homepage](https://www.docker.com/products/docker-desktop/) and install Docker Desktop for Windows 10 ARM-64 edition. It will require a restart, please say yes to all the default options and allow all of its popups.

### Step 2: Setup Docker for VS Code

Check the VS code extensions tab and install Docker.
![VS Code Extensions](markdown_imgs/vs_code_docker_extensions.PNG)
After you are done installing it you should see it on the left hand side of your editor
![Docker VS Code Extension](markdown_imgs/docker_vs_code_icon.PNG)

When you click on it you should see `Containers`, `Images` and `Registries` pop up. Ok everything is ready and we can finally spin our first container.

### Step 3: Checking `WSL` and `nvidia-smi`

Usually after you install Docker you see your GPU on the command line. Type:

```bash
nvidia-smi
```

If you see some output you are all good but if you don't then `WSL` wasn't properly installed. You will need `Powershell` as **admin**. If you launch `Powershell` as **admin** then type:

```powershell
wsl --install
```

When it's done set it to `WSL 2`:

```powershell
wsl --set-default-version 2
```

### Step 4: Create a Dockerfile

Use `File > New Text File` and paste the following:

```dockerfile
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
```
### Step 5: Create `.devcontainer.json`

Note: That already exists for this project! In fact most of these files are there in this repo. This is for a fresh repo.

Example devcontainer:

```json
{
  "name": "TGNN Dev Container",
  "dockerFile": "Dockerfile",
  "runArgs": ["--gpus", "all"],
  "settings": {
    "terminal.integrated.defaultProfile.linux": "bash"
  },
  "extensions": [
    "ms-python.python",
    "ms-vscode-remote.remote-containers"
  ],
  "forwardPorts": [8888],
  "postCreateCommand": "pip install -r requirements.txt",
  "remoteUser": "root"
}
```

### Step 6: Create a `requirements.txt`

Note we are reusing bits of the `requirements.txt` from earlier on:

```
torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
dgl-cu121 -f https://data.dgl.ai/wheels/cu121/repo.html
```

### Step 7: Configure your GPU with Docker

**Note**: This step will cause your system to freeze a lot, this is normal

Before you begin running docker commands, run the Windows Powershell to confirm everything works:

```powershell
docker info
```

If this outputs something and you don't see: `Cannot connect to the Docker daemon` then you will need to reinstall and follow steps 3-6 again.

If it works open VS Code and do `Terminal > New Terminal`:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-devel-ubuntu22.04 nvidia-smi
```

It will check if there is a gpu toolkit for docker and if there isn't it will resort to installing it for you (about 2GB).

If it didn't resort to installing it try:

```powershell
docker pull nvidia/cuda:12.1.1-base-ubuntu22.04
```

The install should complete and then you should run `nvidia-smi` in the container:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If it prints out the chart of GPUs on your system **in the container** we are looking solid!

### Step 8: Spinning the container

1. Under VS Code choose the `Docker` icon on the left sidebar
2. Choose the `+` icon next to `CONTAINERS` and pick `Open Folder in a Container`
3. Navigate to the `.devcontainer` folder of your project and click `Open`

## Installing CUDA Drivers Locally

Alternatively, you could set up an anaconda virtual environment and if you have a local GPU then you can follow these steps to install `CUDA` drivers. Note, this guide covers Windows, the Mac OS X and Linux installs are not covered but are actually easier.

First download the version of `CUDA` compatible with this project: `CUDA 12.1` available on the [NVIDIA CUDA toolkit archive](https://developer.nvidia.com/cuda-12-1-0-download-archive). Follow all the steps and use `Express Installation` and leave every option on the default option.

When that is done, press your Windows icon and search for `Environment Variables`, you should see an option to edit the environment variables. When that pops up click on `Environment Variables`, select `PATH` under `User Variables` and press `New`, then add **each** of these:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\libnvvp
```

After that open `System Variables` and create a **new system variable** called `CUDA_HOME`. Paste the following in it:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
```

Launch a terminal (preferably anaconda prompt) and confirm your driver is working:

```bash
nvcc --version
```

If it says `12.1` you are good to go.

Additionally confirm GPU usage:

```bash
nvidia-smi
```

**Note**: This command will show a much higher version than what is installed, that is normal -> it is not indicative of the highest or lowest version supported, rather the version at the time this GPU was made.

## Installing packages

**Note**: Only attempt this step if you have confirmed there is a working GPU/container/something that can run `torch 12.1` ready on your system.

Now you will need to install the provided `requirements.txt` file:

```bash
pip install -r requirements.txt 
```
**Note**: It will fail for `torch 12.1` this is normal, it's hard to keep that command in `requirements.txt`. When it fails install it separately:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then with your `torch geometric` installation there will be [1 issue](https://github.com/pyg-team/pytorch_geometric/discussions/9023). You will need to modify the import in your conda environment.


