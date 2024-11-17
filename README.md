# diffuse-plan-learn
PRM + Diffusion RL

## Installation
System requirements: Ubuntu 20.04 / Ubuntu 22.04 and CUDA 11.8
#### 1. Install CUDA 11.8
#### 2. Update GLIBC package to 2.34+
You can ignore this step if you are using Ubuntu 22.04
```
# Add sources in Ubuntu
sudo vi /etc/apt/sources.list

# Add the following line to the file and save
deb http://th.archive.ubuntu.com/ubuntu jammy main

# Update the package list
sudo apt-get update

# Install the latest GLIBC package
sudo apt install libc6

# Verify the GLIBC version (should be 2.34+)
ldd --version
```

#### 3. Create virtual environment
```
# create conda environment
conda create -n prm python=3.10
conda activate prm
```

#### 4. Install IsaacSim packages
```
# install PyTorch 2.2.2 with CUDA 11.8
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# install the Isaac Sim packages necessary for running Isaac Lab
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com
```

#### 5. Install IsaacLab packages
```
# clone the Isaac Lab repository
git clone git@github.com:isaac-sim/IsaacLab.git

# install dependencies using apt
sudo apt install cmake build-essential

# Installs learning frameworks using pip
cd IsaacLab && ./isaaclab.sh --install
```

#### 6. Install other dependencies
```
cd ../ && pip install -r requirements.txt
```

## Testing the MazeBot implementation
You can modify the task configuration in cfg/task/Maze.yaml
```
python tasks/maze/test_mazebot.py
```




