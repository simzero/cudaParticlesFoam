NOTICE: This is a work in progress, subject to change.

# Description

GPU-accelerated particle tracking for OpenFOAM. You can use this repository without any compilation. Jump to [Running with docker](#running-with-docker) for more details, or follow the instructions below for a native installation.

# Credits

This repository is built upon the following repositories:

- [RTXAdvect](https://github.com/BinWang0213/RTXAdvect)
- [tetMeshQueries](https://github.com/owl-project/tetMeshQueries)
- [OpenFOAM](https://develop.openfoam.com/Development/openfoam)

Kudos to the authors!

# Requirements

- NVIDIA CUDA Toolkit 10.1
- OptiX 7.0
- OpenFOAM v2106

# Installing dependencies

Check the [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for installing the NVIDIA CUDA Toolkit 10.1. Note that the maximum supported GCC version is 8.

Download [OptiX 7.0 for Linux](https://developer.nvidia.com/designworks/optix/downloads/7.0.0/linux64). You must be a member of the NVIDIA Developer Program to download OptiX.

The following command will install OptiX in a local folder:

```console
./NVIDIA-OptiX-SDK-7.0.0-linux64.sh --include-subdir --skip-license
```

Only OpenFOAM v2106 is currently supported. Compile and [install OpenFOAM v2106](https://develop.openfoam.com/Development/openfoam/-/blob/OpenFOAM-v2106.211215/doc/Build.md) environment before continuing.

# Building

Set the environment variables in `etc/bashrc` pointing to your installation paths, for example:

```bash
export RTX=false
export OptiX_INSTALL_DIR=${HOME}/cudaParticlesFoam/NVIDIA-OptiX-SDK-7.0.0-linux64
export CUDA_HOME=/usr/local/cuda-10.1
```

If your graphics card has ray tracing cores set `RTX=true` for additional hardware acceleration [1].


Set the environment for this repository with:

```console
source etc/bashrc
```

Load OpenFOAM's environment:

```console
source ${HOME}/OpenFOAM/OpenFOAM-v2106/etc/bashrc
```

Run the following commnad for building the `cudaParticleAdvection` library:

```console
make lib
```

And finally, build the OpenFOAM solvers:

```console
make applications
```

If you change the variables on `etc/bashrc` do a `make clean` anre repeat the process before building `lib` and `applications` again.

# Running

## Running with native installation

Go to one of the tutorials:

```console
cd tutorials/incompressible/cudaParticlesUncoupledFoam/pitzDaily
```

Finally, run the tutorial with:

```console
./Allrun
```

## Running with Docker

You need to first configure your machine for using GPUs within Docker containers. Follow this [link](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html) for instructions.

Set up the environment with:

```console
source etc/bashrc
```

Go to one of the tutorials:

```console
cd tutorials/incompressible/cudaParticlesUncoupledFoam/pitzDaily
```

Finally, run the tutorial with:

```console
runWithDocker ./Allrun
```

You will see the results and logs. Running containers can be checked with `docker ps`. Containers can be killed with `docker kill CONTAINER_ID`.

# References

- [1] Wang, Bin, et al. "[An GPU-accelerated particle tracking method for Eulerian–Lagrangian simulations using hardware ray tracing cores.](https://www.sciencedirect.com/science/article/abs/pii/S0010465521003337)" Computer Physics Communications 271 (2022): 108221. 


# Licenses

The whole project is licensed under the GNU General Public License v3.0 except the code inside the `third_party` directory which is licensed under Apache Licence 2.0.

OPENFOAM® is a registered trade mark of OpenCFD Limited, producer and distributor of the OpenFOAM software via www.openfoam.com.
