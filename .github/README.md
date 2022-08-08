NOTICE: This is a work in progress, subject to change.

# Building

Download OptiX 7.0 for [Linux](https://developer.nvidia.com/designworks/optix/downloads/7.0.0/linux64). You must be a member of the NVIDIA Developer Program to download OptiX.

The following command will install OptiX in a local folder:

```
./NVIDIA-OptiX-SDK-7.0.0-linux64.sh --include-subdir --skip-license
```

Set the enviroment variables in `etc/bashrc` pointing to your installation paths, for example:

```
export RTX=false
export OptiX_INSTALL_DIR=${HOME}/cudaParticlesFoam/NVIDIA-OptiX-SDK-7.0.0-linux64
export CUDA_HOME=/usr/local/cuda-10.1
```

If your graphics card has ray tracing cores set `RTX=true` for additional hardware acceleration.


Set the environment with:

```
source etc/bashrc
```

Only OpenFOAM v2106 is currently supported. Install and load OpenFOAM's environment before continuing, for example:

```
source ${HOME}/OpenFOAM/OpenFOAM-v2106/etc/bashrc
```

Run the following commnad for building the `cudaParticleAdvection` library:

```
make lib
```

And finally build the OpenFOAM solvers:

```
make applications
```

# Running

## Running with native installation

Go to one of the tutorials:

```
cd tutorials/incompressible/cudaParticlesUncoupledFoam
```

and execute:

```
./Allrun
```

## Running with Docker

You need to first configure your machine for using GPUs within Docker containers. Follow this [link](https://docs.nvidia.com/ai-enterprise/deployment-guide/dg-docker.html) for instructions.

Once you can run Docker containers with GPUs run:

```
source etc/bashrc
```

Go to one of the tutorials:

```
cd tutorials/incompressible/cudaParticlesUncoupledFoam
```

and execute:

```
runWithDocker ./Allrun
```

You will see the results and logs. Running containers can be checked with `docker ps`. Containers can be killed with `docker kill CONTAINER_ID`.

# Credits

This repository is built upon the following repositories:

- [RTXAdvect](https://github.com/BinWang0213/RTXAdvect)
- [tetMeshQueries](https://github.com/owl-project/tetMeshQueries)
- [OpenFOAM](https://develop.openfoam.com/Development/openfoam)

Kudos to the authors!

# Licenses

All the project is licensed under the GNU General Public License v3.0 except the code inside the `third_party` directory which is licensed under Apache Licence 2.0.

OPENFOAM® is a registered trade mark of OpenCFD Limited, producer and distributor of the OpenFOAM software via www.openfoam.com.
