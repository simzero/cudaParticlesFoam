NOTICE: This is a work in progress, subject to change.

# Building

Download OptiX 7.0 for [Linux](https://developer.nvidia.com/designworks/optix/downloads/7.0.0/linux64). You must be a member of the NVIDIA Developer Program to download OptiX.

The following command will install OptiX in a local folder:

```
./NVIDIA-OptiX-SDK-7.0.0-linux64.sh --include-subdir --skip-license
```

Set the following `OptiX_INSTALL_DIR` variable pointing to the OptiX installation path:

```
export OptiX_INSTALL_DIR=${HOME}/cudaParticlesFoam/NVIDIA-OptiX-SDK-7.0.0-linux64
```

and check that the `CUDA_HOME` is set up to `CUDA_HOME`/usr/local/cuda-10.1`:


```
export CUDA_HOME=/usr/local/cuda-10.1
```

Run the following commnad for building the `cudaParticleAdvection` library:

```
make lib
```

And finally build the

```
make applications
```

# Running

## Running locally

Go to one of the tutorials:

```
cd tutorials/incompressible/cudaParticlesUncoupledFoam
```

and execute:

```
./Allrun
```

## Running With docker

You need to first configure your machine for using GPUs within Docker containers. Follow X for instructions.

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

You will see the results and logs. The running containers can be checked with `docker ps` and any container stopped with `docker kill CONTAINER_ID`.

# Credit

This repository is built upon the following repositories:

[RTXAdvect](https://github.com/BinWang0213/RTXAdvect)
[tetMesh](https://github.com/owl-project/tetMeshQueries)
[OpenFOAM](https://develop.openfoam.com/Development/openfoam)

Kudos to the authors!

# Licenses

All the project is licensed under the GNU Lesser General Public License v3.0 except all the code in the `third_party` directory which is Apache Licence 2.0.

OPENFOAM® is a registered trade mark of OpenCFD Limited, producer and distributor of the OpenFOAM software via www.openfoam.com.
