//cudaTimer timer;

#include <iostream>
#include <fstream>

HostTetMesh* hostTetMesh = new HostTetMesh;
DeviceTetMesh devMesh;

pointList allCells;
vectorList allU;
List<vectorList> gatheredU(Pstream::nProcs());

double BVHTime = 0.0;
Particle* d_particles = nullptr;//x,y,z,statusID
int* d_particles_tetIDs = nullptr;
int* d_particles_triIDs = nullptr;
int* d_particles_ConvextetIDs = nullptr;
vec4d* d_particle_disps = nullptr;//x,y,z,tetID_last
vec4d* d_particle_vels =nullptr;//vx,vy,vz
curandState_t* rand_states = nullptr;;
std::vector<std::vector<vec3f>> trajectories;

boundBox bb(point(0, 0, 0), point(30, 30, 30));
boundBox seedingBox =  cudaParticleAdvectionDict.getOrDefault("seedingBox", bb);
label numParticles = cudaParticleAdvectionDict.getOrDefault("numParticles", 1000);
scalar particleStartTime = cudaParticleAdvectionDict.getOrDefault("startTime", 0.0);
scalar particleEndTime = cudaParticleAdvectionDict.getOrDefault("endTime", 1e05);
scalar dt = cudaParticleAdvectionDict.getOrDefault("dt", 1e-4);
scalar diffusionCoeff = cudaParticleAdvectionDict.getOrDefault("diffusionCoeff", 5.7e-6);
int saveInterval = cudaParticleAdvectionDict.getOrDefault("saveInterval", 10);

vector minSeedingBox = seedingBox.min();
vector maxSeedingBox = seedingBox.max();
box3d initBox(vec3d(minSeedingBox.x(), minSeedingBox.y(), minSeedingBox.z()),
vec3d(maxSeedingBox.x(), maxSeedingBox.y(), maxSeedingBox.z()));

int tetsPerCell = 12;
bool usingAdvection = true;
bool usingBrownianMotion = true;
bool reflectWall = true;
bool saveStreamlinetoFile = false;

std::string objTrajectoryFileName;
std::string vtkStreamlineFileName = "Streamline.vtk";
std::string VelocityInterpMethod = "TetVelocity";

if(Pstream::nProcs() == 1)
{
    int nCells = mesh.cells().size();
    int nPoints = mesh.points().size();
    int nTets = nCells*tetsPerCell;

    hostTetMesh->positions.reserve(nCells + nPoints);
    hostTetMesh->indices.reserve(nTets);
    hostTetMesh->velocities.reserve(nTets);

    allCells = mesh.C();

    forAll(mesh.cells(), celli)
    {
        List<tetIndices> cellTets = polyMeshTetDecomposition::cellTetIndices
        (
            mesh,
            celli
        );

        forAll(cellTets, tetI)
        {
            const tetIndices& tetIs = cellTets[tetI];
            const triFace triIs = tetIs.faceTriIs(mesh);

	    label v0 = celli;
	    label v1 = triIs[0];
	    label v2 = triIs[1];
            label v3 = triIs[2];
            
            vec4i v(nPoints + v0, v1, v2, v3);
	    hostTetMesh->indices.push_back(v);
            hostTetMesh->velocities.push_back(vec3d(U[celli].x(),
                                              U[celli].y(),
                                              U[celli].z()));
        }
    }

    forAll(mesh.points(), pointi)
    {
        vector pos = mesh.points()[pointi];
        hostTetMesh->positions.push_back(vec3d(pos.x(), pos.y(), pos.z()));
        hostTetMesh->worldBounds.extend(vec3d(pos.x(), pos.y(), pos.z()));
    }

    forAll(mesh.C(), celli)
    {
        vector pos = mesh.C()[celli];
        hostTetMesh->positions.push_back(vec3d(pos.x(), pos.y(), pos.z()));
        hostTetMesh->worldBounds.extend(vec3d(pos.x(), pos.y(), pos.z()));
    }
   
    HostTetMesh hostBoundaryMesh = hostTetMesh->getBoundaryMesh();

    devMesh.upload(*hostTetMesh);
    DeviceBdMesh devBdMesh;
    devBdMesh.upload(hostBoundaryMesh);

    cudaTimer timer_BVH;
    timer_BVH.start();
    OptixQuery tetQueryAccelerator((double3 *) hostTetMesh->positions.data(),
                                       hostTetMesh->positions.size(),
                                       (int4 *) hostTetMesh->indices.data(),
                                       hostTetMesh->indices.size());
    BVHTime = timer_BVH.stop();
    printf("#adv BVH Construction Time=%lf  ms\n", BVHTime);

    cudaCheck(cudaMalloc(&d_particles, numParticles * sizeof(Particle)));
    cudaCheck(cudaMalloc(&d_particles_tetIDs, numParticles * sizeof(int)));
    cudaCheck(cudaMalloc(&d_particles_triIDs, numParticles * sizeof(int)));
    cudaCheck(cudaMalloc(&d_particles_ConvextetIDs, numParticles * sizeof(int)));
    cudaCheck(cudaMemset(d_particles_ConvextetIDs, -1, numParticles * sizeof(int)));
    cudaCheck(cudaMalloc(&d_particle_disps, numParticles * sizeof(vec4d)));
    cudaCheck(cudaMemset(d_particle_disps, 0.0, numParticles * sizeof(vec4d)));
    cudaCheck(cudaMalloc(&d_particle_vels, numParticles * sizeof(vec4d)));
    cudaCheck(cudaMemset(d_particle_vels, 0.0, numParticles * sizeof(vec4d)));
    cudaCheck(cudaMalloc(&rand_states, numParticles * sizeof(curandState_t)));

    initRandomGenerator(numParticles, rand_states);

    size_t bytes = numParticles * sizeof(Particle)
                   + numParticles * sizeof(int)
                   + numParticles * sizeof(vec4d)
                   + numParticles * sizeof(vec4d)
                   + numParticles * sizeof(curandState_t);

    std::cout << OWL_TERMINAL_YELLOW
              << "#adv: particle mem: "
              << prettyNumber(bytes)
              << OWL_TERMINAL_DEFAULT
              << std::endl;

    cudaInitParticles(d_particles, numParticles, initBox);
    cudaCheck(cudaDeviceSynchronize());

#ifdef  ConvexPoly
    Info<< "ADVECT_MODE: ConvexPoly" <<endl;
    RTQuery(tetQueryAccelerator, devMesh,
            d_particles, d_particles_ConvextetIDs, numParticles);
#else
    Info<< "ADVECT_MODE: RTX" <<endl;
    RTQuery(tetQueryAccelerator, devMesh,
            d_particles, d_particles_tetIDs, numParticles);
#endif

#ifndef  ConvexPoly
    cudaReportParticles(numParticles, d_particles_tetIDs);
#else
    cudaReportParticles(numParticles, d_particles_ConvextetIDs);
#endif
    if (usingAdvection) {
        cudaAdvect(d_particles,
#ifndef  ConvexPoly
                   d_particles_tetIDs,
#else
                   d_particles_ConvextetIDs,
#endif
                   d_particle_vels,
                   d_particle_disps,
                   dt,
                   numParticles,
                   devMesh.d_indices,
                   devMesh.d_positions,
                   devMesh.d_velocities,
                   VelocityInterpMethod);
    }

    writeParticles2VTU(0, d_particles, d_particle_vels, d_particles_tetIDs,
                       numParticles, d_particles_ConvextetIDs);
      

    delete hostTetMesh;
}
else
{
    gatheredU[Pstream::myProcNo()] = U;
    Pstream::gatherList(gatheredU);

    List<pointList> gatheredPoints(Pstream::nProcs());
    gatheredPoints[Pstream::myProcNo()] = mesh.points();
    Pstream::gatherList(gatheredPoints);

    List<pointList> gatheredCells(Pstream::nProcs());
    gatheredCells[Pstream::myProcNo()] = mesh.C();
    Pstream::gatherList(gatheredCells);

    List<labelList> gatheredV0(Pstream::nProcs());
    List<labelList> gatheredV1(Pstream::nProcs());
    List<labelList> gatheredV2(Pstream::nProcs());
    List<labelList> gatheredV3(Pstream::nProcs());

    labelList listV0(mesh.cells().size()*tetsPerCell);
    labelList listV1(mesh.cells().size()*tetsPerCell);
    labelList listV2(mesh.cells().size()*tetsPerCell);
    labelList listV3(mesh.cells().size()*tetsPerCell);

    int k = 0;
   
    const globalIndex globalPoints(mesh.points().size());
    const globalIndex globalCells(mesh.cells().size());
    
    forAll(mesh.cells(), celli)
    {
        List<tetIndices> cellTets = polyMeshTetDecomposition::cellTetIndices
        (
            mesh,
            celli
        );

        forAll(cellTets, tetI)
        {
            const tetIndices& tetIs = cellTets[tetI];
            const triFace triIs = tetIs.faceTriIs(mesh);

	    label v0 = globalCells.toGlobal(celli);
	    label v1 = globalPoints.toGlobal(triIs[0]);
	    label v2 = globalPoints.toGlobal(triIs[1]);
            label v3 = globalPoints.toGlobal(triIs[2]);

            listV0[k] = v0;
            listV1[k] = v1;
            listV2[k] = v2;
            listV3[k] = v3;

            k++;
        }
    }

    gatheredV0[Pstream::myProcNo()] = listV0;
    gatheredV1[Pstream::myProcNo()] = listV1;
    gatheredV2[Pstream::myProcNo()] = listV2;
    gatheredV3[Pstream::myProcNo()] = listV3;

    Pstream::gatherList(gatheredV0);
    Pstream::gatherList(gatheredV1);
    Pstream::gatherList(gatheredV2);
    Pstream::gatherList(gatheredV3);

    pointList allPoints
    (
        ListListOps::combine<pointList>
        (
            gatheredPoints, accessOp<pointList>()
        )
    );

    allCells = ListListOps::combine<pointList>
               (
                   gatheredCells, accessOp<pointList>()
               );

    labelList allV0
    (
        ListListOps::combine<labelList>
        (
            gatheredV0, accessOp<labelList>()
        )
    );

    labelList allV1
    (
        ListListOps::combine<labelList>
        (
            gatheredV1, accessOp<labelList>()
        )
    );

    labelList allV2
    (
        ListListOps::combine<labelList>
        (
            gatheredV2, accessOp<labelList>()
        )
    );

    labelList allV3
    (
        ListListOps::combine<labelList>
        (
            gatheredV3, accessOp<labelList>()
        )
    );

    allU = ListListOps::combine<vectorList>
           (
               gatheredU, accessOp<vectorList>()
           );

    if(Pstream::master())
    {
        std::vector<std::vector<double>> vec;

        int nCells = allCells.size();
        int nPoints = allPoints.size();
        int nTets = nCells*tetsPerCell;
    
        hostTetMesh->indices.reserve(nTets);
        hostTetMesh->velocities.reserve(nTets);

        int k = 0;

        DynamicList<vector> pointsSeen;
    
        List<label> pointsMap(allPoints.size());

        forAll(allPoints, pointi)
        {
            vector pos = allPoints[pointi];

            label id = pointsSeen.find(pos);
            if (id == -1)
            {
                pointsSeen.append(pos);
	        pointsMap[pointi] = k;
	        k++;
            }
            else
            {
	        pointsMap[pointi] = id;
            }
        }

        hostTetMesh->positions.reserve(pointsSeen.size()+nCells);
    
        forAll(pointsSeen, pointi)
        {
            vector pos = pointsSeen[pointi];
            hostTetMesh->positions.push_back(vec3d(pos.x(), pos.y(), pos.z()));
            hostTetMesh->worldBounds.extend(vec3d(pos.x(), pos.y(), pos.z()));
        }

        forAll(allCells, celli)
        {
            vector pos = allCells[celli];
            hostTetMesh->positions.push_back(vec3d(pos.x(), pos.y(), pos.z()));
            hostTetMesh->worldBounds.extend(vec3d(pos.x(), pos.y(), pos.z()));
            for (int i = 0; i < tetsPerCell; i++)
	    {
                hostTetMesh->velocities.push_back(vec3d(allU[celli].x(),
                                                  allU[celli].y(),
                                                  allU[celli].z()));
            }
        }

        forAll(allV0, teti)
        {
            vec4i v(pointsSeen.size() + allV0[teti], pointsMap[allV1[teti]], pointsMap[allV2[teti]], pointsMap[allV3[teti]]);
            hostTetMesh->indices.push_back(v);
        }

        HostTetMesh hostBoundaryMesh = hostTetMesh->getBoundaryMesh();

        gatheredPoints.clear();
        gatheredCells.clear();
        gatheredV0.clear();
        gatheredV1.clear();
        gatheredV2.clear();
        gatheredV3.clear();

        allV0.clear();
        allV1.clear();
        allV2.clear();
        allV3.clear();
        allPoints.clear();

        // ------------------------------------------------------------------
        // upload our own cuda data
        // ------------------------------------------------------------------
        // DeviceTetMesh devMesh;
        devMesh.upload(*hostTetMesh);
        DeviceBdMesh devBdMesh;
        devBdMesh.upload(hostBoundaryMesh);

        //CPUTimer timer_BVH;
        cudaTimer timer_BVH;
        timer_BVH.start();
        OptixQuery tetQueryAccelerator((double3 *) hostTetMesh->positions.data(),
                                       hostTetMesh->positions.size(),
                                       (int4 *) hostTetMesh->indices.data(),
                                       hostTetMesh->indices.size());
        BVHTime = timer_BVH.stop();
        printf("#adv BVH Construction Time=%lf  ms\n", BVHTime);
    
        cudaCheck(cudaMalloc(&d_particles, numParticles * sizeof(Particle)));
        cudaCheck(cudaMalloc(&d_particles_tetIDs, numParticles * sizeof(int)));
        cudaCheck(cudaMalloc(&d_particles_triIDs, numParticles * sizeof(int)));
        cudaCheck(cudaMalloc(&d_particles_ConvextetIDs, numParticles * sizeof(int)));
        cudaCheck(cudaMemset(d_particles_ConvextetIDs, -1, numParticles * sizeof(int)));
        cudaCheck(cudaMalloc(&d_particle_disps, numParticles * sizeof(vec4d)));
        cudaCheck(cudaMemset(d_particle_disps, 0.0, numParticles * sizeof(vec4d)));
        cudaCheck(cudaMalloc(&d_particle_vels, numParticles * sizeof(vec4d)));
        cudaCheck(cudaMemset(d_particle_vels, 0.0, numParticles * sizeof(vec4d)));
        cudaCheck(cudaMalloc(&rand_states, numParticles * sizeof(curandState_t)));

        initRandomGenerator(numParticles, rand_states);

        size_t bytes = numParticles * sizeof(Particle)
                       + numParticles * sizeof(int)
                       + numParticles * sizeof(vec4d)
                       + numParticles * sizeof(vec4d)
                       + numParticles * sizeof(curandState_t);

        std::cout << OWL_TERMINAL_YELLOW
                  << "#adv: particle mem: "
                  << prettyNumber(bytes)
                  << OWL_TERMINAL_DEFAULT
                  << std::endl;

        cudaInitParticles(d_particles, numParticles, initBox);
        cudaCheck(cudaDeviceSynchronize());

#ifdef  ConvexPoly
        RTQuery(tetQueryAccelerator, devMesh,
                d_particles, d_particles_ConvextetIDs, numParticles);
#else
        RTQuery(tetQueryAccelerator, devMesh,
                d_particles, d_particles_tetIDs, numParticles);
#endif

#ifndef  ConvexPoly
        cudaReportParticles(numParticles, d_particles_tetIDs);
#else
        cudaReportParticles(numParticles, d_particles_ConvextetIDs);
#endif

        if (usingAdvection) {
            cudaAdvect(d_particles,
#ifndef  ConvexPoly
            d_particles_tetIDs,
#else
            d_particles_ConvextetIDs,
#endif
            d_particle_vels,
            d_particle_disps,
            dt,
            numParticles,
            devMesh.d_indices,
            devMesh.d_positions,
            devMesh.d_velocities,
            VelocityInterpMethod);
        }

        writeParticles2VTU(0, d_particles, d_particle_vels, d_particles_tetIDs,
                           numParticles, d_particles_ConvextetIDs);

	delete hostTetMesh;
    }
    else
    {
        allCells.clear();
    }
}


/*
cudaTimer timer_loop;
CPUTimer timer_loop;
double advectionTime = 0.0;
double diffusionTime = 0.0;
double queryTime = 0.0;
double reflectTime = 0.0;
double moveTime = 0.0;
double IOTime = 0.0;
*/

std::vector<vec3d> newVel;
int step = 0;
