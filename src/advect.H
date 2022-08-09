// Copyright (c) 2022 Carlos Peña-Monferrer. All rights reserved.
// CUDA-related code adapted from cudaParticleAdvection.cu
// on https://github.com/BinWang0213/RTXAdvect/tree/ca5f6ae/cuda
//
/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/


if ((runTime.value() >= particleStartTime) && runTime.value() <= particleEndTime)
{
    // Assuming Eulerian timesteps are always much bigger than Lagrangian one
    int nCycles = max(ceil(runTime.deltaT().value() / dt), 1);
    double cycleDt = runTime.deltaT().value() / nCycles;

    Info<< "dtE:" << runTime.deltaT().value() << " dtL: " << dt <<endl;
    Info << "nCycles: " <<  nCycles << " cycleDt: " << cycleDt <<endl;
        
    if(Pstream::nProcs() == 1)
    {
        newVel.clear();
        // TODO: optimize new U update
        forAll(allCells, celli)
        {
            for (int i = 0; i < tetsPerCell; i++)
            {
                newVel.push_back(vec3d(U[celli].x(),
                                       U[celli].y(),
                                       U[celli].z()));
            }
        }
        cudaUpdateVelocity(newVel, allCells.size()*tetsPerCell,
                           devMesh.d_indices,
                           devMesh.d_velocities);
    }
    else
    {
        newVel.clear();
        gatheredU[Pstream::myProcNo()] = U;
        Pstream::gatherList(gatheredU);
        allU = ListListOps::combine<vectorList>
               (
                   gatheredU, accessOp<vectorList>()
               );
        if (Pstream::master())
	{
            // TODO: optimize new U update
            forAll(allCells, celli)
            {
                for (int i = 0; i < tetsPerCell; i++)
		{
                    newVel.push_back(vec3d(allU[celli].x(),
                                           allU[celli].y(),
                                           allU[celli].z()));
                }
            }
            cudaUpdateVelocity(newVel, allCells.size()*tetsPerCell,
                               devMesh.d_indices,
                               devMesh.d_velocities);
	}
    }

    for (int i = 0; i < nCycles; i++)
    {

        if (Pstream::master())
        {
            // TODO: redo timers
            // cudaTimer timer;
            // cudaTimer timer_loop;

            // ... compute advection
            if (usingAdvection) {
                cudaAdvect(d_particles,
#ifndef  ConvexPoly
                           d_particles_tetIDs,
#else
                           d_particles_ConvextetIDs,
#endif
                           d_particle_vels,
                           d_particle_disps,
                           cycleDt,
                           numParticles,
                           devMesh.d_indices,
                           devMesh.d_positions,
                           devMesh.d_velocities,
                           VelocityInterpMethod);
            }

            // ... compute random Brownian motion
            if(usingBrownianMotion)
	    {
               cudaBrownianMotion(d_particles, 
                                  d_particle_disps,
                                  rand_states,
                                  cycleDt,
                                  numParticles,
				  diffusionCoeff);
	    }

#ifndef  ConvexPoly

            RTQuery(devMesh,d_particles,d_particle_disps,
                    d_particles_tetIDs,numParticles);

            if (reflectWall)
                RTWallReflect(devMesh,
                              d_particles_tetIDs,
                              d_particles,
                              d_particle_disps,
                              d_particle_vels,
                              numParticles);
#else  
            // ... Convex Query particle tet location
            convexTetQuery(devMesh, 
                           d_particles,
                           d_particle_disps, 
                           d_particles_ConvextetIDs, 
                           numParticles);

            // ... compute wall reflection
            if (reflectWall)
                convexWallReflect(devMesh, 
                                  d_particles_ConvextetIDs,
                                  d_particles, 
                                  d_particle_vels,
                                  d_particle_disps, 
                                  numParticles);
#endif
#ifndef  ConvexPoly
            // ... Move particles
            cudaMoveParticles(d_particles, d_particle_disps,
                              numParticles, d_particles_tetIDs);
#else
            // ... Move particles
            cudaMoveParticles(d_particles, d_particle_disps,
                              numParticles, d_particles_ConvextetIDs);
#endif

    if (saveStreamlinetoFile)
        if ((step % (saveInterval * 1)) == 0)
            addToTrajectories(d_particles, numParticles, trajectories);
    if ((step % saveInterval) == 0 || runTime.value() == particleEndTime)
        writeParticles2VTU(step + 1, d_particles, d_particle_vels,
                           d_particles_tetIDs, numParticles,
                           d_particles_ConvextetIDs);
    if (saveStreamlinetoFile) {
        if (objTrajectoryFileName.size() > 0)
            saveTrajectories(objTrajectoryFileName, trajectories);
                if (vtkStreamlineFileName.size() > 0)
                    writeStreamline2VTK(vtkStreamlineFileName, trajectories);
       }
    }
/*#ifndef  ConvexPoly
    cudaReportParticles(numParticles, d_particles_tetIDs);
#else
    cudaReportParticles(numParticles, d_particles_ConvextetIDs);
#endif*/
    step++;

    }


    // TODO: print if a debug flag
    /*double runtime = timer.stop();
    printf("#adv: Simulation RunTime=%f ms\n", runtime);
    // printf("#adv: Simulation Performance=%f steps/secs\n", numSteps/(runtime - IOTime )*1000);
	
    double totalTime = BVHTime + advectionTime + diffusionTime + queryTime + reflectTime + moveTime;
    printf("IO is not included to compute time fraction\n");
    printf("\tItem\ttime(s)\tfraction(%%)\n");
    printf("\tBVH init\t%.2f\t\%.2f\n", BVHTime / 1000, BVHTime / totalTime * 100);
    printf("\tAdv\t%.2f\t\%.2f\n", advectionTime/1000, advectionTime / totalTime * 100);
    printf("\tDfs\t%.2f\t\%.2f\n", diffusionTime / 1000, diffusionTime / totalTime * 100);
    printf("\tQry\t%.2f\t\%.2f\n", queryTime / 1000, queryTime / totalTime * 100);
    printf("\tRft\t%.2f\t\%.2f\n", reflectTime / 1000, reflectTime / totalTime * 100);
    printf("\tMov\t%.2f\t\%.2f\n", moveTime / 1000, moveTime / totalTime * 100);
    printf("\tIO\t%.2f\n", IOTime / 1000);
    printf("\tTotal Time = %.2f ms\n", totalTime);
    // printf("\tPerformance = %f steps/secs\n", numSteps / (totalTime-BVHTime) * 1000);*/

}
