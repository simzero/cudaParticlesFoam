// ======================================================================== //
// Copyright 2019-2020 The Collaborators                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "cuda_runtime_api.h"

#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"

#include <thrust/device_vector.h>

#include "ConvexQuery.h"
#include "RTQuery.h"

//#define DEBUG_CONVEX

namespace advect {


    __device__ int traceIntet(int particleID, vec3d &P_start, vec3d P_end, int tetID_current,
        vec4i* d_tetIndices,
        vec3d* d_vertexPositions,
        vec4i* d_tetfacets,
        vec4i* d_facets,
        FaceInfo* d_faceinfos,
        int &outletFace,
        int inletFace=-2) {
        //Trace particle within a tet using convex polygon method

        double tol = 1e-13;

        int tetID_next = tetID_current;

        const vec3d P_0 = P_start;
        const vec3d P_disp = P_end-P_start;

#ifdef  DEBUG_CONVEX
        ///*
        if (particleID == 48000 || particleID == 0)
            printf("\n----Particle %d (%.15f,%.15f,%.15f)->(%.15f,%.15f,%.15f) \n Disp=(%.15f,%.15f,%.15f)\n",
                particleID,
                P_0.x, P_0.y, P_0.z, P_end.x, P_end.y, P_end.z,
                P_disp.x, P_disp.y, P_disp.z);
        //*/
#endif //  DEBUG_CONVEX

        vec3d Pxf = vec3d(-1.0, -1.0, -1.0);
        vec3d norm;
        double face_dist,dT,dT_min=1.1;
        for (int i = 0; i < 4; ++i) {
            bool skipCheck = false;//Skip inlet face

            Pxf = vec3d(-1.0, -1.0, -1.0);

            const int faceID = d_tetfacets[tetID_current][i];
            const vec4i index = d_facets[faceID];

            if (faceID == inletFace)
                skipCheck = true;

            const vec3d A = d_vertexPositions[index.x];
            const vec3d B = d_vertexPositions[index.y];
            const vec3d C = d_vertexPositions[index.z];

            norm = triNorm(A, B, C);//DeviceTetMesh.cuh
            if (d_faceinfos[faceID].back == tetID_current)
                norm = -norm;

            //We skip inlet face if inlet neighbor tetID is given
            //if (d_faceinfos[faceID].back == inletFaceTet || d_faceinfos[faceID].front == inletFaceTet)
            //    skipCheck = true;

            face_dist = dot(A - P_0, norm);
            dT = face_dist / dot(P_disp, norm);

            //Handle special case
            if (isinf(dT)) dT = -1.0; //line parallel to a face
            //if (abs(dT) < tol) dT = tol;//P_start close to a face         
            //if (abs(face_dist) < tol) face_dist = tol;//P_start close to a face         

            //Particle exit from this face min(max(0,dT)), face_dist <0
            if(!skipCheck)
            if (face_dist<tol && dT > tol && dT <= 1.0 && dT<dT_min) {
                dT_min = dT;

                //Next neighboring tetID
                tetID_next = d_faceinfos[faceID].back;
                if (d_faceinfos[faceID].back == tetID_current)
                    tetID_next = d_faceinfos[faceID].front;

                //Update P_start for the next tet
                Pxf = P_0 + dT * P_disp;
                P_start = Pxf;
                outletFace = faceID;
#ifdef  DEBUG_CONVEX
                if (particleID == 48000 || particleID == 0)
                    printf("Hit..");
#endif
            }
#ifdef  DEBUG_CONVEX
            ///*
            if (particleID == 48000 || particleID == 0)
                printf("--Tet%d Checkface %d f/b (%d/%d) Skip %d\n %e %e NextTet %d \n A(%f,%f,%f) B(%f,%f,%f) C(%f,%f,%f) \n Fxt(%f,%f,%f) norm (%f,%f,%f)\n",
                    tetID_current, faceID, 
                    d_faceinfos[faceID].front, d_faceinfos[faceID].back, inletFace,
                    face_dist, dT, tetID_next,
                    A.x, A.y, A.z,  B.x, B.y, B.z,  C.x, C.y, C.z,
                    Pxf.x, Pxf.y, Pxf.z, 
                    norm.x, norm.y, norm.z);
             //*/
#endif
        }


        //boundary face has d_faceinfos[faceID].back/front < 0 
        if (tetID_next < 0) tetID_next = -1;

        return tetID_next;
    }


    //[Device] Pk tet velocity interpolation
    __global__
        void particleLocator(double4* d_particles,
            int* d_tetIDs,
            vec4d* d_disps,
            int    numParticles,
            vec4i* d_tetIndices,
            vec3d* d_vertexPositions,
            vec4i* d_tetfacets,
            vec4i* d_facets,
            FaceInfo* d_faceinfos)
    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        double4& p = d_particles[particleID];
        vec4d& disp = d_disps[particleID];
        if (!p.w) return;

        const vec3d P = vec3d(p.x, p.y, p.z);
        const vec3d P_disp = vec3d(d_disps[particleID].x, d_disps[particleID].y, d_disps[particleID].z);
        const vec3d P_end = P + P_disp;

        /*
       if (particleID == 50606 || particleID == 0)
           printf("\n----Particle %d tetID %d (%.15f,%.15f,%.15f)->(%.15f,%.15f,%.15f) \n Disp=(%.15f,%.15f,%.15f)\n",
               particleID, d_tetIDs[particleID],
               P.x, P.y, P.z, P_end.x, P_end.y, P_end.z,
               P_disp.x, P_disp.y, P_disp.z);
        */

        vec3d P_start = vec3d(p.x, p.y, p.z);
        int tetID_current = d_tetIDs[particleID];
        int tetID_next = -2;
        int OutFace = -2, InFace = -2;
        for(int i=0;i<50;++i)//Search maximum 50 tets
        {
            tetID_next = traceIntet(particleID, P_start, P_end, tetID_current,
                d_tetIndices,
                d_vertexPositions,
                d_tetfacets,
                d_facets,
                d_faceinfos,
                OutFace,
                InFace);
#ifdef  DEBUG_CONVEX
            //debug
            ///*
            if (particleID == 48000 || particleID == 0)
            printf("SearchID=%d !!Particle %d (%f,%f,%f)->(%f,%f,%f) Tet%d->%d\n", i,particleID,
                P_start.x, P_start.y, P_start.z, P_end.x, P_end.y, P_end.z,
               tetID_current, tetID_next);
            //*/
#endif
            //P_start->P_end is wihtin tet
            if (tetID_next == tetID_current)
                break;

            //Update Inlet face to avoid fault hit detection
            InFace = OutFace;
            //P_start->P_end is hit the wall
            if (tetID_next == -1)
                break;

            //Update current tetID
            tetID_current = tetID_next;
        }

        //Hit the wall: set tetID = -id
        //And move particle at the wall to match with tetID
        if (tetID_next == -1) {
            //p.x = P_start.x;
            //p.y = P_start.y;
            //p.z = P_start.z;
            //disp.x = P_end.x - P_start.x;
            //disp.y = P_end.y - P_start.y;
            //disp.z = P_end.z - P_start.z;
            //tetID_next = -(tetID_current + 1); //Convert to 1-index based ID to avoid the case of -0 = 0
            tetID_next = -(d_tetIDs[particleID] + 1); //Convert to 1-index based ID to avoid the case of -0 = 0
        }

        d_tetIDs[particleID] = tetID_next;
    }
	
	void convexTetQuery(DeviceTetMesh d_mesh, double4* d_particles, vec4d* d_disps, int* inout_tetIDs, int numParticles)
	{
        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        particleLocator << <gridDims, blockDims >> > (d_particles,
            inout_tetIDs,
            d_disps,
            numParticles,
            d_mesh.d_indices,
            d_mesh.d_positions,
            d_mesh.d_tetfacets,
            d_mesh.d_facets,
            d_mesh.d_faceinfos);

        cudaCheck(cudaDeviceSynchronize());
	}

    //----------------------Wall Reflect--------------------


    __device__ void reflectInTet(int particleID, vec3d& Pxf, vec3d& P_end, vec3d& vel, int tetID,
        vec4i* d_tetIndices,
        vec3d* d_vertexPositions,
        vec4i* d_tetfacets,
        vec4i* d_facets,
        FaceInfo* d_faceinfos) {

        double tol = 1e-13;
        const vec3d P_disp = P_end - Pxf;

        //Find the Exit face and do reflect
        vec3d norm, P_reflect, u_reflect;
        double dT, face_dist;
        for (int i = 0; i < 4; ++i) {
            const int faceID = d_tetfacets[tetID][i];
            const vec4i index = d_facets[faceID];

            const vec3d A = d_vertexPositions[index.x];
            const vec3d B = d_vertexPositions[index.y];
            const vec3d C = d_vertexPositions[index.z];

            norm = triNorm(A, B, C);//DeviceTetMesh.cuh
            if (d_faceinfos[faceID].back == tetID)//outward normal
                norm = -norm;

            face_dist = dot(A - Pxf, norm);
            dT = face_dist / dot(P_disp, norm);

            //Handle special case
            if (isinf(dT)) dT = -1.0; //line parallel to a face
            if (abs(dT) < tol) dT = tol;//P_start close to a face         
            if (abs(face_dist) < tol) face_dist = tol;

#ifdef  DEBUG_CONVEX
            ///*
            if (particleID == 48000 || particleID == 0)
                printf("\nrCheck----%d (%f,%f,%f)->(%f,%f,%f) Tet%d ExitFaceID %d f/b (%d/%d) \n %e %e \n A(%f,%f,%f) B(%f,%f,%f) C(%f,%f,%f) \n norm (%f,%f,%f)\n",
                    particleID,
                    Pxf.x, Pxf.y, Pxf.z, P_end.x, P_end.y, P_end.z,
                    tetID, faceID,
                    d_faceinfos[faceID].front, d_faceinfos[faceID].back,
                    face_dist, dT,
                    A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z,
                    -norm.x, -norm.y, -norm.z);
            //*/
#endif
            //Particle exit from this face
            if (dT == tol || face_dist==tol) {
                vec3d nw=-norm;//inward normal vector

                //Specular reflection point
                P_reflect = P_end - (1.0 + 1.0) * dot(P_end - A, nw) * nw;
                const vec3d dx_reflect = (P_reflect - Pxf);

                //Specular reflection velocity
                const vec3d Vel = vec3d(vel.x, vel.y, vel.z);
                u_reflect = Vel - (1.0 + 1.0) * dot(Vel, nw) * nw;
#ifdef  DEBUG_CONVEX
                ///*
                if (particleID == 48000 || particleID == 0)
                    printf("rReflect----%d (%.15f,%.15f,%.15f)->(%.15f,%.15f,%.15f)\n Tet%d ExitFaceID %d f/b (%d/%d) \n %e %e \n A(%f,%f,%f) B(%f,%f,%f) C(%f,%f,%f) \n norm (%f,%f,%f) P_reflect(%f,%f,%f)\n",
                    particleID,
                    Pxf.x, Pxf.y, Pxf.z, P_end.x, P_end.y, P_end.z,
                    tetID, faceID,
                    d_faceinfos[faceID].front, d_faceinfos[faceID].back,
                    face_dist, dT,
                    A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z,
                    nw.x, nw.y, nw.z, P_reflect.x, P_reflect.y, P_reflect.z);
                //*/
#endif
                break;
            }
        }

        //Update the end point to the reflect point
        P_end = P_reflect;
        //Update reflect velocity
        vel = u_reflect;
    }

    //[Device] Pk tet velocity interpolation
    __global__
        void convexReflector(double4* d_particles,
            int* d_tetIDs,
            vec4d* d_disps,
            vec4d* d_vels,
            int    numParticles,
            vec4i* d_tetIndices,
            vec3d* d_vertexPositions,
            vec4i* d_tetfacets,
            vec4i* d_facets,
            FaceInfo* d_faceinfos)
    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        double4& p = d_particles[particleID];
        vec4d& vel = d_vels[particleID];
        vec4d& disp = d_disps[particleID];
        if (!p.w) return; //Skip freeze particle

        int tetID = d_tetIDs[particleID];
        if (tetID>=0) return; //Only reflect particle which hit the wall

        vec3d P_start = vec3d(p.x, p.y, p.z);
        vec3d P_disp = vec3d(disp.x, disp.y, disp.z);
        vec3d P_end = P_start + P_disp;
        vec3d u_reflect = vec3d(vel.x, vel.y, vel.z);

        int tetID_current = -tetID-1;//Convert to a postive 0-based id
        int tetID_next = -2;
        int OutFace = -2, InFace = -2;

        vec3d P_hit(-1.0,-1.0,-1.0);
        for (int j = 0; j < 5; ++j) {//Handel multiple wall reflection
            
            //Search and update the next tetID and check another wall hit
            for (int i = 0; i < 50; ++i)//Search maximum 10 tets
            {
                //debug 
                /*
               if (particleID == 16676 && i==0)
                printf("[%d][%d] !!!!!Particle %d Pxf (%.15f,%.15f,%.15f) \n (%.15f,%.15f,%.15f)->(%.15f,%.15f,%.15f)\n Disp(%.15f,%.15f,%.15f)\n", j, i, particleID,
                    P_hit.x, P_hit.y, P_hit.z,
                    P_start.x, P_start.y, P_start.z, P_end.x, P_end.y, P_end.z,
                    P_disp.x, P_disp.y, P_disp.z);
                */

                //Update P_start
                tetID_next = traceIntet(particleID, P_start, P_end, tetID_current,
                    d_tetIndices,
                    d_vertexPositions,
                    d_tetfacets,
                    d_facets,
                    d_faceinfos,
                    OutFace,
                    InFace);
#ifdef  DEBUG_CONVEX
#endif
                /*
                if (particleID == 48000 || particleID == 0)
                    printf("[%d][%d] !!!!!Particle %d Tet%d->%d Good:%d HitWall:%d\n", j, i, particleID,
                    tetID_current, tetID_next, tetID_next == tetID_current, tetID_next == -1);
                */


                //P_start->P_end is wihtin tet
                if (tetID_next == tetID_current)
                    break;

                //Update Inlet face to avoid fault hit detection
                InFace = OutFace;
                //P_start->P_end is hit the wall
                if (tetID_next == -1)
                    break;

                //Update current tetID
                tetID_current = tetID_next;
            }
            
            //Didn't find another hit
            if (tetID_next == tetID_current && tetID_next!=-1) break;

            //Reflect particle from wall, update P_end
            P_hit = P_start;
            reflectInTet(particleID, P_hit, P_end, u_reflect, tetID_current,
                d_tetIndices,
                d_vertexPositions,
                d_tetfacets,
                d_facets,
                d_faceinfos);
            //Set previous ID as -3 after reflection, this will help traceInTet to skip boundary face check
            //InFace = OutFace;
        }

        P_disp = (P_end - P_hit);
        /*
        if (particleID == 63501)
        printf("%d (%f,%f,%f)->(%f,%f,%f) \n Hit (%.15f,%.15f,%.15f)\n Reflect Disp (%.15f,%.15f,%.15f) Vel (%f,%f,%f)\n", particleID,
            p.x, p.y, p.z, P_end.x, P_end.y, P_end.z,
            P_hit.x, P_hit.y, P_hit.z,
            P_disp.x, P_disp.y, P_disp.z, u_reflect.x, u_reflect.y, u_reflect.z);
        */
         
        //Update reflect pos,disp and velocity
        p.x = P_hit.x;
        p.y = P_hit.y;
        p.z = P_hit.z;
        disp.x = P_disp.x;
        disp.y = P_disp.y;
        disp.z = P_disp.z;
        vel.x = u_reflect.x; 
        vel.y = u_reflect.y; 
        vel.z = u_reflect.z;

        //Update tetID
        d_tetIDs[particleID] = tetID_next;
    }

    void convexWallReflect(DeviceTetMesh d_mesh, int* d_tetIDs,
        Particle* d_particles, vec4d* d_vels, vec4d* d_disps, int numParticles)
    {
        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        //printf("-----Convex Wall Reflection-------\n");

        convexReflector << <gridDims, blockDims >> > (d_particles,
            d_tetIDs,
            d_disps,
            d_vels,
            numParticles,
            d_mesh.d_indices,
            d_mesh.d_positions,
            d_mesh.d_tetfacets,
            d_mesh.d_facets,
            d_mesh.d_faceinfos);

        cudaCheck(cudaDeviceSynchronize());
    }


    //---------------Debug--------------------
    __global__
        void printTet(Particle* d_particles, vec4d* d_disps,
            int* d_tetIDs, int numParticles,
            vec3d* d_tetVerts, vec4i* d_tetInds
        ) {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        Particle& p = d_particles[particleID];
        vec4d& disp = d_disps[particleID];

        const vec3d P = vec3d(p.x, p.y, p.z);
        const vec3d P_next = P + vec3d(disp.x, disp.y, disp.z);
        int tetID = d_tetIDs[particleID];
        const vec4i index_tet = d_tetInds[tetID];

        const vec3d A = d_tetVerts[index_tet.x];
        const vec3d B = d_tetVerts[index_tet.y];
        const vec3d C = d_tetVerts[index_tet.z];
        const vec3d D = d_tetVerts[index_tet.w];

        //
        const vec4d baryCoord = tetBaryCoord(P, A, B, C, D);
        const vec4d baryCoord_next = tetBaryCoord(P_next, A, B, C, D);
        printf("\nParticle (%.15lf,%.15lf,%.15lf)->(%.15lf,%.15lf,%.15lf) \nBaryP (%.15lf,%.15lf,%.15lf,%.15f)\nBaryP_next (%.15lf,%.15lf,%.15lf,%.15f)\n",
            P.x, P.y, P.z, P_next.x, P_next.y, P_next.z,
            baryCoord.x, baryCoord.y, baryCoord.z, baryCoord.w,
            baryCoord_next.x, baryCoord_next.y, baryCoord_next.z, baryCoord_next.w);

        printf("Tet%d (%d,%d,%d,%d)\n (%.15lf,%.15lf,%.15lf)\n (%.15lf,%.15lf,%.15lf)\n (%.15lf,%.15lf,%.15lf)\n (%.15lf,%.15lf,%.15lf)\n",
            tetID, index_tet.x, index_tet.y, index_tet.z, index_tet.w,
            A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z, D.x, D.y, D.z);

        printf("TetID orig %d\n", tetID);
    }

    void testNStracing(OptixQuery& cellLocator, DeviceTetMesh devMesh)
    {
        int numParticles = 1;
        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        //Example 1. Tol is too small as 1e-4, Reflector fail
        //thrust::device_vector<Particle> particles(numParticles, make_double4(75.875366330347177, -0.001774039097085, -655.289915166197261, true));
        //thrust::device_vector<vec4d> disps(numParticles, vec4d(0.000207797177225, -0.000140655460453, 0.000291871871468, 0.0));

        //Example 2. cellLocator confused about its front and back
        thrust::device_vector<Particle> particles(numParticles, make_double4(77.407653286554819, -0.226291355139613, -655.599157683661019, true));
        thrust::device_vector<vec4d> disps(numParticles, vec4d(-0.000234741263335, -0.000385634696309, 0.000501310939382, 0.0));
        //thrust::device_vector<int> tetIDs(numParticles, 70173);
        thrust::device_vector<int> tetIDs(numParticles, 10107);

        //thrust::device_vector<Particle> particles(numParticles, make_double4(76.898119015792389, -0.000194660156722, -655.079991263042189, true));
        //thrust::device_vector<vec4d> disps(numParticles, vec4d(0.000098090620440, 0.000242651898842, 0.000352390406405, 0.0));
          
        Particle* d_particles = thrust::raw_pointer_cast(particles.data());
        vec4d* d_disps = thrust::raw_pointer_cast(disps.data());

        thrust::device_vector<vec4d> vels(numParticles, vec4d(0.1, -0.1, 0.01, 0.0));
        vec4d* d_vels = thrust::raw_pointer_cast(vels.data());

        int* d_tetIDs = thrust::raw_pointer_cast(tetIDs.data());


        //Get the initial cell location
        //cellLocator.query_sync(d_particles, d_tetIDs, numParticles);
        RTQuery(cellLocator, devMesh,
            d_particles, d_tetIDs, numParticles);

        //Some movement to obtain the disp
        printTet << <gridDims, blockDims >> > (d_particles, d_disps,
            d_tetIDs,numParticles,
            devMesh.d_positions, devMesh.d_indices);
        cudaCheck(cudaDeviceSynchronize());

        //Locate cell and move particle location to hit point
        particleLocator << <gridDims, blockDims >> > (d_particles,
            d_tetIDs,
            d_disps,
            numParticles,
            devMesh.d_indices,
            devMesh.d_positions,
            devMesh.d_tetfacets,
            devMesh.d_facets,
            devMesh.d_faceinfos);
        cudaCheck(cudaDeviceSynchronize());


        convexReflector << <gridDims, blockDims >> > (d_particles,
            d_tetIDs,
            d_disps,
            d_vels,
            numParticles,
            devMesh.d_indices,
            devMesh.d_positions,
            devMesh.d_tetfacets,
            devMesh.d_facets,
            devMesh.d_faceinfos);
        cudaCheck(cudaDeviceSynchronize());

        //Some movement to obtain the disp
        printTet << <gridDims, blockDims >> > (d_particles, d_disps,
            d_tetIDs, numParticles,
            devMesh.d_positions, devMesh.d_indices);
        cudaCheck(cudaDeviceSynchronize());

        //exit(1);
    }

}
