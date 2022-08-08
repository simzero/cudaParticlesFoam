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

#include "RTQuery.h"

//#define DEBUG_RT

namespace advect {

    struct SearchInfo {
        int tetID;//global tetID, 1-based index(tetID<0), 0-based index (tetID>0)
        int faceID;//global face ID
    };

    __device__ SearchInfo baryTetSearch(int particleID,vec3d P, int tetID_start, vec3d* d_tetVerts, vec4i* d_tetInds,
        vec4i* d_facets, vec4i* d_tetfacets, FaceInfo* d_faceinfos) {
        //Search hosting cell of point P start from tetID_start

        int tetID_search = tetID_start;
        int faceID = -1;
        int tetID_previous = tetID_search;
        for (int i = 0; i < 50; ++i) {//Search maximum 50 times
            const vec4i index_tet = d_tetInds[tetID_search];
            const vec3d A = d_tetVerts[index_tet.x];
            const vec3d B = d_tetVerts[index_tet.y];
            const vec3d C = d_tetVerts[index_tet.z];
            const vec3d D = d_tetVerts[index_tet.w];
            const vec4d bary = tetBaryCoord(P, A, B, C, D);

            const double wmin = reduce_min(bary);

            //Case1. This is a right cell
            if (wmin >= 0.0) break;
            else {//Start to move one of its neighbors
                  //where the facet has minimum negative number has largest possibility
                const int exitfaceID = arg_min(bary);
                faceID = d_tetfacets[tetID_search][exitfaceID];

                //If facet neighbor is out of domain(ID<0), tetID_previous is the exit tet
                tetID_previous = tetID_search;
                //Update search tetID to facet neighbor 
                tetID_search =
                    d_faceinfos[faceID].front == tetID_search
                    ? d_faceinfos[faceID].back
                    : d_faceinfos[faceID].front;

                /*
                if (particleID== 9670)
                printf("%d SearchID%d tetID=%d tetID_i-1 =%d tetSearch=%d \nP (%f,%f,%f)\n Bary(%f,%f,%f,%f)\n", 
                    particleID,i,
                    tetID_start, tetID_previous,tetID_search+1,//1-based to 0-based index facetID
                    P.x, P.y, P.z,
                    bary.x, bary.y, bary.z, bary.w);
                */

                //Case 2. The particle is out of domain, tetID_search = -(last in domain tet+1)
                if (tetID_search < 0) {
                    tetID_search = -(tetID_previous+1);//Set as 1-based index to avoid -0=0 
                    break;
                }
                //Case3. start to next loop of search
            }

        }

        SearchInfo info;
        info.tetID = tetID_search;
        info.faceID = faceID;
        return info;
    }

    __device__ void specularReflect(vec3d P_end, vec3d Vel, int tetID, int faceID,
        vec3d* d_tetVerts, FaceInfo* d_faceinfos, vec4i* d_facets,
        vec3d &P_reflect, vec3d &u_reflect) {
        const vec4i index = d_facets[faceID];
        const vec3d A = d_tetVerts[index.x];
        const vec3d B = d_tetVerts[index.y];
        const vec3d C = d_tetVerts[index.z];

        vec3d norm = triNorm(A, B, C);//DeviceTetMesh.cuh
        if (d_faceinfos[faceID].back == tetID)//inner normal vector 
            norm = -norm;

        //this equation is not sensitive to norm direction
        P_reflect = P_end - (1.0 + 1.0) * dot(P_end - A, norm) * norm;
        u_reflect = Vel - (1.0 + 1.0) * dot(Vel, norm) * norm;
    }

    __global__ void RTreflection(Particle* d_particles, vec4d* d_disps, vec4d* d_vels,
        int* d_tetIDs, int numParticles,
        vec3d* d_tetVerts, vec4i* d_tetInds,
        vec4i* d_facets, vec4i* d_tetfacets, FaceInfo* d_faceinfos)
    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        int tetID = d_tetIDs[particleID];
        if (tetID >= 0) return;

        tetID = -(tetID + 1);//Convert 1-based index back to 0-based index
        Particle& p = d_particles[particleID];
        vec4d& disp = d_disps[particleID];
        vec4d& vel = d_vels[particleID];

        const vec3d P = vec3d(p.x, p.y, p.z);
        const vec3d P_end = P + vec3d(disp.x, disp.y, disp.z);

        vec3d u_reflect = vec3d(vel.x, vel.y, vel.z);
        vec3d P_reflect = P_end;
        int tetID_bd = tetID;
        for (int i = 0; i < 10; ++i) {//Maximum reflect 10 times
            /*
            const vec4i index_tet = d_tetInds[tetID_bd];
            const vec3d A = d_tetVerts[index_tet.x];
            const vec3d B = d_tetVerts[index_tet.y];
            const vec3d C = d_tetVerts[index_tet.z];
            const vec3d D = d_tetVerts[index_tet.w];
            const vec4d bary = tetBaryCoord(P, A, B, C, D);
            const vec4d bary_end = tetBaryCoord(P_reflect, A, B, C, D);
             
            if (particleID == 9670)
                printf("%d ReflectID%d tetID=%d tetbd=%d \nP (%f,%f,%f)->(%f,%f,%f)\nBary(%f,%f,%f,%f)->(%f,%f,%f,%f)\n",
                    particleID, i,
                    tetID, tetID_bd,
                    P.x, P.y, P.z, P_end.x, P_end.y, P_end.z,
                    bary.x, bary.y, bary.z, bary.w,
                    bary_end.x, bary_end.y, bary_end.z, bary_end.w);
            */

            //P_end may still out of domain after few reflection iters
            //and perform reflection again
            SearchInfo info = baryTetSearch(particleID, P_reflect, tetID_bd, d_tetVerts, d_tetInds,
                d_facets, d_tetfacets, d_faceinfos);

            //P_end is within the domain now
            if (info.tetID >= 0) {
                tetID_bd = info.tetID;
                break;
            }

            //P_end is still out of domain
            tetID_bd = -(info.tetID + 1);//convert to 0-based index
            const int faceID = info.faceID;

            //[TODO] We can perform many different boundary treatment if based on faceID tags
            //[TODO] Now we perform specular reflection on all boundaries

            //Reflect P_end back to domain
            specularReflect(P_reflect, u_reflect, //P_input, U_input
                tetID_bd, faceID,d_tetVerts, d_faceinfos, d_facets,
                            P_reflect, u_reflect);//P_reflect, U_reflect
        }

        //Update reflect pos, disp and velocity
        vec3d P_disp = P_reflect - P;
        disp.x = P_disp.x;
        disp.y = P_disp.y;
        disp.z = P_disp.z;
        vel.x = u_reflect.x;
        vel.y = u_reflect.y;
        vel.z = u_reflect.z;

        //Update tetID
        //disp.w = double(tetID);
        d_tetIDs[particleID] = tetID_bd;
    }

    //Point-wise barycentric searching (initial valid tetID (>=0) should be provided by RTX locator)
    __global__ void baryQuery(Particle* d_particles,
        int* d_tetIDs, int numParticles,
        vec3d* d_tetVerts, vec4i* d_tetInds,
        vec4i* d_facets, vec4i* d_tetfacets, FaceInfo* d_faceinfos
    )
    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        if (d_tetIDs[particleID] < 0) {
            printf("[Warnning!] Particle %d is out of domain %d\n", particleID, d_tetIDs[particleID]);
            return;
        }

        Particle& p = d_particles[particleID];
        vec3d P = vec3d(p.x, p.y, p.z);

        //Displacement mode is query pts+disp
        //Also we can access the tetID in previous timestep (useful for particle exit the domain)

        int tetID_init = d_tetIDs[particleID];
        SearchInfo info = baryTetSearch(particleID,P, tetID_init, d_tetVerts, d_tetInds,
            d_facets, d_tetfacets, d_faceinfos);
        int tetID_search = info.tetID;

        //if (tetID_search != tetID_init)
        //    printf("Particle%d RTX TetID is not correct! %d->%d\n", particleID, tetID_init, tetID_search);

        d_tetIDs[particleID] = tetID_search;
    }

    //Displacement mode searching (initial tetID from last timestep)
    __global__ void baryQueryDisp(Particle* d_particles, vec4d* d_disps,
        int* d_tetIDs, int numParticles,
        vec3d* d_tetVerts, vec4i* d_tetInds,
        vec4i* d_facets, vec4i* d_tetfacets, FaceInfo* d_faceinfos
        )
    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        Particle& p = d_particles[particleID];
        vec4d& disp = d_disps[particleID];

        vec3d P = vec3d(p.x, p.y, p.z);
        P = P + vec3d(disp.x, disp.y, disp.z);

        //Displacement mode is query pts+disp
        //Also we can access the tetID in previous timestep (useful for particle exit the domain)

        int tetID_init = d_tetIDs[particleID];
        SearchInfo info = baryTetSearch(particleID, P, tetID_init, d_tetVerts, d_tetInds,
            d_facets, d_tetfacets, d_faceinfos);
        int tetID_search = info.tetID;
        
        //if (tetID_search != tetID_init)
        //    printf("Particle%d RTX TetID is not correct! %d->%d\n", particleID, tetID_RTX, tetID_search);
        //disp.w = double(tetID_init);
        d_tetIDs[particleID] = tetID_search;
    }


    __global__ void baryQueryDisp_RTX(Particle* d_particles, vec4d* d_disps,
        int* d_tetIDs, int numParticles,
        vec3d* d_tetVerts, vec4i* d_tetInds,
        vec4i* d_facets, vec4i* d_tetfacets, FaceInfo* d_faceinfos
		) 
	    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;


        Particle& p = d_particles[particleID];
        vec4d& disp = d_disps[particleID];
        
        vec3d P = vec3d(p.x, p.y, p.z);
        P = P + vec3d(disp.x, disp.y, disp.z);

        //Displacement mode is query pts+disp
        //RTX locator enabled tetID is searched and old tetID is now storaged in disp.w
        
        int tetID_RTX = d_tetIDs[particleID];
        int tetID_last = -(tetID_RTX + 1);//Convert 1-based index back to 0-based index
        tetID_RTX = tetID_RTX < 0 //Barycentric searching requires a good start ID
                ? tetID_last
                : tetID_RTX;
        
        if (tetID_RTX < 0 && tetID_last < 0) {//No GOOD tetID available for this particle
            printf("[Warnning] particle %d is really out of domain %d/%d\n", particleID, tetID_RTX, tetID_last);
            return;
        }

        tetID_RTX = tetID_last;
            
        SearchInfo info = baryTetSearch(particleID, P, tetID_RTX, d_tetVerts, d_tetInds,
            d_facets, d_tetfacets, d_faceinfos);
        int tetID_search = info.tetID;

        //if (tetID_search != tetID_RTX)
        //    printf("Particle%d RTX TetID is not correct! %d->%d\n", particleID, tetID_RTX, tetID_search);
        //disp.w = double(tetID_RTX);
        d_tetIDs[particleID] = tetID_search;
    }



	void RTQuery(OptixQuery& cellLocator, DeviceTetMesh devMesh, double4* d_particles, int* out_tetIDs, int numParticles)
	{
        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

		//Board-phase fast RTX location
		cellLocator.query_sync(d_particles, out_tetIDs, numParticles);
        cudaCheck(cudaDeviceSynchronize());

		//Narrow-phase barycentric location
        baryQuery << <gridDims, blockDims >> > (d_particles,
            out_tetIDs, numParticles,
            devMesh.d_positions, devMesh.d_indices,
            devMesh.d_facets, devMesh.d_tetfacets, devMesh.d_faceinfos);
        cudaCheck(cudaDeviceSynchronize());
	}


	void RTQuery(OptixQuery& cellLocator, DeviceTetMesh devMesh, 
		double4* d_particles, vec4d* d_disps, int* out_tetIDs, int numParticles)
	{
        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        //Board-phase fast RTX location (optional)
        cellLocator.query_sync(d_particles, (double4*)d_disps, out_tetIDs, numParticles);
        cudaCheck(cudaDeviceSynchronize());

        //Narrow-phase barycentric searching location to fix some floating point error (optional)
        //If barycentric searching is not enabled, some particle may locate in one of neigoring cell
        //or considered as out-of-domain particle
        /*
        baryQueryDisp_RTX << <gridDims, blockDims >> > (d_particles, d_disps,
            out_tetIDs, numParticles,
            devMesh.d_positions, devMesh.d_indices,
            devMesh.d_facets, devMesh.d_tetfacets, devMesh.d_faceinfos);
        cudaCheck(cudaDeviceSynchronize());
        */
	}

    void RTQuery(DeviceTetMesh devMesh, double4* d_particles, vec4d* d_disps, int* out_tetIDs, int numParticles)
    {
        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        //Narrow-phase barycentric searching location (initial tetID is from last timestep)
        baryQueryDisp << <gridDims, blockDims >> > (d_particles, d_disps,
            out_tetIDs, numParticles,
            devMesh.d_positions, devMesh.d_indices,
            devMesh.d_facets, devMesh.d_tetfacets, devMesh.d_faceinfos);
        cudaCheck(cudaDeviceSynchronize());
    }



    void RTWallReflect(DeviceTetMesh devMesh, int* d_tetIDs, Particle* d_particles,  vec4d* d_disps, vec4d* d_vels, int numParticles)
    {
        /*
        Reflect particle back to domain if it hit a wall

        Parameters
        ----------
        a : array_like

        Returns
        -------
        x : ndarray, shape Q

        Notes
        -----
        Multiply reflection will be applied if a particle exit from a corner

        */
        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        //Boundary condition: currently it is specular reflection
        RTreflection << <gridDims, blockDims >> > (d_particles, d_disps, d_vels,
            d_tetIDs, numParticles,
            devMesh.d_positions, devMesh.d_indices,
            devMesh.d_facets, devMesh.d_tetfacets, devMesh.d_faceinfos);
        cudaCheck(cudaDeviceSynchronize());

    }


    //-----------------------Debug------------------------
    __global__
        void printRTTet(Particle* d_particles, vec4d* d_disps,
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

    /*void advect::testRT(OptixQuery& cellLocator, DeviceTetMesh devMesh)
    {
        int numParticles = 1;
        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        //Example 1. Cell locator find a wrong tetId
        thrust::device_vector<Particle> particles(numParticles, make_double4(77.407653286554819, -0.226291355139613, -655.599157683661019, true));
        thrust::device_vector<vec4d> disps(numParticles, vec4d(-0.000234741263335, -0.000385634696309, 0.000501310939382, 0.0));


        Particle* d_particles = thrust::raw_pointer_cast(particles.data());
        vec4d* d_disps = thrust::raw_pointer_cast(disps.data());

        thrust::device_vector<vec4d> vels(numParticles, vec4d(0.1, -0.1, 0.01, 0.0));
        vec4d* d_vels = thrust::raw_pointer_cast(vels.data());

        thrust::device_vector<int> triIDs(numParticles, -2);
        int* d_triIDs = thrust::raw_pointer_cast(triIDs.data());

        thrust::device_vector<int> tetIDs(numParticles, -1);
        int* d_tetIDs = thrust::raw_pointer_cast(tetIDs.data());

        //Get the initial cell location
        cellLocator.query_sync(d_particles, d_tetIDs, numParticles);
        cudaCheck(cudaDeviceSynchronize());

        //Narrow-phase barycentric location
        baryQuery << <gridDims, blockDims >> > (d_particles,
            d_tetIDs, numParticles,
            devMesh.d_positions, devMesh.d_indices,
            devMesh.d_facets, devMesh.d_tetfacets, devMesh.d_faceinfos);
        cudaCheck(cudaDeviceSynchronize());

        //Some movement to obtain the disp
        printRTTet << <gridDims, blockDims >> > (d_particles, d_disps,
            d_tetIDs, numParticles,
            devMesh.d_positions, devMesh.d_indices);
        cudaCheck(cudaDeviceSynchronize());

    }*/


}
