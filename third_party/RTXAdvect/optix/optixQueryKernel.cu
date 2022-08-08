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

#include <owl/owl.h>
#include "internalTypes.h"

namespace advect {
  using namespace owl;
  using namespace owl::common;
  
  struct PerRayData
  {
    int tetID;
    int tetID_neigh;
	int faceID;
  };

  extern "C" __constant__ union {
      LaunchParams nonBD;
      LaunchParams_BD bd;
  } optixLaunchParams;

  //extern "C" __constant__ LaunchParams optixLaunchParams;

  // closest hit for the shared faces method
  OPTIX_CLOSEST_HIT_PROGRAM(sharedFacesCH)()
  {
    PerRayData  &prd = owl::getPRD<PerRayData>();
    const SharedFacesGeom &self = getProgramData<SharedFacesGeom>();
    const int   faceID = optixGetPrimitiveIndex();
    const int   tetID
      = optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE
      ? self.tetForFace[faceID].front
      : self.tetForFace[faceID].back;
    const int tetID_neigh
        = optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE
        ? self.tetForFace[faceID].back
        : self.tetForFace[faceID].front;
    prd.tetID = tetID;
    prd.tetID_neigh = tetID_neigh;
    prd.faceID= faceID;
  }

  OPTIX_MISS_PROGRAM(miss)()
  {
    /* nothing */
  }

  
  OPTIX_RAYGEN_PROGRAM(queryKernel)()
  {
    int particleID
      = getLaunchIndex().x
      + getLaunchDims().x
      * getLaunchIndex().y;

    LaunchParams &lp = optixLaunchParams.nonBD;
    const SharedFacesGeom& selfGeom = getProgramData<SharedFacesGeom>();

    if (particleID >= lp.numParticles) return;

    vec3f pos;
    vec3d pos_double;
    if (lp.isFloat) {
	  if (!lp.particlesFloat[particleID].isActive) return;
      float4 particle = (const float4&)lp.particlesFloat[particleID];
      pos = vec3f(particle.x,particle.y,particle.z);
      pos_double = vec3d(particle.x, particle.y, particle.z);
    } else {
	  if (!lp.particlesDouble[particleID].isActive) return;
      double4 particle = (const double4&)lp.particlesDouble[particleID];
      pos = vec3f(particle.x,particle.y,particle.z);
      pos_double = vec3d(particle.x, particle.y, particle.z);
    }
    
    //Check if query mode is pts + disp
    if (lp.isDisp) {
        //Backup the original tetID in case of the next fake missing hit (tetID=-1)
        //and we want to check the real tetID by tracking it from original tetID
        //lp.disps[particleID].w = double(lp.out_tetIDs[particleID]);

        vec3d dx = vec3d(lp.disps[particleID].x,
            lp.disps[particleID].y,
            lp.disps[particleID].z);
        pos_double += dx;
        pos = vec3f(pos_double.x, pos_double.y, pos_double.z);
        //printf("Disp Mode (%.15lf,%.15lf,%.15lf)\n", pos_double.x, pos_double.y, pos_double.z);
    }


    // create 'dummy' ray from particle pos, with particle itself as per-ray data
    const RayGen &self = getProgramData<RayGen>();
    PerRayData prd;
    prd.tetID = -1;
    prd.faceID = -2;
	owl::Ray ray(pos,
				 vec3f(1.f, 1e-10f, 1e-10f),
				 0.f, self.maxEdgeLength);//1e10f);self.maxEdgeLength);
    owl::traceRay(self.faces,ray,prd,
                  OPTIX_RAY_FLAG_DISABLE_ANYHIT);

    //if (particleID == 0) {
    //    printf("[Tet Locator] %d (%f,%f,%f) faceID %d tetID %d tetID_neigh %d \n",
    //        particleID, pos.x, pos.y, pos.z, prd.faceID, prd.tetID, prd.tetID_neigh);
    //}

    if (lp.isDisp && prd.tetID == -1) //Set missed particle as -previous tetID (1-based index)
        prd.tetID = -(lp.out_tetIDs[particleID] + 1);

    lp.out_tetIDs[particleID] = prd.tetID;
  }

  //-------------------------Boundary Ray-Tracing Hit programe----------------

  // closest hit for the shared faces method
  OPTIX_CLOSEST_HIT_PROGRAM(boundaryCH)()
  {
      PerRayData& prd = owl::getPRD<PerRayData>();
      const SharedFacesGeom& self = getProgramData<SharedFacesGeom>();
      const int   faceID = optixGetPrimitiveIndex();
      const int   tetID
          = optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE
          ? 1
          : 0;
      prd.tetID = tetID;
      prd.faceID = faceID;
  }

  OPTIX_MISS_PROGRAM(missBD)()
  {
      /* nothing */
  }


  OPTIX_RAYGEN_PROGRAM(queryKernelBD)()
  {
      int particleID
          = getLaunchIndex().x
          + getLaunchDims().x
          * getLaunchIndex().y;

      LaunchParams_BD& lp = optixLaunchParams.bd;

      if (particleID >= lp.numParticles) return;
      //int debugParticle = 52747;

      //if (particleID == debugParticle) printf("[Check particle] %d triIDs=%d\n", particleID, lp.out_triIDs[particleID]);

      //Set input tetID = -1 will skip check for this particle
      if (lp.out_triIDs[particleID] == -1) return;

      vec3f pos;
      vec3d pos_double;
      if (lp.isFloat) {
          //if (!lp.particlesFloat[particleID].isActive) return;
          float4 particle = (const float4&)lp.particlesFloat[particleID];
          pos = vec3f(particle.x, particle.y, particle.z);
          pos_double = vec3d(particle.x, particle.y, particle.z);
      }
      else {
          //if (!lp.particlesDouble[particleID].isActive) return;
          double4 particle = (const double4&)lp.particlesDouble[particleID];
          pos = vec3f(particle.x, particle.y, particle.z);
          pos_double = vec3d(particle.x, particle.y, particle.z);
      }

      // P^n--------->|-------->P^(n+1)
      // P^(n+1) = P^n + disp

      vec3f dir;
      vec3f dx = vec3f(lp.disps[particleID].x,
          lp.disps[particleID].y,
          lp.disps[particleID].z);
      dir = normalize(dx);
      float tmax = length(dx);

      const RayGen& self = getProgramData<RayGen>();
      PerRayData prd;

      prd.faceID = -1;
      prd.tetID = -1;//front = 1, back = 0

      //Ray length may very short (1e-3)
      //float precsion query can only be used as board search
      //so we extend ray to hit the potential collision face
      // and then check collision using double precsion
      pos = pos - 1e-2f * self.maxEdgeLength * dir;
      tmax = self.maxEdgeLength;
      owl::Ray ray(pos, dir,
          0.0, tmax);

      //OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES is used 
      //as the only possibile hit scheme is a particle hit from domain side
      //Boundary mesh normal vector is OUTWARD oriented 
      owl::traceRay(self.faces, ray, prd,
          OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES);

      //Double-check some miss hit particles due to floating-point error
      /*
      if (prd.faceID == -1) {
          vec3f pos_pertub(0.0f,0.0f,0.0f);
          
          //Move particle position towards double
          //pos_pertub = pertubPos(particleID, pos, pos_double);
          //Extend ray (1e-2f,1)maxEdgeLength to avoid miss hit due to float tolerance
          pos_pertub = pos - (7e-3f * self.maxEdgeLength) * dir;
          tmax = self.maxEdgeLength;

          owl::Ray ray_pertub(pos_pertub, dir,
              0.0, tmax);
          owl::traceRay(self.faces, ray_pertub, prd,
              OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES);

          //Reverse checking test

          //printf("[2nd check]%d (%.15f,%.15f,%.15f)\n", particleID,
          //    pos_pertub.x, pos_pertub.y, pos_pertub.z);
      }
      */

      /*
            
      if (particleID == -100)
      //if (particleID == 52747 || particleID == 98550 || particleID == 83144)
          printf("[2st check]%d (%.15f,%.15f,%.15f)->(%.15f,%.15f,%.15f) front/back %d \nfaceIDExit %d/%d disp(%.10f,%.10f,%.10f)/(%.10f,%.10f,%.10f) Dir (%f,%f,%f)\n", particleID,
              pos.x, pos.y, pos.z,
              pos.x + dx.x, pos.y + dx.y, pos.z + dx.z, prd.tetID,
              lp.out_triIDs[particleID], prd.faceID,
              dx.x, dx.y, dx.z,
              lp.disps[particleID].x,
              lp.disps[particleID].y,
              lp.disps[particleID].z,
              dir.x, dir.y, dir.z);
     
      if (particleID == 0)
          //if (particleID == 52747 || particleID == 98550 || particleID == 83144)
          printf("[kernel check]%d tmax=%.15lf/%.15lf (%.15f,%.15f,%.15f)->(%.15f,%.15f,%.15f) \n Disp(%.15f,%.15f,%.15f) front/back %d faceIDExit %d/%d \n", particleID,
              tmax, length(dx),
              pos_double.x, pos_double.y, pos_double.z,
              pos.x + dx.x, pos.y + dx.y, pos.z + dx.z, 
              lp.disps[particleID].x, lp.disps[particleID].y, lp.disps[particleID].z,
              prd.tetID,
              lp.out_triIDs[particleID], prd.faceID);
       */

      //Return Exit FaceID, -1 = HitNothing or TrifaceID
      lp.out_triIDs[particleID] = prd.faceID;
  }

}
