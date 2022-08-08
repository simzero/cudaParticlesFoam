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

#pragma once

#include "OptixQuery.h"
#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"

namespace advect {

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
    {
        if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }

  /*! this is the gpu representation - do not put any std:: etc stuff
    in here.  note we do NOT store the triangle indices/vertices here;
    optix will keep track of them once we contructed them. */
  struct SharedFacesGeom {
    struct FaceInfo { int front=-1, back=-1; };
    FaceInfo *tetForFace;
  };

  struct RayGen {
    OptixTraversableHandle faces;
    float                  maxEdgeLength;
  };
  
  struct LaunchParams {
    union {
      FloatParticle  *particlesFloat;
      DoubleParticle *particlesDouble;
    };
    int     numParticles;
    int     isFloat;
	int     *out_tetIDs;
    int     isDisp;
    double4 *disps;
  };

  struct TriMeshGeom {
      struct FaceInfo { int tagID = -1, dummy = -1; };
      FaceInfo* tagForFace;
  };

  struct LaunchParams_BD {
      union {
          FloatParticle* particlesFloat;
          DoubleParticle* particlesDouble;
      };
      int      numParticles;
      int      isFloat;
      int      *out_triIDs;
      double4  *disps;
  };
  
}
