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

#include <map>
#include <vector>
#include "cuda_runtime_api.h"

#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"

#include "cuda/DeviceTetMesh.cuh"
#include "cuda/HostTetMesh.h"

#include "cuda/common.h"

namespace advect {
	using namespace owl;
	using namespace owl::common;
	
	void convexTetQuery(
		DeviceTetMesh d_mesh, 
		double4* d_particles, 
		vec4d* d_disps, 
		int* out_tetIDs, 
		int numParticles);

	void convexWallReflect(
		DeviceTetMesh d_mesh,
		int* d_tetIDs,
		Particle* d_particles,
		vec4d* d_vels,
		vec4d* d_disps,
		int    numParticles);

	//
	//  Debug routines
	//
	void testNStracing(OptixQuery& cellLocator, DeviceTetMesh devMesh);

}
