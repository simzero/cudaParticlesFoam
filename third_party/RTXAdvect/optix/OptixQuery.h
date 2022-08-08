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

#include <vector>
#include <map>
#include "owl/owl.h"
#include "owl/common/math/vec.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace advect {
	using namespace owl;
	using namespace owl::common;

	/*! specifies one query's input/output data, with single precision
	  query position.  Queries will only perofrmed for partiles whose
	  isActive is not True, so marking particles isActive will make it
	  skipped during query. After the query, out_tetID will be set to
	  the tet that the particle is in, or -1, if not within any tet (if
	  particle is invalid/inactive, out_tetID remains unchanged)
	 */
	struct FloatParticle {
		float pos[3];
		/*! if the query is valid (ie, isActive is True) then this
			value will, after the query, be either the ID of the tet
			that 'pos' is in, or -1 if not within any tet */
		float isActive;
	};

	/* specifies one querie's input/output data, with a double-precision
	   query position (the query itself will use single-precision float,
	   though). Invalid particles (ie, particles in an array for which
	   _no_ query should be performed) can be specified by setting
	   isActive to False */
	struct DoubleParticle {
		double pos[3];
		/*! if 0, this particle will not be done */
		double isActive;
	};

	/*! class that builds a 'shared faces' accel on a given tet mesh,
		and performs (batch-style) point location queries within
		this. "batch" in that sense means you cannot directly call this
		from a cuda kernel, but will have to architect your program in a
		way that the simulatoin coe generates a (device-side) array of
		queries (in CUDA), then calls this kernel (on the host) to do
		all those queries, and then goes back to processing the results
		in a cuda kernel */
	struct OptixQuery {
		OptixQuery(const double3* vertex, int numVertices,
			const int4* index, int numIndices, bool isBoundaryMesh = false) {
			this->isBdMesh = isBoundaryMesh;

			std::vector<double4> vertex_align(numVertices);
			for (int i = 0; i < numVertices; i++)
				vertex_align[i] = make_double4(vertex[i].x, vertex[i].y, vertex[i].z, 0.0);

			if (isBoundaryMesh)
				this->initBoundarySystem(vertex_align.data(), numVertices, index, numIndices);
			else
				this->initSystem(vertex_align.data(), numVertices, index, numIndices);
		}
		OptixQuery(const double4* vertex, int numVertices,
			const int4* index, int numIndices, bool isBoundaryMesh = false) {
			this->isBdMesh = isBoundaryMesh;

			if(isBoundaryMesh)
				this->initBoundarySystem(vertex, numVertices, index, numIndices);
			else
				this->initSystem(vertex, numVertices, index, numIndices);
		}

		/*! perform a _synchronous_ query with given device-side array of
		  particle 
		  
		  Two query mode support:
		  Point-based mode: query point location without given any direction information
		  Displacement-based mode: point + displacement location with direction
		  
		  For robust double-precsion query, board phase + narrow phase checking is required
		  also, the mesh information should be provided. In the current implementation, only
		  tetrahedron mesh is supported where barycentric coord is used for narrow phase checking
		  */
		void query_sync(float4 *d_particles, int* out_tetIDs, int numParticles);
		
		void query_sync(double4 *d_particles, int* out_tetIDs, int numParticles);
		
		void query_sync(double4* d_particles, double4* d_disps, int* out_tri_tet_IDs, int numParticles) {
			//Boundary mesh query (only support pts+disp)
			if (this->isBdMesh) this->query_disp_Bd(d_particles, d_disps, out_tri_tet_IDs, numParticles);
			//Volume mesh displacement-mode query (query [pts+disp] instead of [pts])
			else this->query_disp(d_particles, d_disps, out_tri_tet_IDs, numParticles);
		};

		// NOT IMPLEMENTED YET:
#ifdef HAVE_ASYNC_QUERIES
	/*! if you do want to run queries asynchronously from multiple
	  threads, then every thread running queries must have its _own_
		asynch query context */
		struct AsyncQueryContext {

			/*! launches a query in the given stream; this will *not*
				synchronize at the end of this call, so you may
				asynchrnously launch another cuda kernel that depends on
				this result into the given this->stream without blocking
				other kernels */
			void query(Particle *d_particles, int numParticles);

			/*! the cuda stream that the query will be launched into */
			cudaStream_t    stream;
		private:
			OWLLaunchParams launchParams;

			/*! for sanity checking, to make sure the user doesn't
				accidentally run the same context with multiple threads
				after all ... */
			std::mutex      mutex;
		};

		/*! create a new async query context; if you do want to
			asynchronously run quries from multiple host threads in
			parallel, you _have_ to have a separte context per thread */
		AsyncQueryContext *createAsyncQueryContext();
#endif

	private:
		bool isBdMesh=false;
		OWLContext owl = 0;
		OWLModule  module = 0;

		OWLGroup   faceBVH = 0;
		OWLRayGen  rayGen = 0;
		OWLLaunchParams launchParams = 0;


		OWLGroup   faceBVH_BD = 0;
		OWLRayGen  rayGen_BD = 0;
		OWLLaunchParams launchParams_BD = 0;

		void initSystem(const double4 * vertex, int numVertices,
			const int4* index, int numIndices);
		void initBoundarySystem(const double4* vertex, int numVertices,
			const int4* index, int numIndices);

		// pts + disp query  
		void query_disp(double4* d_particles, double4* d_disps, int* out_tetIDs, int numParticles);
		void query_disp_Bd(double4* d_particles, double4* d_disps, int* out_triIDs, int numParticles);
	};

}
