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

#include "OptixQuery.h"
#include "internalTypes.h"
#include "cuda_runtime_api.h"

#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"

extern "C" char ptxCode[];

namespace advect {
	using namespace owl;
	using namespace owl::common;

	//VTK Mesh writer
	void writeVTKMesh(int NumVerts, int NumTets, 
		const vec3da *vertices, const vec4i *tetIDs, 
		std::vector<vec3i> &faceIDs, std::vector<SharedFacesGeom::FaceInfo> &faceinfos);

	/*! contains temporary data for shared-face strcuture
		computation. after the respective geom and accel have been
		built, these can be released, and are thus in a separate
		class */
	struct SharedFacesBuilder {

		SharedFacesBuilder(const vec3da *vertex, int numVertices,
			const vec4i *index, int numIndices);

		std::vector<SharedFacesGeom::FaceInfo> faceInfos;
		std::vector<vec3i> faceIndices;
		std::vector<vec3f> faceVertices;
		std::map<uint64_t, int> knownFaces;
		float maxEdgeLength = 0.f;

		void add(int tetID, vec3i face);
	};

	SharedFacesBuilder::SharedFacesBuilder(const vec3da *vertices, int numVertices,
		const vec4i *indices, int numIndices)
	{
		std::cout << "#adv: creating shared faces" << std::endl;
		for (int i = 0; i < numVertices; i++)
			faceVertices.push_back(vec3f(vertices[i].x, vertices[i].y, vertices[i].z));

		for (int tetID = 0; tetID < numIndices; tetID++) {
			vec4i index = indices[tetID];
			if (index.x == index.y) continue;
			if (index.x == index.z) continue;
			if (index.x == index.w) continue;
			if (index.y == index.z) continue;
			if (index.y == index.w) continue;
			if (index.z == index.w) continue;

			const vec3f A = faceVertices[index.x];
			const vec3f B = faceVertices[index.y];
			const vec3f C = faceVertices[index.z];
			const vec3f D = faceVertices[index.w];

			maxEdgeLength = std::max(maxEdgeLength, length(B - A));
			maxEdgeLength = std::max(maxEdgeLength, length(C - A));
			maxEdgeLength = std::max(maxEdgeLength, length(D - A));
			maxEdgeLength = std::max(maxEdgeLength, length(C - B));
			maxEdgeLength = std::max(maxEdgeLength, length(D - B));
			maxEdgeLength = std::max(maxEdgeLength, length(D - C));

			const float volume = dot(D - A, cross(B - A, C - A));
			if (volume == 0.f) {
				// ideally, remove this tet from the input; for now, just
				// don't create any faces for it, and instead write in dummies
				// (to not mess up indices)
				continue;
			}
			else if (volume < 0.f) {
				std::swap(index.x, index.y);
			}
			//{x,0},{y,1},{z,2},{w,3}
			//add(tetID, vec3i(index.x, index.y, index.z)); // 0,1,2
			//add(tetID, vec3i(index.y, index.w, index.z)); // 1,3,2
			//add(tetID, vec3i(index.x, index.w, index.y)); // 0,3,1
			//add(tetID, vec3i(index.z, index.w, index.x)); // 2,3,0

			//Gmsh face order
			add(tetID, vec3i(index.y, index.z, index.w)); // 1,2,3
			add(tetID, vec3i(index.z, index.x, index.w)); // 2,0,3
			add(tetID, vec3i(index.x, index.y, index.w)); // 0,1,3
			add(tetID, vec3i(index.x, index.z, index.y)); // 0,2,1
		}

		std::cout << "#adv: maximum edge length " << maxEdgeLength << std::endl;
		std::cout << "#adv: #shared face = " << faceIndices.size() << std::endl;

		writeVTKMesh(numVertices, numIndices, vertices, indices, faceIndices,faceInfos);
	}


	void SharedFacesBuilder::add(int tetID, vec3i face)
	{
		//int front = true;
		int front = false;//Gmsh face order
		if (face.x > face.z) { std::swap(face.x, face.z); front = !front; }
		if (face.y > face.z) { std::swap(face.y, face.z); front = !front; }
		if (face.x > face.y) { std::swap(face.x, face.y); front = !front; }
		assert(face.x < face.y && face.x < face.z);

		int faceID = -1;
		uint64_t key = ((((uint64_t)face.z << 20) | face.y) << 20) | face.x;
		auto it = knownFaces.find(key);
		if (it == knownFaces.end()) {
			faceID = faceIndices.size();
			faceIndices.push_back(face);
			SharedFacesGeom::FaceInfo newFace;
                        newFace.front = -1;
                        newFace.back = -1;
			faceInfos.push_back(newFace);
			knownFaces[key] = faceID;
		}
		else {
			faceID = it->second;
		}

		if (front)
			faceInfos[faceID].front = tetID;
		else
			faceInfos[faceID].back = tetID;
	}


	void advect::OptixQuery::initSystem(const double4 * vertex, int numVertices,
		const int4 * index, int numIndices)
	{
		// ------------------------------------------------------------------
		// create the shared facet geom part
		// ------------------------------------------------------------------
		vec3da * vertex_owl = (vec3da *)vertex;
		vec4i * index_owl = (vec4i *)index;
		SharedFacesBuilder sharedFaces(vertex_owl, numVertices, index_owl, numIndices);

		std::cout << "#adv: initializing owl" << std::endl;
		owl = owlContextCreate(nullptr, 1);
		owlSetMaxInstancingDepth(owl, 1);
		std::cout << "#adv: initializing owl done" << std::endl;

		module
			= owlModuleCreate(owl, ptxCode);

		std::cout << "#adv: creating tet mesh 'shared faces' geom type" << std::endl;
		OWLVarDecl sharedFacesGeomVars[]
			= {
			   { "tetForFace", OWL_BUFPTR, OWL_OFFSETOF(SharedFacesGeom,tetForFace) },
			   { /* end of list sentinel: */nullptr },
		};
		OWLGeomType facesGeomType
			= owlGeomTypeCreate(owl, OWL_GEOM_TRIANGLES, sizeof(SharedFacesGeom),
				sharedFacesGeomVars, -1);
		owlGeomTypeSetClosestHit(facesGeomType, 0,
			module, "sharedFacesCH");

		// ------------------------------------------------------------------
		// create the triangles geom part
		// ------------------------------------------------------------------
		std::cout << "#adv: creating geom" << std::endl;
		OWLGeom facesGeom
			= owlGeomCreate(owl, facesGeomType);
		OWLBuffer faceVertexBuffer
			= owlDeviceBufferCreate(owl, OWL_FLOAT3,
				sharedFaces.faceVertices.size(),
				sharedFaces.faceVertices.data());
		OWLBuffer faceIndexBuffer
			= owlDeviceBufferCreate(owl, OWL_INT3,
				sharedFaces.faceIndices.size(),
				sharedFaces.faceIndices.data());
		OWLBuffer faceInfoBuffer
			= owlDeviceBufferCreate(owl, OWL_INT2,
				sharedFaces.faceInfos.size(),
				sharedFaces.faceInfos.data());

		owlTrianglesSetVertices(facesGeom, faceVertexBuffer,
			sharedFaces.faceVertices.size(),
			sizeof(sharedFaces.faceVertices[0]), 0);
		owlTrianglesSetIndices(facesGeom, faceIndexBuffer,
			sharedFaces.faceIndices.size(),
			sizeof(sharedFaces.faceIndices[0]), 0);

		// ------------------------------------------------------------------
		// create the group, to force accel build
		// ------------------------------------------------------------------
		// iw: todo - set disable-anyhit flag on group (needs addition to owl)d
		std::cout << "#adv: building BVH" << std::endl;
		OWLGroup faces
			= owlTrianglesGeomGroupCreate(owl, 1, &facesGeom);
		owlGroupBuildAccel(faces);
		this->faceBVH
			= owlInstanceGroupCreate(owl, 1, &faces);
		owlGroupBuildAccel(this->faceBVH);

		owlBufferDestroy(faceIndexBuffer);
		owlBufferDestroy(faceVertexBuffer);

		// ------------------------------------------------------------------
		// upload/set the 'shading' data
		// ------------------------------------------------------------------
		owlGeomSetBuffer(facesGeom, "tetForFace", faceInfoBuffer);
		std::cout << "#adv: done setting up optix tet-mesh" << std::endl;

		// ------------------------------------------------------------------
		// create a raygen that we can launch for the query kernel
		// ------------------------------------------------------------------
		OWLVarDecl rayGenVars[]
			= {
			   { "faces",        OWL_GROUP, OWL_OFFSETOF(RayGen,faces) },
			   { "maxEdgeLength",OWL_FLOAT, OWL_OFFSETOF(RayGen,maxEdgeLength) },
			   { /* sentinel */ nullptr },
		};
		this->rayGen = owlRayGenCreate(owl,
			module,
			"queryKernel",
			sizeof(RayGen),
			rayGenVars, -1);

		owlRayGenSetGroup(rayGen, "faces", faceBVH);
		owlRayGenSet1f(rayGen, "maxEdgeLength", sharedFaces.maxEdgeLength);
		// ------------------------------------------------------------------
		// create a dummy miss program, to make optix happy
		// ------------------------------------------------------------------
		OWLMissProg miss = owlMissProgCreate(owl,
			module,
			"miss",
			0, nullptr, 0);

		// ------------------------------------------------------------------
		// have all programs, geometries, groups, etc - build the SBT
		// ------------------------------------------------------------------
		owlBuildPrograms(owl);
		owlBuildPipeline(owl);
		owlBuildSBT(owl);


		// ------------------------------------------------------------------
		// FINALLY: create a launch params that we can use to pass array
		// of queries
		// ------------------------------------------------------------------
		OWLVarDecl lpVars[]
			= {
			   { "particles",    OWL_ULONG,  OWL_OFFSETOF(LaunchParams,particlesFloat) },
			   { "numParticles", OWL_INT,    OWL_OFFSETOF(LaunchParams,numParticles) },
			   { "isFloat",      OWL_INT,    OWL_OFFSETOF(LaunchParams,isFloat) },
			   { "out_tetIDs",   OWL_ULONG,  OWL_OFFSETOF(LaunchParams,out_tetIDs) },
			   { "isDisp",       OWL_INT,    OWL_OFFSETOF(LaunchParams,isDisp) },
			   { "disps",        OWL_ULONG,  OWL_OFFSETOF(LaunchParams,disps) },
			   { /* sentinel */ nullptr },
		};

		launchParams = owlLaunchParamsCreate(owl,
			sizeof(LaunchParams),
			lpVars, -1);
	}


	


	/*! perform a _synchronous_ query with given device-side array of
	 particle */
	void OptixQuery::query_sync(float4 *d_particles_float4, int* out_tetIDs, int numParticles)
	{
		FloatParticle *d_particles = (FloatParticle *)d_particles_float4;
		int launchWidth = 64 * 1024;
		int launchHeight = divRoundUp(numParticles, launchWidth);

		owlLaunchParamsSet1ul(launchParams, "particles", (uint64_t)d_particles);
		owlLaunchParamsSet1i(launchParams, "numParticles", numParticles);
		owlLaunchParamsSet1i(launchParams, "isFloat", 1);
		owlLaunchParamsSet1ul(launchParams, "out_tetIDs", (uint64_t)out_tetIDs);
		owlLaunchParamsSet1i(launchParams, "isDisp", 0);
		owlParamsLaunch2D(rayGen, launchWidth, launchHeight, launchParams);
		cudaDeviceSynchronize();
	}

	/*! perform a _synchronous_ query with given device-side array of
	 particle */
	void OptixQuery::query_sync(double4 *d_particles_double4, int* out_tetIDs, int numParticles)
	{
		DoubleParticle *d_particles = (DoubleParticle *)d_particles_double4;
		int launchWidth = 64 * 1024;
		int launchHeight = divRoundUp(numParticles, launchWidth);

		owlLaunchParamsSet1ul(launchParams, "particles", (uint64_t)d_particles);
		owlLaunchParamsSet1i(launchParams, "numParticles", numParticles);
		owlLaunchParamsSet1i(launchParams, "isFloat", 0);
		owlLaunchParamsSet1ul(launchParams, "out_tetIDs", (uint64_t)out_tetIDs);
		owlLaunchParamsSet1i(launchParams, "isDisp", 0);
		owlParamsLaunch2D(rayGen, launchWidth, launchHeight, launchParams);
		cudaDeviceSynchronize();
	}

	/*! perform a _synchronous_ query with given device-side array of
	 particle and displacement */
	void OptixQuery::query_disp(double4* d_particles_double4, double4* d_disps, int* out_tetIDs, int numParticles)
	{
		DoubleParticle* d_particles = (DoubleParticle*)d_particles_double4;
		int launchWidth = 64 * 1024;
		int launchHeight = divRoundUp(numParticles, launchWidth);

		owlLaunchParamsSet1ul(launchParams, "particles", (uint64_t)d_particles);
		owlLaunchParamsSet1i(launchParams, "numParticles", numParticles);
		owlLaunchParamsSet1i(launchParams, "isFloat", 0);
		owlLaunchParamsSet1ul(launchParams, "out_tetIDs", (uint64_t)out_tetIDs);
		owlLaunchParamsSet1i(launchParams, "isDisp", 1);
		owlLaunchParamsSet1ul(launchParams, "disps", (uint64_t)d_disps);

		owlParamsLaunch2D(rayGen, launchWidth, launchHeight, launchParams);
		cudaDeviceSynchronize();
	}


	void writeVTKMesh(int NumVerts, int NumTets,
		const vec3da *vertices, const vec4i *tetIDs,
		std::vector<vec3i> &faceIDs, std::vector<SharedFacesGeom::FaceInfo> &faceinfos)
	{
		int i;
		FILE *fp;
		char fileName[1024];

		//----------Write facet mesh-----------
		sprintf(fileName, "mesh_faces.vtk");
		printf("#GCPS: Write mesh shared face to file %s...\n", fileName);

		fp = fopen(fileName, "w");
		fprintf(fp, "# vtk DataFile Version 4.1\n");
		fprintf(fp, "vtk output\n");
		fprintf(fp, "ASCII\n");
		fprintf(fp, "DATASET POLYDATA\n");

		//Output vertices
		fprintf(fp, "POINTS %d float\n", NumVerts);
		for (auto vi = 0; vi < NumVerts; ++vi) {
			fprintf(fp, "%lf %lf %lf\n", vertices[vi].x,
				vertices[vi].y,
				vertices[vi].z);
		}

		//Output facets
		int NumFacets = faceIDs.size();
		fprintf(fp, "POLYGONS %d %d\n", NumFacets, NumFacets * 4);//4 value define a triangle facet
		for (auto fi = 0; fi < NumFacets; ++fi) {
			vec3i face = faceIDs[fi];
			fprintf(fp, "3 %d %d %d\n", face.x, face.y, face.z);
		}
		fprintf(fp, "\n");

		//Output scalar
		fprintf(fp, "CELL_DATA %d\n", NumFacets);
		fprintf(fp, "FIELD FieldData %d\n", 1);

		fprintf(fp, "Tet_front_back 2 %d int\n", NumFacets);
		for (auto fi = 0; fi < NumFacets; ++fi) {
			fprintf(fp, "%d %d\n", faceinfos[fi].front, faceinfos[fi].back);
		}
		fprintf(fp, "\n");

		fclose(fp);


		//------Write volume mesh-------------
		FILE *fp_v;
		sprintf(fileName, "mesh.vtk");
		printf("#GCPS: Write mesh to file %s...\n", fileName);

		fp_v = fopen(fileName, "w");
		fprintf(fp_v, "# vtk DataFile Version 4.1\n");
		fprintf(fp_v, "vtk output\n");
		fprintf(fp_v, "ASCII\n");
		fprintf(fp_v, "DATASET UNSTRUCTURED_GRID\n");

		//Output vertices
		fprintf(fp_v, "POINTS %d float\n", NumVerts);
		for (auto vi = 0; vi < NumVerts; ++vi) {
			fprintf(fp_v, "%lf %lf %lf\n", vertices[vi].x,
				vertices[vi].y,
				vertices[vi].z);
		}
		fprintf(fp_v, "\n");

		//Output elements
		fprintf(fp_v, "CELLS %d %d\n", NumTets, NumTets * 5);//5 value define a tet ele
		for (auto ei = 0; ei < NumTets; ++ei) {
			vec4i ele = tetIDs[ei];
			fprintf(fp_v, "4 %d %d %d %d\n", ele.x, ele.y, ele.z, ele.w);
		}
		fprintf(fp_v, "\n");

		fprintf(fp_v, "CELL_TYPES %d\n", NumTets);
		for (auto ei = 0; ei < NumTets; ++ei) {
			if (ei % 10 == 0 && ei > 0) fprintf(fp_v, "\n");
			fprintf(fp_v, "%d ", 10);
		}
		fprintf(fp_v, "\n");
		fprintf(fp_v, "\n");

		fclose(fp_v);

	}

}
