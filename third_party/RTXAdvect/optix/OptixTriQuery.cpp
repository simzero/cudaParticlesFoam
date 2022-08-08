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

#include "owl/common/math/vec.h"
#include "owl/common/math/AffineSpace.h"


extern "C" char ptxCode[];

namespace advect {
	using namespace owl;
	using namespace owl::common;

	//VTK Mesh writer
	void writeVTKMesh(int NumVerts, int NumTris,
		const vec3da* vertices, const vec4i* tetIDs,
		std::vector<TriMeshGeom::FaceInfo>& faceinfos);

	struct TriMeshBuilder {

		TriMeshBuilder(const vec3da* vertex, int numVertices,
			const vec4i* index, int numIndices);

		std::vector<TriMeshGeom::FaceInfo> faceInfos;
		std::vector<vec3i> faceIndices;
		std::vector<vec3f> faceVertices;
		float maxEdgeLength = 0.f;
	};

	TriMeshBuilder::TriMeshBuilder(const vec3da* vertices, int numVertices,
		const vec4i* indices, int numIndices)
	{
		std::cout << "#adv: creating boundary tri mesh" << std::endl;
		for (int i = 0; i < numVertices; i++)
			faceVertices.push_back(vec3f(vertices[i].x, vertices[i].y, vertices[i].z));

		for (int i = 0; i < numIndices; ++i) {
			faceIndices.push_back(vec3i(indices[i].x, indices[i].y, indices[i].z));
                        TriMeshGeom::FaceInfo newFace;
                        newFace.tagID = indices[i].w;
                        newFace.dummy = -1;
			faceInfos.push_back(newFace);

			const vec3f A = faceVertices[indices[i].x];
			const vec3f B = faceVertices[indices[i].y];
			const vec3f C = faceVertices[indices[i].z];
			maxEdgeLength = std::max(maxEdgeLength, length(B - A));
			maxEdgeLength = std::max(maxEdgeLength, length(C - A));
		}
		
		std::cout << "#adv: maximum bd edge length " << maxEdgeLength << std::endl;

		writeVTKMesh(numVertices, numIndices, vertices, indices, faceInfos);
	}

	
	void OptixQuery::initBoundarySystem(const double4* vertex, int numVertices,
		const int4* index, int numIndices) {

		vec3da* vertex_owl = (vec3da*)vertex;
		vec4i* index_owl = (vec4i*)index;
		TriMeshBuilder sharedFaces(vertex_owl, numVertices, index_owl, numIndices);

		std::cout << "#adv: initializing owl" << std::endl;
		owl = owlContextCreate(nullptr, 1);
		owlSetMaxInstancingDepth(owl, 1);

		module
			= owlModuleCreate(owl, ptxCode);
		
		std::cout << "#adv: creating tri mesh geom type" << std::endl;
		OWLVarDecl sharedFacesGeomVars[]
			= {
			   { "tagForFace", OWL_BUFPTR, OWL_OFFSETOF(TriMeshGeom,tagForFace) },
			   { /* end of list sentinel: */nullptr },
		};
		OWLGeomType facesGeomType
			= owlGeomTypeCreate(owl, OWL_GEOM_TRIANGLES, sizeof(TriMeshGeom),
				sharedFacesGeomVars, -1);
		owlGeomTypeSetClosestHit(facesGeomType, 0,
			module, "boundaryCH");

		// ------------------------------------------------------------------
		// create the triangles geom part
		// ------------------------------------------------------------------
		std::cout << "#adv: creating boundary mesh geom" << std::endl;
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
		std::cout << "#adv: building boundary BVH" << std::endl;
		OWLGroup faces
			= owlTrianglesGeomGroupCreate(owl, 1, &facesGeom);
		owlGroupBuildAccel(faces);
		this->faceBVH_BD
			= owlInstanceGroupCreate(owl, 1, &faces);
		owlGroupBuildAccel(this->faceBVH_BD);

		owlBufferDestroy(faceIndexBuffer);
		owlBufferDestroy(faceVertexBuffer);

		// ------------------------------------------------------------------
		// upload/set the 'shading' data
		// ------------------------------------------------------------------
		owlGeomSetBuffer(facesGeom, "tagForFace", faceInfoBuffer);
		std::cout << "#adv: done setting up optix tri-mesh" << std::endl;


		// ------------------------------------------------------------------
		// create a raygen that we can launch for the query kernel
		// ------------------------------------------------------------------
		OWLVarDecl rayGenVars[]
			= {
			   { "faces",        OWL_GROUP, OWL_OFFSETOF(RayGen,faces) },
			   { "maxEdgeLength",OWL_FLOAT, OWL_OFFSETOF(RayGen,maxEdgeLength) },
			   { /* sentinel */ nullptr },
		};

		this->rayGen_BD = owlRayGenCreate(owl,
			module,
			"queryKernelBD",
			sizeof(RayGen),
			rayGenVars, -1);
		owlRayGenSetGroup(rayGen_BD, "faces", faceBVH_BD);
		owlRayGenSet1f(rayGen_BD, "maxEdgeLength", sharedFaces.maxEdgeLength); //dummpy parameters

		// ------------------------------------------------------------------
		// create a dummy miss program, to make optix happy
		// ------------------------------------------------------------------
		OWLMissProg miss = owlMissProgCreate(owl,
			module,
			"missBD",
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
			   { "particles",    OWL_ULONG,  OWL_OFFSETOF(LaunchParams_BD,particlesFloat) },
			   { "numParticles", OWL_INT,    OWL_OFFSETOF(LaunchParams_BD,numParticles) },
			   { "isFloat",      OWL_INT,    OWL_OFFSETOF(LaunchParams_BD,isFloat) },
			   { "out_triIDs",   OWL_ULONG,  OWL_OFFSETOF(LaunchParams_BD,out_triIDs) },
			   { "disps",         OWL_ULONG, OWL_OFFSETOF(LaunchParams_BD,disps) },
			   { /* sentinel */ nullptr },
		};

		//Custom type hint
		//"dt", OWL_USER_TYPE(double), OWL_OFFSETOF(LaunchParams_BD, dt)
		//owlLaunchParamsSetRaw(launchParams_BD, "dt", dt)

		launchParams_BD = owlLaunchParamsCreate(owl,
			sizeof(LaunchParams_BD),
			lpVars, -1);
	}


	/*! perform a _synchronous_ query with given device-side array of
	particle and velocity */
	void advect::OptixQuery::query_disp_Bd(double4* d_particles_double4, double4* d_disps, int* out_triIDs, int numParticles)
	{
		DoubleParticle* d_particles = (DoubleParticle*)d_particles_double4;
		int launchWidth = 64 * 1024;
		int launchHeight = divRoundUp(numParticles, launchWidth);

		owlLaunchParamsSet1ul(launchParams_BD, "particles", (uint64_t)d_particles);
		owlLaunchParamsSet1i(launchParams_BD, "numParticles", numParticles);
		owlLaunchParamsSet1i(launchParams_BD, "isFloat", 0);
		owlLaunchParamsSet1ul(launchParams_BD, "out_triIDs", (uint64_t)out_triIDs);
		owlLaunchParamsSet1ul(launchParams_BD, "disps", (uint64_t)d_disps);
		
		owlParamsLaunch2D(rayGen_BD, launchWidth, launchHeight, launchParams_BD);
		cudaCheck(cudaDeviceSynchronize());
	}


	void writeVTKMesh(int NumVerts, int NumTris,
		const vec3da* vertices, const vec4i* triIDs,
		std::vector<TriMeshGeom::FaceInfo>& faceinfos) {

		//Output mesh facets into VTK file
		int i;
		FILE* fp;
		char fileName[1024];

		//----------Write facet mesh-----------
		sprintf(fileName, "mesh_bdfaces.vtk");
		printf("#GCPS: Write mesh boundary face to file %s...\n", fileName);

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
		fprintf(fp, "\n");

		//Output bdfacets
		int NumFacets = NumTris;
		fprintf(fp, "POLYGONS %d %d\n", NumFacets, NumFacets * 4);//4 value define a triangle facet
		for (auto fi = 0; fi < NumFacets; ++fi) {
			vec4i face = triIDs[fi];
			fprintf(fp, "3 %d %d %d\n", face.x, face.y, face.z);
		}
		fprintf(fp, "\n");


		//Output scalar
		fprintf(fp, "CELL_DATA %d\n", NumFacets);
		fprintf(fp, "FIELD FieldData %d\n", 1);

		//Output bdfacets tag
		fprintf(fp, "BoundaryTag 1 %d int\n", NumFacets);
		for (auto fi = 0; fi < NumFacets; ++fi) {
			fprintf(fp, "%d\n", triIDs[fi].w);
		}
		fprintf(fp, "\n");

		fclose(fp);

	}

}
