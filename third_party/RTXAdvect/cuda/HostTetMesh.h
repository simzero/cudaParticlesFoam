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

#ifndef TET_MESH_HOST
#define TET_MESH_HOST

#include "owl/common/math/box.h"
#include <fstream>

namespace advect {

  using namespace owl;
  using namespace owl::common;

  typedef owl::common::box_t<vec3d> box3d;
  struct FaceInfo { int front = -1, back = -1; };

  struct HostTetMesh {
    std::vector<vec3d> positions;
    std::vector<vec3d> velocities;
    std::vector<vec4i> indices;

	std::vector<vec4i> facets;       //facet node indices
	std::vector<vec4i> tetfacets;    //tetID->facetID
	std::vector<FaceInfo> faceInfos; //front,back

	box3d worldBounds;

    /*! creates a simple test data set of NxNxN cells with some dummy
        velocity field */
	static HostTetMesh createBoxMesh(int nx, int ny, int nz);

	static HostTetMesh readDataSet(std::string vert_fname, std::string cell_fname, std::string solv_fname="", std::string solc_fname="");

	HostTetMesh getBoundaryMesh();

	size_t bytes() {
		return positions.capacity() * sizeof(vec3d)
			+ velocities.capacity() * sizeof(vec3d)
			+ indices.capacity() * sizeof(vec4i)
			+ facets.capacity() * sizeof(vec4i)
			+ tetfacets.capacity() * sizeof(vec4i)
			+ faceInfos.capacity() * sizeof(FaceInfo);
	}
  };

  inline HostTetMesh advect::HostTetMesh::createBoxMesh(int nx, int ny, int nz){
	  //Create abritary size box shape mesh
	  HostTetMesh model;

	  vec3d p0 = vec3d(0.0,0.0,0.0);
	  vec3d p1 = vec3d(nx*1.0, ny*1.0, nz*1.0);

	  // Extract minimum and maximum coordinates
	  const double x0 = std::min(p0.x, p1.x);
	  const double x1 = std::max(p0.x, p1.x);
	  const double y0 = std::min(p0.y, p1.y);
	  const double y1 = std::max(p0.y, p1.y);
	  const double z0 = std::min(p0.z, p1.z);
	  const double z1 = std::max(p0.z, p1.z);

	  const double a = x0;
	  const double b = x1;
	  const double c = y0;
	  const double d = y1;
	  const double e = z0;
	  const double f = z1;
	  
	  model.worldBounds.extend(p0);
	  model.worldBounds.extend(p1);
	  vec3d center =model.worldBounds.center();

	  // assert((nx >= 1 && ny >= 1 && nz >= 1,"BoxMesh: number of vertices must be at least 1 in each dimension"));

	  // Create vertices
	  std::vector<double> x(3);
	  std::size_t vertex = 0;
	  for (std::size_t iz = 0; iz <= nz; iz++)
	  {
		  x[2] = e + (static_cast<double>(iz))*(f - e) / static_cast<double>(nz);
		  for (std::size_t iy = 0; iy <= ny; iy++)
		  {
			  x[1] = c + (static_cast<double>(iy))*(d - c) / static_cast<double>(ny);
			  for (std::size_t ix = 0; ix <= nx; ix++)
			  {
				  x[0] = a + (static_cast<double>(ix))*(b - a) / static_cast<double>(nx);

				  vec3d pos = vec3d(x[0],x[1],x[2]);
				  vec3d vel = (pos == center) ? vec3d(1.f, 0.f, 0.f) : normalize(pos - center);
				  model.positions.push_back(pos);
				  model.velocities.push_back(vel);
				  vertex++;
			  }
		  }
	  }

	  // Create tetrahedra
	  std::size_t cell = 0;
	  for (std::size_t iz = 0; iz < nz; iz++)
	  {
		  for (std::size_t iy = 0; iy < ny; iy++)
		  {
			  for (std::size_t ix = 0; ix < nx; ix++)
			  {
				  const std::size_t v0 = iz * (nx + 1)*(ny + 1) + iy * (nx + 1) + ix;
				  const std::size_t v1 = v0 + 1;
				  const std::size_t v2 = v0 + (nx + 1);
				  const std::size_t v3 = v1 + (nx + 1);
				  const std::size_t v4 = v0 + (nx + 1)*(ny + 1);
				  const std::size_t v5 = v1 + (nx + 1)*(ny + 1);
				  const std::size_t v6 = v2 + (nx + 1)*(ny + 1);
				  const std::size_t v7 = v3 + (nx + 1)*(ny + 1);

				  // Add cells
				  // Note that v0 < v1 < v2 < v3 < vmid.
				  model.indices.push_back(vec4i(v0, v1, v3, v7));
				  model.indices.push_back(vec4i(v0, v1, v7, v5));
				  model.indices.push_back(vec4i(v0, v5, v7, v4));
				  model.indices.push_back(vec4i(v0, v3, v2, v7));
				  model.indices.push_back(vec4i(v0, v6, v4, v7));
				  model.indices.push_back(vec4i(v0, v2, v6, v7));
			  }
		  }
	  }

	  std::cout << "created test geom w/ " << model.indices.size() << " tets" << std::endl;
	  std::cout << "geom bounding box w/ " << model.worldBounds.lower << ", " << model.worldBounds.upper << std::endl;
	  return model;
  }

  inline HostTetMesh HostTetMesh::readDataSet(std::string vert_fname, std::string cell_fname, std::string solv_fname, std::string solc_fname)
  {
	  /* Read tets mesh and velocity (vector) field from ascii field

	ASCII file format

	vert.dat
	--------
	NumTetVerts = 16226
	x y z
	-10.0 -17.0 46.0
	-10.0 -17.0 -10.0

	cell.dat
	--------
	NumTetCells = 74430
	id1 id2 id3 id4
	3559 3653 3710 11699
	10154 10852 11785 14048

	solution.dat
	------------
	p u v w
	-0.000518217 -4.31E-17 -8.19E-16 3.38E-17
	-0.000388584 3.77E-17 -3.18E-16 3.95E-17

	*/
	  HostTetMesh model;
	  int NumVerts = 0;
	  int NumTets = 0;

	  std::string word;
	  double number;

	  //Read vertices
	  std::ifstream vfile(vert_fname);
	  if (vfile.is_open()) {
		  vfile >> word >> NumVerts;//Header Line
		  printf("%s %d\n", word.c_str(), NumVerts);
		  vfile >> word >> word >> word;//Comment line

		  model.positions.reserve(NumVerts);
		  for (int i = 0; i < NumVerts; ++i) {
			  vec3d pos;
			  vfile >> pos.x >> pos.y >> pos.z;

			  model.worldBounds.extend(pos);
			  model.positions.push_back(pos);
		  }
		  vfile.close();
	  }

	  //Read Tet indices
	  std::ifstream tfile(cell_fname);
	  if (tfile.is_open()) {
		  tfile >> word >> NumTets;//Header Line
		  printf("%s %d\n", word.c_str(), NumTets);
		  tfile >> word >> word >> word >> word;//Comment line

		  model.indices.reserve(NumTets);
		  for (int i = 0; i < NumTets; ++i) {
			  vec4i tetIDs;
			  tfile >> tetIDs.x >> tetIDs.y >> tetIDs.z >> tetIDs.w;
			  model.indices.push_back(tetIDs);
		  }
		  tfile.close();
	  }
	
	  //Read velocity solutions
	  if (solv_fname.size() > 0) {//Vertx-wise solution
		  std::ifstream sfile(solv_fname);
		  if (sfile.is_open()) {
			  sfile >> word >> word >> word >> word;//Comment line

			  model.velocities.reserve(NumVerts);
			  for (int i = 0; i < NumVerts; ++i) {
				  vec3d vel;
				  sfile >> number >> vel.x >> vel.y >> vel.z; //p u v w
				  if (i < 2) std::cout << "Vert Vel" << i << " " << vel << std::endl;
				  model.velocities.push_back(vel);
			  }
			  sfile.close();
		  }
	  }
	  else {//Cell-wise solution
		  std::ifstream sfile(solc_fname);
		  if (sfile.is_open()) {
			  sfile >> word >> word >> word >> word;//Comment line

			  model.velocities.reserve(NumTets);
			  for (int i = 0; i < NumTets; ++i) {
				  vec3d vel;
				  sfile >> number >> vel.x >> vel.y >> vel.z; //p u v w
				  model.velocities.push_back(vel);
				  if(i<2) std::cout << "Tet Vel" << i << " " << vel<<std::endl;
			  }
			  sfile.close();
		  }
	  }
	  

	  // fixing winding order of tets:
	  for (auto &tet : model.indices) {
		  const vec3d &a = model.positions[tet.x];
		  const vec3d &b = model.positions[tet.y];
		  const vec3d &c = model.positions[tet.z];
		  const vec3d &d = model.positions[tet.w];
		  if (dot(d - a, cross(b - a, c - a)) < 0.)
			  std::swap(tet.z, tet.w);
	  }


	  std::cout << "created test geom w/ " << model.indices.size() << " tets" << std::endl;
	  std::cout << "geom bounding box w/ " << model.worldBounds.lower << ", " << model.worldBounds.upper << std::endl;
	  if (solc_fname.size() > 0) std::cout << "cell-wise uniform velocity enabled!" << std::endl;
	  return model;
  }


  inline void add1Facet(int tetID, int fi, vec3i face, std::map<uint64_t, int>& knownFaces, 
	  std::vector<vec4i> &facets, 
	  std::vector<vec4i>& tetfacets,
	  std::vector<FaceInfo> &faceInfos,
	  std::vector<bool> &boundaryMask) {

	  //int front = true;
	  int front = false;//Gmsh face order
	  if (face.x > face.z) { std::swap(face.x, face.z); front = !front; }
	  if (face.y > face.z) { std::swap(face.y, face.z); front = !front; }
	  if (face.x > face.y) { std::swap(face.x, face.y); front = !front; }
	  assert(face.x < face.y&& face.x < face.z);

	  int faceID = -1;
	  uint64_t key = ((((uint64_t)face.z << 20) | face.y) << 20) | face.x;
	  auto it = knownFaces.find(key);
	  if (it == knownFaces.end()) {//New facet
		  faceID = facets.size();
		  facets.push_back(vec4i(face.x,face.y,face.z,-1));
		  // faceInfos.push_back({ -1,-1 });
		  FaceInfo newFace;
		  newFace.front = -1;
		  newFace.back = -1;
		  faceInfos.push_back(newFace);
		  knownFaces[key] = faceID;
		  // std::cout<< "Boundary: " << face.x << " " << face.y << " " << face.z  <<std::endl;
		  boundaryMask.push_back(true); //Assume it is boundary 
	  }
	  else {//Already know
		  faceID = it->second;
		  boundaryMask[faceID] = false; //boundary facet is not shared by other tets
	  }
	  tetfacets[tetID][fi] = faceID;

	  
	  if (front)
		  faceInfos[faceID].front = tetID;
	  else
		  faceInfos[faceID].back = tetID;
  }


  inline HostTetMesh advect::HostTetMesh::getBoundaryMesh() {
	  //Get the its boundary mesh from a tet mesh


	  //Collect all facets
	  std::map<uint64_t, int> knownFaces;
	  std::vector<bool> boundaryMask;

          // std::cout<< "List:" << this->indices.size() <<std::endl;

	  for (int tetID = 0; tetID < this->indices.size(); tetID++) {
		  vec4i index = indices[tetID];
		  // std::cout<< "Index: " << index <<std::endl;
		  if (index.x == index.y) continue;
		  if (index.x == index.z) continue;
		  if (index.x == index.w) continue;
		  if (index.y == index.z) continue;
		  if (index.y == index.w) continue;
		  if (index.z == index.w) continue;

		  this->tetfacets.push_back(vec4i(-1, -1, -1, -1));

		  const vec3d A = this->positions[index.x];
		  const vec3d B = this->positions[index.y];
		  const vec3d C = this->positions[index.z];
		  const vec3d D = this->positions[index.w];

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
		  add1Facet(tetID, 0,vec3i(index.y, index.z, index.w),  
			  knownFaces, this->facets, this->tetfacets, this->faceInfos, boundaryMask); // 1,2,3
		  add1Facet(tetID, 1, vec3i(index.z, index.x, index.w), 
			  knownFaces, this->facets, this->tetfacets, this->faceInfos, boundaryMask); // 2,0,3
		  add1Facet(tetID, 2,vec3i(index.x, index.y, index.w), 
			  knownFaces, this->facets, this->tetfacets, this->faceInfos, boundaryMask); // 0,1,3
		  add1Facet(tetID, 3,vec3i(index.x, index.z, index.y), 
			  knownFaces, this->facets, this->tetfacets, this->faceInfos, boundaryMask); // 0,2,1
	  }


	  //Find boundary mesh
	  HostTetMesh boundaryMesh;
	  //printf("-%d facets, %d mask, %d facetinfos\n", facets.size(), boundaryMask.size(),this->faceInfos.size());
	  std::map<int, int> idxMap;

	  //Find boundary mesh verts
	  //printf("debug\n");
	  int TriVertId = -1;
	  for (int i = 0; i < facets.size(); ++i) 
		  if (boundaryMask[i] == true)
			  for (int j = 0; j < 3; ++j) {
				  int TetVertId = facets[i][j];

				  auto it = idxMap.find(TetVertId);
				  if (it == idxMap.end()) {//New vert
					  TriVertId = idxMap.size();
					  idxMap[TetVertId] = TriVertId;
					  //printf("-Map %d->%d\n", TetVertId, TriVertId);
					  //system("pause");
				  }
			  }

	  //Mapping vertex to boundary mesh
	  boundaryMesh.positions.resize(idxMap.size());
	  for (const auto& idxpair : idxMap) {
		  boundaryMesh.positions[idxpair.second]=this->positions[idxpair.first];
		  boundaryMesh.worldBounds.extend(boundaryMesh.positions[idxpair.second]);
		  std::cout<< "Bound pos: " << this->positions[idxpair.first] <<std::endl;
		  //printf("Map %d->%d (%f,%f,%f)\n", idxpair.first, idxpair.second,this->positions[idxpair.first].x, this->positions[idxpair.first].y, this->positions[idxpair.first].z);
	  }

	  //Mapping indice to boundary mesh, faceInfos -1 is replaced by -BoundaryMesh cell ID
	  int bdCellID = 0;
	  for (int i = 0; i < facets.size(); ++i)
		  if (boundaryMask[i] == true) {
			  vec4i idx;
			  if (faceInfos[i].front == -1) {
				  idx = vec4i(idxMap[facets[i].x], idxMap[facets[i].y], idxMap[facets[i].z], 0);//i0,i1,i2,boundary_tag
				  faceInfos[i].front = -(bdCellID+1);//1-based index
			  }
			  else//Outward-normal ordering fix
			  {
				  idx = vec4i(idxMap[facets[i].z], idxMap[facets[i].y], idxMap[facets[i].x], 0);//i2,i1,i0,boundary_tag
				  faceInfos[i].back = -(bdCellID+1);//1-based index
			  }
			  boundaryMesh.indices.push_back(idx);
			  //printf("%d (%d,%d,%d)/(%d,%d,%d)-Tag(%d)-front/back(%d/%d)\n", 
			  //	  i, idx.x, idx.y, idx.z, facets[i].x, facets[i].y, facets[i].z, idx.w,faceInfos[i].front, faceInfos[i].back);
			  bdCellID++;
		  }

	  //Check facet info for debug
	  /*
	  for (int tetID = 0; tetID < this->indices.size(); tetID++) {
		  printf("Tet%d Facet (%d,%d,%d,%d)\n", tetID,
			  this->tetfacets[tetID].x, this->tetfacets[tetID].y, this->tetfacets[tetID].z, this->tetfacets[tetID].w);
		  for (int i = 0; i < 4; ++i) {
			  int faceID = this->tetfacets[tetID][i];
			  printf("\t(%d %d %d %d) f/b %d %d\n",
				  this->facets[faceID].x, this->facets[faceID].y, this->facets[faceID].z, this->facets[faceID].w,
				  faceInfos[faceID].front, faceInfos[faceID].back);
		  }
	  }
	  */
	  std::cout << "created boundary mesh geom w/ " << boundaryMesh.indices.size() << " tris" << std::endl;
	  std::cout << "surface mesh bbox" << boundaryMesh.worldBounds.lower << ", " << boundaryMesh.worldBounds.upper << std::endl;

	  return boundaryMesh;
  }

}

#endif



