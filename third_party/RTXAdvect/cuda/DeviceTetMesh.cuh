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
#ifndef TET_MESH_CUDA
#define TET_MESH_CUDA

#include "HostTetMesh.h"
#include "cudaHelpers.cuh"

namespace advect {
  
  struct DeviceTetMesh {

    void upload(const HostTetMesh &hostTetMesh);
    vec3d *d_positions    { nullptr };
    vec3d *d_velocities   { nullptr };
    vec4i *d_indices      { nullptr };

    vec4i* d_facets       { nullptr };
    vec4i* d_tetfacets    { nullptr };
    FaceInfo* d_faceinfos { nullptr };
    box3d worldBounds;
  };

  struct DeviceBdMesh {
      void upload(const HostTetMesh& hostBoundaryMesh);
      vec3d* d_positions{ nullptr };
      vec4i* d_indices{ nullptr };
      box3d worldBounds;
  };

  template<typename T>
  inline void allocAndUpload(T *&d_data,
                      const std::vector<T> &data)
  {
    assert(d_data == nullptr);
    cudaCheck(cudaMalloc(&d_data,
                         data.size()*sizeof(data[0])));
    cudaCheck(cudaMemcpy(d_data,
                         data.data(),
                         data.size()*sizeof(data[0]),
                         cudaMemcpyHostToDevice));
  }
  
  inline void DeviceTetMesh::upload(const HostTetMesh &mesh)
  {
    allocAndUpload(d_positions,mesh.positions);
    allocAndUpload(d_velocities,mesh.velocities);
    allocAndUpload(d_indices,mesh.indices);

    allocAndUpload(d_facets, mesh.facets);
    allocAndUpload(d_tetfacets, mesh.tetfacets);
    allocAndUpload(d_faceinfos, mesh.faceInfos);



    worldBounds = mesh.worldBounds;
  }

  inline void advect::DeviceBdMesh::upload(const HostTetMesh& mesh)
  {
      allocAndUpload(d_positions, mesh.positions);
      allocAndUpload(d_indices, mesh.indices);
      worldBounds = mesh.worldBounds;
  }

  //---------------Geometric Calculation-----------------
  inline __device__ double det(const vec3d A,
      const vec3d B,
      const vec3d C,
      const vec3d D)
  {
      return dot(D - A, cross(B - A, C - A));
  }

  inline __device__ double det(
      double a, double b, double c,
      double d, double e, double f,
      double g, double h, double i)
  {
      /*
     * 3x3 determiant
     * |         |
     * | a  b  c |
     * | d  e  f |
     * | g  h  i |
     * |         |
     */
      return (a * (e * i - f * h)) - (b * (d * i - f * g)) + (c * (d * h - e * g));
  }
  


  inline __device__ vec4d tetBaryCoord(const vec3d P, const vec3d A,
      const vec3d B,
      const vec3d C,
      const vec3d D) {

      const double den = det(A, B, C, D);
      
      const double wA = det(P, B, C, D) * (1. / den);
      const double wB = det(A, P, C, D) * (1. / den);
      const double wC = det(A, B, P, D) * (1. / den);
      //const double wD = det(A, B, C, P) * (1. / den);
      const double wD = 1.0 - wA - wB - wC;
      /*
      const vec3d v0 = B - A;
      const vec3d v1 = C - A;
      const vec3d v2 = D - A;
      const vec3d v3 = P - A;

      double d00 = dot(v0, v0), d01 = dot(v0, v1), d02 = dot(v0, v2);
      double d11 = dot(v1, v1), d21 = dot(v2, v1), d22 = dot(v2, v2);
      double d30 = dot(v3, v0), d31 = dot(v3, v1), d32 = dot(v3, v2);

      double denom =   det(d00, d01, d02,
                           d01, d11, d21,
                           d02, d21, d22);

      double denom_x = det(d30, d01, d02,
                           d31, d11, d21,
                           d32, d21, d22);

      double denom_y = det(d00, d30, d02,
                           d01, d31, d21,
                           d02, d32, d22);

      double denom_z = det(d00, d01, d30,
                           d01, d11, d31,
                           d02, d21, d32);

      double v = denom_x / denom;
      double w = denom_y / denom;
      double z = denom_z / denom;
      double u = 1 - v - w - z;

      printf("%lf,%lf,%lf,%lf-%lf,%lf,%lf,%lf", 
          u, v, w, z,
          wA, wB, wC, wD);
      */
      return vec4d(wA, wB, wC, wD);
  }

  inline __device__ vec3d triBaryCoord(const vec3d P, const vec3d A,
      const vec3d B,
      const vec3d C) 
  {//Compute barycentric coordinate for a point P wihtin a Triangle ABC
      const vec3d v0 = B - A;
      const vec3d v1 = C - A;
      const vec3d v2 = P - A;

      double d00 = dot(v0, v0), d01=dot(v0, v1);
      double d11 = dot(v1, v1);
      double d20 = dot(v2, v0), d21 = dot(v2, v1);

      double denom = d00 * d11 - d01 * d01;

      double v = (d11 * d20 - d01 * d21) / denom;
      double w = (d00 * d21 - d01 * d20) / denom;
      double u = 1 - v - w;

      return vec3d(u, v, w);
  }

  inline __device__ bool maxNegative(vec4d vec, int& ind, double& val) 
  {//Find the max negative value and index from a given vector
      double valmin = reduce_min(vec);
      if (valmin >= 0.0) return false;
      
      valmin -= 0.1;
      for (int i = 0; i < 4; ++i)
          if (vec[i] < 0.0 && vec[i]>valmin) {
              val = vec[i];
              ind = i;
          }
      return true;
  }

  inline __device__ vec3d triNorm(const vec3d A,
      const vec3d B,
      const vec3d C)
  {//Triangle normal vector, order of A,B,C define normal vector direction
      vec3d norm = cross(B - A, C - A);
      return norm / length(norm);
  }

  inline __device__ vec3d triReflect(const vec3d P_start, const vec3d P_end,
      const vec3d A,
      const vec3d B,
      const vec3d C) {
      //Do ray triangle reflection, assume ABC oriented outward
      //Ray shoot from back side to front side, and reflect back to back side
      vec3d nw = -triNorm(A, B, C);//inward normal vector
      vec3d P_reflect = P_end - (1.0 + 1.0) * dot(P_end - A, nw) * nw;

      return P_reflect;
  }

  inline __device__ vec3d triEdgeNorm(const int ei,const vec3d P, 
      const vec3d A,
      const vec3d B,
      const vec3d C,
      const vec3d nw) {
      //Compute the edge outward normal vector for a 3D triangle
      if (ei == 0) {
          vec3d norm_e1 = cross(C - B, nw);
          return norm_e1 / length(norm_e1);
      }
      if (ei == 1) {
          vec3d norm_e2 = cross(A - C, nw);
          return norm_e2 / length(norm_e2);
      }
      if (ei == 2) {
          vec3d norm_e3 = cross(B - A, nw);
          return norm_e3 / length(norm_e3);
      }

  }

  inline __device__ vec3d triEdgeNode(const int ei,
      const vec3d A,
      const vec3d B,
      const vec3d C) {
      if (ei == 0) return C; //B-C
      if (ei == 1) return A; //A-C
      if (ei == 2) return B; //A-B
  }

  inline __device__ double Pts2TriDist(const vec3d P,
      const vec3d A,
      const vec3d B,
      const vec3d C) {
  
  }

  
}


#endif
