
#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <map>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "owl/owl.h"
#include "owl/common/math/box.h"
#include "optix/OptixQuery.h"

#include "cuda/DeviceTetMesh.cuh"

#include <curand_kernel.h>


namespace advect {
	using namespace owl;
	using namespace owl::common;

	typedef owl::common::box_t<vec3d> box3d;

	typedef double4 Particle;
	//typedef float4 Particle;

	//
	// Simulation routines
	//
	void cudaInitParticles(Particle* d_particles, int N,
		const box3d& worldBounds);

	void cudaInitParticles(Particle* d_particles, int N,
		std::string fileName);

	double cudaEvalTimestep(int NumTets,
		vec4i* d_tetIndices,
		vec3d* d_vertexPositions,
		vec3d* d_Velocities,
		std::string mode,
		double diffusionCoeff);

	void cudaAdvect(Particle* d_particles,
		int* d_tetIDs,
		vec4d* d_vels,
		vec4d* d_disp,
		double dt,
		int numParticles,
		vec4i* d_tetIndices,
		vec3d* d_vertexPositions,
		vec3d* d_Velocities,
		std::string mode);

        //Speical anallytical square tube advector
	void cudaTubeAdvect(Particle* d_particles, 
		int* d_tetIDs,
		vec4d* d_vels,
		vec4d* d_disp,
		double dt,
		int numParticles);

	void initRandomGenerator(int numParticles, curandState_t *states);

	void cudaBrownianMotion(Particle* d_particles,
		vec4d* d_disp,
		curandState_t* states,
		double dt,
		int numParticles,
		double diffusionCoeff);

	void cudaMoveParticles(Particle* d_particles, vec4d* d_vels, double dt,
		int numParticles,int* d_tetIDs);

	void cudaMoveParticles(Particle* d_particles, vec4d* d_disps,
		int numParticles, int* d_tetIDs);

	void cudaReportParticles(int numParticles, int* d_tetIDs);
	
	void cudaUpdateVelocity(std::vector<vec3d> velocities, int numParticles, vec4i* d_tetIndices, vec3d* d_Velocities);

	//
	//  I/O routines
	//
	int loadNumParticles(std::string fileName);
	void addToTrajectories(Particle* d_particles,
		int numParticles, std::vector<std::vector<vec3f>> &trajectories);
	void saveTrajectories(const std::string& fileName, std::vector<std::vector<vec3f>>& trajectories);
	void writeStreamline2VTK(const std::string& fileName, std::vector<std::vector<vec3f>>& trajectories);
	void writeParticles2VTU(unsigned int ti,
		Particle* d_particles,
		vec4d* d_vels,
		int* d_tetIDs,
		int numParticles,
		int* d_tetIDs_Convex = nullptr);
	void writeParticles2OBJ(unsigned int ti,
		Particle* d_particles,
		vec4d* d_vels,
		int* d_tetIDs,
		int numParticles,
		int* d_tetIDs_Convex = nullptr);



	
	

}

#endif
