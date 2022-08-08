#include "common.h"
#include <owl/common/math/random.h>
#include "cudaHelpers.cuh"
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

//Brownian motion
#include <ctime>
#include <curand_kernel.h>

//Mesh
#include "cuda/DeviceTetMesh.cuh"


#include "query/RTQuery.h"


namespace advect {

    double diffusionCoeff = 0.1; // kbT/(6*pi*mu*rp)  um^2/s

    //
    // Thrust helper
    //

    template <typename T>
    struct minmax_pair
    {
        T min_val;
        T max_val;
    };

    // minmax_unary_op is a functor that takes in a value x and
    // returns a minmax_pair whose minimum and maximum values
    // are initialized to x.
    template <typename T>
    struct minmax_unary_op
        : public thrust::unary_function< T, minmax_pair<T> >
    {
        __host__ __device__
            minmax_pair<T> operator()(const T& x) const
        {
            minmax_pair<T> result;
            result.min_val = x;
            result.max_val = x;
            return result;
        }
    };

    // minmax_binary_op is a functor that accepts two minmax_pair 
    // structs and returns a new minmax_pair whose minimum and 
    // maximum values are the min() and max() respectively of 
    // the minimums and maximums of the input pairs
    template <typename T>
    struct minmax_binary_op
        : public thrust::binary_function< minmax_pair<T>, minmax_pair<T>, minmax_pair<T> >
    {
        __host__ __device__
            minmax_pair<T> operator()(const minmax_pair<T>& x, const minmax_pair<T>& y) const
        {
            minmax_pair<T> result;
            result.min_val = thrust::min(x.min_val, y.min_val);
            result.max_val = thrust::max(x.max_val, y.max_val);
            return result;
        }
    };

    template<typename T>
    struct negative
    {
        __host__ __device__ bool operator()(const T& x) const { return x < 0; }
    };

    //----------------End Thrust Helper-------------------

    //[Device] Init Random particles (pos and vel) in a box
    __global__ void initParticlesKernel(Particle* particles, int N,
        const box3d worldBounds)
    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= N) return;

        LCG<16> random;
        random.init(threadIdx.x, blockIdx.x);

        // *any* random position for now:
        vec3d randomPos
            = worldBounds.lower
            + vec3d(random(), random(), random())
            * worldBounds.size();

        particles[particleID].x = randomPos.x;
        particles[particleID].y = randomPos.y;
        particles[particleID].z = randomPos.z;
        particles[particleID].w = true;
    }

    //[Host] Init Random particles (pos and vel) in a box
    void cudaInitParticles(Particle* d_particles, int N,
        const box3d& worldBounds)
    {
        int blockDims = 128;
        int gridDims = divRoundUp(N, blockDims);
        initParticlesKernel << <gridDims, blockDims >> > (d_particles, N, worldBounds);

        cudaCheck(cudaDeviceSynchronize());
    }

    //[Host] Find the number of particles in file
    int loadNumParticles(std::string fileName)
    {
        int numParticles = 0;
        std::string word;

        std::ifstream vfile(fileName);
        if (vfile.is_open()) {
            vfile >> word >> numParticles;//Header Line
            printf("%s %d\n", word.c_str(), numParticles);
        }
        vfile.close();

        return numParticles;
    }

    //[Host] Init particles (pos and vel) from file
    void cudaInitParticles(Particle* d_particles, int N, std::string fileName)
    {
        //Read particles
        std::string word;
        int numParticles = 0;

        std::vector<Particle> particle_loc;
        particle_loc.reserve(N);

        std::ifstream vfile(fileName);
        if (vfile.is_open()) {
            vfile >> word >> numParticles;//Header Line
            printf("%s %d\n", word.c_str(), numParticles);
            vfile >> word >> word >> word >>word;//Comment line, x,y,z,tetID

            for (int i = 0; i < N; ++i) {
                Particle pos;
                vfile >> pos.x >> pos.y >> pos.z >> word;
                pos.w = true;
                particle_loc.push_back(pos);
                if (i < 5) 
                    std::cout << "Seeding Pos" << i << " " << vec4d(particle_loc[i].x, particle_loc[i].y, particle_loc[i].z, particle_loc[i].w) << std::endl;
            }
            vfile.close();
        }


        //Upload to GPU
        cudaCheck(cudaMemcpy(d_particles,
            particle_loc.data(),
            N * sizeof(particle_loc[0]),
            cudaMemcpyHostToDevice));
        cudaCheck(cudaDeviceSynchronize());
    }


    //[Device] Init Random particles (pos and vel) in a box
    __global__ void evalTimestep(int NumTets,
        vec4i* d_tetIndices,
        vec3d* d_vertexPositions,
        vec3d* d_TetVelocities,
        double diffusionCoeff,
        double* d_dt)
    {
        int tetID = threadIdx.x + blockDim.x * blockIdx.x;
        if (tetID >= NumTets) return;

        vec4i index = d_tetIndices[tetID];
        const vec3d A = d_vertexPositions[index.x];
        const vec3d B = d_vertexPositions[index.y];
        const vec3d C = d_vertexPositions[index.z];
        const vec3d D = d_vertexPositions[index.w];

        const double volume = dot(D - A, cross(B - A, C - A));
        const double grid_h = cbrt(volume);
        
        //Velocity timestep constrains : not exceed half of grid size
        const vec3d vel = d_TetVelocities[tetID];
        double max_vel_disp = length(vel);
        const double dt_vel = 0.5*grid_h/ max_vel_disp;

        //Brownian motion timestep constrians : 
        const double dt_vel_brownian =
            (sqrt(6 * diffusionCoeff + 2 * max_vel_disp * grid_h)
                - sqrt(6 * diffusionCoeff))
            / (2 * max_vel_disp);

        const double dt_estimate = abs(min(dt_vel_brownian, dt_vel));
        d_dt[tetID] = dt_estimate<1e-8?1.12345678:dt_estimate;


        //if (d_dt[tetID] < 1e-2)
        //    printf("TetID=%d Vel=%f,%f,%f Velocity Disp=%f\n grid_size=%f, dt_vel=%f, dt_vel+Brownian=%f\n", tetID,
        //        vel.x, vel.y, vel.z, max_vel_disp,
        //        grid_h, dt_vel, dt_vel_brownian);
    }

   
    //[Host] Estimate stable step size based on local element size
    double cudaEvalTimestep(int NumTets,
        vec4i* d_tetIndices, 
        vec3d* d_vertexPositions, 
        vec3d* d_Velocities, std::string mode)
    {
        thrust::device_vector<double> d_dt_thrust(NumTets, 1000.0);
        double* d_dt = thrust::raw_pointer_cast(d_dt_thrust.data());

        int blockDims = 128;
        int gridDims = divRoundUp(NumTets, blockDims);
        evalTimestep << <gridDims, blockDims >> > (NumTets,
            d_tetIndices,
            d_vertexPositions,
            d_Velocities,
            diffusionCoeff,
            d_dt);
        cudaCheck(cudaDeviceSynchronize());

        // setup arguments
        minmax_unary_op<double>  unary_op;
        minmax_binary_op<double> binary_op;
        // initialize reduction with the first value
        minmax_pair<double> init = unary_op(d_dt_thrust[0]);

        // compute minimum and maximum values
        minmax_pair<double> result = thrust::transform_reduce(d_dt_thrust.begin(), d_dt_thrust.end(), unary_op, init, binary_op);

        std::cout << "#adv: minimum dt= " << result.min_val << std::endl;
        std::cout << "#adv: maximum dt= " << result.max_val << std::endl;

        return result.min_val;
    }


    //----------------------Advection--------------------


    //[Device] Pk tet velocity interpolation
    __global__
        void particleAdvectKernel(Particle* d_particles,
            int*   d_tetIDs,
            vec4d* d_vels,
            vec4d* d_disp,
            double dt,
            int    numParticles,
            vec4i* d_tetIndices,
            vec3d* d_vertexPositions,
            vec3d* d_vertexVelocities)
    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        Particle& p = d_particles[particleID];
        if (!p.w) return;

        const int tetID = d_tetIDs[particleID];
        if (tetID < 0) {//tet=-1
            // this particle left the domain
            p.w = false;
            return;
        }

        vec4i index = d_tetIndices[tetID];
        const vec3d P = vec3d(p.x, p.y, p.z);
        const vec3d A = d_vertexPositions[index.x];
        const vec3d B = d_vertexPositions[index.y];
        const vec3d C = d_vertexPositions[index.z];
        const vec3d D = d_vertexPositions[index.w];

        const double den = det(A, B, C, D);
        if (den == 0.f) {//We are in a bad tet, set tetID=-2
            p.w = false;
            return;
        }

        const double wA = det(P, B, C, D) * (1. / den);
        const double wB = det(A, P, C, D) * (1. / den);
        const double wC = det(A, B, P, D) * (1. / den);
        const double wD = det(A, B, C, P) * (1. / den);

        const vec3d velA = d_vertexVelocities[index.x];
        const vec3d velB = d_vertexVelocities[index.y];
        const vec3d velC = d_vertexVelocities[index.z];
        const vec3d velD = d_vertexVelocities[index.w];

        const vec3d vel
            = wA * velA
            + wB * velB
            + wC * velC
            + wD * velD;

        //First-order Euler Integration
        const vec3d P_next = P + dt * vel;
        const vec3d P_disp = P_next - P;

        d_vels[particleID] = vec4d(vel.x, vel.y, vel.z, -1.0);
        d_disp[particleID] = vec4d(P_disp.x, P_disp.y, P_disp.z, -1.0);
        //p.x = P_next.x; p.y = P_next.y; p.z = P_next.z;

        /*
        if (particleID == 98550)
            printf("%d [Advect] TetID=%d P(%f,%f,%f) Disp(%f,%f,%f) Vel(%f,%f,%f)\n",
                particleID, tetID,
                P.x, P.y, P.z,
                P_disp.x, P_disp.y, P_disp.z,
                vel.x, vel.y, vel.z);
        */
    }

    //[Device] RT0 tet velocity interpolation
    __global__
        void particleAdvectKernelTetVel(Particle* d_particles,
            int* d_tetIDs,
            vec4d* d_vels,
            vec4d* d_disp,
            double dt,
            int    numParticles,
            vec4i* d_tetIndices,
            vec3d* d_vertexPositions,
            vec3d* d_TetVelocities)
    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        Particle& p = d_particles[particleID];
        if (!p.w) return;

        const int tetID = d_tetIDs[particleID];
        if (tetID < 0) {//tet=-1
            // this particle left the domain
            p.w = false;
            return;
        }


        vec4i index = d_tetIndices[tetID];
        const vec3d P = vec3d(p.x, p.y, p.z);
        const vec3d A = d_vertexPositions[index.x];
        const vec3d B = d_vertexPositions[index.y];
        const vec3d C = d_vertexPositions[index.z];
        const vec3d D = d_vertexPositions[index.w];

        const double den = det(A, B, C, D);
        if (den == 0.f) {//We are in a bad tet, set tetID=-2
            p.w = false;
            return;
        }

        const vec3d vel
            = (vec3d&)d_TetVelocities[tetID];

        //First-order Euler Integration
        const vec3d P_next = P + dt * vel;
        const vec3d P_disp = P_next - P;

        d_vels[particleID] = vec4d(vel.x, vel.y, vel.z, -1.0);
        d_disp[particleID] = vec4d(P_disp.x, P_disp.y, P_disp.z, -1.0);
        //p.x = P_next.x; p.y = P_next.y; p.z = P_next.z;

        /*
        if (particleID <100)
            printf("%d [Advect] TetID=%d P(%f,%f,%f) Disp(%f,%f,%f) Vel(%f,%f,%f)\n",
                particleID, tetID,
                P.x, P.y, P.z,
                P_disp.x, P_disp.y, P_disp.z,
                vel.x, vel.y, vel.z);
        */
    }

    //[Device] Constant initial velocity interpolation
    __global__
        void particleAdvectConstVel(Particle* d_particles,
            int* d_tetIDs,
            vec4d* d_vels,
            vec4d* d_disp,
            double dt,
            int       numParticles) {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        Particle& p = d_particles[particleID];
        if (!p.w) return;

        const int tetID = d_tetIDs[particleID];
        if (tetID < 0) {//tet=-1
            // this particle left the domain
            p.w = false;
            return;
        }

        //Constant velocity, First-order Euler Integration
        vec4d& vel = d_vels[particleID];
        d_disp[particleID] = vec4d(vel.x*dt, vel.y * dt, vel.z * dt, -1.0);
    }

    // [Velocity] Tet velocity interpolation
    // output: d_vels
    void cudaAdvect(Particle* d_particles,
        int* d_tetIDs,
        vec4d* d_vels,
        vec4d* d_disp,
        double dt,
        int numParticles,
        vec4i* d_tetIndices,
        vec3d* d_vertexPositions,
        vec3d* d_Velocities,
        std::string mode)
    {
        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        if (mode == "TetVelocity") {
            particleAdvectKernelTetVel << <gridDims, blockDims >> > (d_particles,
                d_tetIDs,
                d_vels,
                d_disp,
                dt,
                numParticles,
                d_tetIndices,
                d_vertexPositions,
                d_Velocities);
        }
            
        if (mode == "VertexVelocity")
            particleAdvectKernel << <gridDims, blockDims >> > (d_particles,
                d_tetIDs,
                d_vels,
                d_disp,
                dt,
                numParticles,
                d_tetIndices,
                d_vertexPositions,
                d_Velocities);
        if (mode == "ConstantVelocity")
            particleAdvectConstVel << <gridDims, blockDims >> > (d_particles,
                d_tetIDs,
                d_vels,
                d_disp,
                dt,
                numParticles);

        cudaCheck(cudaDeviceSynchronize());
    }

    //[Device] Constant initial velocity interpolation
    __device__ double SquareDuct_analyticalVel(double x,double y,
        double h,double L, double dp,double mu) {
        //10.1103/PhysRevE.71.057301
        double vz = 0.0;

        for (int i = 0; i < 20; i++) {
            double n = 2.0 * i + 1.0;
            vz += 1 / (n * n * n) * ( 1.0 - cosh(n* M_PI*x/h) / cosh(n*M_PI/2.0) )*sin(n * M_PI * y/ h);
        }

        vz = -dp / L / mu * 4.0 * h * h / M_PI / M_PI / M_PI * vz;
        return vz;
    }

    __global__
        void particleTubeAdvect(Particle* d_particles,
            int* d_tetIDs,
            vec4d* d_vels,
            vec4d* d_disp,
            double dt,
            int numParticles,
            double h, double L, double dp, double mu) {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        Particle& p = d_particles[particleID];
        if (!p.w) return;

        const int tetID = d_tetIDs[particleID];
        if (tetID < 0) {//tet=-1
            // this particle left the domain
            p.w = false;
            return;
        }

        //Constant velocity, First-order Euler Integration
        vec4d& vel = d_vels[particleID];
        vel.x = 0.0; 
        vel.y = 0.0;
        vel.z = SquareDuct_analyticalVel(p.x, p.y, h, L, dp, mu);
        //printf("%lf %lf %lf->vel=%lf,%lf,%lf\n", p.x, p.y, p.z, vel.x, vel.y, vel.z);
        d_disp[particleID] = vec4d(vel.x * dt, vel.y * dt, vel.z * dt, -1.0);

        //if(particleID<5)
        //    printf("(%lf,%lf,%lf) Vel=(%lf,%lf,%lf)\n", p.x, p.y, p.z, vel.x, vel.y, vel.z);
    }

    void cudaTubeAdvect(Particle* d_particles, int* d_tetIDs,
        vec4d* d_vels, vec4d* d_disp, double dt, int numParticles)
    {
        double L = 30;//cm
        double h = 0.1;//cm
        double mu = 0.001072;//Pa s
        double dp = -4.904871302657455;//Pa
        double Q = 0.000536;//cm^3/s

        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);
        particleTubeAdvect << <gridDims, blockDims >> > (d_particles,
            d_tetIDs,
            d_vels,
            d_disp,
            dt,
            numParticles,
            h, L, dp, mu);

        cudaCheck(cudaDeviceSynchronize());
        //system("pause");
    }

    //----------------------Brownian motion--------------------
#include <thrust/execution_policy.h>

    struct InitCURAND
    {
        unsigned long long seed;
        curandState_t* states;
        InitCURAND(unsigned long long _seed, curandState_t* _states)
        {
            seed = _seed;
            states = _states;
        }

        __device__
            void operator()(unsigned int i)
        {
            curand_init(seed, i, 0, &states[i]);
        }
    };

    void initRandomGenerator(int numParticles, curandState_t* rand_states){
        //Each particle has its own random generator for each thread
        long int rng_seed = time(NULL);
        rng_seed = 1591593751;
        printf("#adv: Random Seed=%d\n", rng_seed);
        thrust::counting_iterator<unsigned int> count(0);
        thrust::for_each(count, count + numParticles, InitCURAND(rng_seed, rand_states));
    }

    //[Device] Constant initial velocity interpolation
    __global__
        void particleBrownianMotion(Particle* d_particles,
            vec4d* d_disp,
            double D,
            curandState_t* states,
            double dt,
            int numParticles) {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        Particle& p = d_particles[particleID];
        if (!p.w) return;

        const double randDisp = sqrt(2.00 * D * dt);
        const double randXi0 = curand_normal_double(&states[particleID]);
        const double randXi1 = curand_normal_double(&states[particleID]);
        const double randXi2 = curand_normal_double(&states[particleID]);

        d_disp[particleID] += vec4d(randXi0, randXi1, randXi2, 0.0)*randDisp;

        //if (particleID == 170)
        //printf("%d D=%f dt=%f Random disp (%f,%f,%f)-(%f,%f,%f) Disp %lf\n", particleID, D,dt,
        //    d_disp[particleID].x, d_disp[particleID].y, d_disp[particleID].z, 
        //    randXi0, randXi1, randXi2,randDisp);
    }

    void cudaBrownianMotion(Particle* d_particles, 
        vec4d* d_disp, 
        curandState_t* states,
        double dt, 
        int numParticles,
	double diffusionCoeff)
    {//Brownian motion
        //double diffusionCoef = 0.1;// kbT/(6*pi*mu*rp)  um^2/s

        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        // diffusionCoef = 5.7e-6;

        particleBrownianMotion << <gridDims, blockDims >> > (d_particles,
            d_disp,
            diffusionCoeff,
            states,
            dt,
            numParticles);
        cudaCheck(cudaDeviceSynchronize());    
        //system("pause");
    }

    

    __global__
        void checkParticles(int* d_triIDs, bool * isHit, int numParticles) {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        //if (particleID == 52747)
        //printf("Check Pts %d %d OldStatus=%d\n",particleID, d_triIDs[particleID], isHit[particleID]);

        if (d_triIDs[particleID] == -1) return;
            
        isHit[particleID] = true;
    }

    


    //----------------------Move--------------------
    __global__
        void particleMoveKernel(Particle* d_particles,
            vec4d* d_vels,
            double dt,
            int    numParticles)
    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        Particle& p = d_particles[particleID];
        if (!p.w) return;//particle is out of domain

        const vec4d P = vec4d(p.x, p.y, p.z, p.w);
        const vec4d vel = d_vels[particleID];
        const vec4d P_next = P + vel * dt;

        //if (particleID == 5)
        //    printf("[Move] %d (%f,%f,%f)->(%f,%f,%f) @ Vel(%f,%f,%f)\n", particleID,
        //        p.x, p.y, p.z, P_next.x, P_next.y, P_next.z,
        //        vel.x, vel.y, vel.z);

        p.x = P_next.x; p.y = P_next.y; p.z = P_next.z;
    }


    //[Host] Move particles x1 = x0 + vel*dt and do specular reflection if hit the wall
    // d_tetIDs used to determine the status of a particle
    void cudaMoveParticles(Particle* d_particles, vec4d* d_vels, double dt,
        int numParticles, int* d_tetIDs){

        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        //Move particles
        particleMoveKernel << <gridDims, blockDims >> > (d_particles, d_vels, dt, numParticles);
        cudaCheck(cudaDeviceSynchronize());
    }


    __global__
        void particleMoveKernel(Particle* d_particles,
            vec4d* d_disps,
            int* d_tetIDs,
            int numParticles)
    {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        Particle& p = d_particles[particleID];
        vec4d& disp = d_disps[particleID];
        
        /*
        if (!p.w) {
            //if (particleID == 33216)
                //if (particleID == 52747 || particleID == 98550 || particleID == 83144)
                //printf("[Move] %d (%.15lf,%.15lf,%.15lf)->Disp(%.15f,%.15f,%.15f)\n", particleID,
                //    p.x, p.y, p.z,
                //    disp.x, disp.y, disp.z);
        }
        */
        if (!p.w) return;//particle is out of domain

        

        /*
        const vec4d P = vec4d(p.x, p.y, p.z, p.w);
        const vec4d P_next = P + disp;
        if (particleID == 50606) {
            //if (particleID == 52747 || particleID == 98550 || particleID == 83144)
            printf("[Move] %d tetID=%d (%.15lf,%.15lf,%.15lf)->(%.15f,%.15f,%.15f)  Disp(%.15f,%.15f,%.15f)\n", particleID, d_tetIDs[particleID],
                p.x, p.y, p.z, P_next.x, P_next.y, P_next.z,
                disp.x, disp.y, disp.z);
        }  
        */
        //Update location
        p.x += disp.x;
        p.y += disp.y;
        p.z += disp.z;

        //Reset disp for the next iteration
        disp.x = 0.0;
        disp.y = 0.0;
        disp.z = 0.0;
        //disp.w = 0.0;
    }

    void cudaMoveParticles(Particle* d_particles, vec4d* d_disps,
        int numParticles, int* d_tetIDs) {

        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);


        //Move particles
        particleMoveKernel << <gridDims, blockDims >> > (d_particles, d_disps, d_tetIDs,numParticles);
        cudaCheck(cudaDeviceSynchronize());
    }

    __global__ void updateVelocity( vec3d* velocities, int numTets, vec4i* d_tetIndices, vec3d* d_Velocities)
    {
        int tetID = blockIdx.x * blockDim.x + threadIdx.x;
        
	vec4i index = d_tetIndices[tetID];

	while (tetID < numTets)
	{
             const vec3d vel = velocities[tetID];
	     d_Velocities[tetID] = vel;
	     tetID += gridDim.x*blockDim.x;
	}

    }

    void cudaUpdateVelocity( std::vector<vec3d> velocities, int numTets, vec4i* d_tetIndices, vec3d* d_Velocities)
    {
        int blockDims = 128;
        int gridDims = divRoundUp(numTets, blockDims);

	thrust::device_vector<vec3d> d_vec_thrust = velocities;

        /*for (int i = 0; i < numTets; ++i) {
            // std::cout<< i <<std::endl;
            d_vec_thrust[i] = velocities[i];
	}*/

	vec3d* d_vec = thrust::raw_pointer_cast(d_vec_thrust.data());
        
        updateVelocity << <gridDims, blockDims >> > (d_vec, numTets, d_tetIndices, d_Velocities);
        cudaCheck(cudaDeviceSynchronize());
    }

    //-------------------------Debug-----------------------

    __global__
        void reportParticles(int* tetIDs, int numParticles, int TagID) {
        int particleID = threadIdx.x + blockDim.x * blockIdx.x;
        if (particleID >= numParticles) return;

        if (tetIDs[particleID] <= TagID) 
            printf("--Particle [%d] TagID=%d\n", particleID, tetIDs[particleID]);
    }

    //[Host] Check particle status based on tetID
    void cudaReportParticles(int numParticles, int* d_tetIDs) {
        thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_tetIDs);

        int blockDims = 128;
        int gridDims = divRoundUp(numParticles, blockDims);

        int NumBadParticles = thrust::count_if(thrust::device, dev_ptr, dev_ptr + numParticles, negative<int>());
        printf("#adv: Out-of-domain particles(-tetID) = %d\n", NumBadParticles);
        if (NumBadParticles > 0) {
            reportParticles << <gridDims, blockDims >> > (d_tetIDs, numParticles, -1);
            cudaCheck(cudaDeviceSynchronize());
        }
    }
}
