#include <fstream>

#include "common.h"
#include "cudaHelpers.cuh"

namespace advect {
    void addToTrajectories(Particle* d_particles,
        int numParticles, std::vector<std::vector<vec3f>>& trajectories)
    {
        if (trajectories.empty())
            trajectories.resize(numParticles);

        cudaDeviceSynchronize();

        std::vector<Particle> hostParticles(numParticles);
        cudaCheck(cudaMemcpy(hostParticles.data(),
            d_particles,
            numParticles * sizeof(Particle),
            cudaMemcpyDeviceToHost));
        for (int i = 0; i < numParticles; i++) {
            if (!hostParticles[i].w) continue;

            vec3f p(hostParticles[i].x,
                hostParticles[i].y,
                hostParticles[i].z);
            trajectories[i].push_back(p);
        }
    }

    void saveTrajectories(const std::string& fileName, std::vector<std::vector<vec3f>>& trajectories)
    {
        std::ofstream out(fileName);
        int numVerticesWritten = 0;
        for (auto& traj : trajectories) {
            int firstVertexID = numVerticesWritten + 1; // +1 for obj format
            if (traj.size() <= 1) continue;
            for (auto& p : traj) {
                out << "v " << p.x << " " << p.y << " " << p.z << std::endl;
                numVerticesWritten++;
            }
            for (int i = 0; i < (traj.size() - 1); i++) {
                out << "l " << (firstVertexID + i) << " " << (firstVertexID + i + 1) << std::endl;
            }
        }

        out.close();
    }

	void writeStreamline2VTK(const std::string& fileName, std::vector<std::vector<vec3f>>& trajectories) {

		std::ofstream out(fileName);
		int NumSLs = 0;
		int numVerticesWritten = 0;
		for (auto& traj : trajectories) {
			if (traj.size() <= 1) continue;
			for (auto& p : traj)
				numVerticesWritten++;
			NumSLs += 1;
		}

		out << "# vtk DataFile Version 4.1\n";
		out << "vtk output\n";
		out << "ASCII\n";
		out << "DATASET POLYDATA\n";
		out << "POINTS " << numVerticesWritten << " float\n";
		for (auto& traj : trajectories) {
			if (traj.size() <= 1) continue;
			for (auto& p : traj)
				out << p.x << " " << p.y << " " << p.z << std::endl;
		}
		out << "\n";

		out << "LINES " << NumSLs << " " << numVerticesWritten + NumSLs << "\n";
		int vertexID = 0;
		for (auto& traj : trajectories) {
			if (traj.size() <= 1) continue;
			out << traj.size();
			for (int i = 0; i < traj.size(); i++) {
				out << " " << vertexID;
				vertexID += 1;
			}
			out << "\n";
		}
		out << "\n\n";

		out << "CELL_DATA " << NumSLs << "\n";
		out << "FIELD FieldData 1\n"; //Now we only set one field here

		out << "StreamlineID 1 " << NumSLs << " int\n";
		for (int i = 0; i < NumSLs; ++i)
			out << i << " " << "\n";;

		out.close();
	}

    void writeParticles2OBJ(unsigned int ti,
        Particle* d_particles,
        vec4d* d_vels,
        int* d_tetIDs,
        int numParticles,
        int* d_tetIDs_Convex)
    {
        cudaDeviceSynchronize();
        std::vector<Particle> hostParticles(numParticles);
        cudaCheck(cudaMemcpy(hostParticles.data(),
            d_particles,
            numParticles * sizeof(Particle),
            cudaMemcpyDeviceToHost));

        std::vector<int> h_tetIDs(numParticles);
        cudaCheck(cudaMemcpy(h_tetIDs.data(),
            d_tetIDs,
            numParticles * sizeof(int),
            cudaMemcpyDeviceToHost));

        std::vector<vec4d> h_vels(numParticles);
        cudaCheck(cudaMemcpy(h_vels.data(),
            d_vels,
            numParticles * sizeof(vec4d),
            cudaMemcpyDeviceToHost));

        //Output particle into OBJ file
        int i;
        FILE* fp;
        char fileName[1024];

        sprintf(fileName, "particle_%04d.obj", ti);
        if (ti % 500 == 0) printf("#adv: Write particles to file %s...\n", fileName);

        fp = fopen(fileName, "w");
        int numVerticesWritten = 0;
        for (i = 0; i < numParticles; i++) {
            fprintf(fp, "v %.15lf %.15lf %.15lf\n",
                hostParticles[i].x,
                hostParticles[i].y,
                hostParticles[i].z);
         //       out << "v " << hostParticles[i].x << " " << hostParticles[i].y << " " << hostParticles[i].z << std::endl;
        //        out << "l " << (firstVertexID + i) << " " << (firstVertexID + i + 1) << std::endl;
        }

        fclose(fp);
    }

    void writeParticles2VTU(unsigned int ti,
        Particle* d_particles,
        vec4d* d_vels,
        int* d_tetIDs,
        int numParticles,
        int* d_tetIDs_Convex)
    {
        //Move data from GPU
        cudaDeviceSynchronize();

        std::vector<Particle> hostParticles(numParticles);
        cudaCheck(cudaMemcpy(hostParticles.data(),
            d_particles,
            numParticles * sizeof(Particle),
            cudaMemcpyDeviceToHost));

        std::vector<int> h_tetIDs(numParticles);
        cudaCheck(cudaMemcpy(h_tetIDs.data(),
            d_tetIDs,
            numParticles * sizeof(int),
            cudaMemcpyDeviceToHost));

        std::vector<vec4d> h_vels(numParticles);
        cudaCheck(cudaMemcpy(h_vels.data(),
            d_vels,
            numParticles * sizeof(vec4d),
            cudaMemcpyDeviceToHost));

        //Output particle into VTU file
        int i;
        FILE* fp;
        char fileName[1024];

        sprintf(fileName, "particle_%04d.vtu", ti);
        if (ti % 500 == 0) printf("#adv: Write particles to file %s...\n", fileName);

        fp = fopen(fileName, "w");
        //fprintf(fp, "<?xml version='1.0' encoding='UTF-8'?>\n");
        //fprintf(fp, "<VTKFile xmlns='VTK' byte_order='LittleEndian' version='0.1' type='UnstructuredGrid'>\n");
        fprintf(fp, "<VTKFile type='UnstructuredGrid' version='1.0' byte_order='LittleEndian' header_type='UInt64'>\n");
        fprintf(fp, "<UnstructuredGrid>\n");
        fprintf(fp, "<Piece NumberOfCells='%d' NumberOfPoints='%d'>\n", numParticles, numParticles);
        fprintf(fp, "<Points>\n");
        fprintf(fp, "<DataArray NumberOfComponents='3' type='Float64' Name='Position' format='ascii'>\n");
        for (i = 0; i < numParticles; i++) {
            fprintf(fp, "%.15lf %.15lf %.15lf\n",
                hostParticles[i].x,
                hostParticles[i].y,
                hostParticles[i].z);
            //if (i == 99655)
            //    printf("Timestep %d pID %d %.15lf %.15lf %.15lf\n",ti-1, i,hostParticles[i].x,
            //        hostParticles[i].y,
            //        hostParticles[i].z);
        }
        fprintf(fp, "</DataArray>\n");
        fprintf(fp, "</Points>\n");
        fprintf(fp, "<PointData>\n");
        fprintf(fp, "<DataArray NumberOfComponents='1' type='Int32' Name='ParticleType' format='ascii'>\n");
        for (i = 0; i < numParticles; i++) {
            fprintf(fp, "%d\n", (int)hostParticles[i].w);
        }
        fprintf(fp, "</DataArray>\n");
        fprintf(fp, "<DataArray NumberOfComponents='1' type='Int32' Name='ParticleID' format='ascii'>\n");
        for (i = 0; i < numParticles; i++) {
            fprintf(fp, "%d\n", i);
        }
        fprintf(fp, "</DataArray>\n");
        fprintf(fp, "<DataArray NumberOfComponents='1' type='Int32' Name='ParticleTetID' format='ascii'>\n");
        for (i = 0; i < numParticles; i++) {
            fprintf(fp, "%d\n", h_tetIDs[i]);
        }

        if (d_tetIDs_Convex != nullptr) {
            std::vector<int> h_ctetIDs(numParticles);
            cudaCheck(cudaMemcpy(h_ctetIDs.data(),
                d_tetIDs_Convex,
                numParticles * sizeof(int),
                cudaMemcpyDeviceToHost));

            fprintf(fp, "</DataArray>\n");
            fprintf(fp, "<DataArray NumberOfComponents='1' type='Int32' Name='ConvexTetID' format='ascii'>\n");
            for (i = 0; i < numParticles; i++) {
                fprintf(fp, "%d\n", h_ctetIDs[i]);
            }
        }

        fprintf(fp, "</DataArray>\n");
        fprintf(fp, "<DataArray NumberOfComponents='3' type='Float32' Name='vels' format='ascii'>\n");
        for (i = 0; i < numParticles; i++) {
            if (isnan(h_vels[i].x))
                fprintf(fp, "%lf %lf %lf\n", 0.0, 0.0, 0.0);
            else 
                fprintf(fp, "%lf %lf %lf\n", h_vels[i].x, h_vels[i].y, h_vels[i].z);
        }

        fprintf(fp, "</DataArray>\n");
        fprintf(fp, "<DataArray NumberOfComponents='1' type='Float32' Name='KEs' format='ascii'>\n");
        double mass = 1.0;
        double TotalKEs = 0.0;
        for (i = 0; i < numParticles; i++) {
            double KE = 0.5 * mass * (h_vels[i].x * h_vels[i].x + h_vels[i].y * h_vels[i].y + h_vels[i].z * h_vels[i].z);
            if (KE)
                fprintf(fp, "%lf\n", 0.0);
            else
                fprintf(fp, "%lf\n", KE);
            TotalKEs += KE;
            //if(isnan(KE)) printf("[Warnning] Nan particle=%d\n",i);
        }

        if (isnan(TotalKEs)) {
            printf("#adv: [Warnning] nan particle vels\n");
            system("pause");
            //exit(-1);
        }
        printf("#adv: System Kinetic Energy=%lf\n", TotalKEs);

        fprintf(fp, "</DataArray>\n");
        fprintf(fp, "</PointData>\n");
        fprintf(fp, "<Cells>\n");
        fprintf(fp, "<DataArray type='Int32' Name='connectivity' format='ascii'>\n");
        for (i = 0; i < numParticles; i++) {
            fprintf(fp, "%d\n", i);
        }
        fprintf(fp, "</DataArray>\n");
        fprintf(fp, "<DataArray type='Int32' Name='offsets' format='ascii'>\n");
        for (i = 0; i < numParticles; i++) {
            fprintf(fp, "%d\n", i + 1);
        }
        fprintf(fp, "</DataArray>\n");
        fprintf(fp, "<DataArray type='UInt8' Name='types' format='ascii'>\n");
        for (i = 0; i < numParticles; i++) {
            fprintf(fp, "1\n");
        }
        fprintf(fp, "</DataArray>\n");
        fprintf(fp, "</Cells>\n");
        fprintf(fp, "</Piece>\n");
        fprintf(fp, "</UnstructuredGrid>\n");
        fprintf(fp, "</VTKFile>\n");
        fclose(fp);
    }
}
