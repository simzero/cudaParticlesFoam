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

#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#ifdef WIN32   // Windows system specific
#include <windows.h>
#else          // Unix based system specific
#include <sys/time.h>
#endif


#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}



class cudaTimer
{
private:
  cudaEvent_t t_start;
  cudaEvent_t t_stop;
  cudaStream_t stream;
#define cudaSafeCall(call)                                              \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (cudaSuccess != err)                                             \
      {                                                                 \
        std::cerr << "CUDA error in " << __FILE__ << "(" << __LINE__ << "): " \
                  << cudaGetErrorString(err);                           \
        exit(EXIT_FAILURE);                                             \
      }                                                                 \
  } while(0)
public:
  cudaTimer()
  {
    cudaSafeCall(cudaEventCreate(&t_start));
    cudaSafeCall(cudaEventCreate(&t_stop));
  }

  ~cudaTimer()
  {
    cudaSafeCall(cudaEventDestroy(t_start));
    cudaSafeCall(cudaEventDestroy(t_stop));
  }

  void start(cudaStream_t st = 0)
  {
    stream = st;
    cudaSafeCall(cudaEventRecord(t_start, stream));
  }

  float stop()
  {
    float milliseconds = 0;
    cudaSafeCall(cudaEventRecord(t_stop, stream));
    cudaSafeCall(cudaEventSynchronize(t_stop));
    cudaSafeCall(cudaEventElapsedTime(&milliseconds, t_start, t_stop));
    return milliseconds;
  }
};

/*class CPUTimer
{

// typedef unsigned long long LARGE_INTEGER;


private:
    double startTimeInMicroSec;                 // starting time in micro-second
    double endTimeInMicroSec;                   // ending time in micro-second
    int    stopped;                             // stop flag 
#ifdef WIN32
    LARGE_INTEGER tFreq, tStart, tEnd;                    // ticks per second
#else
    timeval tStart, tEnd;                         //
#endif


private:
#if defined(_WIN32) || defined(_WIN64)
	LARGE_INTEGER tFreq, tStart, tEnd;
#endif

public:

#if defined(_WIN32) || defined(_WIN64)
	CPUTimer(void)
	{
		QueryPerformanceFrequency(&tFreq);
		return;
	}

	void start(void)
	{
		QueryPerformanceCounter(&tStart);
	}

	double stop(void)
	{
		QueryPerformanceCounter(&tEnd);
		return this->TimeInSeconds() * 1000.0;
	}

	long TimeInTicks(void)
	{
		return((long)(tEnd.QuadPart - tStart.QuadPart));
	}

	double TimeInSeconds(void)
	{
		return ((double)(tEnd.QuadPart - tStart.QuadPart) / (tFreq.QuadPart));
	}
#else

    rusage start;
    rusage stop;

    void start()
    {
        getrusage(RUSAGE_SELF, &start);
    }

    void stop()
    {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis()
    {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec / 1000);
    }

#endif
};*/
