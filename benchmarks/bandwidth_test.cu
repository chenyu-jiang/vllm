// CUDA runtime
#include <cuda_runtime.h>

#include <cuda.h>

#include <cassert>
#include <iostream>
#include <memory>
#include <thread>
#include <atomic>

using std::atomic_bool;
using std::atomic_int;

void HtoDBandwidth(atomic_bool& startCpy, atomic_int& readyCnt, size_t memSize, int repeat, int warmup){
  cudaSetDevice(0);
  unsigned char *h_odata;
  unsigned char *d_idata;
  float totalBandwidth = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

//   h_odata = (unsigned char *) malloc(memSize);
  cudaMallocHost((void **)&h_odata, memSize);
  cudaMalloc((void **)&d_idata, memSize);

  readyCnt++;

  while (!startCpy){}
  for (int i = 0; i < repeat; i++) {
    cudaEventRecord(start);
    cudaMemcpy(d_idata, h_odata, memSize, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (i < warmup) {
      continue;
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float bw = memSize/milliseconds/1e6;
    totalBandwidth += bw;
  }

  printf("H2D avg Bandwidth GB/s %f \n", totalBandwidth / (repeat - warmup));
}

void DtoHBandwidth(atomic_bool& startCpy, atomic_int& readyCnt, size_t memSize, int repeat, int warmup) {
  cudaSetDevice(0);
  unsigned char *h_idata;
  unsigned char *d_odata;
  float totalBandwidth = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

//   h_idata = (unsigned char *) malloc(memSize);
  cudaMallocHost((void **)&h_idata, memSize);
  cudaMalloc((void **)&d_odata, memSize);
  readyCnt++;

  while (!startCpy){}
  for (int i=0; i<repeat; i++) {
    cudaEventRecord(start);
    cudaMemcpy(h_idata, d_odata, memSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    if (i < warmup) {
      continue;
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float bw = memSize/milliseconds/1e6;
    totalBandwidth += bw;
  }

  printf("D2H avg Bandwidth GB/s %f \n", totalBandwidth / (repeat - warmup));
}

int main(){
  atomic_bool startFlag{false};
  atomic_int readyCnt{0};
  size_t memSize = (size_t)4 * 1024 * 1024 * 1024;
  int repeat = 20;
  int warmup = 5;

  std::thread t1(HtoDBandwidth, std::ref(startFlag), std::ref(readyCnt), memSize, repeat, warmup);
  std::thread t2(DtoHBandwidth, std::ref(startFlag), std::ref(readyCnt),memSize, repeat, warmup);
  while(readyCnt < 2) {}
  printf("ready to start bandwdith tests\n");
  startFlag = true;
  t1.join();
  t2.join();

  return 0;
}