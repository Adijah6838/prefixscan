// scan.cu - Exclusive Prefix Scan, Blelloch, 3-phase multi-block
// Compile: nvcc -O3 -arch=sm_80 scan.cu -o scan
// Run: ./scan [N]   e.g. ./scan 10000000
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define BLOCK 256
#define ELEM (BLOCK*2)
#define CHECK(x) do{auto e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__); return 1;}}while(0)
void cpu_scan(const float* x, float* y, int N){
  float s=0; for(int i=0;i<N;i++){ y[i]=s; s+=x[i]; }
}
// Blelloch block scan: ELEM per block, in-place in shared
// Up-sweep: tree reduce, Down-sweep: distribute prefix
__device__ void blelloch(float* s, int tid){
  for(int off=1; off<ELEM; off<<=1){
    int idx=(tid+1)*off*2-1;
    if(idx<ELEM) s[idx]+=s[idx-off];
    __syncthreads();
  }
  if(tid==0) s[ELEM-1]=0;
  __syncthreads();
  for(int off=ELEM/2; off>0; off>>=1){
    int idx=(tid+1)*off*2-1;
    if(idx<ELEM){ float t=s[idx-off]; s[idx-off]=s[idx]; s[idx]+=t; }
    __syncthreads();
  }
}
__global__ void scan_phaseA(const float* x, float* y, float* blockSums, int N){
  __shared__ float s[ELEM];
  int tid=threadIdx.x; int base=blockIdx.x*ELEM;
  int i1=base+tid, i2=base+tid+BLOCK;
  s[tid]=(i1<N)?x:0; s[tid+BLOCK]=(i2<N)?x:0; __syncthreads();
  float blockTotal=s[ELEM-1]+s[ELEM-2+1];
  blelloch(s,tid);
  if(tid==0) blockSums[blockIdx.x]=0;
  if(i1<N) y=s[tid];
  if(i2<N) y=s[tid+BLOCK];
  if(tid==0){
    float tot=0; for(int k=0;k<ELEM;k++){ int gi=base+k; if(gi<N) tot+=x[gi]; }
    blockSums[blockIdx.x]=tot;
  }
}
__global__ void scan_phaseB(float* blockSums, int numBlocks){
  __shared__ float s[ELEM];
  int tid=threadIdx.x;
  s[tid]=(tid<numBlocks)?blockSums[tid]:0; s[tid+BLOCK]=(tid+BLOCK<numBlocks)?blockSums[tid+BLOCK]:0; __syncthreads();
  blelloch(s,tid);
  if(tid<numBlocks) blockSums[tid]=s[tid];
  if(tid+BLOCK<numBlocks) blockSums[tid+BLOCK]=s[tid+BLOCK];
}
__global__ void scan_phaseC(float* y, const float* blockSums, int N){
  int base=blockIdx.x*ELEM;
  float add=blockSums[blockIdx.x];
  int tid=threadIdx.x;
  int i1=base+tid, i2=base+tid+BLOCK;
  if(i1<N) y+=add;
  if(i2<N) y+=add;
}
int main(int argc,char** argv){
  int N=argc>1?atoi(argv[1]):10000000;
  std::vector<float> x(N), y_cpu(N), y_gpu(N);
  srand(42); for(int i=0;i<N;i++) x[i]=(float)(rand()%100)/10.0f;
  cpu_scan(x.data(), y_cpu.data(), N);
  int numBlocks=(N+ELEM-1)/ELEM;
  float *dx,*dy,*db; CHECK(cudaMalloc(&dx,N*sizeof(float))); CHECK(cudaMalloc(&dy,N*sizeof(float))); CHECK(cudaMalloc(&db,numBlocks*sizeof(float)));
  CHECK(cudaMemcpy(dx,x.data(),N*sizeof(float),cudaMemcpyHostToDevice));
  scan_phaseA<<<numBlocks,BLOCK>>>(dx,dy,db,N); CHECK(cudaGetLastError());
  scan_phaseB<<<1,BLOCK>>>(db,numBlocks); CHECK(cudaGetLastError());
  scan_phaseC<<<numBlocks,BLOCK>>>(dy,db,N); CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(y_gpu.data(),dy,N*sizeof(float),cudaMemcpyDeviceToHost));
  float maxErr=0; for(int i=0;i<N;i++) maxErr=fmaxf(maxErr,fabsf(y_cpu[i]-y_gpu[i]));
  printf("N=%d blocks=%d y[0]=%.2f y[1]=%.2f y[N-1]=%.2f maxErr=%.2e %s\n",N,numBlocks,y_gpu[0],y_gpu[1],y_gpu[N-1],maxErr,maxErr<1e-2?"PASS":"FAIL");
  CHECK(cudaFree(dx)); CHECK(cudaFree(dy)); CHECK(cudaFree(db)); CHECK(cudaDeviceReset());
  return maxErr<1e-2?0:1;
}
