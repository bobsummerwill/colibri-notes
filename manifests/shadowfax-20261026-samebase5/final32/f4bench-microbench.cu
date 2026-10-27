#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
__device__ __forceinline__ float f4_dev(uint8_t q){const float a[8]={0,.5f,1,1.5f,2,3,4,6};return q&8?-a[q&7]:a[q];}
__device__ __forceinline__ float f4_new(uint8_t bits){unsigned m=bits&7;unsigned v=m?(((m/2+126)<<23)|((m>1?m&1:0)<<22)):0;return __uint_as_float(((unsigned)(bits&8)<<28)|v);}
__constant__ float e8_table[256];
__device__ __forceinline__ float e8(uint8_t b){return e8_table[b];}
/* shaped like mv_fp4_expert_major: one warp per row, each decoded weight used by TB tokens */
template<int V,int TB> __global__ void k(const uint8_t*w,const uint8_t*sc,const float*x,float*y,int O,int I){
  int lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);if(o>=O)return;
  float p[TB]={};const uint8_t*wr=w+(size_t)o*(I/2);const uint8_t*sr=sc+(size_t)o*(I/32);
  for(int i=lane;i<I;i+=32){uint8_t b=wr[i>>1],q=(i&1)?b>>4:b&15;float v=(V?f4_new(q):f4_dev(q))*e8(sr[i/32]);
    #pragma unroll
    for(int t=0;t<TB;t++)p[t]+=x[(size_t)t*I+i]*v;}
  #pragma unroll
  for(int t=0;t<TB;t++){float s=p[t];for(int m=16;m;m>>=1)s+=__shfl_xor_sync(~0u,s,m);if(!lane)y[(size_t)t*O+o]=s;}
}
template<int V,int TB> float run(const uint8_t*w,const uint8_t*sc,const float*x,float*y,int O,int I){
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);k<V,TB><<<O/8,256>>>(w,sc,x,y,O,I);cudaEventRecord(a);
  for(int r=0;r<50;r++)k<V,TB><<<O/8,256>>>(w,sc,x,y,O,I);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);return ms/50;}
int main(){int O=4096,I=4096;uint8_t*w,*sc;float*x,*y;cudaMalloc(&w,(size_t)O*I/2);cudaMalloc(&sc,(size_t)O*I/32);
  cudaMalloc(&x,64*(size_t)I*4);cudaMalloc(&y,64*(size_t)O*4);cudaMemset(w,0x5a,(size_t)O*I/2);cudaMemset(sc,127,(size_t)O*I/32);cudaMemset(x,0,64*(size_t)I*4);
  float t[256];for(int i=0;i<256;i++)t[i]=1;cudaMemcpyToSymbol(e8_table,t,sizeof t);
  printf("TB=1  dev %.3f ms  new %.3f ms\n",run<0,1>(w,sc,x,y,O,I),run<1,1>(w,sc,x,y,O,I));
  printf("TB=4  dev %.3f ms  new %.3f ms\n",run<0,4>(w,sc,x,y,O,I),run<1,4>(w,sc,x,y,O,I));
  printf("TB=8  dev %.3f ms  new %.3f ms\n",run<0,8>(w,sc,x,y,O,I),run<1,8>(w,sc,x,y,O,I));
  printf("TB=16 dev %.3f ms  new %.3f ms\n",run<0,16>(w,sc,x,y,O,I),run<1,16>(w,sc,x,y,O,I));
  printf("%s\n",cudaGetErrorString(cudaGetLastError()));}
