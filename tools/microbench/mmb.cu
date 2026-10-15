// Microbenchmark: colibri mm_batch<F> (dense batched matmul, generic path) vs
// warp-per-row variants. Checks bit-identity against the original.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__constant__ float e8_table[256];
__device__ float e4m3(uint8_t b){int sign=b>>7,e=(b>>3)&15,m=b&7;float v;
    if(!e)v=ldexpf((float)m,-9);else if(e==15)v=m==7?NAN:ldexpf(1.f+m/8.f,8);else v=ldexpf(1.f+m/8.f,e-7);
    return sign?-v:v;}
__device__ float e8(uint8_t b){return e8_table[b];}
__device__ float f4(uint8_t q){const float a[8]={0,.5f,1,1.5f,2,3,4,6};return q&8?-a[q&7]:a[q];}

// ---- original (copied verbatim from backend_cuda_dsv4.cu) ----
template<int F> __global__ void mm_batch(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int T){
    int o=blockIdx.x;if(o>=O)return;int tbase=blockIdx.y*16,tcount=T-tbase<16?T-tbase:16;
    float sum[16]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){float v;
        if constexpr(F==8||F==9){v=e4m3(w[(long long)o*I+i])*e8(sc[(o/128)*((I+127)/128)+i/128]);if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));}
        else {uint8_t p=w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;v=f4(q)*e8(sc[(long long)o*(I/32)+i/32]);}
        #pragma unroll
        for(int t=0;t<16;t++)if(t<tcount)sum[t]+=x[(long long)(tbase+t)*I+i]*v;
    }
    __shared__ float s[16][256];
    #pragma unroll
    for(int t=0;t<16;t++)s[t][threadIdx.x]=sum[t];
    __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int t=0;t<16;t++)s[t][threadIdx.x]+=s[t][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){
        #pragma unroll
        for(int t=0;t<16;t++)if(t<tcount)y[(long long)(tbase+t)*O+o]=s[t][0];}
}

// ---- candidate: one warp per output row, TB tokens per pass; lane l owns
// partials j = l + 32q; same operands & order as the 256-thread tree. ----
// LUT=1 decodes e4m3 via a shared 256-entry table (exact same floats).
template<int F,int TB,int LUT> __global__ void __launch_bounds__(256) mm_batch_w(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int T){
    __shared__ float lut[256];
    if constexpr(LUT&&(F==8||F==9)){lut[threadIdx.x]=e4m3((uint8_t)threadIdx.x);__syncthreads();}
    int lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);if(o>=O)return;
    int tbase=blockIdx.y*TB,tcount=T-tbase<TB?T-tbase:TB;
    float p[TB][8];
    #pragma unroll
    for(int t=0;t<TB;t++)
        #pragma unroll
        for(int q=0;q<8;q++)p[t][q]=0.f;
    for(int c0=0;c0<I/32;c0+=8){
        #pragma unroll
        for(int q=0;q<8;q++){int i=lane+32*(c0+q);if(i>=I)break;float v;
            if constexpr(F==8||F==9){uint8_t b=w[(long long)o*I+i];v=(LUT?lut[b]:e4m3(b))*e8(sc[(o/128)*((I+127)/128)+i/128]);if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));}
            else {uint8_t pk=w[((long long)o*I+i)>>1],qq=(i&1)?pk>>4:pk&15;v=f4(qq)*e8(sc[(long long)o*(I/32)+i/32]);}
            #pragma unroll
            for(int t=0;t<TB;t++)if(t<tcount)p[t][q]+=x[(long long)(tbase+t)*I+i]*v;
        }
    }
    #pragma unroll
    for(int t=0;t<TB;t++){
        float v0=p[t][0]+p[t][4],v1=p[t][1]+p[t][5],v2=p[t][2]+p[t][6],v3=p[t][3]+p[t][7];
        v0=v0+v2;v1=v1+v3;float v=v0+v1;
        #pragma unroll
        for(int m=16;m;m>>=1)v+=__shfl_down_sync(0xffffffffu,v,m);
        if(!lane&&t<tcount)y[(long long)(tbase+t)*O+o]=v;
    }
}

int main(){
    std::mt19937 rng(7);
    float tab[256];for(int b=0;b<256;b++)tab[b]=ldexpf(1.f,(b%16)-12);
    CK(cudaMemcpyToSymbol(e8_table,tab,sizeof(tab)));
    struct Shape{int F,O,I;const char*name;};
    Shape shapes[]={{8,1024,4096,"fp8 1024x4096"},{8,4096,4096,"fp8 4096x4096"},{8,2048,4096,"fp8 2048x4096 (shared gate/up)"},
                    {8,4096,2048,"fp8 4096x2048 (shared down)"},{9,1024,4096,"bf16r 1024x4096"},{8,4096,8192,"fp8 4096x8192"}};
    const int T=128;
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
    for(auto sh:shapes){
        int O=sh.O,I=sh.I;size_t wb=(size_t)O*I,sb=(size_t)((O+127)/128)*((I+127)/128);
        std::vector<uint8_t> hw(wb),hs(sb);for(auto&v:hw){v=rng();if(((v>>3)&15)==15&&(v&7)==7)v^=1;}for(auto&v:hs)v=rng();
        std::vector<float> hx((size_t)T*I);std::normal_distribution<float> nd(0,1);for(auto&v:hx)v=nd(rng);
        uint8_t*dw,*ds;float*dx,*y0,*y1;CK(cudaMalloc(&dw,wb));CK(cudaMalloc(&ds,sb));CK(cudaMalloc(&dx,hx.size()*4));
        CK(cudaMalloc(&y0,(size_t)T*O*4));CK(cudaMalloc(&y1,(size_t)T*O*4));
        CK(cudaMemcpy(dw,hw.data(),wb,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,hs.data(),sb,cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
        auto timeit=[&](auto fn){fn();CK(cudaDeviceSynchronize());cudaEventRecord(a);for(int i=0;i<10;i++)fn();cudaEventRecord(b);CK(cudaEventSynchronize(b));float ms;cudaEventElapsedTime(&ms,a,b);return ms/10;};
        std::vector<float> r((size_t)T*O),c(r.size());
        float tref;
        if(sh.F==8)tref=timeit([&]{mm_batch<8><<<dim3(O,(T+15)/16),256>>>(dw,ds,dx,y0,O,I,T);});
        else tref=timeit([&]{mm_batch<9><<<dim3(O,(T+15)/16),256>>>(dw,ds,dx,y0,O,I,T);});
        CK(cudaMemcpy(r.data(),y0,r.size()*4,cudaMemcpyDeviceToHost));
        printf("%-32s ref %.3f ms\n",sh.name,tref);
        auto cand=[&](const char*nm,auto fn){CK(cudaMemset(y1,0xff,(size_t)T*O*4));float t=timeit(fn);CK(cudaMemcpy(c.data(),y1,c.size()*4,cudaMemcpyDeviceToHost));
            long bad=0;for(size_t i=0;i<r.size();i++)bad+=memcmp(&r[i],&c[i],4)!=0;printf("   %-14s %.3f ms  x%.2f  mismatches %ld\n",nm,t,tref/t,bad);};
#define C(TB,LUT) if(sh.F==8)cand("w TB" #TB " lut" #LUT,[&]{mm_batch_w<8,TB,LUT><<<dim3((O+7)/8,(T+TB-1)/TB),256>>>(dw,ds,dx,y1,O,I,T);}); \
                  else cand("w TB" #TB " lut" #LUT,[&]{mm_batch_w<9,TB,LUT><<<dim3((O+7)/8,(T+TB-1)/TB),256>>>(dw,ds,dx,y1,O,I,T);});
        C(4,0) C(4,1) C(8,1) C(16,1) C(2,1)
        cudaFree(dw);cudaFree(ds);cudaFree(dx);cudaFree(y0);cudaFree(y1);
    }
    return 0;
}
