// Standalone microbenchmark: per-route mv_fp4_grouped vs expert-major variants.
// Kernels copied from colibri c/backend_cuda_dsv4.cu (f4, e8, mv_fp4_grouped) plus candidates.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__constant__ float e8_table[256];
__device__ float e8(uint8_t b){return e8_table[b];}
__device__ float f4(uint8_t q){const float a[8]={0,.5f,1,1.5f,2,3,4,6};return q&8?-a[q&7]:a[q];}
struct MvDesc { const uint8_t *w,*scale; const float *x; float *y; };

template<int R> __global__ void mv_fp4_grouped(const MvDesc *desc,int count,int O,int I){
    int matrix=blockIdx.y,base=blockIdx.x*R;if(matrix>=count)return;MvDesc d=desc[matrix];float sum[R]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){
        #pragma unroll
        for(int r=0;r<R;r++){int o=base+r;if(o<O){uint8_t p=d.w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;
            sum[r]+=d.x[i]*f4(q)*e8(d.scale[(long long)o*(I/32)+i/32]);}}
    }
    __shared__ float s[R][256];
    #pragma unroll
    for(int r=0;r<R;r++)s[r][threadIdx.x]=sum[r];__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int r=0;r<R;r++)s[r][threadIdx.x]+=s[r][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){
        #pragma unroll
        for(int r=0;r<R;r++)if(base+r<O)d.y[base+r]=s[r][0];}
}

// Candidate: R rows x TB routes per block, reduction in passes of 32 (smem 32 KB).
template<int R,int TB> __global__ void __launch_bounds__(256) mv_fp4_em(const uint8_t*bank_w,const uint8_t*bank_s,size_t wstride,size_t sstride,
        const int*experts,const int*offsets,const int*routes,const float*x,long long xstride,int xdiv,
        float*y0,float*y1,int split,long long ystride,int O,int I){
    int slot=blockIdx.y,base=blockIdx.x*R,e=experts[slot],begin=offsets[slot],end=offsets[slot+1];
    const uint8_t*w=bank_w+(size_t)e*wstride,*sc=bank_s+(size_t)e*sstride;
    constexpr int K=R*TB,P=K<32?K:32;
    __shared__ float s[P][256];
    for(int g=begin;g<end;g+=TB){
        int n=end-g<TB?end-g:TB;float sum[R][TB];const float*xs[TB];
        #pragma unroll
        for(int t=0;t<TB;t++){xs[t]=t<n?x+(long long)(routes[g+t]/xdiv)*xstride:x;
            #pragma unroll
            for(int r=0;r<R;r++)sum[r][t]=0.f;}
        for(int i=threadIdx.x;i<I;i+=blockDim.x){
            float xv[TB];
            #pragma unroll
            for(int t=0;t<TB;t++)xv[t]=t<n?xs[t][i]:0.f;
            #pragma unroll
            for(int r=0;r<R;r++){int o=base+r;if(o<O){uint8_t pk=w[((long long)o*I+i)>>1],q=(i&1)?pk>>4:pk&15;float f=f4(q),es=e8(sc[(long long)o*(I/32)+i/32]);
                #pragma unroll
                for(int t=0;t<TB;t++)if(t<n)sum[r][t]+=xv[t]*f*es;}}
        }
        #pragma unroll
        for(int k0=0;k0<K;k0+=P){
            #pragma unroll
            for(int kk=0;kk<P;kk++)s[kk][threadIdx.x]=sum[(k0+kk)/TB][(k0+kk)%TB];
            __syncthreads();
            for(int m=128;m;m>>=1){if(threadIdx.x<m){
                #pragma unroll
                for(int kk=0;kk<P;kk++)s[kk][threadIdx.x]+=s[kk][threadIdx.x+m];}__syncthreads();}
            if(!threadIdx.x){
                #pragma unroll
                for(int kk=0;kk<P;kk++){int r=(k0+kk)/TB,t=(k0+kk)%TB,o=base+r;if(o<O&&t<n){long long route=routes[g+t];
                    if(o<split)y0[route*ystride+o]=s[kk][0];else y1[route*ystride+(o-split)]=s[kk][0];}}}
            __syncthreads();
        }
    }
}


// Candidate 2: same accumulation; reduction spreads the 128/64/32 steps of
// every slot over all threads, then each warp finishes its slots with
// shuffles. Each surviving lane adds the same operands in the same order as
// the shared-memory tree, so results are bit-identical.
template<int R,int TB> __global__ void __launch_bounds__(256) mv_fp4_em2(const uint8_t*bank_w,const uint8_t*bank_s,size_t wstride,size_t sstride,
        const int*experts,const int*offsets,const int*routes,const float*x,long long xstride,int xdiv,
        float*y0,float*y1,int split,long long ystride,int O,int I){
    int slot=blockIdx.y,base=blockIdx.x*R,e=experts[slot],begin=offsets[slot],end=offsets[slot+1];
    const uint8_t*w=bank_w+(size_t)e*wstride,*sc=bank_s+(size_t)e*sstride;
    constexpr int K=R*TB,P=K<32?K:32;
    __shared__ float s[P][257];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int g=begin;g<end;g+=TB){
        int n=end-g<TB?end-g:TB;float sum[R][TB];const float*xs[TB];
        #pragma unroll
        for(int t=0;t<TB;t++){xs[t]=t<n?x+(long long)(routes[g+t]/xdiv)*xstride:x;
            #pragma unroll
            for(int r=0;r<R;r++)sum[r][t]=0.f;}
        for(int i=threadIdx.x;i<I;i+=blockDim.x){
            float xv[TB];
            #pragma unroll
            for(int t=0;t<TB;t++)xv[t]=t<n?xs[t][i]:0.f;
            #pragma unroll
            for(int r=0;r<R;r++){int o=base+r;if(o<O){uint8_t pk=w[((long long)o*I+i)>>1],q=(i&1)?pk>>4:pk&15;float f=f4(q),es=e8(sc[(long long)o*(I/32)+i/32]);
                #pragma unroll
                for(int t=0;t<TB;t++)if(t<n)sum[r][t]+=xv[t]*f*es;}}
        }
        #pragma unroll
        for(int k0=0;k0<K;k0+=P){
            #pragma unroll
            for(int kk=0;kk<P;kk++)s[kk][threadIdx.x]=sum[(k0+kk)/TB][(k0+kk)%TB];
            __syncthreads();
            #pragma unroll
            for(int m=128;m>=32;m>>=1){
                for(int idx=threadIdx.x;idx<P*m;idx+=256){int kk=idx/m,j=idx%m;s[kk][j]+=s[kk][j+m];}
                __syncthreads();}
            for(int kk=warp;kk<P;kk+=8){
                float v=s[kk][lane];
                #pragma unroll
                for(int m=16;m;m>>=1)v+=__shfl_down_sync(0xffffffffu,v,m);
                if(!lane){int r=(k0+kk)/TB,t=(k0+kk)%TB,o=base+r;if(o<O&&t<n){long long route=routes[g+t];
                    if(o<split)y0[route*ystride+o]=v;else y1[route*ystride+(o-split)]=v;}}}
            __syncthreads();
        }
    }
}

// Candidate 3: one WARP per (R rows x TB routes). Lane l owns the 256-wide
// tree's partials j = l + 32q (q = 0..7): partial j sums i = j, j+256, ... in
// order exactly like thread j of mv_fp4_grouped. The tree's 128/64/32 steps
// pair partials inside a lane (register adds), 16..1 are shuffles: same
// operands, same order, bit-identical; no shared memory, no barriers.
template<int R,int TB> __global__ void __launch_bounds__(256) mv_fp4_em3(const uint8_t*bank_w,const uint8_t*bank_s,size_t wstride,size_t sstride,
        const int*experts,const int*offsets,const int*routes,const float*x,long long xstride,int xdiv,
        float*y0,float*y1,int split,long long ystride,int O,int I){
    int slot=blockIdx.y,lane=threadIdx.x&31,warp=threadIdx.x>>5,base=(blockIdx.x*8+warp)*R;
    int e=experts[slot],begin=offsets[slot],end=offsets[slot+1];
    if(base>=O)return;
    const uint8_t*w=bank_w+(size_t)e*wstride,*sc=bank_s+(size_t)e*sstride;
    for(int g=begin;g<end;g+=TB){
        int n=end-g<TB?end-g:TB;const float*xs[TB];
        #pragma unroll
        for(int t=0;t<TB;t++)xs[t]=t<n?x+(long long)(routes[g+t]/xdiv)*xstride:x;
        float p[R][TB][8];
        #pragma unroll
        for(int r=0;r<R;r++)
            #pragma unroll
            for(int t=0;t<TB;t++)
                #pragma unroll
                for(int q=0;q<8;q++)p[r][t][q]=0.f;
        for(int c0=0;c0<I/32;c0+=8){
            #pragma unroll
            for(int q=0;q<8;q++){int i=lane+32*(c0+q);
                float xv[TB];
                #pragma unroll
                for(int t=0;t<TB;t++)xv[t]=t<n?xs[t][i]:0.f;
                #pragma unroll
                for(int r=0;r<R;r++){int o=base+r;if(o<O){uint8_t pk=w[((long long)o*I+i)>>1],qq=(i&1)?pk>>4:pk&15;float f=f4(qq),es=e8(sc[(long long)o*(I/32)+i/32]);
                    #pragma unroll
                    for(int t=0;t<TB;t++)if(t<n)p[r][t][q]+=xv[t]*f*es;}}
            }
        }
        #pragma unroll
        for(int r=0;r<R;r++){int o=base+r;
            #pragma unroll
            for(int t=0;t<TB;t++){
                float v0=p[r][t][0]+p[r][t][4],v1=p[r][t][1]+p[r][t][5],v2=p[r][t][2]+p[r][t][6],v3=p[r][t][3]+p[r][t][7];
                v0=v0+v2;v1=v1+v3;float v=v0+v1;
                #pragma unroll
                for(int m=16;m;m>>=1)v+=__shfl_down_sync(0xffffffffu,v,m);
                if(!lane&&o<O&&t<n){long long route=routes[g+t];
                    if(o<split)y0[route*ystride+o]=v;else y1[route*ystride+(o-split)]=v;}}}
    }
}

// Candidate 4: v3 + the block's TB token inputs staged in shared memory, 256
// columns at a time (all 8 warps = 8 rows of the same expert share them).
// WPB warps per block. Arithmetic per output unchanged (bit-identical).
template<int TB,int WPB> __global__ void __launch_bounds__(WPB*32) mv_fp4_em4(const uint8_t*bank_w,const uint8_t*bank_s,size_t wstride,size_t sstride,
        const int*experts,const int*offsets,const int*routes,const float*x,long long xstride,int xdiv,
        float*y0,float*y1,int split,long long ystride,int O,int I){
    __shared__ float xt[TB][256];
    int slot=blockIdx.y,lane=threadIdx.x&31,o=blockIdx.x*WPB+(threadIdx.x>>5);
    int e=experts[slot],begin=offsets[slot],end=offsets[slot+1];
    bool live=o<O;
    const uint8_t*w=bank_w+(size_t)e*wstride+(size_t)(live?o:0)*(I/2);const uint8_t*sc=bank_s+(size_t)e*sstride+(size_t)(live?o:0)*(I/32);
    for(int g=begin;g<end;g+=TB){
        int n=end-g<TB?end-g:TB;const float*xs[TB];float p[TB][8];
        #pragma unroll
        for(int t=0;t<TB;t++){xs[t]=t<n?x+(long long)(routes[g+t]/xdiv)*xstride:x;
            #pragma unroll
            for(int q=0;q<8;q++)p[t][q]=0.f;}
        for(int c0=0;c0<I/32;c0+=8){
            __syncthreads();
            for(int k=threadIdx.x;k<TB*256;k+=WPB*32){int t=k>>8,col=k&255;xt[t][col]=t<n?xs[t][c0*32+col]:0.f;}
            __syncthreads();
            if(live){
            #pragma unroll
            for(int q=0;q<8;q++){int i=lane+32*(c0+q);
                uint8_t pk=w[i>>1],qq=(i&1)?pk>>4:pk&15;float f=f4(qq),es=e8(sc[i/32]);
                #pragma unroll
                for(int t=0;t<TB;t++)if(t<n)p[t][q]+=xt[t][lane+32*q]*f*es;
            }}
        }
        if(live){
        #pragma unroll
        for(int t=0;t<TB;t++){
            float v0=p[t][0]+p[t][4],v1=p[t][1]+p[t][5],v2=p[t][2]+p[t][6],v3=p[t][3]+p[t][7];
            v0=v0+v2;v1=v1+v3;float v=v0+v1;
            #pragma unroll
            for(int m=16;m;m>>=1)v+=__shfl_down_sync(0xffffffffu,v,m);
            if(!lane&&t<n){long long route=routes[g+t];
                if(o<split)y0[route*ystride+o]=v;else y1[route*ystride+(o-split)]=v;}}}
    }
}

int main(int argc,char**argv){
    const int H=4096,I=2048,K=6,T=128,E=132,routes=T*K;
    size_t w1=(size_t)I*(H/2),s1=(size_t)I*(H/32),w2=(size_t)H*(I/2),s2=(size_t)H*(I/32);
    std::mt19937 rng(42);
    float tab[256];for(int b=0;b<256;b++)tab[b]=ldexpf(1.f,(b%16)-10);  // modest scales
    CK(cudaMemcpyToSymbol(e8_table,tab,sizeof(tab)));
    std::vector<uint8_t> hw(E*2*w1),hs(E*2*s1),hw2(E*w2),hs2(E*s2);
    for(auto&v:hw)v=rng();for(auto&v:hs)v=rng();for(auto&v:hw2)v=rng();for(auto&v:hs2)v=rng();
    std::vector<float> hx((size_t)T*H);std::normal_distribution<float> nd(0,1);for(auto&v:hx)v=nd(rng);
    // routes: skewed expert popularity, 6 distinct experts per token
    std::vector<double> pop(E);for(int e=0;e<E;e++)pop[e]=1.0/(1+e*0.05);
    std::discrete_distribution<int> dd(pop.begin(),pop.end());
    std::vector<int> ids(routes);std::vector<int> perm(E);for(int e=0;e<E;e++)perm[e]=e;std::shuffle(perm.begin(),perm.end(),rng);
    for(int t=0;t<T;t++){int k=0;while(k<K){int e=perm[dd(rng)];bool dup=false;for(int j=0;j<k;j++)dup|=ids[t*K+j]==e;if(!dup)ids[t*K+k++]=e;}}
    uint8_t*dw,*ds,*dw2,*ds2;float*dx,*p1,*p2,*q1,*q2,*a3,*p4,*q4;
    CK(cudaMalloc(&dw,hw.size()));CK(cudaMalloc(&ds,hs.size()));CK(cudaMalloc(&dw2,hw2.size()));CK(cudaMalloc(&ds2,hs2.size()));
    CK(cudaMemcpy(dw,hw.data(),hw.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,hs.data(),hs.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw2,hw2.data(),hw2.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(ds2,hs2.data(),hs2.size(),cudaMemcpyHostToDevice));
    size_t rb=(size_t)routes*I*4,pb=(size_t)routes*H*4;
    CK(cudaMalloc(&dx,hx.size()*4));CK(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMalloc(&p1,rb));CK(cudaMalloc(&p2,rb));CK(cudaMalloc(&q1,rb));CK(cudaMalloc(&q2,rb));CK(cudaMalloc(&a3,rb));CK(cudaMalloc(&p4,pb));CK(cudaMalloc(&q4,pb));
    {std::vector<float> ha((size_t)routes*I);for(auto&v:ha)v=nd(rng)*0.1f;CK(cudaMemcpy(a3,ha.data(),rb,cudaMemcpyHostToDevice));}
    // reference descriptors
    std::vector<MvDesc> hd1(routes*2),hd2(routes);
    for(int r=0;r<routes;r++){int e=ids[r],t=r/K;const uint8_t*fc1=dw+(size_t)e*2*w1,*fc1s=ds+(size_t)e*2*s1;
        hd1[2*r]={fc1,fc1s,dx+(long long)t*H,p1+(long long)r*I};hd1[2*r+1]={fc1+w1,fc1s+s1,dx+(long long)t*H,p2+(long long)r*I};
        hd2[r]={dw2+(size_t)e*w2,ds2+(size_t)e*s2,a3+(long long)r*I,p4+(long long)r*H};}
    MvDesc*d1,*d2;CK(cudaMalloc(&d1,hd1.size()*sizeof(MvDesc)));CK(cudaMalloc(&d2,hd2.size()*sizeof(MvDesc)));
    CK(cudaMemcpy(d1,hd1.data(),hd1.size()*sizeof(MvDesc),cudaMemcpyHostToDevice));CK(cudaMemcpy(d2,hd2.data(),hd2.size()*sizeof(MvDesc),cudaMemcpyHostToDevice));
    // expert-major lists
    int cnt[256]={0},slot_of[256],nex=0;for(int r=0;r<routes;r++)cnt[ids[r]]++;
    std::vector<int> hex,hoff{0};for(int e=0;e<E;e++)if(cnt[e]){slot_of[e]=nex++;hex.push_back(e);hoff.push_back(hoff.back()+cnt[e]);}
    std::vector<int> hrt(routes),fill(hoff.begin(),hoff.end()-1);for(int r=0;r<routes;r++)hrt[fill[slot_of[ids[r]]]++]=r;
    int maxc=0;for(int e=0;e<E;e++)maxc=std::max(maxc,cnt[e]);
    printf("experts used %d of %d, routes %d, max routes/expert %d\n",nex,E,routes,maxc);
    int*dex,*doff,*drt;CK(cudaMalloc(&dex,nex*4));CK(cudaMalloc(&doff,(nex+1)*4));CK(cudaMalloc(&drt,routes*4));
    CK(cudaMemcpy(dex,hex.data(),nex*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(doff,hoff.data(),(nex+1)*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(drt,hrt.data(),routes*4,cudaMemcpyHostToDevice));
    cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);
    auto timeit=[&](auto fn,int reps){fn();CK(cudaDeviceSynchronize());cudaEventRecord(a);for(int i=0;i<reps;i++)fn();cudaEventRecord(b);CK(cudaEventSynchronize(b));float ms;cudaEventElapsedTime(&ms,a,b);return ms/reps;};
    auto ref1=[&]{mv_fp4_grouped<4><<<dim3((I+3)/4,routes*2),256>>>(d1,routes*2,I,H);};
    auto ref2=[&]{mv_fp4_grouped<4><<<dim3((H+3)/4,routes),256>>>(d2,routes,H,I);};
    printf("reference  fc1 %.2f ms  fc2 %.2f ms\n",timeit(ref1,5),timeit(ref2,5));
    std::vector<float> r1((size_t)routes*I),r2((size_t)routes*I),r4((size_t)routes*H),c1(r1.size()),c2(r2.size()),c4(r4.size());
    CK(cudaMemcpy(r1.data(),p1,rb,cudaMemcpyDeviceToHost));CK(cudaMemcpy(r2.data(),p2,rb,cudaMemcpyDeviceToHost));CK(cudaMemcpy(r4.data(),p4,pb,cudaMemcpyDeviceToHost));
    auto check=[&](const char*name,auto f1,auto f2){
        CK(cudaMemset(q1,0xff,rb));CK(cudaMemset(q2,0xff,rb));CK(cudaMemset(q4,0xff,pb));
        float t1=timeit(f1,5),t2=timeit(f2,5);
        CK(cudaMemcpy(c1.data(),q1,rb,cudaMemcpyDeviceToHost));CK(cudaMemcpy(c2.data(),q2,rb,cudaMemcpyDeviceToHost));CK(cudaMemcpy(c4.data(),q4,pb,cudaMemcpyDeviceToHost));
        long bad=0;for(size_t i=0;i<r1.size();i++){bad+=memcmp(&r1[i],&c1[i],4)!=0;bad+=memcmp(&r2[i],&c2[i],4)!=0;}for(size_t i=0;i<r4.size();i++)bad+=memcmp(&r4[i],&c4[i],4)!=0;
        printf("%-10s fc1 %.2f ms  fc2 %.2f ms  total %.2f ms  bit-mismatches %ld\n",name,t1,t2,t1+t2,bad);};
#define CAND(R,TB) check(#R "x" #TB, \
        [&]{mv_fp4_em<R,TB><<<dim3((2*I+R-1)/R,nex),256>>>(dw,ds,2*w1,2*s1,dex,doff,drt,dx,H,K,q1,q2,I,I,2*I,H);}, \
        [&]{mv_fp4_em<R,TB><<<dim3((H+R-1)/R,nex),256>>>(dw2,ds2,w2,s2,dex,doff,drt,a3,I,1,q4,q4,H,H,H,I);})
#define CAND2(R,TB) check("v2 " #R "x" #TB, \
        [&]{mv_fp4_em2<R,TB><<<dim3((2*I+R-1)/R,nex),256>>>(dw,ds,2*w1,2*s1,dex,doff,drt,dx,H,K,q1,q2,I,I,2*I,H);}, \
        [&]{mv_fp4_em2<R,TB><<<dim3((H+R-1)/R,nex),256>>>(dw2,ds2,w2,s2,dex,doff,drt,a3,I,1,q4,q4,H,H,H,I);})
#define CAND3(R,TB) check("v3 " #R "x" #TB, \
        [&]{mv_fp4_em3<R,TB><<<dim3((2*I+8*R-1)/(8*R),nex),256>>>(dw,ds,2*w1,2*s1,dex,doff,drt,dx,H,K,q1,q2,I,I,2*I,H);}, \
        [&]{mv_fp4_em3<R,TB><<<dim3((H+8*R-1)/(8*R),nex),256>>>(dw2,ds2,w2,s2,dex,doff,drt,a3,I,1,q4,q4,H,H,H,I);})
    CAND3(1,4);
#define CAND4(TB,WPB) check("v4 TB" #TB " w" #WPB, \
        [&]{mv_fp4_em4<TB,WPB><<<dim3((2*I+WPB-1)/WPB,nex),WPB*32>>>(dw,ds,2*w1,2*s1,dex,doff,drt,dx,H,K,q1,q2,I,I,2*I,H);}, \
        [&]{mv_fp4_em4<TB,WPB><<<dim3((H+WPB-1)/WPB,nex),WPB*32>>>(dw2,ds2,w2,s2,dex,doff,drt,a3,I,1,q4,q4,H,H,H,I);})
    CAND4(4,8);CAND4(4,16);CAND4(4,32);CAND4(8,16);CAND4(8,32);CAND4(6,16);
    return 0;
}
