import os
os.chdir("/tmp/cw/pr2")
p = "c/backend_cuda_dsv4.cu"
s = open(p).read()
a = s.index("<<<<<<< HEAD\n")
m = s.index("=======\n", a)
b = s.index(">>>>>>> ", m)
e = s.index("\n", b) + 1
ours = s[a + len("<<<<<<< HEAD\n"):m]
old = 'cudaError_t e=cudaDeviceEnablePeerAccess(dev[j],0);if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled)return ok(e,"enable peer access");}'
assert ours.count(old) == 1, ours.count(old)
ours = ours.replace(old, 'cudaError_t e=cudaDeviceEnablePeerAccess(dev[j],0);if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled)return ok(e,"enable peer access");if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();if(dev[i]>=0&&dev[i]<16&&dev[j]>=0&&dev[j]<16)g_peer[dev[i]][dev[j]]=1;}')
anchor = 'extern "C" int dsv4_cuda_init(const int *dev,int n){'
assert ours.count(anchor) == 1
ours = ours.replace(anchor, 'extern "C" int dsv4_cuda_peer_access(int a,int b){return a!=b&&dev_ok(a,b)&&dev_ok(b,a);}\n' + anchor, 1)
s = s[:a] + ours + s[e:]
open(p, "w").write(s)
print("resolved")
