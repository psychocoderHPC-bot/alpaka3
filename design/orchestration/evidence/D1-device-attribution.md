# D1 device attribution (fwk394)

Proved by running a tiny standalone CUDA binary from the same process as the
timing loop. Both cards are driven independently with `CUDA_VISIBLE_DEVICES`.

## Standalone probe (compiled with nvcc 12.5.82, driver 555.52.04)

Compiled `/tmp/probe.cpp`:
```cpp
#include <iostream>
#include <cuda_runtime_api.h>
int main() {
  int n; cudaGetDeviceCount(&n);
  for(int j=0;j<n;j++){
    cudaDeviceProp p; cudaGetDeviceProperties(&p,j);
    std::cout << "DEVICE " << j << " " << p.name
      << " SMs=" << p.multiProcessorCount << " cc=" << p.major << "." << p.minor
      << " maxSM=" << p.clockRate/1000 << "MHz"
      << " sharedMemBlk=" << p.sharedMemPerBlock
      << " maxThrBlk=" << p.maxThreadsPerBlock << std::endl;
  }
}
```

### From a process that sees both GPUs (no CUDA_VISIBLE_DEVICES):
```
DEVICE 0 Quadro GV100 SMs=80 cc=7.0 maxSM=1627MHz sharedMemBlk=49152 maxThrBlk=1024
DEVICE 1 NVIDIA GeForce RTX 2080 Ti SMs=68 cc=7.5 maxSM=1545MHz sharedMemBlk=49152 maxThrBlk=1024
```

### With `CUDA_VISIBLE_DEVICES=0` (runtime device 0 is ...):
```
DEVICE 0 Quadro GV100 SMs=80 cc=7.0 maxSM=1627MHz sharedMemBlk=49152 maxThrBlk=1024
```

### With `CUDA_VISIBLE_DEVICES=1`:
```
DEVICE 0 NVIDIA GeForce RTX 2080 Ti SMs=68 cc=7.5 maxSM=1545MHz sharedMemBlk=49152 maxThrBlk=1024
```

### `nvidia-smi` reference (same host, same instant, for cross-check):
```
0, NVIDIA GeForce RTX 2080 Ti
1, Quadro GV100
```

## Conclusion

The mapping on fwk394 is the reverse of what the B3 report's legend claimed in
`B3-conditions.md§device-mapping`: the **card at nvidia-smi index 0 is the
2080 Ti (physical PCI-order index), but as a CUDA device it is exposed at
runtime index 1**; the GV100 (nvidia-smi index 1) is exposed as runtime CUDA
device 0. Effect: the B3 driver (which bound `--filter` by nvidia-smi index)
measured the GV100 under the "2080 Ti" label and the 2080 Ti under the
"GV100" label. That is the direct cause of the B3 report's "identical numbers
to the ms" for GV100 vs 2080 Ti — B3 did not run two different cards per
case, it ran the two cards in the wrong order and reported the swap.

For D1 this doc therefore uses:
- "2080 Ti"  = nvidia-smi index 0 = **CUDA_VISIBLE_DEVICES=1**
- "GV100"    = nvidia-smi index 1 = **CUDA_VISIBLE_DEVICES=0**
