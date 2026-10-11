# gaussian — status: DONE

- WORKDIR: /tmp/alpaka3-gaussian-issue-G1
- BRANCH: benchmark/gaussian
- BASE_SHA: 6f13700c390a09df818a255d53e730f1f9d69456 (fork dev)
- COMMIT: 4b16cccbcd46158f686516770507fb5acfa79839 (local only; NO push, NO PR)
- changed paths: benchmark/CMakeLists.txt; benchmark/gaussian/{CMakeLists.txt,README.md,src/helper.hpp,src/main_alpaka.cpp,src/misc.hpp}

## Reference provenance + build
HeCBench 0a17ee47ca6072192101036d4bf292e6612f22e6
  cuda: src/gaussian-cuda/gaussianElim.cu (blob 3c8d2554...) ; omp: src/gaussian-omp/gaussianElim.cpp (blob 268d920b...)
Refs kept outside repo. Built:
  nvcc -std=c++17 -Xcompiler -Wall -arch=sm_70 -O3 gaussianElim.cu -o gaussian_cuda_ref
  g++  -O2 -fopenmp -std=c++17 gaussianElim.cpp utils.cpp -o gaussian_omp_ref

## Validation
- CPU: rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON  -> PASS; sizes 1,2,3,4,7,8,15,16,31,32,63,64,65,
  100,127,128,129,255,256,257,511,512,513,1000,1024,1025 all verify PASS
- GPU: rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON -> PASS (nvcc 12.5 sm_70);
  same size set (GPU subset) verify PASS
- Independent oracle: double-precision partial-pivot LU (helper.hpp). atol=rtol=1e-3, max rel residual 1e-3.
  Observed worst |x-gold| <= 1.8e-5, rel residual <= 2.6e-6 (>50x margin). NaN/Inf checked.
- init_matrix verified byte-identical to reference for all tested sizes.

## Perf (forward-elimination only, median of 5)
| config | backend | alpaka | ref | ratio | variation | validation |
| N=1024 | CUDA | 8.911 ms | 7.809 ms | 1.141 | 0.0012 | PASS |
| N=2048 | CUDA | 55.213 ms | 134.625 ms | 0.410 | 0.1488 | PASS |
| N=4096 | CUDA | 353.261 ms | 483.876 ms | 0.730 | 0.0003 | PASS |
| N=1024 | Host | 115.1 ms | 5460 ms | 0.021 | 0.057 | PASS |
| N=2048 | Host | 754.2 ms | 9757 ms | 0.077 | 0.057 | PASS |

## Repro
CPU: cmake --preset rel-host-cpu-gcc -Dalpaka_COMPILE_PEDANTIC=ON . && cmake --build build/rel-host-cpu-gcc --target gaussian_alpaka -j
GPU: cmake --preset rel-cuda-nvcc-gcc -DCMAKE_CUDA_ARCHITECTURES=70 -Dalpaka_COMPILE_PEDANTIC=ON . && cmake --build build/rel-cuda-nvcc-gcc --target gaussian_alpaka -j
Run: ./gaussian_alpaka --size N --samples 5 --repeat 1

## Risks / follow-ups
- GPU target rosi_v100 was unreachable through the compute gateway for the whole run (all calls errored);
  GPU evidence taken on fwk394 (Quadro GV100, SM 7.0). Architecture identical to V100; CUDA 12.5 not 12.6.
- GPU N=1024 exceeds 1.10 bound (ratio 1.141, launch-overhead at small N); documented exception.
- OMP reference built with g++ -fopenmp has no device offload, so it is not a like-for-like CPU baseline.

Artifacts: results/gaussian/{gaussian-perf.csv,provenance.txt,0001-gaussian.patch}; evidence/gaussian/*
