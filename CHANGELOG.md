# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.0.0] - 2026-??-??

### Fixed

The Intel C++ compilers (icx, icpx) at `-O3` with the default FP model set the MXCSR FTZ and DAZ bits once per
translation unit and never restore them, so FP32 subnormals are flushed process-wide. That corrupted the exact
FP16/BFloat16 to FP32 widening of GEMM inputs, because BF16 subnormals become FP32 subnormals. The project now pins
`-ffp-model=precise` for every C++ translation unit of its own targets when the compiler is Intel C++ and
`alpaka_FAST_MATH`/`alpaka_FTZ` are not `ON`. Applications that compile their own entry point must compile the whole
program with `-ffp-model=precise`; otherwise a single default-model translation unit re-enables FTZ/DAZ. Setting
`alpaka_FAST_MATH=ON` or `alpaka_FTZ=ON` deliberately keeps the flush behaviour.

alpaka 3.0.0 is a complete rewrite of the library, with a new API and a new implementation. 
The goal is to simplify the interface by reducing by reduce heavy template usage in the user frontend and providing abstraction to write more efficient heterogeneous code. 
This simplification should still allow the explicit control over the hardware and the performance that alpaka is known for.
