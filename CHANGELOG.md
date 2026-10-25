# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [3.0.0] - 2026-??-??

### Fixed

The Intel C++ compilers (icx, icpx) flush FP32 denormals to zero in host code by default, which corrupted the exact
FP16/BFloat16 to FP32 widening of GEMM inputs (BF16 subnormals become FP32 denormals) for host executors. Host
builds with an Intel C++ compiler now pin the host FP32 semantics to `-ffp-model=precise` unless `alpaka_FAST_MATH`
or `alpaka_FTZ` is set to `ON`.

alpaka 3.0.0 is a complete rewrite of the library, with a new API and a new implementation. 
The goal is to simplify the interface by reducing by reduce heavy template usage in the user frontend and providing abstraction to write more efficient heterogeneous code. 
This simplification should still allow the explicit control over the hardware and the performance that alpaka is known for.
