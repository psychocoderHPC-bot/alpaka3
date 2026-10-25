// SPDX-License-Identifier: MPL-2.0
//
// Benchmark-only reference baseline for the external alpakaVendor BLAS
// wrappers. Compiled exclusively into the gemm_vendor_bench target; no in-tree
// alpaka header or library target includes alpakaVendor.

#include "vendor_baseline.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/blas.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The minimum viable integration wires the host OpenBLAS backend only. The
// CUDA, HIP and oneMKL paths need a backend-specific device in this functor and
// are tracked as a follow-up; refusing to build is better than silently
// measuring the wrong path.
#if !defined(ALPAKAV_DEP_OPENBLAS) || !ALPAKAV_DEP_OPENBLAS
#    error "gemm_vendor_bench requires the alpakaVendor OpenBLAS host backend (alpakaV_DEP_OPENBLAS=ON)."
#endif

namespace gemm::bench
{
    namespace
    {
        constexpr char const* backendName = "vendor:openblas";

        template<typename T>
        [[nodiscard]] T storedFromFloat(double value)
        {
            return static_cast<T>(value);
        }

        [[nodiscard]] bool isTransposed(std::string const& op)
        {
            return op == "transpose";
        }

        // The manifest only encodes untransposed and plain transposed operands.
        // Anything else cannot be expressed with an alpakaVendor annotation.
        [[nodiscard]] std::optional<std::string> unsupportedLayoutReason(Problem const& problem)
        {
            auto const valid = [](std::string const& op) { return op == "none" || op == "transpose"; };
            if(!valid(problem.transpose_a) || !valid(problem.transpose_b))
                return "alpakaVendor baseline supports only 'none' and 'transpose' operand layouts";
            if(problem.mode != "fp32" && problem.mode != "fp64")
                return "alpakaVendor BLAS has no scalar type for mode '" + problem.mode
                       + "'; the interface supports float, double and complex only";
            for(std::size_t const value : {problem.m, problem.n, problem.k, problem.lda, problem.ldb, problem.ldc})
            {
                if(value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                    return "alpakaVendor host backend narrows descriptor integers to 32 bits";
            }
            return std::nullopt;
        }
    } // namespace

    struct VendorGemmFunctor::Impl
    {
        using Device = decltype(alpaka::onHost::makeHostDevice());
        using Queue = decltype(std::declval<Device&>().makeQueue());

        template<typename T>
        using DeviceBuffer
            = std::decay_t<decltype(alpaka::onHost::alloc<T>(std::declval<Device&>(), alpaka::Vec{std::size_t{1u}}))>;

        template<typename T>
        struct Buffers
        {
            std::size_t m{};
            std::size_t n{};
            std::size_t k{};
            std::size_t rowsA{};
            std::size_t colsA{};
            std::size_t rowsB{};
            std::size_t colsB{};
            std::size_t lda{};
            std::size_t ldb{};
            std::size_t ldc{};
            bool transposeA{false};
            bool transposeB{false};
            std::vector<T> a;
            std::vector<T> b;
            std::vector<T> c;
            std::vector<T> cOriginal;
            std::optional<DeviceBuffer<T>> dA;
            std::optional<DeviceBuffer<T>> dB;
            std::optional<DeviceBuffer<T>> dC;
        };

        Device device = alpaka::onHost::makeHostDevice();
        Queue queue = device.makeQueue();
        std::string deviceName = device.getName();

        std::optional<Buffers<float>> f32;
        std::optional<Buffers<double>> f64;

        template<typename T>
        [[nodiscard]] DeviceBuffer<T> allocate(std::size_t elements)
        {
            return alpaka::onHost::alloc<T>(device, alpaka::Vec{std::max<std::size_t>(elements, 1u)});
        }

        template<typename T>
        [[nodiscard]] static std::vector<T> makeBuffer(
            std::size_t rows,
            std::size_t cols,
            std::size_t ld,
            std::uint32_t seed)
        {
            if(rows == 0u || cols == 0u)
                return std::vector<T>(1u, storedFromFloat<T>(0.0));
            std::size_t const elements = (rows - 1u) * ld + cols;
            std::vector<T> buffer(elements);
            for(std::size_t i = 0u; i < elements; ++i)
                buffer[i] = storedFromFloat<T>(std::sin(0.001 * static_cast<double>(i + seed)));
            return buffer;
        }

        template<typename T>
        void prepareImpl(Problem const& problem, Buffers<T>& buffers)
        {
            buffers.m = problem.m;
            buffers.n = problem.n;
            buffers.k = problem.k;
            buffers.lda = problem.lda;
            buffers.ldb = problem.ldb;
            buffers.ldc = problem.ldc;
            buffers.transposeA = isTransposed(problem.transpose_a);
            buffers.transposeB = isTransposed(problem.transpose_b);
            // Stored operand shapes follow the manifest semantics: a transposed
            // operand is stored as the transposed matrix and annotated.
            buffers.rowsA = buffers.transposeA ? problem.k : problem.m;
            buffers.colsA = buffers.transposeA ? problem.m : problem.k;
            buffers.rowsB = buffers.transposeB ? problem.n : problem.k;
            buffers.colsB = buffers.transposeB ? problem.k : problem.n;

            buffers.a = makeBuffer<T>(buffers.rowsA, buffers.colsA, buffers.lda, 1u);
            buffers.b = makeBuffer<T>(buffers.rowsB, buffers.colsB, buffers.ldb, 2u);
            buffers.c = makeBuffer<T>(problem.m, problem.n, buffers.ldc, 3u);
            buffers.cOriginal = buffers.c;

            buffers.dA = allocate<T>(buffers.a.size());
            buffers.dB = allocate<T>(buffers.b.size());
            buffers.dC = allocate<T>(buffers.c.size());
            alpaka::onHost::memcpy(queue, *buffers.dA, buffers.a);
            alpaka::onHost::memcpy(queue, *buffers.dB, buffers.b);
            alpaka::onHost::memcpy(queue, *buffers.dC, buffers.c);
            alpaka::onHost::wait(queue);
        }

        template<typename T>
        void resetImpl(std::optional<Buffers<T>>& buffers)
        {
            if(buffers && !buffers->cOriginal.empty())
                alpaka::onHost::memcpy(queue, *buffers->dC, buffers->cOriginal);
        }

        template<typename T>
        [[nodiscard]] RunResult runImpl(Problem const& problem, Buffers<T>& buffers)
        {
            // Row-major dense alpaka views. Pitches are byte strides; the
            // column pitch must be exactly one element for the vendor wrapper.
            auto const extA = alpaka::Vec{buffers.rowsA, buffers.colsA};
            auto const pitchA = alpaka::Vec{buffers.lda * sizeof(T), sizeof(T)};
            auto const extB = alpaka::Vec{buffers.rowsB, buffers.colsB};
            auto const pitchB = alpaka::Vec{buffers.ldb * sizeof(T), sizeof(T)};
            auto const extC = alpaka::Vec{buffers.m, buffers.n};
            auto const pitchC = alpaka::Vec{buffers.ldc * sizeof(T), sizeof(T)};

            auto viewA = alpaka::makeView(alpaka::api::Host{}, alpaka::onHost::data(*buffers.dA), extA, pitchA);
            auto viewB = alpaka::makeView(alpaka::api::Host{}, alpaka::onHost::data(*buffers.dB), extB, pitchB);
            auto viewC = alpaka::makeView(alpaka::api::Host{}, alpaka::onHost::data(*buffers.dC), extC, pitchC);

            using Transpose = alpaka::blas::Transpose;
            alpaka::blas::AnnotatedView<decltype(viewA)> const annotatedA{
                viewA,
                buffers.transposeA ? Transpose::transposed : Transpose::none};
            alpaka::blas::AnnotatedView<decltype(viewB)> const annotatedB{
                viewB,
                buffers.transposeB ? Transpose::transposed : Transpose::none};
            alpaka::blas::AnnotatedView<decltype(viewC)> annotatedC{viewC, Transpose::none};

            alpaka::blas::onHost::gemm(
                queue,
                static_cast<T>(problem.alpha),
                annotatedA,
                annotatedB,
                static_cast<T>(problem.beta),
                annotatedC);
            return RunResult{true, {}};
        }
    };

    std::string VendorGemmFunctor::unavailableReason(Problem const& problem)
    {
        if(auto const reason = unsupportedLayoutReason(problem))
            return *reason;
        return {};
    }

    VendorGemmFunctor::VendorGemmFunctor() : m_impl{std::make_unique<Impl>()}
    {
    }

    VendorGemmFunctor::~VendorGemmFunctor() = default;

    std::string_view VendorGemmFunctor::name() const
    {
        return "alpakaVendor-openblas";
    }

    std::string_view VendorGemmFunctor::backend() const
    {
        return backendName;
    }

    std::string_view VendorGemmFunctor::deviceName() const
    {
        return m_impl->deviceName;
    }

    bool VendorGemmFunctor::supports(Problem const&) const
    {
        // Always claim support so the harness asks run() for the authoritative
        // answer. That lets an unsupported mode write status
        // "vendor_unavailable" with a reason instead of the generic
        // "unsupported" record used by in-tree functors.
        return true;
    }

    std::string VendorGemmFunctor::unsupportedReason(Problem const& problem) const
    {
        return unavailableReason(problem);
    }

    void VendorGemmFunctor::prepare(Problem const& problem)
    {
        if(!unavailableReason(problem).empty())
            return;
        if(problem.mode == "fp32")
            m_impl->prepareImpl(problem, m_impl->f32.emplace());
        else if(problem.mode == "fp64")
            m_impl->prepareImpl(problem, m_impl->f64.emplace());
    }

    void VendorGemmFunctor::resetOutput(Problem const& problem)
    {
        if(!unavailableReason(problem).empty())
            return;
        if(problem.mode == "fp32")
            m_impl->resetImpl(m_impl->f32);
        else if(problem.mode == "fp64")
            m_impl->resetImpl(m_impl->f64);
    }

    RunResult VendorGemmFunctor::run(Problem const& problem)
    {
        if(std::string const reason = unavailableReason(problem); !reason.empty())
            return RunResult{false, reason, "vendor_unavailable"};
        if(problem.mode == "fp32")
            return m_impl->runImpl(problem, *m_impl->f32);
        if(problem.mode == "fp64")
            return m_impl->runImpl(problem, *m_impl->f64);
        return RunResult{false, "alpakaVendor baseline supports fp32 and fp64 only", "vendor_unavailable"};
    }

    void VendorGemmFunctor::synchronize()
    {
        alpaka::onHost::wait(m_impl->queue);
    }
} // namespace gemm::bench
