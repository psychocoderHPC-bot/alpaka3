// SPDX-License-Identifier: MPL-2.0
//
// Minimal interface between the benchmark harness and a GEMM implementation.
// A real alpaka GEMM driver (added after M0) implements this contract; the
// harness never owns matrices or kernel launch details.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace gemm::bench
{
    struct Problem
    {
        std::size_t m{};
        std::size_t n{};
        std::size_t k{};
        std::string mode{"fp32"};
        std::string transpose_a{"none"};
        std::string transpose_b{"none"};
        std::size_t lda{};
        std::size_t ldb{};
        std::size_t ldc{};
        double alpha{1.0};
        double beta{0.0};

        // Useful floating-point operations for throughput reporting. Padded
        // storage and tile tails never inflate this value.
        [[nodiscard]] constexpr double usefulFlops() const
        {
            return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
        }
    };

    struct RunResult
    {
        bool executed{true};
        std::string detail{};
        // Optional machine-readable status for a functor that declines to run a
        // problem. An empty value keeps the harness default ("not_implemented").
        // External baselines use this to report e.g. "vendor_unavailable".
        std::string status{};
    };

    class GemmFunctor
    {
    public:
        virtual ~GemmFunctor() = default;

        [[nodiscard]] virtual std::string_view name() const = 0;

        // Optional provenance recorded on every sample. Empty values are not
        // written, so the in-tree functor keeps the original samples.json shape.
        [[nodiscard]] virtual std::string_view backend() const
        {
            return {};
        }

        [[nodiscard]] virtual std::string_view deviceName() const
        {
            return {};
        }

        // Optional device identity recorded next to the timing samples. The
        // harness writes api, device_kind and executor into samples.json so a
        // record can never be mistaken for a host measurement.
        [[nodiscard]] virtual std::string_view api() const
        {
            return {};
        }

        [[nodiscard]] virtual std::string_view deviceKind() const
        {
            return {};
        }

        [[nodiscard]] virtual std::string_view executorName() const
        {
            return {};
        }

        // Return false when the functor cannot run this precision, layout, or
        // problem size on the compiled target. The reason is recorded verbatim.
        [[nodiscard]] virtual bool supports(Problem const& problem) const = 0;

        [[nodiscard]] virtual std::string unsupportedReason(Problem const& problem) const
        {
            (void) problem;
            return "not supported";
        }

        // Optional allocation/planning performed before warm-up. Planning time
        // is not part of the measured warm samples.
        virtual void prepare(Problem const& problem)
        {
            (void) problem;
        }

        // Called between repetitions when beta != 0 so a repeated run does not
        // accumulate into already-modified output storage.
        virtual void resetOutput(Problem const& problem)
        {
            (void) problem;
        }

        // Enqueue one C := alpha * op(A) * op(B) + beta * C operation. The call
        // must be asynchronous where the underlying queue is asynchronous.
        virtual RunResult run(Problem const& problem) = 0;

        // Wait until all work enqueued by run() has completed.
        virtual void synchronize() = 0;
    };
} // namespace gemm::bench
