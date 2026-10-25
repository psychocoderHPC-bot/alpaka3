// SPDX-License-Identifier: MPL-2.0
//
// Optional benchmark-only reference baseline that drives the external
// alpakaVendor BLAS wrappers instead of the in-tree common-source GEMM.
//
// This header deliberately includes no alpakaVendor header: the shared harness
// translation unit only sees this declaration, and all vendor includes stay in
// vendor_baseline.cpp, which is compiled only into the gemm_vendor_bench target.

#pragma once

#include "gemm_interface.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace gemm::bench
{
    /** alpakaVendor-backed reference GEMM functor.
     *
     * Supports the fp32 and fp64 manifest modes. Other numerical modes are
     * reported as "vendor_unavailable" with a reason instead of being measured.
     * The vendor interface exposes no capability predicate, so support is
     * decided here from the manifest mode and the compiled backend.
     */
    class VendorGemmFunctor final : public GemmFunctor
    {
    public:
        VendorGemmFunctor();
        ~VendorGemmFunctor() override;

        VendorGemmFunctor(VendorGemmFunctor const&) = delete;
        VendorGemmFunctor& operator=(VendorGemmFunctor const&) = delete;

        [[nodiscard]] std::string_view name() const override;
        [[nodiscard]] std::string_view backend() const override;
        [[nodiscard]] std::string_view deviceName() const override;

        [[nodiscard]] bool supports(Problem const& problem) const override;
        [[nodiscard]] std::string unsupportedReason(Problem const& problem) const override;

        void prepare(Problem const& problem) override;
        void resetOutput(Problem const& problem) override;
        RunResult run(Problem const& problem) override;
        void synchronize() override;

        // Reason why the vendor path cannot measure this problem, or an empty
        // string when it can.
        [[nodiscard]] static std::string unavailableReason(Problem const& problem);

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
} // namespace gemm::bench
