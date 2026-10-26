// SPDX-License-Identifier: MPL-2.0
//
// Alpaka adapter for the benchmark harness: times the real common-source GEMM
// kernel on a device chosen at run time. The default remains the host CPU.

#pragma once

#include "gemm_interface.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace gemm::bench
{
    /** Device and executor requested on the command line.
     *
     * An empty executor selects the device's default executor, matching the
     * pre-selection behaviour of the harness. A non-empty value must name an
     * executor that is compiled in and supported by the selected device.
     */
    struct DeviceRequest
    {
        std::string deviceKind{"cpu"};
        std::string executor{};
        alpaka::onHost::GemmStaging stagingPolicy = alpaka::onHost::GemmStaging::automatic;
    };

    namespace detail
    {
        template<typename T>
        [[nodiscard]] T storedFromFloat(float value)
        {
            if constexpr(std::same_as<T, float>)
                return value;
            else
                return T::fromFloat(value);
        }

        [[nodiscard]] inline std::string toLower(std::string_view text)
        {
            std::string result{text};
            std::transform(
                result.begin(),
                result.end(),
                result.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return result;
        }

        [[nodiscard]] inline bool equalsIgnoreCase(std::string_view lhs, std::string_view rhs)
        {
            return toLower(lhs) == toLower(rhs);
        }

        /** Parse the command-line staging policy. Every other value is rejected, so a typo never silently selects
         * the default. */
        [[nodiscard]] inline alpaka::onHost::GemmStaging parseStagingPolicy(std::string_view name)
        {
            if(equalsIgnoreCase(name, "automatic"))
                return alpaka::onHost::GemmStaging::automatic;
            if(equalsIgnoreCase(name, "direct"))
                return alpaka::onHost::GemmStaging::direct;
            if(equalsIgnoreCase(name, "cooperative"))
                return alpaka::onHost::GemmStaging::cooperative;
            throw std::runtime_error(
                "unknown staging policy '" + std::string{name} + "'; expected one of automatic, direct, cooperative");
        }

        [[nodiscard]] inline char const* stagingPolicyName(alpaka::onHost::GemmStaging policy)
        {
            switch(policy)
            {
            case alpaka::onHost::GemmStaging::automatic:
                return "automatic";
            case alpaka::onHost::GemmStaging::direct:
                return "direct";
            case alpaka::onHost::GemmStaging::cooperative:
                return "cooperative";
            case alpaka::onHost::GemmStaging::asynchronous:
                return "asynchronous";
            }
            return "unknown";
        }

        /** True when the executor is usable with the concrete device type. */
        template<typename T_Device, typename T_Executor>
        inline constexpr bool executorIsSupported
            = alpaka::onHost::trait::IsExecutorSupportedBy::Op<T_Executor, typename T_Device::element_type>::value;
    } // namespace detail

    /** Alpaka GEMM functor templated on the concrete device and executor.
     *
     * The functor owns the device, queue and the matrix buffers of the active
     * numerical mode. prepare() allocates and seeds them for a problem; run()
     * only enqueues the real kernel, so the harness measures queue latency plus
     * device work rather than allocation or host packing.
     *
     * Strict FP32, FP16-to-FP32 and BF16-to-FP32 are supported. The mixed modes
     * keep A and B in their 16-bit storage type and C in FP32; the common kernel
     * performs the exact conversion. FP64 is not exposed by this adapter.
     */
    template<typename T_Device, typename T_Executor>
    class AlpakaGemmFunctor final : public GemmFunctor
    {
    public:
        using Device = T_Device;
        using Executor = T_Executor;

        AlpakaGemmFunctor(
            Device device,
            Executor exec,
            alpaka::onHost::GemmStaging stagingPolicy = alpaka::onHost::GemmStaging::automatic)
            : m_device{std::move(device)}
            , m_exec{exec}
            , m_staging{stagingPolicy, 1u}
            , m_queue{m_device.makeQueue()}
            , m_deviceName{m_device.getName()}
            , m_apiName{m_device.getApi().getName()}
            , m_deviceKindName{m_device.getDeviceKind().getName()}
            , m_executorName{Executor::getName()}
            , m_adapterName{"alpaka-" + detail::toLower(m_deviceKindName) + "-" + detail::toLower(m_executorName)}
        {
        }

        [[nodiscard]] std::string_view name() const override
        {
            return m_adapterName;
        }

        [[nodiscard]] std::string_view deviceName() const override
        {
            return m_deviceName;
        }

        [[nodiscard]] std::string_view api() const override
        {
            return m_apiName;
        }

        [[nodiscard]] std::string_view deviceKind() const override
        {
            return m_deviceKindName;
        }

        [[nodiscard]] std::string_view executorName() const override
        {
            return m_executorName;
        }

        [[nodiscard]] std::optional<PlanSummary> plan() const override
        {
            return m_planSummary;
        }

        [[nodiscard]] bool supports(Problem const& problem) const override
        {
            return problem.mode == "fp32" || problem.mode == "fp16_fp32" || problem.mode == "bf16_fp32";
        }

        [[nodiscard]] std::string unsupportedReason(Problem const& problem) const override
        {
            return "alpaka adapter supports fp32, fp16_fp32 and bf16_fp32, requested " + problem.mode;
        }

        void prepare(Problem const& problem) override
        {
            // The plan belongs to the prepared problem, so never carry one over
            // from the previous case.
            m_planCaptured = false;
            m_planDiagnostics.reset();
            m_planSummary.reset();
            setStrides(problem);
            if(problem.mode == "fp32")
                prepareImpl(problem, m_f32.emplace());
            else if(problem.mode == "fp16_fp32")
                prepareImpl(problem, m_f16.emplace());
            else if(problem.mode == "bf16_fp32")
                prepareImpl(problem, m_bf16.emplace());
            else
                throw std::runtime_error(unsupportedReason(problem));
        }

        void resetOutput(Problem const& problem) override
        {
            if(problem.mode == "fp32")
                resetImpl(m_f32);
            else if(problem.mode == "fp16_fp32")
                resetImpl(m_f16);
            else if(problem.mode == "bf16_fp32")
                resetImpl(m_bf16);
        }

        RunResult run(Problem const& problem) override
        {
            if(problem.mode == "fp32")
                return runImpl(problem, *m_f32, alpaka::onHost::GemmNumericMode::fp32);
            if(problem.mode == "fp16_fp32")
                return runImpl(problem, *m_f16, alpaka::onHost::GemmNumericMode::fp16ToFp32);
            if(problem.mode == "bf16_fp32")
                return runImpl(problem, *m_bf16, alpaka::onHost::GemmNumericMode::bf16ToFp32);
            throw std::runtime_error(unsupportedReason(problem));
        }

        void synchronize() override
        {
            alpaka::onHost::wait(m_queue);
        }

    private:
        template<typename T>
        using DeviceBuffer
            = std::decay_t<decltype(alpaka::onHost::alloc<T>(std::declval<Device>(), alpaka::Vec{std::size_t{1u}}))>;

        template<typename T_Operand, typename T_Output>
        struct Buffers
        {
            std::vector<T_Operand> a;
            std::vector<T_Operand> b;
            std::vector<T_Output> c;
            std::vector<T_Output> cOriginal;
            std::optional<DeviceBuffer<T_Operand>> dA;
            std::optional<DeviceBuffer<T_Operand>> dB;
            std::optional<DeviceBuffer<T_Output>> dC;
        };

        void setStrides(Problem const& problem)
        {
            m_lda = problem.lda;
            m_ldb = problem.ldb;
            m_ldc = problem.ldc;
            m_rowsA = problem.transpose_a == "transpose" ? problem.k : problem.m;
            m_colsA = problem.transpose_a == "transpose" ? problem.m : problem.k;
            m_rowsB = problem.transpose_b == "transpose" ? problem.n : problem.k;
            m_colsB = problem.transpose_b == "transpose" ? problem.k : problem.n;
        }

        template<typename T_Operand, typename T_Output>
        void prepareImpl(Problem const& problem, Buffers<T_Operand, T_Output>& buffers)
        {
            buffers.a = makeBuffer<T_Operand>(m_rowsA, m_colsA, m_lda, 1u);
            buffers.b = makeBuffer<T_Operand>(m_rowsB, m_colsB, m_ldb, 2u);
            buffers.c = makeBuffer<T_Output>(problem.m, problem.n, m_ldc, 3u);
            buffers.cOriginal = buffers.c;

            buffers.dA = allocate<T_Operand>(buffers.a.size());
            buffers.dB = allocate<T_Operand>(buffers.b.size());
            buffers.dC = allocate<T_Output>(buffers.c.size());
            alpaka::onHost::memcpy(m_queue, *buffers.dA, buffers.a);
            alpaka::onHost::memcpy(m_queue, *buffers.dB, buffers.b);
            alpaka::onHost::memcpy(m_queue, *buffers.dC, buffers.c);
            alpaka::onHost::wait(m_queue);
        }

        template<typename T_Operand, typename T_Output>
        void resetImpl(std::optional<Buffers<T_Operand, T_Output>>& buffers)
        {
            if(buffers && !buffers->cOriginal.empty())
                alpaka::onHost::memcpy(m_queue, *buffers->dC, buffers->cOriginal);
        }

        template<typename T_Operand, typename T_Output>
        RunResult runImpl(
            Problem const& problem,
            Buffers<T_Operand, T_Output>& buffers,
            alpaka::onHost::GemmNumericMode mode)
        {
            auto const viewC = alpaka::onHost::GemmView<T_Output>{
                alpaka::onHost::data(*buffers.dC),
                problem.m,
                problem.n,
                m_ldc,
                alpaka::onHost::GemmLayout::rowMajor,
                alpaka::onHost::GemmOp::none};
            auto const viewA = alpaka::onHost::GemmView<T_Operand>{
                alpaka::onHost::data(*buffers.dA),
                problem.m,
                problem.k,
                m_lda,
                alpaka::onHost::GemmLayout::rowMajor,
                problem.transpose_a == "transpose" ? alpaka::onHost::GemmOp::transpose : alpaka::onHost::GemmOp::none};
            auto const viewB = alpaka::onHost::GemmView<T_Operand>{
                alpaka::onHost::data(*buffers.dB),
                problem.k,
                problem.n,
                m_ldb,
                alpaka::onHost::GemmLayout::rowMajor,
                problem.transpose_b == "transpose" ? alpaka::onHost::GemmOp::transpose : alpaka::onHost::GemmOp::none};

            // Fill the planner diagnostics once per prepared problem. Later
            // repetitions of the same case pass no output so the timed path
            // does not pay for copying the diagnostics.
            alpaka::onHost::GemmPlanDiagnostics* diagnosticsOut = nullptr;
            if(!m_planCaptured)
            {
                m_planDiagnostics.emplace();
                diagnosticsOut = &*m_planDiagnostics;
            }
            alpaka::onHost::gemm(
                m_queue,
                m_exec,
                viewC,
                viewA,
                viewB,
                static_cast<T_Output>(problem.alpha),
                static_cast<T_Output>(problem.beta),
                mode,
                alpaka::onHost::GemmKernel::registerBlocked,
                m_staging,
                std::nullopt,
                diagnosticsOut);
            if(diagnosticsOut != nullptr)
            {
                auto const& selected = m_planDiagnostics->selected;
                // The scale-only and non-candidate paths do not select a plan;
                // they leave the diagnostics default-constructed.
                if(selected.tileM != 0u)
                    m_planSummary = PlanSummary{
                        selected.tileM,
                        selected.tileN,
                        selected.workerRows,
                        selected.workerCols,
                        selected.kBlock,
                        selected.stages,
                        detail::stagingPolicyName(selected.policy),
                        selected.simdWidth,
                        m_planDiagnostics->stagingBytes};
                m_planCaptured = true;
            }
            return RunResult{true, {}};
        }

        template<typename T>
        [[nodiscard]] DeviceBuffer<T> allocate(std::size_t elements)
        {
            return alpaka::onHost::alloc<T>(m_device, alpaka::Vec{std::max<std::size_t>(elements, 1u)});
        }

        template<typename T>
        [[nodiscard]] static std::vector<T> makeBuffer(
            std::size_t rows,
            std::size_t cols,
            std::size_t ld,
            std::uint32_t seed)
        {
            if(rows == 0u || cols == 0u)
                return std::vector<T>(1u, detail::storedFromFloat<T>(0.0f));
            std::size_t const elements = (rows - 1u) * ld + cols;
            std::vector<T> buffer(elements);
            for(std::size_t i = 0u; i < elements; ++i)
                buffer[i]
                    = detail::storedFromFloat<T>(static_cast<float>(std::sin(0.001 * static_cast<double>(i + seed))));
            return buffer;
        }

        Device m_device;
        Executor m_exec;
        alpaka::onHost::GemmStagingConfig m_staging;
        decltype(m_device.makeQueue()) m_queue;
        std::string m_deviceName;
        std::string m_apiName;
        std::string m_deviceKindName;
        std::string m_executorName;
        std::string m_adapterName;
        std::size_t m_lda{};
        std::size_t m_ldb{};
        std::size_t m_ldc{};
        std::size_t m_rowsA{};
        std::size_t m_colsA{};
        std::size_t m_rowsB{};
        std::size_t m_colsB{};
        std::optional<Buffers<float, float>> m_f32;
        std::optional<Buffers<alpaka::Float16, float>> m_f16;
        std::optional<Buffers<alpaka::BFloat16, float>> m_bf16;
        bool m_planCaptured{false};
        std::optional<alpaka::onHost::GemmPlanDiagnostics> m_planDiagnostics;
        std::optional<PlanSummary> m_planSummary;
    };

    namespace detail
    {
        [[nodiscard]] inline std::string supportedExecutorNames()
        {
            std::string names;
            auto append = [&](auto exec)
            {
                if(!names.empty())
                    names += ", ";
                names += std::decay_t<decltype(exec)>::getName();
            };
            std::apply([&](auto... exec) { (append(exec), ...); }, alpaka::exec::enabledExecutors);
            return names;
        }

        template<typename T_Device>
        [[nodiscard]] std::unique_ptr<GemmFunctor> makeFunctorWithExecutor(
            T_Device device,
            DeviceRequest const& request,
            std::string_view requestedKind)
        {
            std::string available;
            std::unique_ptr<GemmFunctor> functor;
            auto tryExecutor = [&](auto exec)
            {
                using Executor = std::decay_t<decltype(exec)>;
                if constexpr(executorIsSupported<T_Device, Executor>)
                {
                    if(!available.empty())
                        available += ", ";
                    available += Executor::getName();
                    if(functor)
                        return;
                    if(!request.executor.empty() && !equalsIgnoreCase(request.executor, Executor::getName()))
                        return;
                    functor = std::make_unique<AlpakaGemmFunctor<T_Device, Executor>>(
                        std::move(device),
                        exec,
                        request.stagingPolicy);
                }
            };
            std::apply([&](auto... exec) { (tryExecutor(exec), ...); }, alpaka::exec::enabledExecutors);

            if(!functor)
            {
                if(request.executor.empty())
                    throw std::runtime_error(
                        "no enabled executor is supported by device kind '" + std::string{requestedKind} + "'");
                throw std::runtime_error(
                    "executor '" + request.executor + "' is not available for device kind '"
                    + std::string{requestedKind} + "'; supported executors: " + available);
            }
            return functor;
        }

        template<typename T_Api, typename T_Kind>
        inline constexpr bool deviceSpecIsValid = alpaka::onHost::DeviceSpec<T_Api, T_Kind>::isValid();

        template<typename T_Api, typename T_Kind>
        [[nodiscard]] bool deviceIsAvailable()
        {
            if constexpr(deviceSpecIsValid<T_Api, T_Kind>)
            {
                auto selector
                    = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec<T_Api, T_Kind>{T_Api{}, T_Kind{}});
                return selector.isAvailable();
            }
            else
            {
                return false;
            }
        }

        template<typename T_Api, typename T_Kind>
        [[nodiscard]] std::unique_ptr<GemmFunctor> makeFunctorForSpec(
            DeviceRequest const& request,
            std::string_view requestedKind)
        {
            if constexpr(!deviceSpecIsValid<T_Api, T_Kind>)
            {
                throw std::runtime_error(
                    "device kind '" + std::string{requestedKind}
                    + "' is not enabled in this build; the required toolkit was not enabled at configure time");
            }
            else
            {
                auto selector
                    = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec<T_Api, T_Kind>{T_Api{}, T_Kind{}});
                if(!selector.isAvailable())
                    throw std::runtime_error(
                        "no device of kind '" + std::string{requestedKind} + "' is available on this host");
                auto device = selector.makeDevice(0);
                // The driver-backed device must report the api and kind this
                // code requested; a mismatch would mean the benchmark measured
                // an unrelated device.
                if(device.getDeviceKind().getName() != T_Kind{}.getName()
                   || device.getApi().getName() != T_Api{}.getName())
                    throw std::runtime_error(
                        "device driver reported api '" + device.getApi().getName() + "' and kind '"
                        + device.getDeviceKind().getName() + "', expected api '" + T_Api{}.getName() + "' and kind '"
                        + T_Kind{}.getName() + "'");
                return makeFunctorWithExecutor(std::move(device), request, requestedKind);
            }
        }
    } // namespace detail

    /** Build the alpaka GEMM functor for the requested device.
     *
     * Every failure is reported by throwing. There is deliberately no fallback
     * to the host: requesting a GPU on a CPU-only build or missing driver must
     * fail instead of silently producing CPU timings labelled as GPU data.
     */
    [[nodiscard]] inline std::unique_ptr<GemmFunctor> makeAlpakaGemmFunctor(DeviceRequest const& request)
    {
        std::string const kind = detail::toLower(request.deviceKind);
        if(kind == "cpu")
            return detail::makeFunctorForSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>(request, "cpu");
        if(kind == "cuda")
            return detail::makeFunctorForSpec<alpaka::api::Cuda, alpaka::deviceKind::NvidiaGpu>(request, "cuda");
        if(kind == "hip")
            return detail::makeFunctorForSpec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>(request, "hip");
        if(kind == "oneapi-cpu")
            return detail::makeFunctorForSpec<alpaka::api::OneApi, alpaka::deviceKind::Cpu>(request, "oneapi-cpu");
        if(kind == "oneapi-gpu")
        {
            if(detail::deviceIsAvailable<alpaka::api::OneApi, alpaka::deviceKind::IntelGpu>())
                return detail::makeFunctorForSpec<alpaka::api::OneApi, alpaka::deviceKind::IntelGpu>(
                    request,
                    "oneapi-gpu");
            if(detail::deviceIsAvailable<alpaka::api::OneApi, alpaka::deviceKind::NvidiaGpu>())
                return detail::makeFunctorForSpec<alpaka::api::OneApi, alpaka::deviceKind::NvidiaGpu>(
                    request,
                    "oneapi-gpu");
            if(detail::deviceIsAvailable<alpaka::api::OneApi, alpaka::deviceKind::AmdGpu>())
                return detail::makeFunctorForSpec<alpaka::api::OneApi, alpaka::deviceKind::AmdGpu>(
                    request,
                    "oneapi-gpu");
            throw std::runtime_error(
                "device kind 'oneapi-gpu' is not enabled in this build or no oneAPI GPU is available on this host");
        }
        throw std::runtime_error(
            "unknown device kind '" + request.deviceKind
            + "'; expected one of cpu, cuda, hip, oneapi-cpu, oneapi-gpu");
    }
} // namespace gemm::bench
