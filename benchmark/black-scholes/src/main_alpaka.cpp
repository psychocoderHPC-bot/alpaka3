/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** @file Portable, single-source alpaka implementation of the HeCBench `black-scholes` benchmark.
 *
 * The pinned HeCBench reference (`src/black-scholes-cuda/blackScholesAnalyticEngine.cu`, with the
 * OpenMP twin `src/black-scholes-omp/blackScholesAnalyticEngine.cpp`) prices a long stream of
 * European options with the Black-Scholes analytic engine ported from QuantLib. Every option is a
 * plain-vanilla call or put and every entry of the input stream repeats one of the 37 test settings
 * defined in the reference `main`.
 *
 * This port keeps the reference algorithm exactly:
 *  - the same per-option pipeline (dividend/risk-free discounts, forward, standard deviation,
 *    `initBlackCalculator` / `initBlackCalcVars`, `getResultVal`),
 *  - the same single-precision Cody-Waite `erf` approximation (`errorFunct`) and its constants,
 *    used through `cumNormDistOp` for the normal CDF,
 *  - the same input stream (`settings[i mod 37]`) and single-precision arithmetic.
 *
 * One portable compute source drives the host CPU backend and the CUDA backend; there is no native
 * CUDA/HIP kernel and no backend branch selecting a different algorithm. The HeCBench reference
 * sources are kept outside this repository and are not vendored.
 */

#include <alpaka/alpaka.hpp>

#include "helper.hpp"
#include "misc.hpp"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace blackScholesAlpaka
{
    using namespace blackScholesBench;

    //! Thread block size of the reference (`THREAD_BLOCK_SIZE`).
    constexpr std::uint32_t threadBlockSize = 256u;

    //! Constants of the reference `errorFunct`, reproduced verbatim from `errorFunctConsts.cuh`.
    constexpr float M_1_SQRTPI = 0.564189583547756286948f;
    constexpr float M_SQRT_2 = 0.7071067811865475244008443621048490392848359376887f;
    constexpr float ERROR_FUNCT_tiny = 0.000000000000000000001f;
    constexpr float ERROR_FUNCT_one = 1.00000000000000000000e+00f;
    constexpr float ERROR_FUNCT_erx = 8.45062911510467529297e-01f;
    constexpr float ERROR_FUNCT_efx = 1.28379167095512586316e-01f;
    constexpr float ERROR_FUNCT_efx8 = 1.02703333676410069053e+00f;
    constexpr float ERROR_FUNCT_pp0 = 1.28379167095512558561e-01f;
    constexpr float ERROR_FUNCT_pp1 = -3.25042107247001499370e-01f;
    constexpr float ERROR_FUNCT_pp2 = -2.84817495755985104766e-02f;
    constexpr float ERROR_FUNCT_pp3 = -5.77027029648944159157e-03f;
    constexpr float ERROR_FUNCT_pp4 = -2.37630166566501626084e-05f;
    constexpr float ERROR_FUNCT_qq1 = 3.97917223959155352819e-01f;
    constexpr float ERROR_FUNCT_qq2 = 6.50222499887672944485e-02f;
    constexpr float ERROR_FUNCT_qq3 = 5.08130628187576562776e-03f;
    constexpr float ERROR_FUNCT_qq4 = 1.32494738004321644526e-04f;
    constexpr float ERROR_FUNCT_qq5 = -3.96022827877536812320e-06f;
    constexpr float ERROR_FUNCT_pa0 = -2.36211856075265944077e-03f;
    constexpr float ERROR_FUNCT_pa1 = 4.14856118683748331666e-01f;
    constexpr float ERROR_FUNCT_pa2 = -3.72207876035701323847e-01f;
    constexpr float ERROR_FUNCT_pa3 = 3.18346619901161753674e-01f;
    constexpr float ERROR_FUNCT_pa4 = -1.10894694282396677476e-01f;
    constexpr float ERROR_FUNCT_pa5 = 3.54783043256182359371e-02f;
    constexpr float ERROR_FUNCT_pa6 = -2.16637559486879084300e-03f;
    constexpr float ERROR_FUNCT_qa1 = 1.06420880400844228286e-01f;
    constexpr float ERROR_FUNCT_qa2 = 5.40397917702171048937e-01f;
    constexpr float ERROR_FUNCT_qa3 = 7.18286544141962662868e-02f;
    constexpr float ERROR_FUNCT_qa4 = 1.26171219808761642112e-01f;
    constexpr float ERROR_FUNCT_qa5 = 1.36370839120290507362e-02f;
    constexpr float ERROR_FUNCT_qa6 = 1.19844998467991074170e-02f;
    constexpr float ERROR_FUNCT_ra0 = -9.86494403484714822705e-03f;
    constexpr float ERROR_FUNCT_ra1 = -6.93858572707181764372e-01f;
    constexpr float ERROR_FUNCT_ra2 = -1.05586262253232909814e+01f;
    constexpr float ERROR_FUNCT_ra3 = -6.23753324503260060396e+01f;
    constexpr float ERROR_FUNCT_ra4 = -1.62396669462573470355e+02f;
    constexpr float ERROR_FUNCT_ra5 = -1.84605092906711035994e+02f;
    constexpr float ERROR_FUNCT_ra6 = -8.12874355063065934246e+01f;
    constexpr float ERROR_FUNCT_ra7 = -9.81432934416914548592e+00f;
    constexpr float ERROR_FUNCT_sa1 = 1.96512716674392571292e+01f;
    constexpr float ERROR_FUNCT_sa2 = 1.37657754143519042600e+02f;
    constexpr float ERROR_FUNCT_sa3 = 4.34565877475229228821e+02f;
    constexpr float ERROR_FUNCT_sa4 = 6.45387271733267880336e+02f;
    constexpr float ERROR_FUNCT_sa5 = 4.29008140027567833386e+02f;
    constexpr float ERROR_FUNCT_sa6 = 1.08635005541779435134e+02f;
    constexpr float ERROR_FUNCT_sa7 = 6.57024977031928170135e+00f;
    constexpr float ERROR_FUNCT_sa8 = -6.04244152148580987438e-02f;
    constexpr float ERROR_FUNCT_rb0 = -9.86494292470009928597e-03f;
    constexpr float ERROR_FUNCT_rb1 = -7.99283237680523006574e-01f;
    constexpr float ERROR_FUNCT_rb2 = -1.77579549177547519889e+01f;
    constexpr float ERROR_FUNCT_rb3 = -1.60636384855821916062e+02f;
    constexpr float ERROR_FUNCT_rb4 = -6.37566443368389627722e+02f;
    constexpr float ERROR_FUNCT_rb5 = -1.02509513161107724954e+03f;
    constexpr float ERROR_FUNCT_rb6 = -4.83519191608651397019e+02f;
    constexpr float ERROR_FUNCT_sb1 = 3.03380607434824582924e+01f;
    constexpr float ERROR_FUNCT_sb2 = 3.25792512996573918826e+02f;
    constexpr float ERROR_FUNCT_sb3 = 1.53672958608443695994e+03f;
    constexpr float ERROR_FUNCT_sb4 = 3.19985821950859553908e+03f;
    constexpr float ERROR_FUNCT_sb5 = 2.55305040643316442583e+03f;
    constexpr float ERROR_FUNCT_sb6 = 4.74528541206955367215e+02f;
    constexpr float ERROR_FUNCT_sb7 = -2.24409524465858183362e+01f;

    /** Single-precision `erf` approximation, identical to `errorFunct` in the reference
     *  `blackScholesAnalyticEngineKernels.cu`.
     */
    ALPAKA_FN_ACC inline float errorFunct(float x)
    {
        float R, S, P, Q, s, y, z, r, ax;

        ax = std::fabs(x);

        if(ax < 0.84375f)
        {
            if(ax < 3.7252902984e-09f)
            {
                if(ax < FLT_MIN * 16.0f)
                    return 0.125f * (8.0f * x + ERROR_FUNCT_efx8 * x);
                return x + ERROR_FUNCT_efx * x;
            }
            z = x*x;
            r = ERROR_FUNCT_pp0+z*(ERROR_FUNCT_pp1+z*(ERROR_FUNCT_pp2+z*(ERROR_FUNCT_pp3+z*ERROR_FUNCT_pp4)));
            s = ERROR_FUNCT_one+z*(ERROR_FUNCT_qq1+z*(ERROR_FUNCT_qq2+z*(ERROR_FUNCT_qq3+z*(ERROR_FUNCT_qq4+z*ERROR_FUNCT_qq5))));
            y = r/s;
            return x + x*y;
        }
        if(ax <1.25f)
        {
            s = ax-ERROR_FUNCT_one;
            P = ERROR_FUNCT_pa0+s*(ERROR_FUNCT_pa1+s*(ERROR_FUNCT_pa2+s*(ERROR_FUNCT_pa3+s*(ERROR_FUNCT_pa4+s*(ERROR_FUNCT_pa5+s*ERROR_FUNCT_pa6)))));
            Q = ERROR_FUNCT_one+s*(ERROR_FUNCT_qa1+s*(ERROR_FUNCT_qa2+s*(ERROR_FUNCT_qa3+s*(ERROR_FUNCT_qa4+s*(ERROR_FUNCT_qa5+s*ERROR_FUNCT_qa6)))));
            if(x>=0.0f)
                return ERROR_FUNCT_erx + P/Q;
            else
                return -1.0f*ERROR_FUNCT_erx - P/Q;
        }
        if (ax >= 6.0f)
        {
            if(x>=0.0f)
                return ERROR_FUNCT_one-ERROR_FUNCT_tiny;
            else
                return ERROR_FUNCT_tiny-ERROR_FUNCT_one;
        }

        /* Starts to lose accuracy when ax~5 */
        s = ERROR_FUNCT_one/(ax*ax);

        if(ax < 2.85714285714285f) { /* |x| < 1/0.35 */
            R = ERROR_FUNCT_ra0+s*(ERROR_FUNCT_ra1+s*(ERROR_FUNCT_ra2+s*(ERROR_FUNCT_ra3+s*(ERROR_FUNCT_ra4+s*(ERROR_FUNCT_ra5+s*(ERROR_FUNCT_ra6+s*ERROR_FUNCT_ra7))))));
            S = ERROR_FUNCT_one+s*(ERROR_FUNCT_sa1+s*(ERROR_FUNCT_sa2+s*(ERROR_FUNCT_sa3+s*(ERROR_FUNCT_sa4+s*(ERROR_FUNCT_sa5+s*(ERROR_FUNCT_sa6+s*(ERROR_FUNCT_sa7+s*ERROR_FUNCT_sa8)))))));
        } else {    /* |x| >= 1/0.35 */
            R = ERROR_FUNCT_rb0+s*(ERROR_FUNCT_rb1+s*(ERROR_FUNCT_rb2+s*(ERROR_FUNCT_rb3+s*(ERROR_FUNCT_rb4+s*(ERROR_FUNCT_rb5+s*ERROR_FUNCT_rb6)))));
            S = ERROR_FUNCT_one+s*(ERROR_FUNCT_sb1+s*(ERROR_FUNCT_sb2+s*(ERROR_FUNCT_sb3+s*(ERROR_FUNCT_sb4+s*(ERROR_FUNCT_sb5+s*(ERROR_FUNCT_sb6+s*ERROR_FUNCT_sb7))))));
        }

        r = std::exp(-ax * ax - 0.5625f + R / S);
        if(x >= 0.0f)
            return ERROR_FUNCT_one - r / ax;
        else
            return r / ax - ERROR_FUNCT_one;
    }

    //! Operator function of the cumulative normal distribution (`cumNormDistOp`).
    ALPAKA_FN_ACC inline float cumNormDistOp(float z)
    {
        // average = 0, sigma = 1 for the cumulative normal distribution used here.
        float result = 0.5f * (1.0f + errorFunct(z * M_SQRT_2));
        return result;
    }

    //! Gaussian density of the standard normal distribution (`gaussianFunctNormDist` with
    //! average = 0, sigma = 1, denominator = 2; normalizationFactor = M_SQRT_2 * M_1_SQRTPI).
    ALPAKA_FN_ACC inline float gaussianFunctNormDist(float x)
    {
        float const exponent = -(x * x) / 2.0f;
        return exponent <= -690.0f ? 0.0f : (M_SQRT_2 * M_1_SQRTPI) * std::exp(exponent);
    }

    //! Derivative of the cumulative normal distribution (`cumNormDistDeriv`).
    ALPAKA_FN_ACC inline float cumNormDistDeriv(float x)
    {
        return gaussianFunctNormDist(x);
    }

    /** Price one option, reproducing the reference `getOutValOption` pipeline field by field.
     *
     * The reference also fills `n_d1`/`n_d2` (`DalphaDd1`/`DbetaDd2`) via `cumNormDistDeriv`; they
     * are computed here too for fidelity, although `getResultVal` never reads them.
     */
    ALPAKA_FN_ACC inline float priceOption(OptionInput const& o)
    {
        float const strike = o.strike;
        float const spot = o.spot;

        // dividend (q) and risk-free (r) discounts: the reference computes expf(forward * t) and
        // takes its reciprocal.
        float const dividendDiscount = 1.0f / std::exp(o.q * o.t);
        float const riskFreeDiscount = 1.0f / std::exp(o.r * o.t);

        float const variance = o.vol * o.vol * o.t;
        float const stdDev = std::sqrt(variance);
        float const forward = spot * dividendDiscount / riskFreeDiscount;

        // initBlackCalculator + initBlackCalcVars
        float const d1 = std::log(forward / strike) / stdDev + 0.5f * stdDev;
        float const d2 = d1 - stdDev;

        float const cum_d1 = cumNormDistOp(d1);
        float const cum_d2 = cumNormDistOp(d2);
        float const n_d1 = cumNormDistDeriv(d1);
        float const n_d2 = cumNormDistDeriv(d2);
        (void)n_d1;
        (void)n_d2;

        float alpha;
        float beta;
        if(o.type == CALL)
        {
            alpha = cum_d1;
            beta = -1.0f * cum_d2;
        }
        else
        {
            alpha = -1.0f + cum_d1;
            beta = 1.0f - cum_d2;
        }

        // getResultVal
        return riskFreeDiscount * (forward * alpha + strike * beta);
    }

    /** Compute the price of every option in the stream. */
    struct BlackScholesKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            alpaka::concepts::IMdSpan auto options,
            alpaka::concepts::IMdSpan auto output,
            std::uint32_t numOptions) const
        {
            using namespace alpaka;

            for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{numOptions}))
            {
                output[i] = priceOption(options[i]);
            }
        }
    };

    struct Stats
    {
        double median = 0.0;
        double min = 0.0;
        double max = 0.0;
        double spread = 0.0;
    };

    inline Stats summarize(std::vector<double> samples)
    {
        Stats s;
        std::sort(samples.begin(), samples.end());
        s.min = samples.front();
        s.max = samples.back();
        std::size_t const n = samples.size();
        s.median = (n % 2u == 0u) ? 0.5 * (samples[n / 2u - 1u] + samples[n / 2u]) : samples[n / 2u];
        s.spread = (s.median > 0.0) ? (s.max - s.min) / s.median : 0.0;
        return s;
    }

    /** Enqueue the pricing kernel once over the whole option stream. */
    inline void runPricing(auto& queue, auto const& exec, std::uint32_t numOptions, auto& options, auto& output)
    {
        using namespace alpaka;

        std::uint32_t const frames = (numOptions + threadBlockSize - 1u) / threadBlockSize;
        auto const frameSpec = onHost::FrameSpec{Vec{frames}, Vec{threadBlockSize}, exec};
        queue.enqueue(frameSpec, KernelBundle{BlackScholesKernel{}, options, output, numOptions});
    }

    /** Run and validate the benchmark for one device and executor. */
    auto example(auto const& deviceSpec, auto const& exec, blackScholesBench::Params const& params) -> int
    {
        using namespace alpaka;

        auto devSelector = onHost::makeDeviceSelector(deviceSpec);
        if(!devSelector.isAvailable())
            return EXIT_SUCCESS;
        onHost::Device device = devSelector.makeDevice(0);
        onHost::Queue queue = device.makeQueue(queueKind::nonBlocking);
        std::string const backend = deviceSpec.getApi().getName();

        std::uint32_t const numOptions = static_cast<std::uint32_t>(params.options);

        std::printf(
            "Using alpaka backend %s on %s executor %s, options=%u repeat=%d samples=%d\n",
            backend.c_str(),
            deviceSpec.getDeviceKind().getName().c_str(),
            exec.getName().c_str(),
            numOptions,
            params.repeat,
            params.samples);

        auto hInput = onHost::allocHost<OptionInput>(numOptions);
        blackScholesBench::generateOptions(onHost::data(hInput), numOptions);

        auto dOptions = onHost::alloc<OptionInput>(device, numOptions);
        auto dOutput = onHost::alloc<float>(device, numOptions);
        onHost::memcpy(queue, dOptions, hInput);
        onHost::wait(queue);

        if(!params.verifyOnly)
        {
            // warm-up
            runPricing(queue, exec, numOptions, dOptions, dOutput);
            onHost::wait(queue);

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(params.samples));
            for(int s = 0; s < params.samples; ++s)
            {
                onHost::wait(queue);
                auto const start = std::chrono::steady_clock::now();
                for(int r = 0; r < params.repeat; ++r)
                    runPricing(queue, exec, numOptions, dOptions, dOutput);
                onHost::wait(queue);
                auto const end = std::chrono::steady_clock::now();
                // One measured sample runs `repeat` kernel launches end-to-end; report the
                // per-launch time so it is directly comparable to the reference's
                // `Average kernel execution time on GPU`.
                double const ms = static_cast<double>(
                                      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count())
                                  * 1e-6 / static_cast<double>(params.repeat);
                samples.push_back(ms);
            }
            Stats const stats = summarize(samples);
            std::printf(
                "RESULT backend=%s executor=%s options=%u median_ms=%.5f min_ms=%.5f max_ms=%.5f "
                "relspread=%.4f\n",
                backend.c_str(),
                exec.getName().c_str(),
                numOptions,
                stats.median,
                stats.min,
                stats.max,
                stats.spread);
            std::printf("  raw_samples_ms=");
            for(std::size_t i = 0; i < samples.size(); ++i)
                std::printf("%s%.5f", i == 0 ? "" : ",", samples[i]);
            std::printf("\n");
        }

        // Validation: run once into dOutput and compare to the double oracle.
        runPricing(queue, exec, numOptions, dOptions, dOutput);
        auto hResult = onHost::allocHost<float>(numOptions);
        onHost::memcpy(queue, hResult, dOutput);
        onHost::wait(queue);

        std::printf("Comparing against host double-precision reference...\n");
        int const rc = blackScholesBench::verify(
            onHost::data(hResult),
            onHost::data(hInput),
            numOptions,
            blackScholesBench::defaultTolerance,
            "black-scholes alpaka");
        if(rc != EXIT_SUCCESS)
        {
            std::printf("black-scholes verification FAILED\n");
            return rc;
        }
        std::printf("black-scholes options=%u verification PASSED\n", numOptions);
        return EXIT_SUCCESS;
    }
} // namespace blackScholesAlpaka

auto main(int argc, char* argv[]) -> int
{
    blackScholesBench::Params params;
    if(int const ret = blackScholesBench::parseCmd(argc, argv, params))
        return ret;

    using namespace alpaka;

    return onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        { return blackScholesAlpaka::example(onHost::makeDeviceSpec(backend), getExecutor(backend), params); },
        onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors));
}
