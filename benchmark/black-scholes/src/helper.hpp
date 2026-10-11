/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Independent host reference and verification helpers for the HeCBench `black-scholes`
 * benchmark.
 *
 * The pinned HeCBench reference (`src/black-scholes-cuda/blackScholesAnalyticEngine.cu` and the
 * OpenMP twin `src/black-scholes-omp/blackScholesAnalyticEngine.cpp`) prices a long stream of
 * European options with the Black-Scholes analytic engine (ported from QuantLib). Each option is a
 * plain-vanilla call or put; the input stream repeats the 37 QuantLib test settings
 * (`optionInputStruct`) defined in the reference, each carrying the published price `value` and its
 * tolerance `tol = 1.0e-4`.
 *
 * The reference computes the normal CDF with a hand-rolled single-precision Cody-Waite `erf`
 * approximation (`errorFunct`). This file contains a deliberately different oracle: the same
 * Black-Scholes formulas evaluated in double precision with the C library `std::erfc`, which has no
 * shared code with the port. It is used only for validation, never for timing.
 *
 * Nothing in this file is device code.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace blackScholesBench
{
    //! Option kind, matching the reference `CALL`/`PUT` defines.
    enum : int
    {
        CALL = 0,
        PUT = 1
    };

    /** One option input, field order identical to the reference `optionInputStruct` (without the
     *  trailing expected value and tolerance, which are validation metadata rather than input).
     */
    struct OptionInput
    {
        int type; //!< CALL or PUT
        float strike;
        float spot;
        float q; //!< dividend yield
        float r; //!< risk-free rate
        float t; //!< time to maturity (year fraction)
        float vol; //!< Black volatility
    };

    /** A pinned reference setting: the option input plus the published analytic price and the
     *  reference's own comparison tolerance.
     */
    struct Setting
    {
        OptionInput in;
        float value;
        float tol;
    };

    //! Number of distinct settings in the reference `main` (`NUM_DIFF_SETTINGS`).
    constexpr int numSettings = 37;

    //! The 37 QuantLib test settings, copied verbatim from the pinned reference `main`.
    inline constexpr std::array<Setting, numSettings> settings = {{
        {{CALL, 40.00f, 42.00f, 0.08f, 0.04f, 0.75f, 0.35f}, 5.0975f, 1.0e-4f},
        {{CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.15f}, 0.0205f, 1.0e-4f},
        {{CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.15f}, 1.8734f, 1.0e-4f},
        {{CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.15f}, 9.9413f, 1.0e-4f},
        {{CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.25f}, 0.3150f, 1.0e-4f},
        {{CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.25f}, 3.1217f, 1.0e-4f},
        {{CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.25f}, 10.3556f, 1.0e-4f},
        {{CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.35f}, 0.9474f, 1.0e-4f},
        {{CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.35f}, 4.3693f, 1.0e-4f},
        {{CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.35f}, 11.1381f, 1.0e-4f},
        {{CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.15f}, 0.8069f, 1.0e-4f},
        {{CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.15f}, 4.0232f, 1.0e-4f},
        {{CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.15f}, 10.5769f, 1.0e-4f},
        {{CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.25f}, 2.7026f, 1.0e-4f},
        {{CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.25f}, 6.6997f, 1.0e-4f},
        {{CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.25f}, 12.7857f, 1.0e-4f},
        {{CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.35f}, 4.9329f, 1.0e-4f},
        {{CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.35f}, 9.3679f, 1.0e-4f},
        {{CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.35f}, 15.3086f, 1.0e-4f},
        {{PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.15f}, 9.9210f, 1.0e-4f},
        {{PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.15f}, 1.8734f, 1.0e-4f},
        {{PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.15f}, 0.0408f, 1.0e-4f},
        {{PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.25f}, 10.2155f, 1.0e-4f},
        {{PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.25f}, 3.1217f, 1.0e-4f},
        {{PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.25f}, 0.4551f, 1.0e-4f},
        {{PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.35f}, 10.8479f, 1.0e-4f},
        {{PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.35f}, 4.3693f, 1.0e-4f},
        {{PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.35f}, 1.2376f, 1.0e-4f},
        {{PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.15f}, 10.3192f, 1.0e-4f},
        {{PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.15f}, 4.0232f, 1.0e-4f},
        {{PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.15f}, 1.0646f, 1.0e-4f},
        {{PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.25f}, 12.2149f, 1.0e-4f},
        {{PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.25f}, 6.6997f, 1.0e-4f},
        {{PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.25f}, 3.2734f, 1.0e-4f},
        {{PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.35f}, 14.4452f, 1.0e-4f},
        {{PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.35f}, 9.3679f, 1.0e-4f},
        {{PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.35f}, 5.7963f, 1.0e-4f},
    }};

    /** Fill `count` option inputs exactly like the pinned reference `main`: entry `i` is
     *  `settings[i % 37]`. The reference uses a chain of `if (numOption % 37 == k)` tests; the
     *  modulo is the same mapping.
     */
    inline void generateOptions(OptionInput* out, std::size_t count)
    {
        for(std::size_t i = 0; i < count; ++i)
            out[i] = settings[i % static_cast<std::size_t>(numSettings)].in;
    }

    //! Standard normal CDF in double precision, independent of the reference's custom `erf`.
    inline double normalCdf(double x)
    {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    }

    /** Independent double-precision Black-Scholes price.
     *
     * Same formula and the same call/put conventions as the reference, but evaluated in double and
     * with `std::erfc` for the normal CDF:
     *   dividendDiscount = exp(-q*t), riskFreeDiscount = exp(-r*t),
     *   forward = spot * dividendDiscount / riskFreeDiscount,
     *   stdDev = sqrt(vol^2 * t), d1 = log(forward/strike)/stdDev + stdDev/2, d2 = d1 - stdDev,
     *   call: alpha = N(d1),  beta = -N(d2),
     *   put:  alpha = -N(-d1), beta = N(-d2),
     *   price = riskFreeDiscount * (forward*alpha + strike*beta).
     */
    inline double referencePrice(OptionInput const& o)
    {
        double const t = static_cast<double>(o.t);
        double const dividendDiscount = std::exp(-static_cast<double>(o.q) * t);
        double const riskFreeDiscount = std::exp(-static_cast<double>(o.r) * t);
        double const spot = static_cast<double>(o.spot);
        double const strike = static_cast<double>(o.strike);
        double const forward = spot * dividendDiscount / riskFreeDiscount;
        double const vol = static_cast<double>(o.vol);
        double const stdDev = std::sqrt(vol * vol * t);
        double const d1 = std::log(forward / strike) / stdDev + 0.5 * stdDev;
        double const d2 = d1 - stdDev;

        double alpha;
        double beta;
        if(o.type == CALL)
        {
            alpha = normalCdf(d1);
            beta = -normalCdf(d2);
        }
        else
        {
            alpha = -normalCdf(-d1);
            beta = normalCdf(-d2);
        }
        return riskFreeDiscount * (forward * alpha + strike * beta);
    }

    /** Verify a float result against the independent double oracle.
     *
     * Rejects any NaN or Inf, then requires the per-option absolute deviation from the double
     * reference to stay within `tolerance`. The largest deviation, the largest deviation from the
     * pinned published prices, and the first offending index are reported. A nonzero return means
     * the process must exit non-zero.
     *
     * @return `EXIT_SUCCESS` when every value is finite and within `tolerance`, else `EXIT_FAILURE`.
     */
    inline int verify(
        float const* got,
        OptionInput const* in,
        std::size_t count,
        double tolerance,
        char const* label)
    {
        double maxAbs = 0.0;
        double maxPublished = 0.0;
        std::size_t worst = 0;
        std::size_t firstNonFinite = count;
        std::size_t firstOver = count;
        for(std::size_t i = 0; i < count; ++i)
        {
            if(!std::isfinite(got[i]))
            {
                firstNonFinite = i;
                break;
            }
            double const ref = referencePrice(in[i]);
            double const d = std::fabs(static_cast<double>(got[i]) - ref);
            if(d > maxAbs)
            {
                maxAbs = d;
                worst = i;
            }
            if(d > tolerance && firstOver == count)
                firstOver = i;

            // Secondary, informational: deviation from the published analytic price of the setting
            // this entry was generated from.
            Setting const& s = settings[i % static_cast<std::size_t>(numSettings)];
            double const dp = std::fabs(static_cast<double>(got[i]) - static_cast<double>(s.value));
            if(dp > maxPublished)
                maxPublished = dp;
        }

        if(firstNonFinite != count)
        {
            std::printf(
                "  %s: non-finite value (NaN/Inf) at index %llu\n",
                label,
                static_cast<unsigned long long>(firstNonFinite));
            std::printf("%s FAIL\n", label);
            return EXIT_FAILURE;
        }

        std::printf(
            "  %s: max abs vs double oracle = %.3e (index %llu), max abs vs published = %.3e, "
            "tolerance = %.1e\n",
            label,
            maxAbs,
            static_cast<unsigned long long>(worst),
            maxPublished,
            tolerance);

        if(firstOver != count)
        {
            std::printf(
                "  %s: first value over tolerance at index %llu (got %.8g, ref %.8g)\n",
                label,
                static_cast<unsigned long long>(firstOver),
                static_cast<double>(got[firstOver]),
                referencePrice(in[firstOver]));
            std::printf("%s FAIL\n", label);
            return EXIT_FAILURE;
        }
        std::printf("%s PASS\n", label);
        return EXIT_SUCCESS;
    }

} // namespace blackScholesBench
