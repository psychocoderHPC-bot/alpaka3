/* Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

/** @file Host reference, deterministic input generation and verification for the `histogram` benchmark.
 *
 * The pinned HeCBench reference generates its input pixels with a Mersenne Twister (ext/mersenne.h in
 * the reference tree) seeded by `init_by_array({0x123, 0x234, 0x345, 0x456}, 4)` and then produces an
 * exact integer histogram of 256 bins per active channel. This file reproduces that generator and the
 * reference bucket scheme so the same pixels and the same buckets are used, and it supplies an
 * independent host histogram plus an exact (integer) comparison.
 *
 * RNG provenance:
 *  - The MT19937 implementation reproduces the algorithm of `mersenne.h` in HeCBench
 *    (`src/histogram-cuda/mersenne.h`), which is
 *    Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura, BSD-3-Clause.
 *  - `randomBits()` reproduces the semantics of `RandomBits()` from `src/histogram-cuda/test_util.h`
 *    (Copyright (c) 2011, Duane Merrill; Copyright (c) 2011-2018, NVIDIA CORPORATION., BSD-3-Clause).
 *
 * Nothing in this file is device code; it only produces the input and the trusted result.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace histogramBench
{
    //! Number of histogram bins, matching `TestGenres()` in the pinned HeCBench reference.
    constexpr int numBins = 256;

    //! Pixel used by the 1-channel test (256 bins).
    struct Uchar1
    {
        std::uint8_t x;
    };

    //! Pixel used by the 3-channel 8-bit test (256 bins).
    struct Uchar4
    {
        std::uint8_t x;
        std::uint8_t y;
        std::uint8_t z;
        std::uint8_t w;
    };

    //! Pixel used by the 3-channel float test (256 bins, samples scaled to [0, 1)).
    struct Float4
    {
        float x;
        float y;
        float z;
        float w;
    };

    /** MT19937, the Mersenne Twister.
     *
     * A faithful reimplementation of the reference `mersenne::` namespace: `init_by_array`, `init_genrand`
     * and `genrand_int32` produce the identical word stream for the same seed array.
     */
    class Mersenne
    {
        static constexpr int N = 624;
        static constexpr int M = 397;
        static constexpr std::uint32_t MATRIX_A = 0x9908b0dfu;
        static constexpr std::uint32_t UPPER_MASK = 0x80000000u;
        static constexpr std::uint32_t LOWER_MASK = 0x7fffffffu;

    public:
        Mersenne() = default;

        explicit Mersenne(std::uint32_t seed)
        {
            initGenrand(seed);
        }

        void initGenrand(std::uint32_t s)
        {
            m_mt[0] = s & 0xffffffffu;
            for(m_mti = 1; m_mti < N; m_mti++)
            {
                m_mt[m_mti] = (1812433253u * (m_mt[m_mti - 1] ^ (m_mt[m_mti - 1] >> 30)) + static_cast<std::uint32_t>(m_mti));
                m_mt[m_mti] &= 0xffffffffu;
            }
        }

        void initByArray(std::uint32_t const init_key[], int key_length)
        {
            initGenrand(19650218u);
            int i = 1;
            int j = 0;
            int k = (N > key_length ? N : key_length);
            for(; k; k--)
            {
                m_mt[i] = (m_mt[i] ^ ((m_mt[i - 1] ^ (m_mt[i - 1] >> 30)) * 1664525u)) + init_key[j]
                          + static_cast<std::uint32_t>(j);
                m_mt[i] &= 0xffffffffu;
                i++;
                j++;
                if(i >= N)
                {
                    m_mt[0] = m_mt[N - 1];
                    i = 1;
                }
                if(j >= key_length)
                    j = 0;
            }
            for(k = N - 1; k; k--)
            {
                m_mt[i] = (m_mt[i] ^ ((m_mt[i - 1] ^ (m_mt[i - 1] >> 30)) * 1566083941u)) - static_cast<std::uint32_t>(i);
                m_mt[i] &= 0xffffffffu;
                i++;
                if(i >= N)
                {
                    m_mt[0] = m_mt[N - 1];
                    i = 1;
                }
            }
            m_mt[0] = 0x80000000u;
        }

        std::uint32_t genrandInt32()
        {
            std::uint32_t y;
            static std::uint32_t const mag01[2] = {0x0u, MATRIX_A};

            if(m_mti >= N)
            {
                int kk;
                if(m_mti == N + 1)
                    initGenrand(5489u);

                for(kk = 0; kk < N - M; kk++)
                {
                    y = (m_mt[kk] & UPPER_MASK) | (m_mt[kk + 1] & LOWER_MASK);
                    m_mt[kk] = m_mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1u];
                }
                for(; kk < N - 1; kk++)
                {
                    y = (m_mt[kk] & UPPER_MASK) | (m_mt[kk + 1] & LOWER_MASK);
                    m_mt[kk] = m_mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1u];
                }
                y = (m_mt[N - 1] & UPPER_MASK) | (m_mt[0] & LOWER_MASK);
                m_mt[N - 1] = m_mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1u];

                m_mti = 0;
            }

            y = m_mt[m_mti++];
            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= (y >> 18);
            return y;
        }

    private:
        std::uint32_t m_mt[N]{};
        int m_mti = N + 1;
    };

    //! NaN predicate for integral keys, matching the reference `IsNaN()` fallback.
    template<typename T>
    inline bool isNan(T)
    {
        return false;
    }

    inline bool isNan(float val)
    {
        return std::isnan(val);
    }

    /** Generate a random value, bit-for-bit identical to the reference `RandomBits()` for a 1-byte key.
     *
     * Words are the bitwise AND of `entropy_reduction + 1` MT words, then the low byte is kept. For the
     * default entropy of 0 this is exactly `low_byte(rand)`.
     */
    template<typename T_Key>
    inline void randomBits(Mersenne& rng, T_Key& key, int entropy_reduction = 0, int begin_bit = 0, int end_bit = -1)
    {
        constexpr int num_bytes = static_cast<int>(sizeof(T_Key));
        constexpr int word_bytes = static_cast<int>(sizeof(std::uint32_t));
        constexpr int num_words = (num_bytes + word_bytes - 1) / word_bytes;

        std::uint32_t word_buff[num_words];

        if(entropy_reduction == -1)
        {
            std::memset(static_cast<void*>(&key), 0, sizeof(key));
            return;
        }

        if(end_bit < 0)
            end_bit = num_bytes * 8;

        while(true)
        {
            for(int j = 0; j < num_words; j++)
            {
                int const current_bit = j * word_bytes * 8;

                std::uint32_t word = 0xffffffffu;
                word &= 0xffffffffu << std::max(0, begin_bit - current_bit);
                word &= 0xffffffffu >> std::max(0, (current_bit + (word_bytes * 8)) - end_bit);

                for(int i = 0; i <= entropy_reduction; i++)
                    word &= rng.genrandInt32();

                word_buff[j] = word;
            }

            std::memcpy(&key, word_buff, sizeof(T_Key));

            T_Key copy = key;
            if(!isNan(copy))
                break; // avoids NaNs when generating random floating point numbers
        }
    }

    /** Generate @p numPixels 8-bit RGBA pixels with the reference RNG and seeding.
     *
     * The seed array and the per-channel order (x, y, z, w) match `CommandLineArgs` + `GenerateRandomImage`
     * in the pinned HeCBench reference.
     */
    inline void generateRandomPixels(Uchar4* pixels, std::size_t numPixels, int entropyReduction)
    {
        Mersenne rng;
        std::uint32_t const seed[4] = {0x123u, 0x234u, 0x345u, 0x456u};
        rng.initByArray(seed, 4);

        for(std::size_t i = 0; i < numPixels; ++i)
        {
            randomBits(rng, pixels[i].x, entropyReduction);
            randomBits(rng, pixels[i].y, entropyReduction);
            randomBits(rng, pixels[i].z, entropyReduction);
            randomBits(rng, pixels[i].w, entropyReduction);
        }
    }

    //! 1-channel view: integer average of the first three channels, matching `TestGenres()`.
    inline void makeUchar1(Uchar1* out, Uchar4 const* in, std::size_t numPixels)
    {
        for(std::size_t i = 0; i < numPixels; ++i)
        {
            out[i].x = static_cast<std::uint8_t>(
                (static_cast<unsigned>(in[i].x) + static_cast<unsigned>(in[i].y) + static_cast<unsigned>(in[i].z)) / 3u);
        }
    }

    //! 3-channel float view scaled to [0, 1), matching `TestGenres()`.
    inline void makeFloat4(Float4* out, Uchar4 const* in, std::size_t numPixels)
    {
        for(std::size_t i = 0; i < numPixels; ++i)
        {
            out[i].x = static_cast<float>(in[i].x) / 256;
            out[i].y = static_cast<float>(in[i].y) / 256;
            out[i].z = static_cast<float>(in[i].z) / 256;
            out[i].w = static_cast<float>(in[i].w) / 256;
        }
    }

    /** Decode a pixel into `ACTIVE_CHANNELS` bin indices, matching the reference bucket scheme.
     *
     * 8-bit inputs use the sample value directly; float inputs use `(unsigned)(sample * NUM_BINS)`.
     */
    template<int NUM_BINS, int ACTIVE_CHANNELS>
    inline void decodePixel(Uchar1 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        bins[0] = static_cast<unsigned int>(pixel.x);
    }

    template<int NUM_BINS, int ACTIVE_CHANNELS>
    inline void decodePixel(Uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        unsigned char const samples[4] = {pixel.x, pixel.y, pixel.z, pixel.w};
        for(int channel = 0; channel < ACTIVE_CHANNELS; ++channel)
            bins[channel] = static_cast<unsigned int>(samples[channel]);
    }

    template<int NUM_BINS, int ACTIVE_CHANNELS>
    inline void decodePixel(Float4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
    {
        float const samples[4] = {pixel.x, pixel.y, pixel.z, pixel.w};
        for(int channel = 0; channel < ACTIVE_CHANNELS; ++channel)
            bins[channel] = static_cast<unsigned int>(samples[channel] * static_cast<float>(NUM_BINS));
    }

    /** Independent host reference histogram, recomputed directly from the input pixels.
     *
     * @param pixels pointer to @p numPixels pixels
     * @param hist output array of `ACTIVE_CHANNELS * NUM_BINS` counters, zeroed by this function
     */
    template<int ACTIVE_CHANNELS, int NUM_BINS, typename T_Pixel>
    inline void histogramGold(T_Pixel const* pixels, std::size_t numPixels, unsigned int* hist)
    {
        std::memset(hist, 0, static_cast<std::size_t>(ACTIVE_CHANNELS) * NUM_BINS * sizeof(unsigned int));

        for(std::size_t i = 0; i < numPixels; ++i)
        {
            unsigned int bins[ACTIVE_CHANNELS];
            decodePixel<NUM_BINS>(pixels[i], bins);
            for(int channel = 0; channel < ACTIVE_CHANNELS; ++channel)
                hist[NUM_BINS * channel + bins[channel]]++;
        }
    }

    /** Check that all floating point samples are finite.
     *
     * @return EXIT_SUCCESS when every sample is finite, EXIT_FAILURE otherwise.
     */
    inline int checkFinite(Float4 const* pixels, std::size_t numPixels, char const* label)
    {
        for(std::size_t i = 0; i < numPixels; ++i)
        {
            if(!std::isfinite(pixels[i].x) || !std::isfinite(pixels[i].y) || !std::isfinite(pixels[i].z)
               || !std::isfinite(pixels[i].w))
            {
                std::printf("error: non-finite input sample at pixel %zu (%s)\n", i, label);
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

    /** Exact integer comparison of a computed histogram against the host reference.
     *
     * No tolerance is applied: a histogram is an integer reduction of exact bucket assignments, so any
     * difference is a real bug. On the first mismatch the global bin index and both counts are printed and
     * `EXIT_FAILURE` is returned. The total-count invariant `sum(hist) == ACTIVE_CHANNELS * numPixels` is
     * also validated.
     */
    inline int verify(
        unsigned int const* computed,
        unsigned int const* reference,
        int activeChannels,
        int binsPerChannel,
        std::size_t numPixels,
        char const* label)
    {
        int error = EXIT_SUCCESS;
        for(int channel = 0; channel < activeChannels; ++channel)
        {
            for(int bin = 0; bin < binsPerChannel; ++bin)
            {
                int const idx = channel * binsPerChannel + bin;
                if(computed[idx] != reference[idx])
                {
                    std::printf(
                        "error: %s channel=%d bin=%d expected=%u got=%u\n",
                        label,
                        channel,
                        bin,
                        reference[idx],
                        computed[idx]);
                    error = EXIT_FAILURE;
                    return error;
                }
            }
        }

        // Invariant: every pixel contributes exactly one count per active channel.
        unsigned long long computedTotal = 0;
        unsigned long long referenceTotal = 0;
        for(int i = 0; i < activeChannels * binsPerChannel; ++i)
        {
            computedTotal += computed[i];
            referenceTotal += reference[i];
        }
        unsigned long long const expected = static_cast<unsigned long long>(activeChannels) * numPixels;
        if(computedTotal != expected || referenceTotal != expected)
        {
            std::printf(
                "error: %s total count mismatch: computed=%llu reference=%llu expected=%llu\n",
                label,
                computedTotal,
                referenceTotal,
                expected);
            return EXIT_FAILURE;
        }

        return error;
    }

} // namespace histogramBench
