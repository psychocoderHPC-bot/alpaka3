// SPDX-License-Identifier: MPL-2.0
//
// GEMM benchmark harness skeleton. It loads benchmark/gemm/workloads.json,
// drives a caller-supplied GemmFunctor, and writes raw timing samples to an
// output directory outside the source checkout. The M0 placeholder functor
// reports "not implemented" instead of inventing timings.

#include "gemm_alpaka.hpp"
#include "gemm_interface.hpp"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef GEMM_DEFAULT_MANIFEST
#    define GEMM_DEFAULT_MANIFEST "workloads.json"
#endif

#ifndef GEMM_SOURCE_DIR
#    define GEMM_SOURCE_DIR "."
#endif

namespace
{
    namespace fs = std::filesystem;
    using gemm::bench::GemmFunctor;
    using gemm::bench::Problem;
    using gemm::bench::RunResult;

    struct Arguments
    {
        std::string manifestPath{GEMM_DEFAULT_MANIFEST};
        std::string outputDir{"/tmp/alpaka3-gemm-benchmark"};
        std::string revision{"unknown"};
        std::string target{"cpu-host"};
        std::string subset{"all"};
        std::string filter{};
        std::size_t samplesPerRound{10};
        std::size_t rounds{3};
        std::size_t warmups{1};
        std::size_t maxCases{0};
        bool smoke{false};
        bool listCases{false};
        bool help{false};
    };

    [[noreturn]] void usageError(std::string const& message)
    {
        throw std::runtime_error(message);
    }

    [[nodiscard]] std::size_t parseSize(std::string_view text, std::string_view option)
    {
        std::size_t consumed = 0;
        unsigned long long value = 0;
        try
        {
            value = std::stoull(std::string{text}, &consumed);
        }
        catch(std::exception const&)
        {
            usageError("invalid value for " + std::string{option} + ": " + std::string{text});
        }
        if(consumed != text.size())
            usageError("invalid value for " + std::string{option} + ": " + std::string{text});
        return static_cast<std::size_t>(value);
    }

    [[nodiscard]] Arguments parseArguments(int argc, char** argv)
    {
        Arguments args;
        auto const next = [&](int& index, std::string_view option) -> std::string
        {
            if(index + 1 >= argc)
                usageError("missing value for " + std::string{option});
            ++index;
            return argv[index];
        };

        for(int i = 1; i < argc; ++i)
        {
            std::string const arg = argv[i];
            if(arg == "--manifest")
                args.manifestPath = next(i, arg);
            else if(arg == "--output-dir")
                args.outputDir = next(i, arg);
            else if(arg == "--revision")
                args.revision = next(i, arg);
            else if(arg == "--target")
                args.target = next(i, arg);
            else if(arg == "--subset")
                args.subset = next(i, arg);
            else if(arg == "--filter")
                args.filter = next(i, arg);
            else if(arg == "--samples")
                args.samplesPerRound = parseSize(next(i, arg), arg);
            else if(arg == "--rounds")
                args.rounds = parseSize(next(i, arg), arg);
            else if(arg == "--warmups")
                args.warmups = parseSize(next(i, arg), arg);
            else if(arg == "--max-cases")
                args.maxCases = parseSize(next(i, arg), arg);
            else if(arg == "--smoke")
                args.smoke = true;
            else if(arg == "--list-cases")
                args.listCases = true;
            else if(arg == "--help" || arg == "-h")
                args.help = true;
            else
                usageError("unknown option: " + arg);
        }
        return args;
    }

    void printUsage()
    {
        std::cout << "Usage: gemm_bench [options]\n"
                     "  --manifest PATH       workload manifest (default: " GEMM_DEFAULT_MANIFEST ")\n"
                     "  --output-dir PATH     output directory outside the checkout (default: "
                     "/tmp/alpaka3-gemm-benchmark)\n"
                     "  --revision SHA        source revision recorded in samples.json (default: unknown)\n"
                     "  --target NAME         target identifier (default: cpu-host)\n"
                     "  --subset all|ci-fast  select the full manifest or its bounded CI subset\n"
                     "  --filter TEXT         keep only case ids containing TEXT\n"
                     "  --samples N           samples per round, N >= 10 (default: 10)\n"
                     "  --rounds N            rounds, N >= 3 (default: 3)\n"
                     "  --warmups N           untimed warm-up iterations (default: 1)\n"
                     "  --max-cases N         stop after N selected cases, 0 = all (default: 0)\n"
                     "  --smoke               run the placeholder functor and record not_implemented\n"
                     "  --list-cases          print selected case ids and exit\n"
                     "  --help                print this message\n";
    }

    struct Case
    {
        std::string id;
        std::string family;
        std::size_t m{};
        std::size_t n{};
        std::size_t k{};
        std::vector<std::string> requiredTargets;
        bool capacityQualified{false};
    };

    struct Variant
    {
        std::string mode;
        std::string transposeA;
        std::string transposeB;
        std::string strideClass;
        std::string alphaBetaId;
        double alpha{1.0};
        double beta{0.0};
    };

    struct Manifest
    {
        std::string schemaVersion;
        std::vector<Case> cases;
        std::vector<std::string> modes;
        std::vector<std::pair<std::string, std::string>> transposes;
        std::vector<std::string> strideClasses;
        std::vector<Variant> alphaBeta;
        std::set<std::string> ciCaseIds;
        std::vector<std::string> ciModes;
        std::vector<std::string> ciTransposes;
        std::vector<std::string> ciStrideClasses;
        std::vector<std::string> ciAlphaBetaIds;
    };

    [[nodiscard]] std::string requiredString(gemm::json::Value const& object, std::string_view key)
    {
        return object.at(key).asString();
    }

    [[nodiscard]] std::size_t requiredSize(gemm::json::Value const& object, std::string_view key)
    {
        double const value = object.at(key).asNumber();
        if(value < 0.0 || value > static_cast<double>(std::numeric_limits<std::size_t>::max()))
            throw std::runtime_error("manifest size out of range for key " + std::string{key});
        return static_cast<std::size_t>(value);
    }

    [[nodiscard]] std::vector<std::string> stringList(gemm::json::Value const& value)
    {
        std::vector<std::string> result;
        for(auto const& item : value.asArray())
            result.push_back(item.asString());
        return result;
    }

    [[nodiscard]] Manifest loadManifest(std::string const& path, std::string const& subset)
    {
        std::ifstream stream{path};
        if(!stream)
            throw std::runtime_error("cannot open workload manifest: " + path);
        std::ostringstream buffer;
        buffer << stream.rdbuf();
        auto const root = gemm::json::parse(buffer.str());

        Manifest manifest;
        manifest.schemaVersion = requiredString(root, "schema_version");

        for(auto const& item : root.at("numerical_modes").asArray())
            manifest.modes.push_back(requiredString(item, "id"));

        for(auto const& item : root.at("transposes").asArray())
        {
            manifest.transposes.emplace_back(requiredString(item, "op_a"), requiredString(item, "op_b"));
        }

        for(auto const& item : root.at("stride_classes").asArray())
            manifest.strideClasses.push_back(requiredString(item, "id"));

        Variant alphaBeta;
        for(auto const& item : root.at("alpha_beta_cases").asArray())
        {
            alphaBeta.mode.clear();
            alphaBeta.alphaBetaId = requiredString(item, "id");
            alphaBeta.alpha = item.at("alpha").asNumber();
            alphaBeta.beta = item.at("beta").asNumber();
            manifest.alphaBeta.push_back(alphaBeta);
        }

        auto const& ci = root.at("ci_fast_subset");
        auto const ciCaseList = stringList(ci.at("case_ids"));
        manifest.ciCaseIds = std::set<std::string>{ciCaseList.begin(), ciCaseList.end()};
        manifest.ciModes = stringList(ci.at("modes"));
        manifest.ciTransposes = stringList(ci.at("transposes"));
        manifest.ciStrideClasses = stringList(ci.at("stride_classes"));
        manifest.ciAlphaBetaIds = stringList(ci.at("alpha_beta_cases"));

        auto const& families = root.at("families").asObject();
        for(auto const& [familyName, familyCases] : families)
        {
            for(auto const& item : familyCases.asArray())
            {
                Case entry;
                entry.id = requiredString(item, "id");
                entry.family = familyName;
                entry.m = requiredSize(item, "m");
                entry.n = requiredSize(item, "n");
                entry.k = requiredSize(item, "k");
                if(item.contains("required_targets"))
                    entry.requiredTargets = stringList(item.at("required_targets"));
                entry.capacityQualified
                    = item.contains("capacity_qualified") && item.at("capacity_qualified").asBool();
                manifest.cases.push_back(std::move(entry));
            }
        }

        if(subset != "all" && subset != "ci-fast")
            usageError("--subset must be 'all' or 'ci-fast'");
        if(subset == "ci-fast")
        {
            manifest.cases.erase(
                std::remove_if(
                    manifest.cases.begin(),
                    manifest.cases.end(),
                    [&](Case const& entry) { return manifest.ciCaseIds.find(entry.id) == manifest.ciCaseIds.end(); }),
                manifest.cases.end());
        }
        return manifest;
    }

    [[nodiscard]] bool contains(std::vector<std::string> const& list, std::string const& value)
    {
        return list.empty() || std::find(list.begin(), list.end(), value) != list.end();
    }

    [[nodiscard]] std::string transposeId(std::string const& opA, std::string const& opB)
    {
        auto const letter
            = [](std::string const& op) { return op == "transpose" ? std::string{"t"} : std::string{"n"}; };
        return letter(opA) + letter(opB);
    }

    [[nodiscard]] std::vector<Variant> buildVariants(Manifest const& manifest, std::string const& subset)
    {
        std::vector<Variant> variants;
        auto const modes = subset == "ci-fast" ? manifest.ciModes : manifest.modes;
        auto const strideClasses = subset == "ci-fast" ? manifest.ciStrideClasses : manifest.strideClasses;
        auto const alphaBetaIds = subset == "ci-fast" ? manifest.ciAlphaBetaIds : std::vector<std::string>{};

        std::vector<std::pair<std::string, std::string>> transposes;
        if(subset == "ci-fast")
        {
            for(auto const& item : manifest.transposes)
            {
                if(contains(manifest.ciTransposes, transposeId(item.first, item.second)))
                    transposes.push_back(item);
            }
        }
        else
        {
            transposes = manifest.transposes;
        }

        for(auto const& mode : modes)
        {
            for(auto const& [opA, opB] : transposes)
            {
                for(auto const& stride : strideClasses)
                {
                    for(auto const& item : manifest.alphaBeta)
                    {
                        if(subset == "ci-fast" && !contains(alphaBetaIds, item.alphaBetaId))
                            continue;
                        Variant variant;
                        variant.mode = mode;
                        variant.transposeA = opA;
                        variant.transposeB = opB;
                        variant.strideClass = stride;
                        variant.alphaBetaId = item.alphaBetaId;
                        variant.alpha = item.alpha;
                        variant.beta = item.beta;
                        variants.push_back(std::move(variant));
                    }
                }
            }
        }
        return variants;
    }

    [[nodiscard]] std::string isoTimestamp()
    {
        std::time_t const now = std::time(nullptr);
        std::tm utc{};
        gmtime_r(&now, &utc);
        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
        return buffer;
    }

    [[nodiscard]] std::string compilerIdentity()
    {
#if defined(__clang__)
        return std::string{"clang "} + __clang_version__;
#elif defined(__GNUC__)
        return std::string{"gcc "} + __VERSION__;
#else
        return "unknown compiler";
#endif
    }

    [[nodiscard]] bool pathIsInside(fs::path const& child, fs::path const& parent)
    {
        auto const resolvedChild = fs::weakly_canonical(child);
        auto const resolvedParent = fs::weakly_canonical(parent);
        auto childIt = resolvedChild.begin();
        for(auto parentIt = resolvedParent.begin(); parentIt != resolvedParent.end(); ++parentIt)
        {
            if(childIt == resolvedChild.end() || *childIt != *parentIt)
                return false;
            ++childIt;
        }
        return true;
    }

    void ensureOutsideCheckout(fs::path const& outputDir)
    {
        if(pathIsInside(outputDir, fs::path{GEMM_SOURCE_DIR}))
            usageError(
                "output directory is inside the source checkout; pass --output-dir outside "
                + std::string{GEMM_SOURCE_DIR});
    }

    struct Statistics
    {
        double medianNs{};
        double minNs{};
        double maxNs{};
        double meanNs{};
    };

    [[nodiscard]] Statistics computeStatistics(std::vector<double> samples)
    {
        if(samples.empty())
            throw std::runtime_error("cannot compute statistics without samples");
        std::sort(samples.begin(), samples.end());
        double sum = 0.0;
        for(double const sample : samples)
            sum += sample;
        Statistics stats;
        stats.minNs = samples.front();
        stats.maxNs = samples.back();
        stats.meanNs = sum / static_cast<double>(samples.size());
        std::size_t const middle = samples.size() / 2u;
        if(samples.size() % 2u == 0u)
            stats.medianNs = 0.5 * (samples[middle - 1u] + samples[middle]);
        else
            stats.medianNs = samples[middle];
        return stats;
    }

    [[nodiscard]] Problem makeProblem(Case const& entry, Variant const& variant)
    {
        Problem problem;
        problem.m = entry.m;
        problem.n = entry.n;
        problem.k = entry.k;
        problem.mode = variant.mode;
        problem.transpose_a = variant.transposeA;
        problem.transpose_b = variant.transposeB;

        // Logical strides before padding. A transposed view swaps the fast
        // dimension, so its leading dimension is the other logical extent.
        std::size_t const lda = variant.transposeA == "transpose" ? entry.m : entry.k;
        std::size_t const ldb = variant.transposeB == "transpose" ? entry.k : entry.n;
        std::size_t const padding = variant.strideClass == "padded" ? 7u : 0u;
        problem.lda = lda + padding;
        problem.ldb = ldb + padding;
        problem.ldc = entry.n + padding;
        problem.alpha = variant.alpha;
        problem.beta = variant.beta;
        return problem;
    }

    [[nodiscard]] std::string caseKey(Case const& entry, Variant const& variant)
    {
        return entry.id + "|" + variant.mode + "|" + variant.transposeA + variant.transposeB + "|"
               + variant.strideClass + "|" + variant.alphaBetaId;
    }

    class PlaceholderFunctor final : public GemmFunctor
    {
    public:
        [[nodiscard]] std::string_view name() const override
        {
            return "placeholder-not-implemented";
        }

        [[nodiscard]] bool supports(Problem const&) const override
        {
            return true;
        }

        [[nodiscard]] RunResult run(Problem const&) override
        {
            return RunResult{false, "GEMM kernel not implemented at M0"};
        }

        void synchronize() override
        {
        }
    };

    gemm::json::Value makeNumber(double value)
    {
        return gemm::json::Value::makeNumber(value);
    }

    gemm::json::Value makeSize(std::size_t value)
    {
        return gemm::json::Value::makeNumber(static_cast<double>(value));
    }

    gemm::json::Value runBenchmark(
        Arguments const& args,
        Manifest const& manifest,
        std::vector<Case> const& cases,
        GemmFunctor& functor)
    {
        auto const variants = buildVariants(manifest, args.subset);
        gemm::json::Value::Array records;
        bool allOk = !cases.empty();
        bool anyNotImplemented = false;

        for(auto const& entry : cases)
        {
            for(auto const& variant : variants)
            {
                Problem const problem = makeProblem(entry, variant);

                gemm::json::Value::Object record;
                record["case_id"] = gemm::json::Value::makeString(entry.id);
                record["family"] = gemm::json::Value::makeString(entry.family);
                record["target"] = gemm::json::Value::makeString(args.target);
                record["executor"] = gemm::json::Value::makeString(std::string{functor.name()});
                record["m"] = makeSize(entry.m);
                record["n"] = makeSize(entry.n);
                record["k"] = makeSize(entry.k);
                record["mode"] = gemm::json::Value::makeString(variant.mode);
                record["transpose_a"] = gemm::json::Value::makeString(variant.transposeA);
                record["transpose_b"] = gemm::json::Value::makeString(variant.transposeB);
                record["stride_class"] = gemm::json::Value::makeString(variant.strideClass);
                record["lda"] = makeSize(problem.lda);
                record["ldb"] = makeSize(problem.ldb);
                record["ldc"] = makeSize(problem.ldc);
                record["alpha_beta_case"] = gemm::json::Value::makeString(variant.alphaBetaId);
                record["alpha"] = makeNumber(problem.alpha);
                record["beta"] = makeNumber(problem.beta);
                record["key"] = gemm::json::Value::makeString(caseKey(entry, variant));
                record["capacity_qualified"] = gemm::json::Value::makeBool(entry.capacityQualified);

                if(!functor.supports(problem))
                {
                    record["status"] = gemm::json::Value::makeString("unsupported");
                    record["detail"] = gemm::json::Value::makeString(functor.unsupportedReason(problem));
                    allOk = false;
                }
                else
                {
                    functor.prepare(problem);
                    RunResult const probe = functor.run(problem);
                    functor.synchronize();
                    if(!probe.executed)
                    {
                        record["status"] = gemm::json::Value::makeString("not_implemented");
                        record["detail"] = gemm::json::Value::makeString(probe.detail);
                        allOk = false;
                        anyNotImplemented = true;
                    }
                    else
                    {
                        for(std::size_t warmup = 0; warmup < args.warmups; ++warmup)
                        {
                            functor.resetOutput(problem);
                            RunResult const result = functor.run(problem);
                            functor.synchronize();
                            if(!result.executed)
                                throw std::runtime_error("functor stopped working during warm-up");
                        }

                        std::vector<double> samples;
                        samples.reserve(args.rounds * args.samplesPerRound);
                        for(std::size_t round = 0; round < args.rounds; ++round)
                        {
                            for(std::size_t sample = 0; sample < args.samplesPerRound; ++sample)
                            {
                                functor.resetOutput(problem);
                                auto const begin = std::chrono::steady_clock::now();
                                RunResult const result = functor.run(problem);
                                functor.synchronize();
                                auto const end = std::chrono::steady_clock::now();
                                if(!result.executed)
                                    throw std::runtime_error("functor stopped working while timing");
                                double const ns = std::chrono::duration<double, std::nano>(end - begin).count();
                                samples.push_back(ns);
                            }
                        }

                        Statistics const stats = computeStatistics(samples);
                        gemm::json::Value::Array sampleValues;
                        sampleValues.reserve(samples.size());
                        for(double const value : samples)
                            sampleValues.push_back(makeNumber(value));

                        record["status"] = gemm::json::Value::makeString("ok");
                        record["rounds"] = makeSize(args.rounds);
                        record["samples_per_round"] = makeSize(args.samplesPerRound);
                        record["samples_ns"] = gemm::json::Value::makeArray(std::move(sampleValues));
                        record["median_ns"] = makeNumber(stats.medianNs);
                        record["min_ns"] = makeNumber(stats.minNs);
                        record["max_ns"] = makeNumber(stats.maxNs);
                        record["mean_ns"] = makeNumber(stats.meanNs);
                        double const gflops = problem.usefulFlops() / stats.medianNs; // flop per ns == GFLOP/s
                        record["gflops"] = makeNumber(gflops);
                    }
                }
                records.push_back(gemm::json::Value::makeObject(std::move(record)));
            }
        }

        std::string const overall
            = !allOk ? (anyNotImplemented ? "not_implemented" : "incomplete") : (records.empty() ? "empty" : "ok");

        gemm::json::Value::Object root;
        root["schema_version"] = gemm::json::Value::makeString("1.0.0");
        root["kind"] = gemm::json::Value::makeString("alpaka3-gemm-samples");
        root["revision"] = gemm::json::Value::makeString(args.revision);
        root["target"] = gemm::json::Value::makeString(args.target);
        root["executor"] = gemm::json::Value::makeString(std::string{functor.name()});
        root["manifest"] = gemm::json::Value::makeObject(
            {{"path", gemm::json::Value::makeString(args.manifestPath)},
             {"schema_version", gemm::json::Value::makeString(manifest.schemaVersion)},
             {"subset", gemm::json::Value::makeString(args.subset)}});
        root["protocol"] = gemm::json::Value::makeObject(
            {{"warmups", makeSize(args.warmups)},
             {"rounds", makeSize(args.rounds)},
             {"samples_per_round", makeSize(args.samplesPerRound)},
             {"metric", gemm::json::Value::makeString("2*M*N*K flops / median wall time")}});
        root["toolchain"]
            = gemm::json::Value::makeObject({{"compiler", gemm::json::Value::makeString(compilerIdentity())}});
        root["generated_at"] = gemm::json::Value::makeString(isoTimestamp());
        root["status"] = gemm::json::Value::makeString(overall);
        root["records"] = gemm::json::Value::makeArray(std::move(records));
        return gemm::json::Value::makeObject(std::move(root));
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        Arguments const args = parseArguments(argc, argv);
        if(args.help)
        {
            printUsage();
            return 0;
        }
        if(!args.smoke && (args.rounds < 3u || args.samplesPerRound < 10u))
            usageError("timed runs require --rounds >= 3 and --samples >= 10");

        Manifest const manifest = loadManifest(args.manifestPath, args.subset);

        std::vector<Case> selected;
        for(auto const& entry : manifest.cases)
        {
            if(!args.filter.empty() && entry.id.find(args.filter) == std::string::npos)
                continue;
            // Empty per-shape required_targets means every configured target.
            if(!entry.requiredTargets.empty()
               && std::find(entry.requiredTargets.begin(), entry.requiredTargets.end(), args.target)
                      == entry.requiredTargets.end())
                continue;
            selected.push_back(entry);
            if(args.maxCases != 0u && selected.size() >= args.maxCases)
                break;
        }

        if(args.listCases)
        {
            for(auto const& entry : selected)
                std::cout << entry.family << "\t" << entry.id << "\t" << entry.m << "x" << entry.n << "x" << entry.k
                          << "\n";
            return 0;
        }
        if(selected.empty())
            usageError("no workload cases selected");

        fs::path const outputDir{args.outputDir};
        ensureOutsideCheckout(outputDir);
        fs::create_directories(outputDir);
        fs::path const outputFile = outputDir / "samples.json";

        gemm::bench::AlpakaGemmFunctor functor;
        auto const document = runBenchmark(args, manifest, selected, functor);

        std::ofstream out{outputFile};
        if(!out)
            throw std::runtime_error("cannot write " + outputFile.string());
        out << gemm::json::serialize(document);
        out.close();
        if(!out)
            throw std::runtime_error("failed while writing " + outputFile.string());

        std::cout << "wrote " << outputFile << " (status " << document.at("status").asString() << ", "
                  << document.at("records").size() << " records)\n";
        return 0;
    }
    catch(std::exception const& error)
    {
        std::cerr << "gemm_bench: " << error.what() << "\n";
        return 1;
    }
}
