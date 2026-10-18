#include <bits/stdc++.h>

#include "pipeline.hpp"

using namespace sim;
using Clock = std::chrono::steady_clock;

static const int REPS = 9;
static volatile double g_sink = 0.0;  // stops the compiler from deleting the work
static bool g_all_match = true;

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Runs fn() once untimed (warm-up), then REPS timed runs; returns median ms.
static double time_ms(const std::function<void()>& fn) {
    fn();
    std::vector<double> t;
    for (int i = 0; i < REPS; ++i) {
        const auto a = Clock::now();
        fn();
        const auto b = Clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    return median(t);
}

struct Row {
    std::string what;
    std::string work;
    double base_ms;
    double fast_ms;
};
static std::vector<Row> rows;

static void report(const std::string& what, const std::string& work, double base_ms, double fast_ms) {
    rows.push_back({what, work, base_ms, fast_ms});
    std::printf("%-38s baseline %9.3f ms   optimised %9.3f ms   speed-up %6.2fx   time cut %5.1f%%\n", what.c_str(),
                base_ms, fast_ms, base_ms / fast_ms, 100.0 * (1.0 - fast_ms / base_ms));
}

// ---- 1. Shortest paths on networks of several sizes
static void bench_routing(int n, int sources, uint64_t seed) {
    const Graph g = generate_network(n, 6, seed);
    std::vector<int> srcs;
    Rng rng(seed + 5);
    for (int i = 0; i < sources; ++i) srcs.push_back(rng.randint(0, n - 1));

    // Correctness first: identical distances from every source used.
    for (int s : srcs) {
        const RouteResult a = dijkstra_array(g, s);
        const RouteResult h = dijkstra_heap(g, s);
        for (int v = 0; v < n; ++v)
            if (std::fabs(a.dist[v] - h.dist[v]) > 1e-9) g_all_match = false;
    }
    const double base = time_ms([&] {
        double acc = 0;
        for (int s : srcs) acc += dijkstra_array(g, s).dist[n - 1];
        g_sink = acc;
    });
    const double fast = time_ms([&] {
        double acc = 0;
        for (int s : srcs) acc += dijkstra_heap(g, s).dist[n - 1];
        g_sink = acc;
    });
    report("Routing, " + std::to_string(n) + " venues", std::to_string(sources) + " full route solves", base, fast);
}

// ---- 2. DP allocation
static void bench_dp(int m, int budget, int kmax, int solves, uint64_t seed) {
    Rng rng(seed);
    std::vector<std::vector<AssetOption>> instances;
    for (int s = 0; s < solves; ++s) {
        std::vector<AssetOption> opts(m);
        for (int i = 0; i < m; ++i) {
            opts[i].value.assign(static_cast<size_t>(kmax) + 1, 0.0);
            const double alpha = rng.uniform(5.0, 40.0), lam = rng.uniform(0.5, 3.0), fixed = rng.uniform(0.0, 15.0);
            for (int k = 1; k <= kmax; ++k) opts[i].value[k] = k * alpha - 0.5 * lam * k * k - fixed;
        }
        instances.push_back(std::move(opts));
    }
    for (const auto& inst : instances)
        if (std::fabs(dp_value_topdown_hash(inst, budget) - dp_value_rolling(inst, budget)) > 1e-9) g_all_match = false;

    const double base = time_ms([&] {
        double acc = 0;
        for (const auto& inst : instances) acc += dp_value_topdown_hash(inst, budget);
        g_sink = acc;
    });
    const double fast = time_ms([&] {
        double acc = 0;
        for (const auto& inst : instances) acc += dp_value_rolling(inst, budget);
        g_sink = acc;
    });
    report("DP, " + std::to_string(m) + " assets, " + std::to_string(budget) + " lots",
           std::to_string(solves) + " allocations", base, fast);
}

// ---- 3. End to end: costs change every tick, so route + allocate every tick
static void bench_pipeline(int nodes, int ticks, int m, int budget, int kmax, uint64_t seed) {
    const Graph base_graph = generate_network(nodes, 6, seed);
    const std::vector<AssetSpec> specs = generate_assets(nodes, 0, m, seed + 1);

    // Pre-compute the cost vector of every tick so both pipelines see exactly
    // the same inputs and the perturbation itself is not timed.
    std::vector<std::vector<double>> tick_costs;
    Rng rng(seed + 2);
    for (int t = 0; t < ticks; ++t) {
        std::vector<double> c = base_graph.cost;
        for (double& x : c) x *= rng.uniform(0.9, 1.1);
        tick_costs.push_back(std::move(c));
    }

    Graph g = base_graph;
    for (int t = 0; t < ticks; ++t) {
        g.cost = tick_costs[t];
        if (std::fabs(pipeline_baseline(g, specs, 0, budget, kmax) - pipeline_fast(g, specs, 0, budget, kmax)) > 1e-9)
            g_all_match = false;
    }
    const double base = time_ms([&] {
        double acc = 0;
        for (int t = 0; t < ticks; ++t) {
            g.cost = tick_costs[t];
            acc += pipeline_baseline(g, specs, 0, budget, kmax);
        }
        g_sink = acc;
    });
    const double fast = time_ms([&] {
        double acc = 0;
        for (int t = 0; t < ticks; ++t) {
            g.cost = tick_costs[t];
            acc += pipeline_fast(g, specs, 0, budget, kmax);
        }
        g_sink = acc;
    });
    report("Full pipeline, " + std::to_string(nodes) + " venues",
           std::to_string(ticks) + " ticks (route + allocate)", base, fast);
}

int main() {
#if defined(__VERSION__)
    const std::string compiler = __VERSION__;
#elif defined(_MSC_VER)
    const std::string compiler = "MSVC " + std::to_string(_MSC_VER);
#else
    const std::string compiler = "unknown compiler";
#endif
    std::printf("Benchmark: median of %d runs after one warm-up. Compiler: %s\n\n", REPS, compiler.c_str());

    bench_routing(500, 200, 11);    // the project's network size
    bench_routing(2000, 40, 12);    // how it scales
    bench_routing(5000, 10, 13);
    bench_dp(30, 300, 30, 20, 21);
    bench_pipeline(500, 100, 30, 300, 30, 31);

    std::printf("\nOutputs identical between baseline and optimised in every run: %s\n", g_all_match ? "YES" : "NO  <-- do not quote these numbers");

    std::printf("\nMarkdown table for the README (numbers are from THIS machine):\n\n");
    std::printf("| Benchmark | Work per run | Baseline | Optimised | Speed-up | Time cut |\n|---|---|---|---|---|---|\n");
    for (const Row& r : rows)
        std::printf("| %s | %s | %.2f ms | %.2f ms | %.1fx | %.0f%% |\n", r.what.c_str(), r.work.c_str(), r.base_ms,
                    r.fast_ms, r.base_ms / r.fast_ms, 100.0 * (1.0 - r.fast_ms / r.base_ms));
    return g_all_match ? 0 : 1;
}