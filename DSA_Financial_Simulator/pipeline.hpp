#pragma once

#include <bits/stdc++.h>

#include "network.hpp"
#include "portfolio_dp.hpp"
#include "routing.hpp"

namespace sim {

struct AssetSpec {
    int node;          // venue where the asset trades
    double alpha_bps;  // expected edge per lot
    double lambda;     // quadratic size penalty
    double fixed;      // fixed cost of opening the position
};

// m assets on m distinct venues, none of them the cash node.
inline std::vector<AssetSpec> generate_assets(int n_nodes, int cash_node, int m, uint64_t seed) {
    if (m < 1 || m > n_nodes - 1) throw std::invalid_argument("bad number of assets");
    Rng rng(seed);
    std::vector<int> pool;
    pool.reserve(static_cast<size_t>(n_nodes) - 1);
    for (int v = 0; v < n_nodes; ++v)
        if (v != cash_node) pool.push_back(v);
    // Partial Fisher-Yates shuffle: the first m entries are a uniform sample.
    for (int i = 0; i < m; ++i) {
        const int j = rng.randint(i, static_cast<int>(pool.size()) - 1);
        std::swap(pool[i], pool[j]);
    }
    std::vector<AssetSpec> specs;
    for (int i = 0; i < m; ++i)
        specs.push_back({pool[i], rng.uniform(5.0, 40.0), rng.uniform(0.5, 3.0), rng.uniform(0.0, 15.0)});
    return specs;
}

// route_cost[v] = cheapest cost from the cash node to venue v (INF if none).
// An unreachable venue gets a hugely negative value for k >= 1, so the DP
// never buys it.
inline std::vector<AssetOption> build_options(const std::vector<AssetSpec>& specs,
                                              const std::vector<double>& route_cost, int max_lots) {
    std::vector<AssetOption> opts(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        const AssetSpec& s = specs[i];
        const double c = route_cost[s.node];
        opts[i].value.assign(static_cast<size_t>(max_lots) + 1, 0.0);
        for (int k = 1; k <= max_lots; ++k) {
            if (!std::isfinite(c)) {
                opts[i].value[k] = -1e18;
            } else {
                opts[i].value[k] = k * (s.alpha_bps - c) - 0.5 * s.lambda * k * k - s.fixed;
            }
        }
    }
    return opts;
}

// Baseline pipeline: O(V^2) Dijkstra + hash-memoised top-down DP.
inline double pipeline_baseline(const Graph& g, const std::vector<AssetSpec>& specs, int cash,
                                int budget, int max_lots) {
    const RouteResult r = dijkstra_array(g, cash);
    return dp_value_topdown_hash(build_options(specs, r.dist, max_lots), budget);
}

// Optimised pipeline: heap Dijkstra + bottom-up rolling-array DP.
inline double pipeline_fast(const Graph& g, const std::vector<AssetSpec>& specs, int cash, int budget,
                            int max_lots) {
    const RouteResult r = dijkstra_heap(g, cash);
    return dp_value_rolling(build_options(specs, r.dist, max_lots), budget);
}

}  // namespace sim