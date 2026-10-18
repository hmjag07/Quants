#include <bits/stdc++.h>

#include "pipeline.hpp"

using namespace sim;

int main(int argc, char** argv) {
    int nodes = 500, assets = 20, budget = 100, max_lots = 30, degree = 6;
    uint64_t seed = 42;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const long long val = std::atoll(argv[i + 1]);
        if (key == "--nodes") nodes = static_cast<int>(val);
        else if (key == "--assets") assets = static_cast<int>(val);
        else if (key == "--lots") budget = static_cast<int>(val);
        else if (key == "--maxlots") max_lots = static_cast<int>(val);
        else if (key == "--degree") degree = static_cast<int>(val);
        else if (key == "--seed") seed = static_cast<uint64_t>(val);
        else {
            std::fprintf(stderr, "unknown option %s\n", key.c_str());
            return 2;
        }
    }

    try {
        const int cash = 0;
        const Graph g = generate_network(nodes, degree, seed);

        double cmin = INF, cmax = 0.0, csum = 0.0;
        for (double c : g.cost) {
            if (c < cmin) cmin = c;
            if (c > cmax) cmax = c;
            csum += c;
        }
        std::printf("Network: %d venues, %d directed edges, edge cost bps: min %.2f  mean %.2f  max %.2f\n",
                    g.n, g.num_edges(), cmin, csum / g.num_edges(), cmax);

        const std::vector<AssetSpec> specs = generate_assets(nodes, cash, assets, seed + 1);
        const RouteResult route = dijkstra_heap(g, cash);

        // Show one example route: to the first asset's venue.
        const int demo = specs[0].node;
        const std::vector<int> path = reconstruct_path(route, cash, demo);
        std::printf("\nExample route, venue %d -> venue %d (cheapest of all paths):\n  ", cash, demo);
        for (size_t i = 0; i < path.size(); ++i) std::printf("%d%s", path[i], i + 1 < path.size() ? " -> " : "");
        std::printf("\n  total cost %.3f bps over %zu hops\n", route.dist[demo], path.size() - 1);

        const std::vector<AssetOption> opts = build_options(specs, route.dist, max_lots);
        const Allocation alloc = allocate_dp(opts, budget);

        std::printf("\nAllocation of %d lots (max %d per asset):\n", budget, max_lots);
        std::printf("  %-6s %-6s %10s %10s %6s %12s\n", "asset", "venue", "route bps", "alpha bps", "lots", "net value");
        int used = 0;
        for (size_t i = 0; i < specs.size(); ++i) {
            const int k = alloc.lots[i];
            used += k;
            std::printf("  %-6zu %-6d %10.2f %10.2f %6d %12.2f\n", i, specs[i].node, route.dist[specs[i].node],
                        specs[i].alpha_bps, k, opts[i].value[k]);
        }
        std::printf("\nLots used: %d of %d.  Total net value: %.2f bps x lots\n", used, budget, alloc.total);

        // Sanity check: the fast pipeline and the baseline must agree.
        const double a = pipeline_baseline(g, specs, cash, budget, max_lots);
        const double b = pipeline_fast(g, specs, cash, budget, max_lots);
        std::printf("Cross-check baseline vs fast pipeline: %.6f vs %.6f -> %s\n", a, b,
                    std::fabs(a - b) < 1e-9 ? "match" : "MISMATCH");
        return std::fabs(a - b) < 1e-9 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}