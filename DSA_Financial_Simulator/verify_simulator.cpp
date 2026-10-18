#include <bits/stdc++.h>

#include "pipeline.hpp"

using namespace sim;

static int g_total = 0;
static int g_failed = 0;

static void check(const std::string& name, bool ok) {
    ++g_total;
    if (!ok) ++g_failed;
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
}

static bool near(double a, double b, double eps = 1e-9) {
    if (a == b) return true;  // also covers INF == INF
    return std::fabs(a - b) <= eps * (1.0 + std::fmax(std::fabs(a), std::fabs(b)));
}

static bool throws_invalid(int n, const std::vector<EdgeRec>& edges) {
    try {
        build_graph(n, edges);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// Random graph with parallel edges, zero-cost edges and unreachable nodes.
static Graph random_graph(Rng& rng) {
    const int n = rng.randint(2, 30);
    const int m = rng.randint(0, n * 3);
    std::vector<EdgeRec> edges;
    for (int i = 0; i < m; ++i) {
        const int u = rng.randint(0, n - 1);
        const int v = rng.randint(0, n - 1);
        if (u == v) continue;
        const double c = rng.uniform() < 0.1 ? 0.0 : rng.uniform(0.0, 10.0);
        edges.push_back({u, v, c});
    }
    return build_graph(n, edges);
}

static std::vector<AssetOption> random_options(Rng& rng, int m, int kmax, bool concave) {
    std::vector<AssetOption> opts(m);
    for (int i = 0; i < m; ++i) {
        const int K = rng.randint(1, kmax);
        opts[i].value.assign(K + 1, 0.0);
        if (concave) {
            double gain = rng.uniform(-5.0, 20.0);
            for (int k = 1; k <= K; ++k) {
                opts[i].value[k] = opts[i].value[k - 1] + gain;
                gain -= rng.uniform(0.0, 5.0);  // marginal gains never increase
            }
        } else {
            for (int k = 1; k <= K; ++k) opts[i].value[k] = rng.uniform(-10.0, 25.0);
        }
    }
    return opts;
}

static double value_of(const std::vector<AssetOption>& opts, const std::vector<int>& lots) {
    double s = 0.0;
    for (size_t i = 0; i < opts.size(); ++i) s += opts[i].value[lots[i]];
    return s;
}

// BFS reachability on the forward graph (or the reversed graph).
static int reach_count(const Graph& g, int start, bool reversed) {
    std::vector<std::vector<int>> adj(g.n);
    for (int u = 0; u < g.n; ++u)
        for (int e = g.offset[u]; e < g.offset[u + 1]; ++e) {
            if (reversed) adj[g.to[e]].push_back(u);
            else adj[u].push_back(g.to[e]);
        }
    std::vector<char> seen(g.n, 0);
    std::vector<int> stack{start};
    seen[start] = 1;
    int count = 1;
    while (!stack.empty()) {
        const int u = stack.back();
        stack.pop_back();
        for (int v : adj[u])
            if (!seen[v]) {
                seen[v] = 1;
                ++count;
                stack.push_back(v);
            }
    }
    return count;
}

int main() {
    std::printf("=== Network ===\n");
    {
        const Graph a = generate_network(500, 6, 7);
        const Graph b = generate_network(500, 6, 7);
        const Graph c = generate_network(500, 6, 8);
        check("same seed gives an identical network", a.offset == b.offset && a.to == b.to && a.cost == b.cost);
        check("different seed gives a different network", a.cost != c.cost);
        check("500 nodes, average out-degree 6 -> 3000 edges", a.n == 500 && a.num_edges() == 3000);

        bool ok = a.offset.size() == 501 && a.offset[0] == 0 && a.offset[500] == a.num_edges();
        for (int i = 0; i < a.n; ++i) ok = ok && a.offset[i] <= a.offset[i + 1];
        check("CSR offsets are monotone and cover every edge", ok);

        bool edges_ok = true;
        for (int u = 0; u < a.n; ++u)
            for (int e = a.offset[u]; e < a.offset[u + 1]; ++e)
                edges_ok = edges_ok && a.to[e] != u && a.to[e] >= 0 && a.to[e] < a.n && a.cost[e] >= 0.0 &&
                           std::isfinite(a.cost[e]);
        check("no self-loops, valid endpoints, costs finite and >= 0", edges_ok);

        check("every venue reachable from venue 0", reach_count(a, 0, false) == a.n);
        check("venue 0 reachable from every venue", reach_count(a, 0, true) == a.n);

        check("negative cost rejected", throws_invalid(3, {{0, 1, -1.0}}));
        check("NaN cost rejected", throws_invalid(3, {{0, 1, std::nan("")}}));
        check("infinite cost rejected", throws_invalid(3, {{0, 1, INF}}));
        check("out-of-range endpoint rejected", throws_invalid(3, {{0, 3, 1.0}}));
        check("zero nodes rejected", throws_invalid(0, {}));
    }

    std::printf("\n=== Routing: hand-worked cases ===\n");
    {
        // Textbook directed graph. From node 0 the cheapest costs are:
        // 1:7  2:9  5:11 (via 2)  3:20 (via 2)  4:20 (via 2,5)
        const Graph g = build_graph(6, {{0, 1, 7}, {0, 2, 9}, {0, 5, 14}, {1, 2, 10}, {1, 3, 15},
                                        {2, 3, 11}, {2, 5, 2}, {3, 4, 6}, {5, 4, 9}});
        for (int variant = 0; variant < 3; ++variant) {
            const RouteResult r = variant == 0 ? dijkstra_array(g, 0)
                                  : variant == 1 ? dijkstra_heap(g, 0)
                                                 : bellman_ford(g, 0);
            const char* nm = variant == 0 ? "array" : variant == 1 ? "heap" : "bellman-ford";
            const bool d_ok = near(r.dist[0], 0) && near(r.dist[1], 7) && near(r.dist[2], 9) &&
                              near(r.dist[3], 20) && near(r.dist[4], 20) && near(r.dist[5], 11);
            check(std::string("textbook graph distances correct (") + nm + ")", d_ok);
            const std::vector<int> p = reconstruct_path(r, 0, 4);
            check(std::string("textbook graph path 0->4 is 0,2,5,4 (") + nm + ")",
                  p == std::vector<int>({0, 2, 5, 4}));
        }

        // Direct edge is expensive; a two-hop detour is cheaper.
        const Graph h = build_graph(3, {{0, 2, 100}, {0, 1, 1}, {1, 2, 1}});
        const RouteResult r = dijkstra_heap(h, 0);
        check("detour (cost 2) beats the direct edge (cost 100)", near(r.dist[2], 2) &&
                                                                   reconstruct_path(r, 0, 2) == std::vector<int>({0, 1, 2}));

        // Unreachable node.
        const Graph u = build_graph(4, {{0, 1, 1}, {1, 2, 1}});
        for (int variant = 0; variant < 2; ++variant) {
            const RouteResult ru = variant == 0 ? dijkstra_array(u, 0) : dijkstra_heap(u, 0);
            check(std::string("unreachable node has INF distance and empty path (") + (variant ? "heap" : "array") + ")",
                  ru.dist[3] == INF && reconstruct_path(ru, 0, 3).empty());
        }

        // Zero-cost chain and parallel edges.
        const Graph z = build_graph(3, {{0, 1, 0}, {1, 2, 0}});
        check("zero-cost edges give zero distance", near(dijkstra_heap(z, 0).dist[2], 0.0));
        const Graph par = build_graph(2, {{0, 1, 5}, {0, 1, 3}, {0, 1, 4}});
        const RouteResult rp = dijkstra_heap(par, 0);
        check("parallel edges: the cheapest is used", near(rp.dist[1], 3) && near(path_cost(par, {0, 1}), 3));

        check("distance from source to itself is 0", near(dijkstra_heap(g, 3).dist[3], 0.0));
        bool threw = false;
        try {
            dijkstra_heap(g, 6);
        } catch (const std::out_of_range&) {
            threw = true;
        }
        check("source out of range throws", threw);
        threw = false;
        try {
            path_cost(g, {0, 4});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check("path_cost throws on a missing edge", threw);
    }

    std::printf("\n=== Routing: randomised cross-checks ===\n");
    {
        Rng rng(2024);
        int agree_ab = 0, agree_bf = 0, certificate = 0, path_ok = 0, early_ok = 0;
        const int trials = 300;
        for (int t = 0; t < trials; ++t) {
            const Graph g = random_graph(rng);
            const int src = rng.randint(0, g.n - 1);
            const RouteResult a = dijkstra_array(g, src);
            const RouteResult h = dijkstra_heap(g, src);
            const RouteResult b = bellman_ford(g, src);

            bool ab = true, bf = true;
            for (int v = 0; v < g.n; ++v) {
                ab = ab && near(a.dist[v], h.dist[v]);
                bf = bf && near(h.dist[v], b.dist[v]);
            }
            agree_ab += ab;
            agree_bf += bf;

            // Optimality certificate: no edge can still improve a distance.
            bool cert = near(h.dist[src], 0.0);
            for (int u = 0; u < g.n; ++u) {
                if (h.dist[u] == INF) continue;
                for (int e = g.offset[u]; e < g.offset[u + 1]; ++e)
                    cert = cert && h.dist[g.to[e]] <= h.dist[u] + g.cost[e] + 1e-9;
            }
            certificate += cert;

            // Every reconstructed path starts at src, ends at dst, uses real
            // edges, and its cost equals the reported distance.
            bool pok = true;
            for (int v = 0; v < g.n; ++v) {
                const std::vector<int> p = reconstruct_path(h, src, v);
                if (h.dist[v] == INF) {
                    pok = pok && p.empty();
                } else {
                    pok = pok && !p.empty() && p.front() == src && p.back() == v &&
                          near(path_cost(g, p), h.dist[v]);
                }
            }
            path_ok += pok;

            const int target = rng.randint(0, g.n - 1);
            early_ok += near(dijkstra_heap(g, src, target).dist[target], h.dist[target]);
        }
        check("array and heap Dijkstra agree on 300 random graphs", agree_ab == trials);
        check("heap Dijkstra matches Bellman-Ford on 300 random graphs", agree_bf == trials);
        check("optimality certificate holds (no edge improves any distance)", certificate == trials);
        check("reconstructed paths are valid and cost exactly the distance", path_ok == trials);
        check("early-exit Dijkstra gives the same target distance", early_ok == trials);

        // And on the real 500-node network.
        const Graph net = generate_network(500, 6, 99);
        bool all_sources = true;
        for (int src : {0, 17, 250, 499}) {
            const RouteResult a = dijkstra_array(net, src);
            const RouteResult h = dijkstra_heap(net, src);
            const RouteResult b = bellman_ford(net, src);
            for (int v = 0; v < net.n; ++v) all_sources = all_sources && near(a.dist[v], h.dist[v]) && near(h.dist[v], b.dist[v]);
        }
        check("all three algorithms agree on the 500-node network", all_sources);
    }

    std::printf("\n=== DP allocation: hand-worked cases ===\n");
    {
        // asset 0: [0,5,8,9]  asset 1: [0,4,6,6.5]  budget 3.
        // (3,0)=9  (2,1)=12  (1,2)=11  (0,3)=6.5  -> best is 12 with lots {2,1}.
        const std::vector<AssetOption> opts = {{{0, 5, 8, 9}}, {{0, 4, 6, 6.5}}};
        const Allocation a = allocate_dp(opts, 3);
        check("hand example: total 12 with lots {2,1}", near(a.total, 12.0) && a.lots == std::vector<int>({2, 1}));
        check("hand example: rolling DP gives 12", near(dp_value_rolling(opts, 3), 12.0));
        check("hand example: top-down DP gives 12", near(dp_value_topdown_hash(opts, 3), 12.0));
        check("hand example: brute force gives 12", near(allocate_bruteforce(opts, 3).total, 12.0));

        // Fixed-fee example: the first lot of asset 0 loses 1, the second makes
        // +11. Greedy never starts it; DP does.
        const std::vector<AssetOption> trap = {{{0, -1, 10}}, {{0, 1, 1.5}}};
        const Allocation dp = allocate_dp(trap, 2);
        const Allocation gr = allocate_greedy(trap, 2);
        check("fixed-fee trap: DP finds 10 (both lots in asset 0)", near(dp.total, 10.0) && dp.lots == std::vector<int>({2, 0}));
        check("fixed-fee trap: greedy only reaches 1.5", near(gr.total, 1.5));

        check("budget 0 buys nothing", allocate_dp(opts, 0).total == 0.0 && allocate_dp(opts, 0).lots == std::vector<int>({0, 0}));
        const std::vector<AssetOption> bad = {{{0, -3, -4}}, {{0, -1}}};
        const Allocation neg = allocate_dp(bad, 5);
        check("all-negative assets: buy nothing, total 0", neg.total == 0.0 && neg.lots == std::vector<int>({0, 0}));
        check("no assets: total 0", allocate_dp({}, 10).total == 0.0 && dp_value_rolling({}, 10) == 0.0);
    }

    std::printf("\n=== DP allocation: randomised cross-checks ===\n");
    {
        Rng rng(555);
        int bf_ok = 0, valid_ok = 0, three_ok = 0, greedy_ok = 0, mono_ok = 0;
        const int trials = 300;
        for (int t = 0; t < trials; ++t) {
            const int m = rng.randint(1, 5);
            const int B = rng.randint(0, 8);
            const std::vector<AssetOption> opts = random_options(rng, m, 4, false);
            const Allocation dp = allocate_dp(opts, B);
            const Allocation bf = allocate_bruteforce(opts, B);
            bf_ok += near(dp.total, bf.total);

            int used = 0;
            bool valid = dp.lots.size() == opts.size();
            for (size_t i = 0; valid && i < opts.size(); ++i) {
                used += dp.lots[i];
                valid = dp.lots[i] >= 0 && dp.lots[i] < static_cast<int>(opts[i].value.size());
            }
            valid = valid && used <= B && near(value_of(opts, dp.lots), dp.total);
            valid_ok += valid;

            mono_ok += allocate_dp(opts, B + 1).total >= dp.total - 1e-12;
        }
        for (int t = 0; t < trials; ++t) {
            const std::vector<AssetOption> opts = random_options(rng, rng.randint(1, 15), 12, false);
            const int B = rng.randint(0, 60);
            const double a = allocate_dp(opts, B).total;
            three_ok += near(a, dp_value_rolling(opts, B)) && near(a, dp_value_topdown_hash(opts, B));

            const std::vector<AssetOption> conc = random_options(rng, rng.randint(1, 15), 12, true);
            greedy_ok += near(allocate_greedy(conc, B).total, allocate_dp(conc, B).total, 1e-8);
        }
        check("DP matches brute force on 300 random small instances (incl. negative values)", bf_ok == trials);
        check("returned lots respect the budget and the per-asset limit, and add up to the total", valid_ok == trials);
        check("more budget never lowers the optimum", mono_ok == trials);
        check("bottom-up, rolling and top-down DP agree on 300 larger instances", three_ok == trials);
        check("greedy equals DP when every asset is concave (300 instances)", greedy_ok == trials);
    }

    std::printf("\n=== Pipeline (routing + allocation together) ===\n");
    {
        const int nodes = 500, cash = 0, B = 100, K = 30;
        const Graph g = generate_network(nodes, 6, 42);
        const std::vector<AssetSpec> specs = generate_assets(nodes, cash, 20, 43);

        bool distinct = specs.size() == 20;
        std::vector<char> used(nodes, 0);
        for (const AssetSpec& s : specs) {
            distinct = distinct && s.node != cash && s.node >= 0 && s.node < nodes && !used[s.node];
            if (s.node >= 0 && s.node < nodes) used[s.node] = 1;
        }
        check("assets sit on 20 distinct venues, none is the cash node", distinct);

        const RouteResult r = dijkstra_heap(g, cash);
        const std::vector<AssetOption> opts = build_options(specs, r.dist, K);
        const AssetSpec& s0 = specs[0];
        const double c0 = r.dist[s0.node];
        const double expect = 3 * (s0.alpha_bps - c0) - 0.5 * s0.lambda * 9 - s0.fixed;
        check("value table: v(0)=0 and v(3) matches the formula", opts[0].value[0] == 0.0 && near(opts[0].value[3], expect));

        const double base = pipeline_baseline(g, specs, cash, B, K);
        const double fast = pipeline_fast(g, specs, cash, B, K);
        const Allocation full = allocate_dp(opts, B);
        check("baseline pipeline, fast pipeline and full allocation give the same total",
              near(base, fast) && near(fast, full.total));
        check("the pipeline is deterministic", near(fast, pipeline_fast(g, specs, cash, B, K)));
        int lots_used = 0;
        for (int k : full.lots) lots_used += k;
        check("allocation uses no more than the budget", lots_used <= B);
        check("total value is never negative (doing nothing is allowed)", full.total >= 0.0);

        // Making every route out of the cash node 5 bps dearer can only hurt.
        Graph worse = g;
        for (int e = worse.offset[cash]; e < worse.offset[cash + 1]; ++e) worse.cost[e] += 5.0;
        check("dearer routing never increases the optimum", pipeline_fast(worse, specs, cash, B, K) <= fast + 1e-9);

        // An unreachable venue is never bought.
        const Graph broken = build_graph(4, {{0, 1, 1}, {1, 0, 1}});  // venues 2 and 3 unreachable
        const std::vector<AssetSpec> two = {{1, 30.0, 0.1, 0.0}, {3, 1000.0, 0.0, 0.0}};
        const RouteResult rb = dijkstra_heap(broken, 0);
        const Allocation ab = allocate_dp(build_options(two, rb.dist, 10), 10);
        check("an unreachable venue gets zero lots even if its alpha is huge", ab.lots[1] == 0 && ab.lots[0] > 0);
    }

    std::printf("\n%d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}