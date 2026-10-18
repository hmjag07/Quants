#pragma once

#include <bits/stdc++.h>
namespace sim {

class Rng {
public:
    explicit Rng(uint64_t seed) : s_(seed) {}

    uint64_t next_u64() {
        s_ += 0x9E3779B97F4A7C15ULL;
        uint64_t z = s_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    // Uniform in [0, 1).
    double uniform() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
    double uniform(double a, double b) { return a + (b - a) * uniform(); }
    // Uniform integer in [lo, hi], inclusive.
    int randint(int lo, int hi) {
        return lo + static_cast<int>(next_u64() % static_cast<uint64_t>(hi - lo + 1));
    }
    // Exponential with the given mean.
    double exponential(double mean) { return -mean * std::log(1.0 - uniform()); }

private:
    uint64_t s_;
};

struct EdgeRec {
    int from;
    int to;
    double cost;
};

struct Graph {
    int n = 0;
    std::vector<int> offset;   // size n + 1
    std::vector<int> to;       // size = number of edges
    std::vector<double> cost;  // size = number of edges, all >= 0

    int num_edges() const { return static_cast<int>(to.size()); }
};

// Builds a CSR graph from an edge list. Rejects bad endpoints and any cost
// that is negative, NaN or infinite (Dijkstra would silently give wrong
// answers on a negative edge, so we refuse it up front).
inline Graph build_graph(int n, const std::vector<EdgeRec>& edges) {
    if (n <= 0) throw std::invalid_argument("graph needs at least one node");
    Graph g;
    g.n = n;
    g.offset.assign(static_cast<size_t>(n) + 1, 0);
    for (const EdgeRec& e : edges) {
        if (e.from < 0 || e.from >= n || e.to < 0 || e.to >= n)
            throw std::invalid_argument("edge endpoint out of range");
        if (!(e.cost >= 0.0) || !std::isfinite(e.cost))
            throw std::invalid_argument("edge cost must be finite and >= 0");
        ++g.offset[static_cast<size_t>(e.from) + 1];
    }
    for (int i = 0; i < n; ++i) g.offset[i + 1] += g.offset[i];

    g.to.resize(edges.size());
    g.cost.resize(edges.size());
    std::vector<int> pos(g.offset.begin(), g.offset.end() - 1);
    for (const EdgeRec& e : edges) {
        const int p = pos[e.from]++;
        g.to[p] = e.to;
        g.cost[p] = e.cost;
    }
    return g;
}

// Random strongly connected network.
//  * A two-way ring 0-1-2-...-(n-1)-0 guarantees every venue can reach every
//    other venue. Ring hops are expensive (5-15 bps).
//  * Extra random shortcut edges are added until the average out-degree is
//    avg_out_degree. Shortcuts are cheap on average (0.3 bps + Exp(mean 2.5)),
//    so the cheapest route is usually NOT the direct or the ring route, which
//    is what makes the routing problem non-trivial.
// The costs are synthetic. They are not calibrated to any real market.
inline Graph generate_network(int n, int avg_out_degree, uint64_t seed) {
    if (n < 3) throw std::invalid_argument("need at least 3 nodes");
    if (avg_out_degree < 2) throw std::invalid_argument("avg_out_degree must be >= 2");
    Rng rng(seed);
    std::vector<EdgeRec> edges;
    const long long extra = static_cast<long long>(n) * (avg_out_degree - 2);
    edges.reserve(static_cast<size_t>(2LL * n + extra));

    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
        edges.push_back({i, j, rng.uniform(5.0, 15.0)});
        edges.push_back({j, i, rng.uniform(5.0, 15.0)});
    }
    for (long long k = 0; k < extra; ++k) {
        const int u = rng.randint(0, n - 1);
        int v = rng.randint(0, n - 2);
        if (v >= u) ++v;  // v != u, uniformly among the other nodes
        edges.push_back({u, v, 0.3 + rng.exponential(2.5)});
    }
    return build_graph(n, edges);
}

}  // namespace sim