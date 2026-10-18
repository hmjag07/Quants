#pragma once

#include <bits/stdc++.h>

#include "network.hpp"

namespace sim {

constexpr double INF = std::numeric_limits<double>::infinity();

struct RouteResult {
    std::vector<double> dist;  // dist[v] = least cost from src to v (INF if unreachable)
    std::vector<int> parent;   // parent[v] = previous node on a cheapest path (-1 for src / unreachable)
};

inline void check_source(const Graph& g, int src) {
    if (src < 0 || src >= g.n) throw std::out_of_range("source node out of range");
}

// Baseline: each round scans all nodes to find the closest unfinished one.
inline RouteResult dijkstra_array(const Graph& g, int src) {
    check_source(g, src);
    RouteResult r;
    r.dist.assign(g.n, INF);
    r.parent.assign(g.n, -1);
    std::vector<char> done(g.n, 0);
    r.dist[src] = 0.0;

    for (int round = 0; round < g.n; ++round) {
        int u = -1;
        double best = INF;
        for (int i = 0; i < g.n; ++i) {
            if (!done[i] && r.dist[i] < best) {
                best = r.dist[i];
                u = i;
            }
        }
        if (u < 0) break;  // everything left is unreachable
        done[u] = 1;
        for (int e = g.offset[u]; e < g.offset[u + 1]; ++e) {
            const int v = g.to[e];
            const double nd = r.dist[u] + g.cost[e];
            if (nd < r.dist[v]) {
                r.dist[v] = nd;
                r.parent[v] = u;
            }
        }
    }
    return r;
}

// Optimised: a min-heap gives the closest unfinished node in O(log V).
// Instead of decreasing a key we push a new entry and skip stale ones.
// If `target` is given (>= 0) we stop as soon as the target is finished; then
// only dist[target] (and the parents along its path) are guaranteed final.
inline RouteResult dijkstra_heap(const Graph& g, int src, int target = -1) {
    check_source(g, src);
    RouteResult r;
    r.dist.assign(g.n, INF);
    r.parent.assign(g.n, -1);
    using Entry = std::pair<double, int>;  // (cost so far, node)
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
    r.dist[src] = 0.0;
    pq.push({0.0, src});

    while (!pq.empty()) {
        const Entry top = pq.top();
        pq.pop();
        const double d = top.first;
        const int u = top.second;
        if (d > r.dist[u]) continue;  // stale entry: a cheaper one was already processed
        if (u == target) break;
        for (int e = g.offset[u]; e < g.offset[u + 1]; ++e) {
            const int v = g.to[e];
            const double nd = d + g.cost[e];
            if (nd < r.dist[v]) {
                r.dist[v] = nd;
                r.parent[v] = u;
                pq.push({nd, v});
            }
        }
    }
    return r;
}

// Reference implementation for the tests: relax every edge up to V-1 times.
inline RouteResult bellman_ford(const Graph& g, int src) {
    check_source(g, src);
    RouteResult r;
    r.dist.assign(g.n, INF);
    r.parent.assign(g.n, -1);
    r.dist[src] = 0.0;
    for (int round = 0; round < g.n - 1; ++round) {
        bool changed = false;
        for (int u = 0; u < g.n; ++u) {
            if (r.dist[u] == INF) continue;
            for (int e = g.offset[u]; e < g.offset[u + 1]; ++e) {
                const double nd = r.dist[u] + g.cost[e];
                if (nd < r.dist[g.to[e]]) {
                    r.dist[g.to[e]] = nd;
                    r.parent[g.to[e]] = u;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }
    return r;
}

// Cheapest route src -> dst as a list of nodes (src first). Empty if dst is
// unreachable.
inline std::vector<int> reconstruct_path(const RouteResult& r, int src, int dst) {
    std::vector<int> path;
    if (dst < 0 || dst >= static_cast<int>(r.dist.size())) return path;
    if (r.dist[dst] == INF) return path;
    int v = dst;
    for (size_t steps = 0; steps <= r.dist.size(); ++steps) {
        path.push_back(v);
        if (v == src) {
            std::reverse(path.begin(), path.end());
            return path;
        }
        v = r.parent[v];
        if (v < 0) break;
    }
    path.clear();  // broken parent chain: should never happen
    return path;
}

// Total cost of walking a path, using the cheapest parallel edge for each hop.
// Throws if two consecutive nodes are not joined by an edge.
inline double path_cost(const Graph& g, const std::vector<int>& path) {
    double total = 0.0;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        const int u = path[i];
        const int v = path[i + 1];
        double best = INF;
        for (int e = g.offset[u]; e < g.offset[u + 1]; ++e)
            if (g.to[e] == v && g.cost[e] < best) best = g.cost[e];
        if (best == INF) throw std::invalid_argument("path uses a missing edge");
        total += best;
    }
    return total;
}

}  // namespace sim