#pragma once

#include <bits/stdc++.h>

namespace sim {

struct AssetOption {
    std::vector<double> value;  // value[k] for k = 0..K, value[0] == 0
};

struct Allocation {
    double total = 0.0;
    std::vector<int> lots;  // lots[i] chosen for asset i
};

// ---------------------------------------------------------------- reference

// Tries every combination. Exponential: only for tiny test instances.
inline void brute_rec(const std::vector<AssetOption>& opts, size_t i, int left,
                      double acc, std::vector<int>& cur, Allocation& best) {
    if (i == opts.size()) {
        if (acc > best.total) {
            best.total = acc;
            best.lots = cur;
        }
        return;
    }
    const int kmax = static_cast<int>(opts[i].value.size()) - 1;
    for (int k = 0; k <= kmax && k <= left; ++k) {
        cur[i] = k;
        brute_rec(opts, i + 1, left - k, acc + opts[i].value[k], cur, best);
    }
    cur[i] = 0;
}

inline Allocation allocate_bruteforce(const std::vector<AssetOption>& opts, int budget) {
    Allocation best;
    best.total = 0.0;
    best.lots.assign(opts.size(), 0);
    std::vector<int> cur(opts.size(), 0);
    brute_rec(opts, 0, budget, 0.0, cur, best);
    return best;
}

// ---------------------------------------------------------------- baseline

// Top-down recursion with a hash-map memo. Correct, but each state costs a
// hash lookup and the recursion adds call overhead. BASELINE for benchmarks.
// f(i, b) = best value using assets i.. with at most b lots left.
inline double topdown_rec(const std::vector<AssetOption>& opts, size_t i, int b,
                          std::unordered_map<long long, double>& memo, int budget) {
    if (i == opts.size()) return 0.0;
    const long long key = static_cast<long long>(i) * (budget + 1) + b;
    auto it = memo.find(key);
    if (it != memo.end()) return it->second;
    const int kmax = static_cast<int>(opts[i].value.size()) - 1;
    double best = 0.0;  // k = 0
    for (int k = 1; k <= kmax && k <= b; ++k) {
        const double cand = opts[i].value[k] + topdown_rec(opts, i + 1, b - k, memo, budget);
        if (cand > best) best = cand;
    }
    // k = 0 case must also include the rest of the assets:
    const double skip = topdown_rec(opts, i + 1, b, memo, budget);
    if (skip > best) best = skip;
    memo.emplace(key, best);
    return best;
}

inline double dp_value_topdown_hash(const std::vector<AssetOption>& opts, int budget) {
    std::unordered_map<long long, double> memo;
    return topdown_rec(opts, 0, budget, memo, budget);
}

// ---------------------------------------------------------------- optimised

// Bottom-up with one flat array. dp[b] = best value using at most b lots over
// the assets processed so far. Going through b from high to low lets us
// update the array in place, because dp[b-k] (k >= 1) has not been touched
// yet for this asset. Memory: O(B) instead of O(M*B).
inline double dp_value_rolling(const std::vector<AssetOption>& opts, int budget) {
    std::vector<double> dp(static_cast<size_t>(budget) + 1, 0.0);
    for (const AssetOption& a : opts) {
        const int kmax = static_cast<int>(a.value.size()) - 1;
        for (int b = budget; b >= 1; --b) {
            double best = dp[b];  // k = 0
            const int lim = kmax < b ? kmax : b;
            for (int k = 1; k <= lim; ++k) {
                const double cand = dp[b - k] + a.value[k];
                if (cand > best) best = cand;
            }
            dp[b] = best;
        }
    }
    return dp[budget];
}

// Full answer (how many lots of each asset) via a table of choices.
// f[i+1][b] = max over k of f[i][b-k] + value[i][k]; choice[i][b] remembers k.
// Ties go to the smaller k (fewer lots).
inline Allocation allocate_dp(const std::vector<AssetOption>& opts, int budget) {
    const size_t m = opts.size();
    const size_t w = static_cast<size_t>(budget) + 1;
    std::vector<double> prev(w, 0.0), cur(w, 0.0);
    std::vector<int> choice(m * w, 0);

    for (size_t i = 0; i < m; ++i) {
        const int kmax = static_cast<int>(opts[i].value.size()) - 1;
        for (int b = 0; b <= budget; ++b) {
            double best = prev[b];  // k = 0
            int best_k = 0;
            const int lim = kmax < b ? kmax : b;
            for (int k = 1; k <= lim; ++k) {
                const double cand = prev[b - k] + opts[i].value[k];
                if (cand > best) {
                    best = cand;
                    best_k = k;
                }
            }
            cur[b] = best;
            choice[i * w + b] = best_k;
        }
        prev.swap(cur);
    }

    Allocation out;
    out.total = prev[budget];
    out.lots.assign(m, 0);
    int b = budget;
    for (size_t i = m; i-- > 0;) {
        const int k = choice[i * w + b];
        out.lots[i] = k;
        b -= k;
    }
    return out;
}

// ---------------------------------------------------------------- greedy

// Repeatedly buys the single lot with the best marginal gain while that gain
// is positive. Optimal ONLY when every asset's value is concave. Kept so the
// tests can show where it fails.
inline Allocation allocate_greedy(const std::vector<AssetOption>& opts, int budget) {
    Allocation out;
    out.lots.assign(opts.size(), 0);
    using Entry = std::pair<double, size_t>;  // (marginal gain of next lot, asset)
    std::priority_queue<Entry> pq;
    auto push_next = [&](size_t i) {
        const int k = out.lots[i];
        if (k + 1 < static_cast<int>(opts[i].value.size()))
            pq.push({opts[i].value[k + 1] - opts[i].value[k], i});
    };
    for (size_t i = 0; i < opts.size(); ++i) push_next(i);
    for (int step = 0; step < budget && !pq.empty(); ++step) {
        const Entry top = pq.top();
        pq.pop();
        if (top.first <= 0.0) break;
        out.total += top.first;
        ++out.lots[top.second];
        push_next(top.second);
    }
    return out;
}

}  // namespace sim