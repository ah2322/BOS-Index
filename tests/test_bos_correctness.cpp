#include "bos/bos_index.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

static void prepareBulkData(const vector<string> &src_keys, char *&data_buf,
                            char **&key_ptrs, uint64_t *&val_ptrs) {
    uint64_t total_len = 0;
    for (const auto &k : src_keys) total_len += k.length() + 1;
    data_buf = new char[total_len];
    key_ptrs = new char *[src_keys.size()];
    val_ptrs = new uint64_t[src_keys.size()];
    uint64_t offset = 0;
    for (size_t i = 0; i < src_keys.size(); ++i) {
        key_ptrs[i] = data_buf + offset;
        val_ptrs[i] = i + 1;
        memcpy(data_buf + offset, src_keys[i].c_str(), src_keys[i].length() + 1);
        offset += src_keys[i].length() + 1;
    }
}

int main() {
    vector<string> base_keys;
    for (int i = 0; i < 3000; ++i) {
        char buf[64];
        snprintf(buf, sizeof(buf), "bb_base_%06d", i);
        base_keys.emplace_back(buf);
    }
    sort(base_keys.begin(), base_keys.end());

    char *base_data = nullptr;
    char **base_key_ptrs = nullptr;
    uint64_t *base_val_ptrs = nullptr;
    prepareBulkData(base_keys, base_data, base_key_ptrs, base_val_ptrs);

    bos::BOSIndex::Options opt;
    opt.prefix_len = 2;
    opt.bucket_flush_threshold = 1000;
    opt.local_merge_run_threshold = 2;
    opt.local_merge_size_threshold = 1800;
    opt.hot_access_threshold = 1000000;
    opt.monitor_interval = 100;

    bos::BOSIndex index(opt);
    index.bulkload_bos_index(base_key_ptrs, base_val_ptrs, (int)base_keys.size());

    uint64_t out = 0;
    if (!index.lookup_bos_index("bb_base_000123", out) || out != 124) {
        cerr << "base lookup failed" << endl;
        return 1;
    }

    // Prefix aa will create a delta run; prefix zz stays active.
    for (int i = 0; i < 1200; ++i) {
        char buf[64];
        snprintf(buf, sizeof(buf), "aa_insert_%06d", i);
        index.insert(buf, 100000 + i);
    }
    for (int i = 0; i < 200; ++i) {
        char buf[64];
        snprintf(buf, sizeof(buf), "zz_insert_%06d", i);
        index.insert(buf, 200000 + i);
    }

    if (!index.lookup_bos_index("aa_insert_000042", out) || out != 100042) {
        cerr << "delta lookup failed" << endl;
        return 2;
    }
    if (!index.lookup_bos_index("zz_insert_000042", out) || out != 200042) {
        cerr << "active lookup failed" << endl;
        return 3;
    }

    index.force_merge_all_buckets();

    if (!index.lookup_bos_index("aa_insert_000999", out) || out != 100999) {
        cerr << "post-merge aa lookup failed" << endl;
        return 4;
    }
    if (!index.lookup_bos_index("zz_insert_000042", out) || out != 200042) {
        cerr << "post-merge zz lookup failed" << endl;
        return 5;
    }

    index.print_stats();
    cout << "BOS correctness test passed" << endl;

    delete[] base_data;
    delete[] base_key_ptrs;
    delete[] base_val_ptrs;
    return 0;
}
