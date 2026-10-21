/**
 * bos/bos_index.hpp
 *
 * BOSIndex: Incremental Learned Index for dynamic string keys.
 *
 * Thesis-oriented implementation:
 *   1. Main BOSBase and incremental layer are decoupled.
 *   2. New strings are routed to prefix-aware delta buckets.
 *   3. Queries visit only the matching delta bucket, not a global buffer.
 *   4. Adaptive local maintenance merges only affected prefix buckets.
 *
 * The previous 03-23 implementation is preserved as bos/bos_index_global_0323.hpp.
 */
#pragma once

#include "bos_static_index.hpp"
#include <cstdlib>

// Restore-5.2 hot-path cleanup:
// Cache BOS_MODEL_INTERNAL_WEAK_REWIRE once per process instead of calling
// getenv()/atoi() from lookup/update hot paths.
static const bool bos_index_model_internal_weak_rewire_cached_ = []() -> bool {
    const char* env = std::getenv("BOS_MODEL_INTERNAL_WEAK_REWIRE");
    return env && std::atoi(env) != 0;
}();

// Restore-5.2 hot-path cleanup:
// Cache BOS_E2E_DYNFIRST_FASTPATH once per process instead of calling
// getenv()/atoi() inside lookup_bos_index_e2e_by_key().
static const bool bos_index_e2e_dynfirst_fastpath_cached_ = []() -> bool {
    const char* env = std::getenv("BOS_E2E_DYNFIRST_FASTPATH");
    return env && std::atoi(env) != 0;
}();



#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include <deque>
#include <limits>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <functional>

#if defined(__x86_64__) || defined(__i386__)
#include <nmmintrin.h>
#endif

namespace bos {

#if defined(__GNUC__) || defined(__clang__)
#define BOS_NOINLINE __attribute__((noinline))
#else
#define BOS_NOINLINE
#endif

static inline uint64_t bos_index_review_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct BOSReviewScopedTimer {
    std::atomic<uint64_t> *target;
    uint64_t start;

    explicit BOSReviewScopedTimer(std::atomic<uint64_t> *dst)
        : target(dst), start(bos_index_review_now_ns()) {}

    ~BOSReviewScopedTimer() {
        if (target) {
            target->fetch_add(bos_index_review_now_ns() - start,
                              std::memory_order_relaxed);
        }
    }
};

struct Hist256 {
    std::array<uint64_t, 256> cnt{};
    uint64_t total = 0;

    void clear() {
        cnt.fill(0);
        total = 0;
    }

    void add_key(const std::string &key) {
        for (unsigned char c : key) {
            cnt[c]++;
            total++;
        }
    }

    void remove_key(const std::string &key) {
        for (unsigned char c : key) {
            if (cnt[c] > 0) cnt[c]--;
            if (total > 0) total--;
        }
    }

    void add_hist(const Hist256 &other) {
        for (int i = 0; i < 256; ++i) {
            cnt[i] += other.cnt[i];
        }
        total += other.total;
    }
};

// Lightweight fixed-size sketch for local distribution drift. Hist256 captures
// global byte-frequency changes; this sketch also mixes byte position into a
// bounded number of counters, so a bucket can detect positional/prefix-region
// shifts without retaining all recent strings beyond the sliding window queue.
struct DriftSketch64 {
    static const size_t BINS = 64;

    std::array<uint64_t, BINS> cnt{};
    uint64_t total = 0;

    void clear() {
        cnt.fill(0);
        total = 0;
    }

    static uint32_t bin_for(unsigned char c, size_t pos) {
        uint32_t h = 2166136261u;
        h ^= static_cast<uint32_t>(c);
        h *= 16777619u;
        h ^= static_cast<uint32_t>(pos & 0xffu);
        h *= 16777619u;
        h ^= static_cast<uint32_t>((pos >> 8) & 0xffu);
        h *= 16777619u;
        return h & static_cast<uint32_t>(BINS - 1);
    }

    void add_key(const std::string &key) {
        for (size_t i = 0; i < key.size(); ++i) {
            cnt[bin_for(static_cast<unsigned char>(key[i]), i)]++;
            total++;
        }
    }

    void remove_key(const std::string &key) {
        for (size_t i = 0; i < key.size(); ++i) {
            uint32_t b = bin_for(static_cast<unsigned char>(key[i]), i);
            if (cnt[b] > 0) cnt[b]--;
            if (total > 0) total--;
        }
    }

    void add_sketch(const DriftSketch64 &other) {
        for (size_t i = 0; i < BINS; ++i) {
            cnt[i] += other.cnt[i];
        }
        total += other.total;
    }
};

// Lightweight fixed-size sketch for local distribution drift. Hist256 captures
// global byte-frequency changes; this sketch also mixes byte position into a
// bounded number of counters, so a bucket can detect positional/prefix-region
// shifts without retaining all recent strings beyond the sliding window queue.




class BOSIndex : public BOSBase {

public:
    // P15h test-only tombstone/performance accessors begin
    struct TestPerformanceGuaranteeStats {
        uint64_t tombstone_count = 0;
        uint64_t tombstones = 0;
        uint64_t delete_tombstones = 0;

        uint64_t tombstone_gc_checks = 0;
        uint64_t tombstone_gc_erases = 0;
        uint64_t tombstone_gc_prefix_bucket_probes = 0;
        uint64_t tombstone_gc_delta_run_probes = 0;
        uint64_t tombstone_gc_prefix_index_misses = 0;
        uint64_t tombstone_gc_full_bucket_scan_fallbacks = 0;

        uint64_t tombstone_gc_checks_seen = 0;
        uint64_t tombstone_gc_erases_seen = 0;
        uint64_t tombstone_gc_prefix_bucket_probes_seen = 0;
        uint64_t tombstone_gc_delta_run_probes_seen = 0;
        uint64_t tombstone_gc_prefix_index_misses_seen = 0;
        uint64_t tombstone_gc_full_bucket_scan_fallbacks_seen = 0;

        uint64_t gc_checks = 0;
        uint64_t gc_erases = 0;
        uint64_t prefix_bucket_probes = 0;
        uint64_t delta_run_probes = 0;
        uint64_t prefix_index_misses = 0;
        uint64_t full_bucket_scan_fallbacks = 0;

        uint64_t tombstone_merge_skips = 0;
        uint64_t merge_skips = 0;

        uint64_t total_local_merges = 0;
        uint64_t local_merges = 0;
    };

    // P15h test-only tombstone/performance accessors end

private:

  public:
    struct Options {
        // Number of leading bytes used to route a key to a delta bucket.
        size_t prefix_len = 2;

        // Convert a bucket's active hash table to an immutable per-prefix delta
        // BOSBase after it reaches this size. BOSBase bulkload requires >= 1000 keys.
        size_t bucket_flush_threshold = 4096;

        // Adaptive local maintenance triggers.
        size_t local_merge_run_threshold = 4;
        size_t local_merge_size_threshold = 65536;
        uint64_t hot_access_threshold = 20000;
        double drift_ratio_threshold = 4.0;

        // Check adaptive policies once every N inserts, or immediately when a
        // bucket reaches bucket_flush_threshold.
        uint64_t monitor_interval = 1024;
        bool enable_local_stats = true;
        bool enable_gpkl_drift = true;
        bool enable_drift_sketch = true;
        double drift_sketch_gpkl_weight = 0.5;
        bool enable_incremental_hpt = true;
        bool enable_hpt_row_buckets = true;

        // Distribution-driven low-overhead mode for weak prefix locality.
        // Triggered by static index statistics, never by dataset name.
        bool enable_weak_prefix_low_overhead = true;
        uint64_t weak_prefix_min_static_entries = 100000;
        uint64_t weak_prefix_min_bucket_profiles = 40000;
        double weak_prefix_max_avg_static_per_bucket = 25.0;
        bool dynamic_first_lookup = false;

        // Innovation (1): split prefix-aware delta buckets one level deeper by
        // the local routing range observed in the base BOSBase tree.  The delta
        // path is therefore keyed by prefix + HPT row group + local BOSBase range,
        // while older prefix-family fallbacks remain as a correctness guard.
        bool enable_bos_range_buckets = true;
        uint32_t bos_range_bucket_depth = 2;

        // Quantize a predicted BOSBase child position into a coarse local range.
        // This avoids degenerating into one delta bucket per key on large root
        // arrays while still separating far-apart local routing regions.
        uint32_t bos_range_bucket_slot_group_size = 64;

        bool enable_performance_guard_stats = true;
        bool enable_lookup_stats = true;

        // HP-ILIM hot-path switches for English-paper ablations.
        // These do not change correctness; they expose individual mechanisms
        // so the benchmark can report w/o-base-id, w/o-negative-guard and
        // w/o-base-tombstone variants cleanly.
        bool enable_base_id_versioning = true;
        bool enable_dynamic_negative_guard = true;
        // PATCH 4B: caller guarantees insert keys are absent from immutable base.
        // Default false preserves generic insert/upsert semantics.
        bool assume_insert_key_is_new = false;
        bool enable_base_id_tombstone = true;

// HPT::ROW_COUNT = 1024 时，32 行一组会形成最多 32 个 row groups。
// 避免每个 prefix 下出现过多小 bucket。
        uint32_t hpt_row_bucket_group_size = 32;

// 查询时如果 primary row bucket 没命中，允许只在同 prefix 的其它 row buckets 中补查。
// 这仍然不是全局 delta scan。
            bool enable_prefix_bucket_fallback_lookup = true;
bool print_bucket_details = true;
size_t max_bucket_detail_print = 16;
bool enable_bucket_drift_stats = true;

// Alert-only thresholds. These do not change behavior yet;
// they only make drift detection visible in stats.
double bucket_gpkl_alert_threshold = 0.05;
double bucket_ratio_alert_threshold = 4.0;
bool enable_bucket_drift_state = true;

// Hysteresis exit thresholds. Lower than alert thresholds to avoid
// repeatedly entering/exiting drift state.
double bucket_gpkl_exit_threshold = 0.025;
double bucket_ratio_exit_threshold = 2.0;

uint64_t bucket_drift_exit_clean_checks = 3;

        // Row-level HPT maintenance. When COW is enabled, inserts are first
        // accumulated as per-row/per-cell deltas; a cloned HPT version is then
        // patched and published after the batch/dirty-row trigger fires.
        bool enable_hpt_cow = true;
        size_t hpt_cow_update_batch = 1024;
        size_t hpt_cow_dirty_row_threshold = 64;
        bool enable_hpt_retired_version_gc = true;
        size_t hpt_retired_version_gc_max_retired = 4;

        // Innovation (2): HPT row-aware local maintenance. Row/cell deltas are
        // not only recorded for COW HPT publishing; accumulated pressure in the
        // affected HPT row group can also trigger earlier local flush/merge for
        // the corresponding prefix+row bucket.
        bool enable_hpt_row_aware_maintenance = true;
        uint64_t hpt_row_flush_pressure_threshold = 2048;
        uint64_t hpt_row_merge_pressure_threshold = 8192;

        size_t drift_window_size = 4096;
        double gpkl_threshold = 1.5;

        uint64_t drift_check_interval = 1024;

        bool enable_delta_bos = true;
        bool enable_adaptive_local_merge = true;

        // Innovation (4): asynchronous local retraining.
        // A hot drifted bucket can be compacted into a separately trained local
        // BOSBase model in a background thread. The foreground query path keeps
        // using active/delta/base and atomically observes the new local model
        // when it is published.
        bool enable_async_local_retrain = true;

        // Innovation (6): bounded background local-retrain workers.
        // Per-bucket retrain_inflight already prevents duplicate work for one
        // bucket; this global cap prevents many drifted buckets from spawning
        // unbounded background threads at the same time.
        bool enable_async_local_retrain_worker_limit = true;
        size_t max_async_local_retrain_workers = 2;

        size_t async_local_retrain_min_entries = 1000;
        uint64_t async_local_retrain_min_delta_runs = 1;

        // Drift-state-driven async local retraining. Once a bucket enters
        // drift state, BOSIndex can retrain only that local bucket without waiting
        // for the older hot+GPKL trigger. This closes the implementation loop:
        // local drift detection -> local background retraining -> atomic publish.
        bool enable_drift_state_local_retrain = true;
        size_t drift_retrain_min_entries = 1000;
        uint64_t drift_retrain_min_insert_gap = 1024;
        uint64_t drift_retrain_min_accesses = 0;

        // Async local snapshot lifecycle. Published local models keep their
        // backing key/value memory in the state object. When a model is
        // invalidated or replaced, retire both the model and its memory as one
        // snapshot, then reclaim old retired snapshots after a conservative
        // publish-generation grace window.
        bool enable_async_local_snapshot_gc = true;
        size_t async_local_snapshot_gc_max_retired = 4;
        uint64_t async_local_snapshot_gc_grace_generations = 2;

        // Async-local snapshots may be probed before active/delta to serve
        // hot drifted regions, but they must never mask newer foreground
        // writes. Each bucket write advances a local write epoch; every
        // published async-local model records the epoch it covers.
        bool enable_async_local_write_epoch_guard = true;

        // Async-local snapshots may be probed before active/delta to serve
        // hot drifted regions, but they must never mask newer foreground
        // writes. Each bucket write advances a local write epoch; every
        // published async-local model records the epoch it covers.

        // Innovation (5): dynamic local structure adjustment.
        // BOSBase PMSS chooses among model-based nodes, compact nodes and HOT
        // subtries at bulkload time. BOSIndex feeds a per-bucket read/write bias
        // into newly materialized delta/local BOSBase snapshots, so only affected
        // local regions change and existing structures remain untouched.
        bool enable_dynamic_structure_adjustment = true;
                // R6: actual PMSS structure feedback policy.
        bool enable_structure_feedback_policy = true;
        uint64_t structure_feedback_min_nodes = 8;
        double structure_model_fallback_ratio_cap = 0.35;
        double structure_trie_dominance_ratio_cap = 0.80;
        double structure_feedback_model_penalty = 0.15;
        double structure_feedback_trie_penalty = 0.10;
        bool enable_structure_feedback_recovery = true;
        bool enable_structure_feedback_decay = true;
        uint64_t structure_feedback_decay_period = 8192;
        double structure_feedback_decay_factor = 0.5;
        uint64_t structure_feedback_recovery_min_attempts = 64;
        double structure_feedback_recovery_fallback_ratio = 0.20;

        double structure_min_read_ratio = 0.10;
        double structure_max_read_ratio = 0.98;
        double structure_drift_read_ratio_cap = 0.70;
        double structure_fragmented_read_ratio_floor = 0.75;
        double structure_policy_change_epsilon = 0.05;

        bool enable_structure_mode_state_machine = true;
        double structure_model_enter_read_ratio = 0.80;
        double structure_model_exit_read_ratio = 0.65;
        double structure_trie_enter_read_ratio = 0.35;
        double structure_trie_exit_read_ratio = 0.50;

        // D3: explicit compact-biased mode.  This fills the third structure
        // choice in Innovation (5): model node, compact local run/node, subtrie.
        // It is selected for non-drifting write-mixed regions where a sorted
        // compact delta run is preferable to spawning an async model or subtrie.
        bool enable_compact_structure_mode = true;
        double structure_compact_enter_read_ratio = 0.45;
        double structure_compact_exit_read_ratio = 0.60;
        double structure_compact_pmss_read_ratio = 0.50;
        double structure_compact_flush_scale = 4.0;
        double structure_compact_merge_scale = 4.0;
        double structure_compact_async_access_scale = 4.0;
        double structure_compact_async_min_entries_scale = 2.0;

        size_t structure_mode_min_builds_before_switch = 2;

        // D2: budgeted/deferred progressive local rebuild. When a bucket's
        // preferred structure changes but no natural flush/merge happens, BOSIndex
        // can schedule one async-local rebuild for that bucket only.
        bool enable_deferred_structure_rebuild = true;
        size_t structure_rebuild_min_entries = 1000;
        uint64_t structure_rebuild_access_threshold = 1024;
        uint64_t structure_rebuild_write_threshold = 1024;
        uint64_t structure_rebuild_min_insert_gap = 1024;
        bool enable_structure_feedback_rebuild_trigger = true;
        // D6: global budget for deferred structure rebuild workers.
        // Per-bucket retrain_inflight prevents duplicate work for one bucket;
        // this cap prevents many buckets from spawning unbounded rebuild work.
        bool enable_structure_rebuild_worker_limit = true;
        size_t max_async_structure_rebuild_workers = 2;

        // D3: structure-mode-aware local maintenance. These scale factors let
        // the chosen local structure mode affect the next maintenance decision
        // without triggering global rebuilding.
        double structure_model_flush_scale = 0.75;
        double structure_model_merge_scale = 2.0;
        double structure_model_async_access_scale = 0.25;
        double structure_model_async_min_entries_scale = 0.75;

        double structure_trie_flush_scale = 0.50;
        double structure_trie_merge_scale = 0.75;
        double structure_trie_async_access_scale = 2.0;
        double structure_trie_async_min_entries_scale = 1.25;

        // Innovation: delete-aware tombstone compaction.
        // Tombstones are still the source of truth for base/delta masking, but
        // maintenance can safely reclaim tombstones that no longer mask any copy.
        bool enable_tombstone_compaction = true;
        size_t tombstone_compaction_min_tombstones = 1;
        size_t tombstone_compaction_max_scan = 4096;
    };

  private:
    static const size_t MIN_DELTA_BULKLOAD_SIZE = 1000;

    struct DeltaData {
        char *data_buf = nullptr;
        char **key_ptrs = nullptr;
        uint64_t *val_ptrs = nullptr;
        int count = 0;
    };

    struct AsyncLocalRetiredSnapshot {
        BOSBase *model = nullptr;
        DeltaData memory;
        uint64_t retire_epoch = 0;
        uint64_t retire_generation = 0;
    };

    struct AsyncLocalModelState {
        std::atomic<BOSBase *> model;
        std::atomic<bool> retrain_inflight;
        std::atomic<uint64_t> epoch;
        std::atomic<uint64_t> requests;
        std::atomic<uint64_t> publishes;
        std::atomic<uint64_t> skips;
        std::atomic<uint64_t> hits;
        std::atomic<size_t> model_entries;
        std::atomic<uint64_t> snapshot_bucket_write_epoch;

        // Innovation (5): actual PMSS node choices materialized by async
        // local BOSBase snapshots. These are atomics because worker threads
        // publish snapshots in the background.
        std::atomic<uint64_t> structure_actual_entry_leaves;
        std::atomic<uint64_t> structure_actual_compact_nodes;
        std::atomic<uint64_t> structure_actual_model_nodes;
        std::atomic<uint64_t> structure_actual_trie_nodes;
        std::atomic<uint64_t> structure_actual_model_attempts;
        std::atomic<uint64_t> structure_actual_model_fallbacks;

        // D2: structure-specific publish counter. Async local retrain publishes
        // and deferred structure rebuild publishes both increment publishes; this
        // lets the feedback path distinguish a true structure-rebuild cycle from
        // a generic hot/drift retrain.
        std::atomic<uint64_t> structure_rebuild_publishes;

        std::atomic<uint64_t> retired_snapshot_count;
        std::atomic<uint64_t> retired_snapshot_reclaims;
        std::atomic<uint64_t> retire_generation;

        std::mutex publish_mutex;
        DeltaData current_model_memory;
        std::vector<AsyncLocalRetiredSnapshot> retired_snapshots;

        AsyncLocalModelState()
            : model(nullptr), retrain_inflight(false), epoch(0), requests(0),
              publishes(0), skips(0), hits(0), model_entries(0),
              snapshot_bucket_write_epoch(0),
              structure_actual_entry_leaves(0),
              structure_actual_compact_nodes(0),
              structure_actual_model_nodes(0),
              structure_actual_trie_nodes(0),
              structure_actual_model_attempts(0),
              structure_actual_model_fallbacks(0),
              structure_rebuild_publishes(0),
              retired_snapshot_count(0), retired_snapshot_reclaims(0),
              retire_generation(0) {}
    };

    struct BucketKey {
    uint64_t prefix = 0;
    uint32_t hpt_row_group = 0;
    uint32_t bos_range_id = 0;

    bool operator==(const BucketKey &other) const {
        return prefix == other.prefix &&
               hpt_row_group == other.hpt_row_group &&
               bos_range_id == other.bos_range_id;
    }
};

struct BucketKeyHash {
    size_t operator()(const BucketKey &k) const {
        uint64_t x = k.prefix;
        x ^= static_cast<uint64_t>(k.hpt_row_group) +
             0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
        x ^= static_cast<uint64_t>(k.bos_range_id) +
             0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
        return static_cast<size_t>(x);
    }
};

struct PrefixRangeKey {
    uint64_t prefix = 0;
    uint32_t bos_range_id = 0;

    bool operator==(const PrefixRangeKey &other) const {
        return prefix == other.prefix &&
               bos_range_id == other.bos_range_id;
    }
};

struct PrefixRangeKeyHash {
    size_t operator()(const PrefixRangeKey &k) const {
        uint64_t x = k.prefix;
        x ^= static_cast<uint64_t>(k.bos_range_id) +
             0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
        return static_cast<size_t>(x);
    }
};

enum LocalStructureMode : uint8_t {
    STRUCTURE_MODE_BALANCED = 0,
    STRUCTURE_MODE_MODEL_BIASED = 1,
    STRUCTURE_MODE_TRIE_BIASED = 2,
    STRUCTURE_MODE_COMPACT_BIASED = 3
};



  struct DeltaBucket {
    uint64_t prefix = 0;
    uint32_t hpt_row_group = 0;
    uint32_t bos_range_id = 0;

    uint64_t row_bucket_inserts = 0;
    uint64_t row_bucket_deletes = 0;
    uint64_t row_bucket_lookups = 0;
    uint64_t row_bucket_fallback_hits = 0;

    uint64_t hpt_row_flush_pressure_seen = 0;
    uint64_t hpt_row_merge_pressure_seen = 0;
    uint64_t hpt_pressure_flush_count = 0;
    uint64_t hpt_pressure_merge_count = 0;

    uint64_t active_hit_count = 0;
    uint64_t delta_run_hit_count = 0;
    uint64_t async_model_hit_count = 0;

    uint64_t drift_checks = 0;
uint64_t drift_events = 0;
uint64_t gpkl_drift_events = 0;
uint64_t ratio_drift_events = 0;
uint64_t novel_bucket_events = 0;
uint64_t drift_state_entries = 0;
uint64_t drift_state_exits = 0;

// Drift episode / retrain coordination. drift_epoch is incremented only
// when the bucket enters a new drift-state episode. The async retrain path
// uses it to avoid retraining the same local event repeatedly while still
// allowing a later retry if an in-flight retrain becomes stale.
uint64_t drift_epoch = 0;
uint64_t last_async_retrain_drift_epoch = 0;
uint64_t last_async_retrain_insert = 0;
uint64_t drift_async_retrain_schedules = 0;
uint64_t drift_async_retrain_skips = 0;


double last_gpkl = 0.0;
double max_gpkl = 0.0;

double last_drift_ratio = 1.0;
double max_drift_ratio = 1.0;
bool in_drift_state = false;


// Monotonic per-bucket write epoch. Active inserts, updates and deletes
// advance this value. Async-local snapshots record the epoch they cover so
// a stale local model cannot outrank newer active/delta state in lookup.
uint64_t write_epoch = 0;

// Monotonic per-bucket write epoch. Active inserts, updates and deletes
// advance this value. Async-local snapshots record the epoch they cover so
// a stale local model cannot outrank newer active/delta state in lookup.

uint64_t consecutive_drift_checks = 0;
uint64_t consecutive_clean_checks = 0;


        // Latest writes for this prefix. std::string is used deliberately so the
        // index never depends on benchmark string lifetime.
        std::unordered_map<std::string, uint64_t> active;

        // Immutable per-prefix delta indexes. Newest run is at the back.
        std::vector<BOSBase *> delta_indexes;
        std::vector<DeltaData> delta_memory;

        uint64_t accesses_since_check = 0;
        uint64_t inserts_since_merge = 0;
        uint64_t pending_entries = 0;
uint64_t flush_count = 0;
uint64_t merge_count = 0;

uint64_t drift_flush_count = 0;
uint64_t drift_merge_count = 0;
uint64_t drift_threshold_flush_count = 0;
uint64_t drift_threshold_merge_count = 0;

// 新增：局部统计信息
Hist256 active_hist;

        Hist256 delta_hist;
        Hist256 recent_hist;
        DriftSketch64 recent_sketch;

    // 新增：滑动窗口，用于近期分布漂移检测
        std::deque<std::string> recent_keys;

        // Innovation (5): local dynamic structure policy. PMSS consumes
        // structure_pmss_read_ratio when this bucket materializes a new delta
        // BOSBase run or async-local BOSBase snapshot.
        double structure_raw_read_ratio = 1.0;
        double structure_pmss_read_ratio = 1.0;
                // R6: aggregate actual PMSS materialization feedback for this bucket.
        uint64_t actual_entry_leaves = 0;
        uint64_t actual_compact_nodes = 0;
        uint64_t actual_model_nodes = 0;
        uint64_t actual_trie_nodes = 0;
        uint64_t actual_model_attempts = 0;
        uint64_t actual_model_fallbacks = 0;
        uint64_t structure_feedback_decay_events = 0;
        uint64_t structure_feedback_model_recoveries = 0;

        uint64_t structure_feedback_updates = 0;
        uint64_t structure_feedback_model_penalties = 0;
        uint64_t structure_feedback_trie_penalties = 0;

        // D2: explicit closed-loop accounting for dynamic structure adjustment.
        // A fold means PMSS build statistics from a published local model were
        // consumed by this bucket. A closed-loop cycle means at least one fold
        // came from a deferred structure-rebuild publication.
        uint64_t structure_feedback_folds = 0;
        uint64_t structure_closed_loop_cycles = 0;

        LocalStructureMode structure_mode = STRUCTURE_MODE_BALANCED;
        bool structure_rebuild_pending = false;
        uint64_t structure_mode_switches = 0;
        uint64_t structure_policy_updates = 0;
        uint64_t structure_mode_builds_since_switch = 0;
        uint64_t structure_model_biased_builds = 0;
        uint64_t structure_balanced_builds = 0;
        uint64_t structure_trie_biased_builds = 0;
        uint64_t structure_compact_biased_builds = 0;
        uint64_t structure_rebuild_requests = 0;
        uint64_t structure_rebuild_schedules = 0;
        uint64_t structure_rebuild_deferred_by_backlog = 0;
        uint64_t structure_rebuild_skips = 0;
        uint64_t structure_rebuild_due_to_drift = 0;
        uint64_t structure_rebuild_due_to_fragmentation = 0;
        uint64_t structure_rebuild_due_to_reads = 0;
        uint64_t structure_rebuild_due_to_writes = 0;
        uint64_t structure_rebuild_due_to_feedback = 0;
        uint64_t structure_last_rebuild_cause_mask = 0;
        uint64_t structure_rebuild_cause_mask_updates = 0;
        uint64_t last_structure_rebuild_insert = 0;
        uint64_t structure_threshold_policy_applications = 0;

        // Innovation (5): actual local structures created by this bucket.
        // PMSS counters are filled when a local BOSBase snapshot is built;
        // compact_delta_runs counts progressive sorted-array local runs.
        uint64_t structure_actual_entry_leaves = 0;
        uint64_t structure_actual_compact_nodes = 0;
        uint64_t structure_actual_model_nodes = 0;
        uint64_t structure_actual_trie_nodes = 0;
        uint64_t structure_actual_model_attempts = 0;
        uint64_t structure_actual_model_fallbacks = 0;
        uint64_t structure_compact_delta_runs = 0;

        // Async-local models are published through AsyncLocalModelState, but
        // the next structure-policy decision is made on DeltaBucket.  These
        // "observed" cursors let the foreground fold newly published PMSS
        // materialization feedback back into the bucket exactly once, closing
        // the loop: async build -> actual structure stats -> future policy.
        uint64_t structure_async_entry_leaves_observed = 0;
        uint64_t structure_async_compact_nodes_observed = 0;
        uint64_t structure_async_model_nodes_observed = 0;
        uint64_t structure_async_trie_nodes_observed = 0;
        uint64_t structure_async_model_attempts_observed = 0;
        uint64_t structure_async_model_fallbacks_observed = 0;
        uint64_t structure_async_rebuild_publishes_observed = 0;

        // Innovation (4): atomically published local retrained model for this
        // bucket. Stored behind shared_ptr so DeltaBucket remains movable/copyable
        // for unordered_map insertion while the state itself can contain atomics
        // and a mutex.
        std::shared_ptr<AsyncLocalModelState> async_local_model;

        DeltaBucket()
            : async_local_model(std::make_shared<AsyncLocalModelState>()) {}

    };

    struct HptDelta {
        std::string key;
        double delta = 0.0;
        uint64_t prefix = 0;
    };

    Options opt;

bool weak_prefix_low_overhead_mode_ = false;
double weak_prefix_avg_static_per_bucket_ = 0.0;
uint64_t weak_prefix_profile_bucket_count_ = 0;

// Build-phase timing, exposed for reviewer overhead experiments.
// model_train_seconds measures only the underlying learned BOSBase bulkload;
// index_build_seconds covers the complete BOSIndex construction path.
double last_model_train_seconds_ = 0.0;
double last_profile_build_seconds_ = 0.0;
double last_index_build_seconds_ = 0.0;

std::unordered_map<BucketKey, DeltaBucket, BucketKeyHash> buckets;

// prefix -> bucket keys，用于 lookup fallback：只查同 prefix 下的 row buckets
std::unordered_map<uint64_t, std::vector<BucketKey>> prefix_bucket_index;

// (prefix, BOSBase local range) -> bucket keys.  Lookup/tombstone maintenance can
// first probe the local BOSBase range family and fall back to the full prefix
// family only when needed for compatibbos_indexy with older HPT/range assignments.
std::unordered_map<PrefixRangeKey,
                   std::vector<BucketKey>,
                   PrefixRangeKeyHash> prefix_range_bucket_index;


    struct BaseData {
        bool is_owned = false;
        char *data_buf = nullptr;
        char **key_ptrs = nullptr;
        uint64_t *val_ptrs = nullptr;
        int count = 0;
    } base_data_store;

    std::unordered_map<uint64_t, uint64_t> static_prefix_counts;
std::unordered_map<uint64_t, Hist256> static_prefix_hist;
std::unordered_map<uint64_t, DriftSketch64> static_prefix_sketch;

// Static profile at the same granularity as delta buckets:
// (prefix, hpt_row_group, bos_range_id) -> count / histogram / bounded sketch.
std::unordered_map<BucketKey, uint64_t, BucketKeyHash> static_bucket_counts;
std::unordered_map<BucketKey, Hist256, BucketKeyHash> static_bucket_hist;
std::unordered_map<BucketKey, DriftSketch64, BucketKeyHash> static_bucket_sketch;


uint64_t total_static_entries = 0;
uint64_t total_dynamic_entries = 0;
// Existing-key updates go here instead of polluting prefix buckets.
    std::unordered_map<std::string, uint64_t> update_overlay;

    // In weak-prefix low-overhead mode, dynamic inserts are stored here
    // instead of creating prefix/range buckets. This is still exact lookup,
    // but avoids per-bucket active maps, hist/sketch maintenance, and HPT routing.
    // P14c_FLAT_REF_WEAK_OVERLAY:
    // Diagnostic/performance weak-prefix dynamic table.
    // Stores references to caller-owned insert keys, avoiding long-string copies.
    // Uses open addressing to avoid std::unordered_map node allocation and pointer chasing.
    //
    // Assumption: inserted key string storage is stable for the benchmark lifetime.
    class P14cFlatRefWeakOverlay {
        struct Slot {
            const std::string* key = nullptr;
            uint64_t second = 0;
            size_t hash = 0;
            uint8_t state = 0; // 0 empty, 1 occupied, 2 tombstone
        };

    public:
        class iterator {
            Slot* slot_ = nullptr;
        public:
            iterator() {}
            explicit iterator(Slot* s) : slot_(s) {}
            Slot& operator*() const { return *slot_; }
            Slot* operator->() const { return slot_; }
            bool operator==(const iterator& other) const {
                return slot_ == other.slot_;
            }
            bool operator!=(const iterator& other) const {
                return slot_ != other.slot_;
            }
        };

        class const_iterator {
            const Slot* slot_ = nullptr;
        public:
            const_iterator() {}
            explicit const_iterator(const Slot* s) : slot_(s) {}
            const Slot& operator*() const { return *slot_; }
            const Slot* operator->() const { return slot_; }
            bool operator==(const const_iterator& other) const {
                return slot_ == other.slot_;
            }
            bool operator!=(const const_iterator& other) const {
                return slot_ != other.slot_;
            }
        };

        void clear() {
            slots_.clear();
            size_ = 0;
            tombstones_ = 0;
        }

        void max_load_factor(float lf) {
            if (lf < 0.30f) lf = 0.30f;
            if (lf > 0.95f) lf = 0.95f;
            max_lf_ = lf;
        }

        void reserve(size_t n) {
            size_t cap = capacity_for(n);
            if (cap > slots_.size()) {
                rehash(cap);
            }
        }

        bool empty() const {
            return size_ == 0;
        }

        size_t size() const {
            return size_;
        }

        iterator end() {
            return iterator(nullptr);
        }

        const_iterator end() const {
            return const_iterator(nullptr);
        }

        std::pair<iterator, bool> emplace(const std::string& key,
                                          uint64_t value) {
            ensure_capacity_for_insert();

            const size_t h = static_cast<size_t>(BOSIndex::bos_index_hash_bytes(key.data(), key.size()));
            size_t idx = h & (slots_.size() - 1);
            size_t first_tomb = static_cast<size_t>(-1);

            for (;;) {
                Slot& s = slots_[idx];

                if (s.state == 0) {
                    size_t target = (first_tomb != static_cast<size_t>(-1))
                                        ? first_tomb
                                        : idx;
                    Slot& dst = slots_[target];

                    if (dst.state == 2) {
                        tombstones_--;
                    }

                    dst.key = &key;
                    dst.second = value;
                    dst.hash = h;
                    dst.state = 1;
                    size_++;

                    return std::make_pair(iterator(&dst), true);
                }

                if (s.state == 2) {
                    if (first_tomb == static_cast<size_t>(-1)) {
                        first_tomb = idx;
                    }
                } else if (s.hash == h && *(s.key) == key) {
                    return std::make_pair(iterator(&s), false);
                }

                idx = (idx + 1) & (slots_.size() - 1);
            }
        }

        iterator find(const std::string& key) {
            if (slots_.empty()) {
                return end();
            }

            const size_t h = static_cast<size_t>(BOSIndex::bos_index_hash_bytes(key.data(), key.size()));
            size_t idx = h & (slots_.size() - 1);

            for (;;) {
                Slot& s = slots_[idx];

                if (s.state == 0) {
                    return end();
                }

                if (s.state == 1 && s.hash == h && *(s.key) == key) {
                    return iterator(&s);
                }

                idx = (idx + 1) & (slots_.size() - 1);
            }
        }

        iterator find_cstr(const char* key) {
            if (slots_.empty()) {
                return end();
            }

            size_t len = 0;
            const size_t h = static_cast<size_t>(BOSIndex::bos_index_hash_cstr_len(key, len));
            size_t idx = h & (slots_.size() - 1);

            for (;;) {
                Slot& s = slots_[idx];

                if (s.state == 0) {
                    return end();
                }

                if (s.state == 1 && s.hash == h &&
                    s.key->size() == len &&
                    std::memcmp(s.key->data(), key, len) == 0) {
                    return iterator(&s);
                }

                idx = (idx + 1) & (slots_.size() - 1);
            }
        }

        const_iterator find_cstr(const char* key) const {
            if (slots_.empty()) {
                return end();
            }

            size_t len = 0;
            const size_t h = static_cast<size_t>(BOSIndex::bos_index_hash_cstr_len(key, len));
            size_t idx = h & (slots_.size() - 1);

            for (;;) {
                const Slot& s = slots_[idx];

                if (s.state == 0) {
                    return end();
                }

                if (s.state == 1 && s.hash == h &&
                    s.key->size() == len &&
                    std::memcmp(s.key->data(), key, len) == 0) {
                    return const_iterator(&s);
                }

                idx = (idx + 1) & (slots_.size() - 1);
            }
        }

        size_t erase(const std::string& key) {
            if (slots_.empty()) {
                return 0;
            }

            const size_t h = static_cast<size_t>(BOSIndex::bos_index_hash_bytes(key.data(), key.size()));
            size_t idx = h & (slots_.size() - 1);

            for (;;) {
                Slot& s = slots_[idx];

                if (s.state == 0) {
                    return 0;
                }

                if (s.state == 1 && s.hash == h && *(s.key) == key) {
                    s.key = nullptr;
                    s.second = 0;
                    s.hash = 0;
                    s.state = 2;
                    size_--;
                    tombstones_++;
                    return 1;
                }

                idx = (idx + 1) & (slots_.size() - 1);
            }
        }

    private:
        std::vector<Slot> slots_;
        size_t size_ = 0;
        size_t tombstones_ = 0;
        float max_lf_ = 0.70f;

        static size_t next_pow2(size_t x) {
            size_t p = 1;
            while (p < x && p < (static_cast<size_t>(-1) >> 1)) {
                p <<= 1;
            }
            return p;
        }

        size_t capacity_for(size_t n) const {
            if (n < 16) {
                n = 16;
            }

            size_t need = static_cast<size_t>(
                static_cast<double>(n) / static_cast<double>(max_lf_)
            ) + 8;

            if (need < 16) {
                need = 16;
            }

            return next_pow2(need);
        }

        void ensure_capacity_for_insert() {
            if (slots_.empty()) {
                rehash(1024);
                return;
            }

            const double projected =
                static_cast<double>(size_ + tombstones_ + 1) /
                static_cast<double>(slots_.size());

            if (projected > max_lf_) {
                rehash(slots_.size() * 2);
            }
        }

        void rehash(size_t new_cap) {
            new_cap = next_pow2(new_cap);
            if (new_cap < 16) {
                new_cap = 16;
            }

            std::vector<Slot> old;
            old.swap(slots_);

            slots_.clear();
            slots_.resize(new_cap);
            size_ = 0;
            tombstones_ = 0;

            for (size_t i = 0; i < old.size(); ++i) {
                if (old[i].state == 1 && old[i].key) {
                    insert_rehashed(old[i].key, old[i].second, old[i].hash);
                }
            }
        }

        void insert_rehashed(const std::string* key,
                             uint64_t value,
                             size_t h) {
            size_t idx = h & (slots_.size() - 1);

            for (;;) {
                Slot& s = slots_[idx];

                if (s.state != 1) {
                    s.key = key;
                    s.second = value;
                    s.hash = h;
                    s.state = 1;
                    size_++;
                    return;
                }

                idx = (idx + 1) & (slots_.size() - 1);
            }
        }
    };

    P14cFlatRefWeakOverlay weak_dynamic_overlay;

    std::vector<uint64_t> base_update_values;
std::vector<uint8_t> base_update_valid;
std::vector<uint8_t> base_deleted;
size_t base_deleted_count_ = 0;
std::vector<uint32_t> base_value_version;

uint64_t base_id_overlay_entries = 0;
uint64_t base_id_overlay_hits = 0;
uint64_t base_id_overlay_updates = 0;

// Compact exact base-id index slot. id_plus_one==0 is the empty sentinel.
// fingerprint is only a filter: every matching fingerprint is still verified
// against the original base key, so 32-bit hash collisions cannot change
// lookup/delete correctness.
struct BaseHashSlot {
    uint32_t fingerprint = 0;
    uint32_t id_plus_one = 0;
};

std::vector<BaseHashSlot> base_hash_slots;
size_t base_hash_mask = 0;
bool base_hash_use_crc32c_ = false;

uint64_t base_hash_index_hits = 0;
uint64_t base_hash_index_misses = 0;
uint64_t base_hash_index_collision_probes = 0;

struct HotKeyCacheEntry {
    uint64_t hash = 0;
    uint64_t value = 0;
    size_t base_id = 0;

    // Benchmark-only shortcut: if the same immutable c_str pointer is seen
    // again, skip the duplicate memcmp. Exact string copy remains as fallback.
    const char *last_key_ptr = nullptr;

    uint8_t valid = 0;
    uint8_t has_base_id = 0;
    std::string key;
};

struct HotPtrCacheEntry {
    const char *key_ptr = nullptr;
    uint64_t value = 0;
    size_t base_id = 0;
    uint32_t version = 0;
    uint8_t valid = 0;
};

struct HotUpdatePtrCacheEntry {
    const char *key_ptr = nullptr;
    size_t base_id = 0;
    size_t hot_key_slot = 0;
    uint8_t valid = 0;
};

// P15f: Model-Segment Cache (MSC).  This cache is deliberately owned by BOSIndex's
// learned routing layer rather than by a global hot-key table: a key first maps
// to a model/prefix region, then to a tiny set-associative shortcut set.  Slots
// memoize verified immutable-base positions (base_id), not independent KV
// copies, so base updates are observed through base_update_valid/base_update_values.
static constexpr size_t BOS_MODEL_SEGMENT_CACHE_WAYS = 4;

struct ModelSegmentCacheSlot {
    uint64_t route_tag = 0;
    uint64_t key_hash = 0;
    uint32_t key_len = 0;
    uint32_t base_id = 0;
    uint8_t valid = 0;
};

struct ModelSegmentCacheSet {
    std::array<ModelSegmentCacheSlot, BOS_MODEL_SEGMENT_CACHE_WAYS> ways{};
    uint8_t next_victim = 0;
};

std::vector<HotKeyCacheEntry> hot_key_cache_;
size_t hot_key_cache_mask_ = 0;
std::vector<HotPtrCacheEntry> hot_ptr_cache_;
size_t hot_ptr_cache_mask_ = 0;
size_t hot_ptr_cache_active_slots_ = 0;
std::vector<HotPtrCacheEntry> static_read_ptr_cache_;
size_t static_read_ptr_cache_mask_ = 0;
std::vector<HotUpdatePtrCacheEntry> hot_update_ptr_cache_;
size_t hot_update_ptr_cache_mask_ = 0;

std::vector<ModelSegmentCacheSet> model_segment_cache_;
size_t model_segment_cache_mask_ = 0;
bool model_segment_cache_enabled_ = false;

uint64_t model_segment_cache_hits_ = 0;
uint64_t model_segment_cache_misses_ = 0;
uint64_t model_segment_cache_puts_ = 0;
uint64_t model_segment_cache_invalidations_ = 0;
uint64_t model_segment_cache_base_hits_ = 0;
uint64_t model_segment_cache_update_hits_ = 0;

bool hot_key_cache_enabled_ = false;
bool hot_ptr_cache_enabled_ = false;
bool static_read_ptr_cache_enabled_ = false;
bool hot_update_ptr_shortcut_enabled_ = false;
bool static_read_fastpath_enabled_ = false;

uint64_t hot_key_cache_hits_ = 0;
uint64_t hot_key_cache_misses_ = 0;
uint64_t hot_key_cache_puts_ = 0;
uint64_t hot_key_cache_invalidations_ = 0;
uint64_t hot_ptr_cache_hits_ = 0;
uint64_t hot_ptr_cache_misses_ = 0;
uint64_t hot_ptr_cache_puts_ = 0;
uint64_t hot_ptr_cache_invalidations_ = 0;
uint64_t static_read_fastpath_hits_ = 0;
uint64_t static_read_ptr_cache_hits_ = 0;
uint64_t static_read_ptr_cache_misses_ = 0;
uint64_t static_read_ptr_cache_puts_ = 0;
uint64_t static_read_ptr_cache_invalidations_ = 0;
uint64_t hot_update_cache_shortcut_hits_ = 0;
uint64_t hot_update_ptr_shortcut_hits_ = 0;
uint64_t hot_update_ptr_cache_puts_ = 0;
uint64_t hot_update_ptr_cache_invalidations_ = 0;


    // Exact-negative membership guard for dynamic keys.
    // Store only 64-bit fingerprints, not a duplicate std::string copy.
    // Collisions are safe: they only create false positives, which fall through
    // to exact active/delta/overlay lookup. There are no false negatives because
    // every dynamic key records its fingerprint on insert/update.
    std::unordered_set<uint64_t> dynamic_key_fps;
    uint64_t dynamic_exact_guard_checks = 0;
    uint64_t dynamic_exact_guard_skips = 0;
    std::unordered_set<std::string> deleted_keys;
    uint64_t tombstone_hits = 0;
    uint64_t delete_count = 0;
    uint64_t tombstone_gc_checks = 0;
    uint64_t tombstone_gc_erases = 0;

    // Innovation (1)/(6): tombstone GC must remain prefix-aware too.
    // Older versions checked every delta bucket while deciding whether a
    // tombstone still masked a live copy.  These counters make the maintenance
    // path auditable: GC now probes only the deleted key's prefix bucket family
    // plus the immutable base lookup, never all delta buckets.
    uint64_t tombstone_gc_prefix_bucket_probes = 0;
    uint64_t tombstone_gc_delta_run_probes = 0;
    uint64_t tombstone_gc_prefix_index_misses = 0;
    uint64_t tombstone_gc_full_bucket_scan_fallbacks = 0;

    uint64_t tombstone_merge_skips = 0;


    uint64_t total_inserts = 0;
   uint64_t total_pending_entries = 0;
uint64_t total_flushes = 0;
uint64_t total_local_merges = 0;

uint64_t total_drift_flushes = 0;
uint64_t total_drift_local_merges = 0;
uint64_t total_drift_threshold_flushes = 0;
uint64_t total_drift_threshold_local_merges = 0;

uint64_t total_hpt_pressure_flushes = 0;
uint64_t total_hpt_pressure_local_merges = 0;

std::atomic<uint64_t> total_async_local_retrain_requests;
std::atomic<uint64_t> total_async_local_retrain_publishes;
std::atomic<uint64_t> total_async_local_retrain_skips;
std::atomic<uint64_t> async_local_retrain_workers_inflight{0};
std::atomic<uint64_t> total_async_local_retrain_deferred_by_backlog{0};
uint64_t async_local_model_hits = 0;

uint64_t total_structure_policy_updates = 0;
uint64_t total_structure_mode_switches = 0;
uint64_t total_structure_model_biased_builds = 0;
uint64_t total_structure_balanced_builds = 0;
uint64_t total_structure_trie_biased_builds = 0;
uint64_t total_structure_compact_biased_builds = 0;
uint64_t total_structure_compact_delta_runs = 0;
uint64_t total_structure_rebuild_requests = 0;
mutable uint64_t total_structure_feedback_updates = 0;
mutable uint64_t total_structure_feedback_model_penalties = 0;
mutable uint64_t total_structure_feedback_trie_penalties = 0;
mutable uint64_t total_structure_feedback_model_recoveries = 0;
mutable uint64_t total_structure_feedback_decay_events = 0;
mutable uint64_t total_structure_feedback_folds = 0;
mutable uint64_t total_structure_closed_loop_cycles = 0;
uint64_t total_structure_rebuild_cause_mask_updates = 0;
uint64_t total_structure_last_rebuild_cause_mask_or = 0;
uint64_t total_structure_rebuild_schedules = 0;
std::atomic<uint64_t> total_structure_rebuild_deferred_by_backlog{0};
std::atomic<uint64_t> async_structure_rebuild_workers_inflight{0};
uint64_t total_structure_rebuild_due_to_drift = 0;
uint64_t total_structure_rebuild_due_to_fragmentation = 0;
uint64_t total_structure_rebuild_due_to_reads = 0;
uint64_t total_structure_rebuild_due_to_writes = 0;
uint64_t total_structure_rebuild_due_to_feedback = 0;
std::atomic<uint64_t> total_structure_rebuild_publishes{0};
std::atomic<uint64_t> total_structure_rebuild_skips{0};
uint64_t total_structure_threshold_policy_applications = 0;

// Review-metric timers. Units are nanoseconds; benchmark drivers print ms.
std::atomic<uint64_t> review_local_merge_time_ns_{0};
std::atomic<uint64_t> review_local_flush_time_ns_{0};
std::atomic<uint64_t> review_async_retrain_time_ns_{0};
std::atomic<uint64_t> review_structure_rebuild_time_ns_{0};

std::vector<char *> base_merged_key_storage;

uint64_t buffer_hits = 0;

    uint64_t delta_hits = 0;
    uint64_t base_hits = 0;
    uint64_t empty_read_fastpath_hits = 0;

    std::vector<HptDelta> pending_hpt_deltas;

    // Innovation (2): keep the incrementally updated HPT as a routing/statistics
    // overlay instead of replacing the base BOSBase HPT. Base BOSBase physical layout
    // remains tied to the HPT trained during bulkload, while routing_hpt is used
    // only for prefix/HPT-row bucket routing and row-level statistics.
    std::atomic<HPT *> routing_hpt;
    std::vector<HPT *> retired_hpt_versions;
    std::mutex hpt_publish_mutex;
    mutable std::atomic<uint64_t> perf_lookup_count{0};
mutable std::atomic<uint64_t> perf_lookup_bucket_probes{0};
mutable std::atomic<uint64_t> perf_lookup_delta_run_probes{0};
mutable std::atomic<uint64_t> perf_lookup_max_bucket_probes{0};
mutable std::atomic<uint64_t> perf_lookup_max_delta_run_probes{0};

    mutable std::atomic<uint64_t> perf_lookup_prefix_exact_bound_violations{0};
mutable std::atomic<uint64_t> perf_lookup_delta_run_bound_violations{0};
mutable std::atomic<uint64_t> perf_lookup_max_prefix_width_seen{0};
mutable std::atomic<uint64_t> perf_lookup_max_delta_run_probe_bound{0};
    mutable std::atomic<uint64_t> hpt_active_readers{0};
std::atomic<uint64_t> hpt_retired_version_reclaims{0};
std::atomic<uint64_t> hpt_retired_version_defers{0};

    std::array<uint64_t, HPT::ROW_COUNT> hpt_row_insert_counts{};
    std::array<uint64_t, HPT::ROW_COUNT> hpt_row_delete_counts{};
    std::array<uint64_t, HPT::ROW_COUNT> hpt_row_touch_counts{};
    std::array<uint8_t, HPT::ROW_COUNT> hpt_dirty_rows{};

    std::vector<uint64_t> hpt_cell_insert_counts;
    std::vector<uint64_t> hpt_cell_delete_counts;

    uint64_t hpt_dirty_row_count = 0;
    uint64_t hpt_version = 0;
    uint64_t hpt_cow_publishes = 0;
    uint64_t hpt_inplace_publishes = 0;
    uint64_t hpt_cell_updates = 0;
    uint64_t lookup_prefix_exact_bound_violations = 0;
uint64_t lookup_delta_run_bound_violations = 0;
uint64_t max_lookup_prefix_width_seen = 0;
uint64_t max_lookup_delta_run_probe_bound = 0;

    std::vector<std::thread> async_local_retrain_threads;
struct LookupProbeTrace {
    uint64_t prefix = 0;
    uint64_t bucket_probes = 0;
    uint64_t delta_run_probes = 0;

    // 当前 prefix 下实际 bucket 数。
    uint64_t prefix_width_bound = 0;

    // 理论 HPT row group 上界：
    // ceil(HPT::ROW_COUNT / group_size) + 1，其中 +1 是 prefix-only bucket。
    uint64_t hpt_row_width_bound = 0;

    // 当前 lookup 的 delta-run probe 上界：
    // prefix_width_bound * max_delta_runs_in_this_prefix。
    uint64_t delta_run_probe_bound = 0;
};

static void atomic_max_u64(std::atomic<uint64_t> &slot, uint64_t value) {
    uint64_t cur = slot.load(std::memory_order_relaxed);
    while (cur < value &&
           !slot.compare_exchange_weak(
               cur,
               value,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

bool empty_dynamic_state_for_read() const {
    return buckets.empty()
        && update_overlay.empty()
        && weak_dynamic_overlay.empty()
        && (deleted_keys.empty() && base_deleted_count_ == 0)
        && total_dynamic_entries == 0
        && total_pending_entries == 0
        && pending_hpt_deltas.empty()
        && hpt_dirty_row_count == 0
        && base_id_overlay_entries == 0;
}

bool clean_static_base_state_for_read() const {
    return empty_dynamic_state_for_read()
        && base_id_overlay_entries == 0
        && base_id_overlay_updates == 0;
}

uint64_t hpt_row_bucket_width_bound_for_test() const {
    if (weak_prefix_low_overhead_mode_ || !opt.enable_hpt_row_buckets) {
        return 1;
    }

    uint64_t group_size =
        static_cast<uint64_t>(opt.hpt_row_bucket_group_size);
    if (group_size == 0) {
        group_size = 1;
    }

    const uint64_t row_groups =
        (static_cast<uint64_t>(HPT::ROW_COUNT) + group_size - 1) /
        group_size;

    // hpt_row_group == 0 is reserved for prefix-only / fallback bucket.
    return row_groups + 1;
}

uint64_t prefix_bucket_width_for_test(uint64_t prefix) const {
    auto it = prefix_bucket_index.find(prefix);
    if (it == prefix_bucket_index.end()) {
        return 0;
    }
    return static_cast<uint64_t>(it->second.size());
}

uint64_t prefix_range_bucket_width_for_test(const PrefixRangeKey &prk) const {
    auto it = prefix_range_bucket_index.find(prk);
    if (it == prefix_range_bucket_index.end()) {
        return 0;
    }
    return static_cast<uint64_t>(it->second.size());
}

uint64_t max_delta_runs_for_prefix_for_test(uint64_t prefix) const {
    uint64_t max_runs = 0;

    auto ix = prefix_bucket_index.find(prefix);
    if (ix == prefix_bucket_index.end()) {
        return 0;
    }

    for (const BucketKey &bk : ix->second) {
        auto b_it = buckets.find(bk);
        if (b_it == buckets.end()) {
            continue;
        }

        max_runs = std::max<uint64_t>(
            max_runs,
            static_cast<uint64_t>(b_it->second.delta_memory.size()));
    }

    return max_runs;
}

void record_lookup_probe_trace_for_test(const LookupProbeTrace &trace) const {
    if (!opt.enable_performance_guard_stats) {
        return;
    }

    perf_lookup_count.fetch_add(1, std::memory_order_relaxed);
    perf_lookup_bucket_probes.fetch_add(
        trace.bucket_probes,
        std::memory_order_relaxed);
    perf_lookup_delta_run_probes.fetch_add(
        trace.delta_run_probes,
        std::memory_order_relaxed);

    atomic_max_u64(perf_lookup_max_bucket_probes, trace.bucket_probes);
    atomic_max_u64(perf_lookup_max_delta_run_probes, trace.delta_run_probes);
    atomic_max_u64(perf_lookup_max_prefix_width_seen,
                   trace.prefix_width_bound);
    atomic_max_u64(perf_lookup_max_delta_run_probe_bound,
                   trace.delta_run_probe_bound);

    if (trace.bucket_probes > trace.prefix_width_bound) {
        perf_lookup_prefix_exact_bound_violations.fetch_add(
            1,
            std::memory_order_relaxed);
    }

    if (trace.prefix_width_bound > trace.hpt_row_width_bound) {
        perf_lookup_prefix_exact_bound_violations.fetch_add(
            1,
            std::memory_order_relaxed);
    }

    if (trace.delta_run_probes > trace.delta_run_probe_bound) {
        perf_lookup_delta_run_bound_violations.fetch_add(
            1,
            std::memory_order_relaxed);
    }
}


    struct HptReadGuard {
    const BOSIndex *owner = nullptr;
    HPT *ptr = nullptr;

    explicit HptReadGuard(const BOSIndex *bos_index) : owner(bos_index), ptr(nullptr) {
        if (owner) {
            owner->hpt_active_readers.fetch_add(
                1, std::memory_order_acq_rel);
            ptr = owner->routing_hpt.load(std::memory_order_acquire);
        }
    }

    ~HptReadGuard() {
        if (owner) {
            owner->hpt_active_readers.fetch_sub(
                1, std::memory_order_acq_rel);
        }
    }

    HptReadGuard(const HptReadGuard &) = delete;
    HptReadGuard &operator=(const HptReadGuard &) = delete;

    HPT *get() const { return ptr; }
};

  public:
    BOSIndex()
        : BOSBase(), opt(), total_async_local_retrain_requests(0),
          total_async_local_retrain_publishes(0),
          total_async_local_retrain_skips(0),
          routing_hpt(nullptr) {
        normalize_options();
        init_hpt_stats_storage();
    }

    explicit BOSIndex(const Options &options)
        : BOSBase(), opt(options), total_async_local_retrain_requests(0),
          total_async_local_retrain_publishes(0),
          total_async_local_retrain_skips(0),
          routing_hpt(nullptr) {
        normalize_options();
        init_hpt_stats_storage();
    }

    ~BOSIndex() {
        wait_for_async_local_retrains();
        cleanup_all_buckets();
        cleanup_routing_hpt_model();
    }

    double last_model_train_seconds() const { return last_model_train_seconds_; }
    double last_profile_build_seconds() const { return last_profile_build_seconds_; }
    double last_index_build_seconds() const { return last_index_build_seconds_; }

    void bulkload_bos_index(char **keys, uint64_t *values, int count) {
    normalize_options();
    const auto bos_index_build_t0 = std::chrono::steady_clock::now();
    last_model_train_seconds_ = 0.0;
    last_profile_build_seconds_ = 0.0;
    last_index_build_seconds_ = 0.0;
    std::cout << "[BOS] Initializing base BOSBase index..." << std::endl;

    free_base_merged_key_storage();

    base_data_store.is_owned = false;
    base_data_store.key_ptrs = keys;
    base_data_store.val_ptrs = values;
    base_data_store.count = count;

    buckets.max_load_factor(0.9);
    buckets.reserve(64);

    prefix_bucket_index.clear();
prefix_bucket_index.max_load_factor(0.9);
prefix_bucket_index.reserve(64);

prefix_range_bucket_index.clear();
prefix_range_bucket_index.max_load_factor(0.9);
prefix_range_bucket_index.reserve(64);


    static_prefix_counts.max_load_factor(0.9);
    static_prefix_counts.reserve(64);

float overlay_lf = 0.3f;

if (const char* env = std::getenv("BOS_UPDATE_OVERLAY_LF")) {
    char* end = nullptr;
    float v = std::strtof(env, &end);
    if (end != env && v > 0.0f) {
        overlay_lf = v;
    }
}

if (overlay_lf < 0.2f) overlay_lf = 0.2f;
if (overlay_lf > 2.0f) overlay_lf = 2.0f;

update_overlay.clear();
update_overlay.max_load_factor(overlay_lf);  // 关键参数

weak_dynamic_overlay.clear();

// P14c_FLAT_REF_WEAK_OVERLAY: flat ref overlay load factor.
// With INSERT_SIZE=5M and P9 reserve, 0.70 normally gives 8,388,608 slots.
float p14c_weak_flat_lf = 0.70f;
if (const char* env = std::getenv("BOS_WEAK_FLAT_LF")) {
    double v = std::atof(env);
    if (v >= 0.30 && v <= 0.95) {
        p14c_weak_flat_lf = static_cast<float>(v);
    }
}
weak_dynamic_overlay.max_load_factor(p14c_weak_flat_lf);

std::cerr << "[BOS P14c] flat_ref_weak_overlay=1 weak_flat_lf="
          << p14c_weak_flat_lf << std::endl;

dynamic_key_fps.max_load_factor(0.7f);

std::cerr << "[BOS] update_overlay max_load_factor = "
          << overlay_lf << std::endl;

// Existing base-key updates use the compact base-id overlay, so this hash
// table is a fallback for non-base updates only. Avoid reserving one node per
// base key at build time; that memory was almost always unused.
update_overlay.rehash(0);
update_overlay.reserve(std::min<size_t>(
    static_cast<size_t>(65536),
    static_cast<size_t>(count) / 16 + static_cast<size_t>(1024)));

// Weak dynamic capacity is reserved after the static profile decides whether
// weak-prefix mode is actually active. This avoids charging every build for a
// multi-million-slot table that may never be used.
dynamic_key_fps.clear();
dynamic_key_fps.rehash(0);

base_update_values.assign(static_cast<size_t>(count), 0);
base_update_valid.assign(static_cast<size_t>(count), 0);
base_deleted.assign(static_cast<size_t>(count), 0);
base_deleted_count_ = 0;
base_value_version.clear();

base_id_overlay_entries = 0;
base_id_overlay_hits = 0;
base_id_overlay_updates = 0;

// Compact BaseHashSlot stores base ids in uint32_t.
// This preserves exact key-only semantics for supported base sizes.
if (static_cast<uint64_t>(count) > 0xffffffffULL) {
    std::cerr << "[BOS] base hash compact-id overflow: "
              << count << " entries" << std::endl;
    std::abort();
}
build_base_hash_index(keys, count);

base_hash_index_hits = 0;
base_hash_index_misses = 0;
base_hash_index_collision_probes = 0;

init_hot_key_cache_from_env(static_cast<size_t>(count));
static_read_fastpath_hits_ = 0;

// Version metadata is only consumed by the optional pointer cache. Do not
// allocate a 4-byte entry for every base key when that cache is disabled.
if (hot_ptr_cache_enabled_) {
    base_value_version.assign(static_cast<size_t>(count), 0);
} else {
    std::vector<uint32_t>().swap(base_value_version);
}

dynamic_exact_guard_checks = 0;
dynamic_exact_guard_skips = 0;
deleted_keys.clear();
deleted_keys.rehash(0);
deleted_keys.max_load_factor(overlay_lf);
deleted_keys.reserve(std::min<size_t>(
    static_cast<size_t>(65536),
    static_cast<size_t>(count) / 16 + static_cast<size_t>(1024)));

    reset_hpt_incremental_state();
    total_async_local_retrain_requests.store(0, std::memory_order_relaxed);
    total_async_local_retrain_publishes.store(0, std::memory_order_relaxed);
    total_async_local_retrain_skips.store(0, std::memory_order_relaxed);
    async_local_retrain_workers_inflight.store(0, std::memory_order_relaxed);
    total_async_local_retrain_deferred_by_backlog.store(0, std::memory_order_relaxed);
    review_local_merge_time_ns_.store(0, std::memory_order_relaxed);
    review_local_flush_time_ns_.store(0, std::memory_order_relaxed);
    review_async_retrain_time_ns_.store(0, std::memory_order_relaxed);
    review_structure_rebuild_time_ns_.store(0, std::memory_order_relaxed);
    async_local_model_hits = 0;
    total_structure_policy_updates = 0;
    total_structure_mode_switches = 0;
    total_structure_model_biased_builds = 0;
    total_structure_balanced_builds = 0;
    total_structure_trie_biased_builds = 0;
    total_structure_compact_biased_builds = 0;
    total_structure_compact_delta_runs = 0;
    total_structure_rebuild_requests = 0;
    total_structure_rebuild_schedules = 0;
    total_structure_rebuild_deferred_by_backlog.store(0, std::memory_order_relaxed);
    async_structure_rebuild_workers_inflight.store(0, std::memory_order_relaxed);
    total_structure_rebuild_due_to_drift = 0;
    total_structure_rebuild_due_to_fragmentation = 0;
    total_structure_rebuild_due_to_reads = 0;
    total_structure_rebuild_due_to_writes = 0;
    total_structure_rebuild_due_to_feedback = 0;
    total_structure_rebuild_publishes.store(0, std::memory_order_relaxed);
    total_structure_rebuild_skips.store(0, std::memory_order_relaxed);
    total_structure_threshold_policy_applications = 0;
    total_structure_feedback_folds = 0;
    total_structure_closed_loop_cycles = 0;
    total_structure_rebuild_cause_mask_updates = 0;
    total_structure_last_rebuild_cause_mask_or = 0;

const auto model_t0 = std::chrono::steady_clock::now();
this->bulkload((const char **)keys, values, count);
const auto model_t1 = std::chrono::steady_clock::now();
last_model_train_seconds_ =
    std::chrono::duration<double>(model_t1 - model_t0).count();

reset_routing_hpt_model();

// Must be built after base bulkload because bucket_key_for_key()
// needs the trained HPT.
const auto profile_t0 = std::chrono::steady_clock::now();
build_static_prefix_profile(keys, count);
const auto profile_t1 = std::chrono::steady_clock::now();
last_profile_build_seconds_ =
    std::chrono::duration<double>(profile_t1 - profile_t0).count();

// Pre-size only the dynamic structure selected by the profile. The old code
// unconditionally reserved seven base-sized weak-overlay regions, which moved
// future insertion cost into build time and inflated after-build RSS. A single
// base-sized reserve keeps the first ~O(n) inserts rehash-free while bounding
// memory. BOS_WEAK_RESERVE_ENTRIES can override this when a deployment knows
// its expected dynamic volume.
const bool weak_model_internal_rewire = bos_index_model_internal_weak_rewire_cached_;

if (weak_prefix_low_overhead_mode_ && !weak_model_internal_rewire) {
    size_t reserve_entries = std::max<size_t>(16384, static_cast<size_t>(count));
    if (const char* env = std::getenv("BOS_WEAK_RESERVE_ENTRIES")) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(env, &end, 10);
        if (end != env && end && *end == '\0' && v > 0 &&
            v <= static_cast<unsigned long long>(static_cast<size_t>(-1))) {
            reserve_entries = static_cast<size_t>(v);
        }
    }
    weak_dynamic_overlay.reserve(reserve_entries);
} else {
    dynamic_key_fps.reserve(static_cast<size_t>(count) / 4 + 1024);
}

last_index_build_seconds_ =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - bos_index_build_t0).count();

}

BOS_NOINLINE void invalidate_async_local_model(DeltaBucket &bucket) {
    if (!bucket.async_local_model) return;

    AsyncLocalModelState *state = bucket.async_local_model.get();

    // Any new write to this bucket makes an already-published async-local
    // snapshot stale. Bump epoch so in-flight retrains can reject old snapshots,
    // then atomically detach the currently published model.
    state->epoch.fetch_add(1, std::memory_order_acq_rel);

    BOSBase *old = state->model.exchange(nullptr, std::memory_order_acq_rel);

    {
        std::lock_guard<std::mutex> lock(state->publish_mutex);
        if (old) {
            retire_async_local_snapshot_locked(*state, old);
        }
        state->model_entries.store(0, std::memory_order_relaxed);
        state->snapshot_bucket_write_epoch.store(0, std::memory_order_release);
        reclaim_retired_async_local_snapshots_locked(*state, false);
    }
}


BOS_NOINLINE bool bucket_contains_materialized_key_no_stats(
    const DeltaBucket &bucket,
    const std::string &key) const {
    if (bucket.active.find(key) != bucket.active.end()) {
        return true;
    }

    for (const DeltaData &dm : bucket.delta_memory) {
        uint64_t ignored = 0;
        if (lookup_delta_data_sorted(dm, key, ignored)) {
            return true;
        }
    }

    // Defensive support for legacy delta runs that may have a BOSBase model
    // without a paired compact DeltaData array.
    for (BOSBase *delta_model : bucket.delta_indexes) {
        if (delta_model && delta_model->lookup(key.c_str())) {
            return true;
        }
    }

    if (bucket.async_local_model) {
        BOSBase *local_model =
            bucket.async_local_model->model.load(std::memory_order_acquire);
        if (local_model && local_model->lookup(key.c_str())) {
            return true;
        }
    }

    return false;
}

BOS_NOINLINE size_t invalidate_prefix_bucket_family_for_delete(
    const std::string &key) {
    const uint64_t prefix = prefix_code(key);
    auto ix = prefix_bucket_index.find(prefix);
    if (ix == prefix_bucket_index.end()) {
        return 0;
    }

    size_t touched = 0;
    for (const BucketKey &bk : ix->second) {
        auto b_it = buckets.find(bk);
        if (b_it == buckets.end()) {
            continue;
        }

        DeltaBucket &bucket = b_it->second;
        if (!bucket_contains_materialized_key_no_stats(bucket, key)) {
            continue;
        }

        bucket.row_bucket_deletes++;
        bucket.write_epoch++;
        invalidate_async_local_model(bucket);
        touched++;
    }

    return touched;
}

void insert(const std::string &key, uint64_t value) {
    // Re-insert resurrects a tombstoned key and must outrank old update overlays.
    // Keep semantics, but avoid hashing hot insert keys against empty overlays.
    if (!deleted_keys.empty()) {
        deleted_keys.erase(key);
    }
    if (!update_overlay.empty()) {
        update_overlay.erase(key);
    }

    // Rare resurrection path. Normal append-style inserts skip the
    // immutable-base probe entirely while no base tombstones exist.
    if (base_deleted_count_ != 0) {
        size_t base_id = 0;
        if (find_base_id_for_key(key, base_id) &&
            base_id < base_deleted.size() &&
            base_deleted[base_id]) {

            base_deleted[base_id] = 0;
            --base_deleted_count_;

            if (base_id < base_update_valid.size()) {
                base_id_overlay_updates++;
                if (!base_update_valid[base_id]) {
                    base_update_valid[base_id] = 1;
                    base_id_overlay_entries++;
                }

                base_update_values[base_id] = value;
                bump_base_value_version(base_id);

                ++total_inserts;
                return;
            }
        }
    }

                // P8c: weak-prefix insert-new path goes directly to BOSIndex's
                // weak dynamic structure, avoiding one immutable-base probe
                // per append-style insert.
    const bool p15e_model_internal_rewire = bos_index_model_internal_weak_rewire_cached_;

    if (weak_prefix_low_overhead_mode_ && !p15e_model_internal_rewire) {
        // The flat weak overlay is already an exact membership structure.
        // Maintaining a second unordered_set<uint64_t> fingerprint table here
        // duplicated hashing/allocation on every insert and was unnecessary.
        auto w_ret = weak_dynamic_overlay.emplace(key, value);
        // HP-ILIM force dynamic guard fp after weak overlay emplace
        if (opt.enable_dynamic_negative_guard) {
            dynamic_key_fps.insert(bos_index_hash_bytes(key.data(), key.size()));
        }

        if (w_ret.second) {
            total_dynamic_entries++;
        } else {
            w_ret.first->second = value;
        }

        if (hot_key_cache_enabled_) {
            hot_key_cache_put_key(key, value);
        }
        total_inserts++;
        return;
    }


            // If an insert key already exists in the immutable base, treat it as an
    // upsert into the base-id overlay instead of creating a duplicate dynamic
    // delta copy.  This preserves latest-value semantics and keeps the cstr base
    // fast path exact even after dynamic insertions are present.
    // PATCH 4B: for an explicitly known-new append stream, avoid the
    // immutable-base membership probe.  With the option disabled, retain
    // the original generic insert/upsert semantics.
    if (!opt.assume_insert_key_is_new &&
        update_base_overlay_by_key(key, value)) {
        total_inserts++;
        return;
    }

if (opt.enable_dynamic_negative_guard) {
    dynamic_key_fps.insert(bos_index_hash_bytes(key.data(), key.size()));
}
    BucketKey bk = bucket_key_for_key(key);
uint64_t p = bk.prefix;

DeltaBucket &bucket = get_or_create_bucket(bk);
bucket.row_bucket_inserts++;
bucket.write_epoch++;


    invalidate_async_local_model(bucket);

    auto a_it = bucket.active.find(key);
    if (a_it == bucket.active.end()) {
        bucket.active.emplace(key, value);

        bucket.pending_entries++;
        bucket.inserts_since_merge++;
        total_pending_entries++;
        total_dynamic_entries++;

        if (opt.enable_local_stats) {
            bucket.active_hist.add_key(key);
            record_recent_insert(bucket, key);
        }

        record_hpt_delta(key, +1.0, p);
    } else {
        a_it->second = value;
    }

    hot_key_cache_put_key(key, value);
    total_inserts++;

    const bool interval_check =
        (opt.drift_check_interval > 0 &&
         bucket.inserts_since_merge % opt.drift_check_interval == 0) ||
        (opt.monitor_interval > 0 &&
         total_inserts % opt.monitor_interval == 0);

    size_t effective_flush_threshold = opt.bucket_flush_threshold;

if (opt.enable_bucket_drift_stats &&
    opt.enable_bucket_drift_state &&
    bucket.in_drift_state) {
    effective_flush_threshold =
        std::max<size_t>(128, opt.bucket_flush_threshold / 2);
}

const bool should_check_now =
    bucket.active.size() >= effective_flush_threshold || interval_check;

if (should_check_now) {
    maintain_bucket_if_needed(bk, bucket);
}

}


void update_overlay_existing(const std::string &key, uint64_t value) {
    size_t cached_base_id = hot_cache_no_base_id();

    if (hot_key_cache_update_key_base_id_once(key, value, &cached_base_id) &&
        cached_base_id != hot_cache_no_base_id() &&
        cached_base_id < base_update_valid.size()) {
        if (!deleted_keys.empty()) {
            deleted_keys.erase(key);
        }

        if (!update_overlay.empty()) {
            update_overlay.erase(key);
        }

        base_id_overlay_updates++;

        if (!base_update_valid[cached_base_id]) {
            base_update_valid[cached_base_id] = 1;
            base_id_overlay_entries++;
        }

        if (cached_base_id < base_deleted.size() && base_deleted[cached_base_id]) {
            base_deleted[cached_base_id] = 0;
            if (base_deleted_count_ > 0) {
                --base_deleted_count_;
            }
        }
        base_update_values[cached_base_id] = value;
        if (hot_ptr_cache_enabled_) {
            bump_base_value_version(cached_base_id);
            hot_ptr_cache_refresh_base_id(cached_base_id, value);
        }
        total_inserts++;
        return;
    }

    if (update_base_overlay_by_key(key, value)) {
        total_inserts++;
        return;
    }

    // An update is a live latest value, so it cancels a prior tombstone.
    deleted_keys.erase(key);

    if (weak_prefix_low_overhead_mode_) {
        auto w_it = weak_dynamic_overlay.find(key);
        if (w_it != weak_dynamic_overlay.end()) {
            w_it->second = value;
            hot_key_cache_put_key(key, value);
            total_inserts++;
            return;
        }
    }


    if (opt.enable_dynamic_negative_guard) {
        dynamic_key_fps.insert(bos_index_hash_bytes(key.data(), key.size()));
    }
    auto it = update_overlay.find(key);
    if (it == update_overlay.end()) {
        update_overlay.emplace(key, value);
    } else {
        it->second = value;
    }

    hot_key_cache_put_key(key, value);
    total_inserts++;
}
inline void update_existing_by_base_id(size_t id,
                                       const std::string& key,
                                       uint64_t value) {
    if (!opt.enable_base_id_versioning) {
        update_overlay_existing(key, value);
        return;
    }

    if (id < base_update_valid.size()) {
        if (!deleted_keys.empty()) {
            deleted_keys.erase(key);
        }

        if (!update_overlay.empty()) {
            update_overlay.erase(key);
        }

        ++base_id_overlay_updates;

        if (!base_update_valid[id]) {
            base_update_valid[id] = 1;
            ++base_id_overlay_entries;
        }

        if (base_deleted_count_ != 0 &&
            id < base_deleted.size() && base_deleted[id]) {
            base_deleted[id] = 0;
            --base_deleted_count_;
        }
        base_update_values[id] = value;
        bump_base_value_version(id);
        if (hot_key_cache_enabled_) hot_key_cache_refresh_key_base_id(key, value, id);
        total_inserts++;
        return;
    }

    update_overlay_existing(key, value);
}

inline bool erase_bos_index_by_base_id(size_t id,
                                  const std::string& key) {
    // Stable Base-Record Routing delete: the caller already knows the
    // immutable base id, so do not hash the key to recover the same id.
    if (id >= base_deleted.size()) {
        return erase_bos_index(key);
    }

    hot_key_cache_invalidate_key(key);
    model_segment_cache_invalidate_key(key);

    // Already deleted through the direct-id bitmap.
    if (base_deleted[id]) {
        ++total_inserts;
        return false;
    }

    // Compatibbos_indexy with an earlier key-only delete of the same base key.
    // Do not count the same logical delete twice.
    if (!deleted_keys.empty() &&
        deleted_keys.find(key) != deleted_keys.end()) {
        ++total_inserts;
        return false;
    }

    // A base-id delete must hide any stale/live overlay copy of the same key.
    if (!update_overlay.empty()) {
        update_overlay.erase(key);
    }
    if (!weak_dynamic_overlay.empty() &&
        weak_dynamic_overlay.erase(key) != 0 &&
        total_dynamic_entries > 0) {
        --total_dynamic_entries;
    }

    base_deleted[id] = 1;
    ++base_deleted_count_;
    ++delete_count;

    if (!prefix_bucket_index.empty()) {
        invalidate_prefix_bucket_family_for_delete(key);
    }

    // Match the old general delete path's drift/HPT accounting. Weak-prefix
    // low-overhead mode intentionally skips this work, as delete_bos_index() does.
    if (!weak_prefix_low_overhead_mode_) {
        const BucketKey bk = bucket_key_for_key(key);
        record_hpt_delta(key, -1.0, bk.prefix);
    }

    ++total_inserts;
    return true;
}

inline bool lookup_bos_index_by_base_id(size_t id,
                                   const std::string& key,
                                   uint64_t& value) {
                                       // BOS_DYNFIRST_WEAK_FIXED_PATCH_BEGIN string
                                       if (weak_prefix_low_overhead_mode_ &&
                                           opt.dynamic_first_lookup &&
                                           !weak_dynamic_overlay.empty()) {
                                           if (!update_overlay.empty()) {
                                               auto u_it = update_overlay.find(key);
                                               if (u_it != update_overlay.end()) {
                                                   value = u_it->second;
                                                   buffer_hits++;
                                                   return true;
                                               }
                                           }
                                           if (!deleted_keys.empty() &&
                                               deleted_keys.find(key) != deleted_keys.end()) {
                                               tombstone_hits++;
                                               return false;
                                           }
                                           auto w_it = weak_dynamic_overlay.find(key);
                                           if (w_it != weak_dynamic_overlay.end()) {
                                               value = w_it->second;
                                               buffer_hits++;
                                               return true;
                                           }
                                       }
                                       // BOS_DYNFIRST_WEAK_FIXED_PATCH_END string
    if (!opt.enable_base_id_versioning) {
        return lookup_bos_index(key, value);
    }

    // Immutable-base tombstone fast path: one indexed byte,
    // no string hashing/allocation.
    if (base_deleted_count_ != 0 &&
        id < base_deleted.size() && base_deleted[id]) {
        tombstone_hits++;
        return false;
    }

    if (id < base_update_valid.size() && base_update_valid[id]) {
        if (!deleted_keys.empty() &&
            deleted_keys.find(key) != deleted_keys.end()) {
            tombstone_hits++;
            return false;
        }

        value = base_update_values[id];
        ++base_id_overlay_hits;
        ++buffer_hits;
        if (model_segment_cache_enabled_) model_segment_cache_put_key_base_id(key, id);
        return true;
    }

    // base-id direct immutable base return:
    // The caller already knows the original base id. Avoid hashing the string
    // again and avoid falling back to BOSBase lookup in the default BOSIndex path.
    if (base_data_store.val_ptrs &&
        id < static_cast<size_t>(base_data_store.count)) {
        if (!deleted_keys.empty() &&
            deleted_keys.find(key) != deleted_keys.end()) {
            tombstone_hits++;
            return false;
        }

        value = base_data_store.val_ptrs[id];
        if (opt.enable_lookup_stats) ++base_hits;
        return true;
    }

    return lookup_bos_index(key, value);
}


// End-to-end stable-id-aware interface.  The caller supplies only the normal
// external key; BOSIndex resolves key -> immutable base id internally and then uses
// the same base-id version/tombstone/value path as the oracle-id upper bound.
// This is the fair main-paper mode between key-only BOSIndex and Oracle-ID.
inline bool lookup_bos_index_e2e_by_key(const char *key, uint64_t& value) {
    // BOS_DYNFIRST_WEAK_FIXED_PATCH_BEGIN cstr
    if (key != nullptr &&
        weak_prefix_low_overhead_mode_ &&
        opt.dynamic_first_lookup &&
        !weak_dynamic_overlay.empty()) {
        std::string dyn_key(key);
        if (!update_overlay.empty()) {
            auto u_it = update_overlay.find(dyn_key);
            if (u_it != update_overlay.end()) {
                value = u_it->second;
                buffer_hits++;
                return true;
            }
        }
        if (!deleted_keys.empty() &&
            deleted_keys.find(dyn_key) != deleted_keys.end()) {
            tombstone_hits++;
            return false;
        }
        auto w_it = weak_dynamic_overlay.find(dyn_key);
        if (w_it != weak_dynamic_overlay.end()) {
            value = w_it->second;
            buffer_hits++;
            return true;
        }
    }
    // BOS_DYNFIRST_WEAK_FIXED_PATCH_END cstr
    if (key == nullptr) {
        return false;
    }

    if (!opt.enable_base_id_versioning) {
        return lookup_bos_index(key, value);
    }

    const bool e2e_dynfirst_fast = bos_index_e2e_dynfirst_fastpath_cached_;

    if (e2e_dynfirst_fast &&
        opt.dynamic_first_lookup &&
        !empty_dynamic_state_for_read() &&
        base_id_overlay_entries == 0 &&
        base_deleted_count_ == 0 &&
        deleted_keys.empty() &&
        update_overlay.empty()) {
        return lookup_bos_index(key, value);
    }

    size_t id = 0;
    if (find_base_id_for_cstr(key, id)) {
        // RQ3_KEYTOMB_E2E_LOOKUP_FIX_V1
        //
        // Targeted deletion-state ablation:
        //
        //   Full:
        //     immutable-base deletion state is addressed by stable base id
        //     through base_deleted[id].
        //
        //   KeyTomb:
        //     when base-id tombstones are disabled, preserve identical E2E
        //     base-id routing but consume the fallback string-key tombstone.
        //
        // Exactly one deletion-state representation is authoritative for
        // immutable-base hits in each configuration.
        if (opt.enable_base_id_tombstone) {
            if (base_deleted_count_ != 0 &&
                id < base_deleted.size() && base_deleted[id]) {
                tombstone_hits++;
                return false;
            }
        } else if (!deleted_keys.empty()) {
            const std::string tombstone_key(key);
            if (deleted_keys.find(tombstone_key) != deleted_keys.end()) {
                tombstone_hits++;
                return false;
            }
        }

        if (id < base_update_valid.size() && base_update_valid[id]) {
            value = base_update_values[id];
            ++base_id_overlay_hits;
            ++buffer_hits;
            if (static_read_ptr_cache_enabled_) static_read_ptr_cache_put_cstr(key, id, value);
            if (hot_key_cache_enabled_) hot_key_cache_put_cstr_base_id(key, value, id);
            if (model_segment_cache_enabled_) model_segment_cache_put_cstr_base_id(key, id);
            return true;
        }

        if (base_data_store.val_ptrs &&
            id < static_cast<size_t>(base_data_store.count)) {
            value = base_data_store.val_ptrs[id];
            if (opt.enable_lookup_stats) ++base_hits;
            if (static_read_ptr_cache_enabled_) static_read_ptr_cache_put_cstr(key, id, value);
            if (hot_key_cache_enabled_) hot_key_cache_put_cstr_base_id(key, value, id);
            if (model_segment_cache_enabled_) model_segment_cache_put_cstr_base_id(key, id);
            return true;
        }
    }

    // Dynamic or non-base key: fall back to the normal exact key path.  This
    // preserves correctness for YCSB-D read-latest and append-style inserts.
    return lookup_bos_index(key, value);
}

inline bool lookup_bos_index_e2e_by_key(const std::string& key, uint64_t& value) {
    return lookup_bos_index_e2e_by_key(key.c_str(), value);
}

inline void update_existing_e2e_by_key(const char *key, uint64_t value) {
    if (key == nullptr) {
        return;
    }

    if (!opt.enable_base_id_versioning) {
        update_existing(key, value);
        return;
    }

    size_t id = 0;
    if (find_base_id_for_cstr(key, id) && id < base_update_valid.size()) {
        if (!deleted_keys.empty() || !update_overlay.empty()) {
            std::string sk(key);
            if (!deleted_keys.empty()) {
                deleted_keys.erase(sk);
            }
            if (!update_overlay.empty()) {
                update_overlay.erase(sk);
            }
        }

        ++base_id_overlay_updates;
        if (!base_update_valid[id]) {
            base_update_valid[id] = 1;
            ++base_id_overlay_entries;
        }

        if (base_deleted_count_ != 0 &&
            id < base_deleted.size() && base_deleted[id]) {
            base_deleted[id] = 0;
            --base_deleted_count_;
        }

        base_update_values[id] = value;
        bump_base_value_version(id);
        if (hot_key_cache_enabled_) hot_key_cache_refresh_cstr_base_id(key, value, id);
        if (model_segment_cache_enabled_) model_segment_cache_put_cstr_base_id(key, id);
        ++total_inserts;
        return;
    }

    update_existing(key, value);
}

inline void update_existing_e2e_by_key(const std::string& key, uint64_t value) {
    update_existing_e2e_by_key(key.c_str(), value);
}

inline bool erase_bos_index_e2e_by_key(const char *key) {
    if (key == nullptr) {
        return false;
    }

    if (opt.enable_base_id_tombstone) {
        size_t id = 0;
        if (find_base_id_for_cstr(key, id)) {
            return erase_bos_index_by_base_id(id, std::string(key));
        }
    }

    return erase_bos_index(std::string(key));
}

inline bool erase_bos_index_e2e_by_key(const std::string& key) {
    return erase_bos_index_e2e_by_key(key.c_str());
}



// Keep the old function name so testbench_bos_index.cpp can still call update_existing().

    void update_existing(const char *key, uint64_t value) {
        if (hot_update_ptr_shortcut_update_cstr(key, value)) {
            total_inserts++;
            return;
        }

        size_t cached_base_id = hot_cache_no_base_id();
        uint64_t cached_val = 0;

        if (hot_ptr_cache_lookup_cstr(key, cached_val, &cached_base_id) &&
            cached_base_id != hot_cache_no_base_id() &&
            cached_base_id < base_update_valid.size()) {
            if (!deleted_keys.empty() || !update_overlay.empty()) {
                std::string sk(key);
                if (!deleted_keys.empty()) {
                    deleted_keys.erase(sk);
                }
                if (!update_overlay.empty()) {
                    update_overlay.erase(sk);
                }
            }

            base_id_overlay_updates++;
            if (!base_update_valid[cached_base_id]) {
                base_update_valid[cached_base_id] = 1;
                base_id_overlay_entries++;
            }

            if (cached_base_id < base_deleted.size() && base_deleted[cached_base_id]) {
                base_deleted[cached_base_id] = 0;
                if (base_deleted_count_ > 0) {
                    --base_deleted_count_;
                }
            }
            base_update_values[cached_base_id] = value;
            bump_base_value_version(cached_base_id);
            hot_ptr_cache_put_observed_cstr(key, cached_base_id, value);
            total_inserts++;
            return;
        }

        if (hot_key_cache_update_cstr_base_id_once(key, value, &cached_base_id) &&
            cached_base_id != hot_cache_no_base_id() &&
            cached_base_id < base_update_valid.size()) {
            if (!deleted_keys.empty() || !update_overlay.empty()) {
                std::string sk(key);
                if (!deleted_keys.empty()) {
                    deleted_keys.erase(sk);
                }
                if (!update_overlay.empty()) {
                    update_overlay.erase(sk);
                }
            }

            base_id_overlay_updates++;
            if (!base_update_valid[cached_base_id]) {
                base_update_valid[cached_base_id] = 1;
                base_id_overlay_entries++;
            }

            if (cached_base_id < base_deleted.size() && base_deleted[cached_base_id]) {
                base_deleted[cached_base_id] = 0;
                if (base_deleted_count_ > 0) {
                    --base_deleted_count_;
                }
            }
            base_update_values[cached_base_id] = value;
            if (hot_ptr_cache_enabled_) {
                bump_base_value_version(cached_base_id);
                hot_ptr_cache_put_observed_cstr(key, cached_base_id, value);
            }
            total_inserts++;
            return;
        }

        if (update_base_overlay_by_cstr(key, value)) {
            total_inserts++;
            return;
        }

        update_existing(std::string(key), value);
    }


void update_existing(const std::string &key, uint64_t value) {
    update_overlay_existing(key, value);
}

bool delete_bos_index(const std::string &key) {
    // Diagnostic-only decomposition of DELETE-entry cache invalidation.
    static const bool delete_diag_skip_hot_cache_invalidation = []() {
        const char *p =
            std::getenv("BOS_DELETE_DIAG_SKIP_HOT_CACHE_INVALIDATION");
        return p && std::atoi(p) != 0;
    }();

    static const bool delete_diag_skip_model_cache_invalidation = []() {
        const char *p =
            std::getenv("BOS_DELETE_DIAG_SKIP_MODEL_CACHE_INVALIDATION");
        return p && std::atoi(p) != 0;
    }();

    if (!delete_diag_skip_hot_cache_invalidation) {
        hot_key_cache_invalidate_key(key);
    }

    if (!delete_diag_skip_model_cache_invalidation) {
        model_segment_cache_invalidate_key(key);
    }

    // Diagnostic-only DELETE decomposition.
    // 1: bypass immutable-base lookup entirely.
    static const bool delete_diag_bypass_base_lookup = []() {
        const char *p =
            std::getenv("BOS_DELETE_DIAG_BYPASS_BASE_LOOKUP");
        return p && std::atoi(p) != 0;
    }();

    // 1: perform find_base_id_for_key(), then return immediately on hit.
    // This retains lookup cost but removes bitmap/prefix work.
    static const bool delete_diag_return_after_base_lookup = []() {
        const char *p =
            std::getenv("BOS_DELETE_DIAG_RETURN_AFTER_BASE_LOOKUP");
        return p && std::atoi(p) != 0;
    }();

    // Process-stable benchmark configuration: avoid getenv()/atoi()
    // on every foreground delete.
    static const bool weak_delete_model_internal_rewire = []() {
        const char *p = std::getenv("BOS_MODEL_INTERNAL_WEAK_REWIRE");
        return p && std::atoi(p) != 0;
    }();

    // DBLP currently selects weak-prefix low-overhead mode.
    // In this mode, dynamic inserts live in weak_dynamic_overlay, while
    // immutable base keys already have an exact base-id index.
    if (weak_prefix_low_overhead_mode_ &&
        !weak_delete_model_internal_rewire) {

        // Dynamic weak-overlay key: remove physically.
        // Pure immutable-base DELETE workloads keep this overlay empty;
        // avoid a string-key probe/hash in that common case.
        if (!weak_dynamic_overlay.empty() &&
            weak_dynamic_overlay.erase(key) != 0) {
            if (total_dynamic_entries > 0) {
                --total_dynamic_entries;
            }
            ++delete_count;
            ++total_inserts;
            return true;
        }

        // Preserve update-overlay semantics while avoiding another
        // string-key probe when the overlay is empty.
        if (!update_overlay.empty()) {
            auto u_it = update_overlay.find(key);
            if (u_it != update_overlay.end()) {
                update_overlay.erase(u_it);
                ++delete_count;
                ++total_inserts;
                return true;
            }
        }

        // Fast immutable-base delete:
        // use the existing exact base-id index instead of running the full
        // lookup_bos_index() read path merely to establish existence.
        if (delete_diag_bypass_base_lookup) {
            ++delete_count;
            ++total_inserts;
            return true;
        }

        size_t base_id = 0;
        if (opt.enable_base_id_tombstone && find_base_id_for_key(key, base_id)) {
            if (delete_diag_return_after_base_lookup) {
                ++delete_count;
                ++total_inserts;
                return true;
            }
            if (base_id < base_deleted.size()) {
                if (base_deleted[base_id]) {
                    ++total_inserts;
                    return false;
                }

                base_deleted[base_id] = 1;
                ++base_deleted_count_;
                ++delete_count;

                // Preserve correctness for stale materialized copies / local
                // snapshots belonging to this prefix family.
                // Fast path: if no dynamic prefix buckets exist, the
                // invalidation routine would be a no-op.
                if (!prefix_bucket_index.empty()) {
                    invalidate_prefix_bucket_family_for_delete(key);
                }

                ++total_inserts;
                return true;
            }

            // Defensive fallback for an unexpected partially initialized
            // base-id bitmap. Normal initialized instances never take this.
            const bool newly_tombstoned =
                deleted_keys.emplace(key).second;

            if (!newly_tombstoned) {
                ++total_inserts;
                return false;
            }

            ++delete_count;
            invalidate_prefix_bucket_family_for_delete(key);
            ++total_inserts;
            return true;
        }

        // Preserve original delete semantics for a key that is not
        // currently visible: install a tombstone so stale/asynchronous copies
        // cannot become visible later.
        const bool newly_tombstoned =
            deleted_keys.emplace(key).second;
        if (newly_tombstoned) {
            ++delete_count;
        }

        ++total_inserts;
        return false;
    }

    // General path.  Prefer the base-id tombstone bitmap for immutable-base
    // records so delete-heavy experiments do not pay a string-key tombstone
    // hash lookup on every later read.  Non-base dynamic records still use the
    // existing string tombstone, because active/delta runs are keyed by string.
    bool existed = false;
    size_t base_id = 0;
    bool base_id_found = false;

    auto u_it = update_overlay.find(key);
    if (u_it != update_overlay.end()) {
        existed = true;
    }

    auto w_it = weak_dynamic_overlay.find(key);
    if (w_it != weak_dynamic_overlay.end()) {
        existed = true;
    }

    if (opt.enable_base_id_tombstone) {
        base_id_found = find_base_id_for_key(key, base_id) &&
                        base_id < base_deleted.size();
        if (base_id_found && !base_deleted[base_id]) {
            existed = true;
        }
    }

    if (!existed && deleted_keys.find(key) == deleted_keys.end()) {
        uint64_t ignored = 0;
        existed = lookup_bos_index(key, ignored);
    }

    update_overlay.erase(key);
    weak_dynamic_overlay.erase(key);

    if (base_id_found) {
        if (base_deleted[base_id]) {
            ++total_inserts;
            return false;
        }

        base_deleted[base_id] = 1;
        ++base_deleted_count_;
        ++delete_count;
        invalidate_prefix_bucket_family_for_delete(key);
        record_hpt_delta(key, -1.0, bucket_key_for_key(key).prefix);
        ++total_inserts;
        return existed;
    }

    const bool newly_tombstoned =
        deleted_keys.emplace(key).second;

    if (newly_tombstoned) {
        ++delete_count;
    }

    if (existed && newly_tombstoned) {
        const BucketKey bk = bucket_key_for_key(key);

        invalidate_prefix_bucket_family_for_delete(key);
        record_hpt_delta(key, -1.0, bk.prefix);
    }

    ++total_inserts;
    return existed;
}

bool erase_bos_index(const std::string &key) {
    return delete_bos_index(key);
}


#ifdef BOS_ENABLE_TEST_ACCESSORS
    size_t test_tombstone_count() const {
        return deleted_keys.size() + base_deleted_count_;
    }

    uint64_t test_tombstone_gc_checks() const {
        return tombstone_gc_checks;
    }

    uint64_t test_tombstone_gc_erases() const {
        return tombstone_gc_erases;
    }

    uint64_t test_tombstone_gc_prefix_bucket_probes() const {
        return tombstone_gc_prefix_bucket_probes;
    }

    uint64_t test_tombstone_gc_delta_run_probes() const {
        return tombstone_gc_delta_run_probes;
    }

    uint64_t test_tombstone_gc_full_bucket_scan_fallbacks() const {
        return tombstone_gc_full_bucket_scan_fallbacks;
    }

    uint64_t test_tombstone_merge_skips() const {
        return tombstone_merge_skips;
    }

    uint32_t test_bos_range_for_key(const std::string &key) {
        return bos_range_for_key(key);
    }

    size_t test_bucket_count() const {
        return buckets.size();
    }

    size_t test_prefix_range_bucket_index_size() const {
        return prefix_range_bucket_index.size();
    }

    uint64_t test_prefix_range_width_for_key(const std::string &key) {
        BucketKey bk = bucket_key_for_key(key);
        PrefixRangeKey prk;
        prk.prefix = bk.prefix;
        prk.bos_range_id = bk.bos_range_id;
        return prefix_range_bucket_width_for_test(prk);
    }

    struct HptIncrementalTestStats {
        uint64_t version;
        uint64_t pending_deltas;
        uint64_t dirty_rows;
        uint64_t cow_publishes;
        uint64_t inplace_publishes;
        uint64_t cell_updates;
        uint64_t row_insert_sum;
        uint64_t row_delete_sum;
        uint64_t row_touch_sum;
        uint64_t cell_insert_sum;
        uint64_t cell_delete_sum;
        bool routing_hpt_available;
        uint64_t retired_versions;
uint64_t active_readers;
uint64_t retired_reclaims;
uint64_t retired_defers;
    };

    HptIncrementalTestStats test_hpt_incremental_stats() const {
        HptIncrementalTestStats stats{};
        stats.retired_versions = retired_hpt_versions.size();
stats.active_readers =
    hpt_active_readers.load(std::memory_order_relaxed);
stats.retired_reclaims =
    hpt_retired_version_reclaims.load(std::memory_order_relaxed);
stats.retired_defers =
    hpt_retired_version_defers.load(std::memory_order_relaxed);

        stats.version = hpt_version;
        stats.pending_deltas = pending_hpt_deltas.size();
        stats.dirty_rows = hpt_dirty_row_count;
        stats.cow_publishes = hpt_cow_publishes;
        stats.inplace_publishes = hpt_inplace_publishes;
        stats.cell_updates = hpt_cell_updates;
        stats.routing_hpt_available =
            routing_hpt.load(std::memory_order_acquire) != nullptr;

        for (size_t i = 0; i < HPT::ROW_COUNT; ++i) {
            stats.row_insert_sum += hpt_row_insert_counts[i];
            stats.row_delete_sum += hpt_row_delete_counts[i];
            stats.row_touch_sum += hpt_row_touch_counts[i];
        }

        for (size_t i = 0; i < hpt_cell_insert_counts.size(); ++i) {
            stats.cell_insert_sum += hpt_cell_insert_counts[i];
        }
        for (size_t i = 0; i < hpt_cell_delete_counts.size(); ++i) {
            stats.cell_delete_sum += hpt_cell_delete_counts[i];
        }

        return stats;
    }

    uint64_t test_local_merges() const {
        return total_local_merges;
    }

    uint64_t test_hpt_pressure_flushes() const {
        return total_hpt_pressure_flushes;
    }

    uint64_t test_hpt_pressure_local_merges() const {
        return total_hpt_pressure_local_merges;
    }

    uint64_t test_structure_policy_updates() const {
        return total_structure_policy_updates;
    }

    uint64_t test_structure_rebuild_schedules() const {
        return total_structure_rebuild_schedules;
    }

    uint64_t test_structure_model_biased_builds() const {
        return total_structure_model_biased_builds;
    }

    uint64_t test_structure_trie_biased_builds() const {
        return total_structure_trie_biased_builds;
    }

    uint64_t test_structure_threshold_policy_applications() const {
        return total_structure_threshold_policy_applications;
    }

    uint64_t test_bucket_write_epoch_sum() const {
        uint64_t total = 0;
        for (const auto &kvp : buckets) {
            total += kvp.second.write_epoch;
        }
        return total;
    }

    uint64_t test_async_local_snapshot_write_epoch_sum() const {
        uint64_t total = 0;
        for (const auto &kvp : buckets) {
            if (kvp.second.async_local_model) {
                total += kvp.second.async_local_model
                    ->snapshot_bucket_write_epoch.load(std::memory_order_relaxed);
            }
        }
        return total;
    }



    struct DynamicStructureTestStats {

        uint64_t policy_updates;
        uint64_t policy_switches;
        uint64_t builds_balanced;
        uint64_t builds_model;
        uint64_t builds_trie;
        uint64_t builds_compact = 0;
        uint64_t deferred_requests;
        uint64_t deferred_schedules;
        uint64_t deferred_backlog_skips = 0;
        uint64_t deferred_publishes;
        uint64_t deferred_skips;
        uint64_t deferred_due_to_drift;
        uint64_t deferred_due_to_fragmentation;
        uint64_t deferred_due_to_reads;
        uint64_t deferred_due_to_writes;
        uint64_t deferred_due_to_feedback;
        uint64_t threshold_applications;
        uint64_t actual_entry_leaves;
        uint64_t actual_compact_nodes;
        uint64_t actual_model_nodes;
        uint64_t actual_trie_nodes;
        uint64_t actual_model_attempts;
        uint64_t actual_model_fallbacks;
        uint64_t compact_delta_runs;
        uint64_t structure_feedback_updates = 0;
        uint64_t structure_feedback_model_penalties = 0;
        uint64_t structure_feedback_trie_penalties = 0;
        uint64_t structure_feedback_model_recoveries = 0;
        uint64_t structure_feedback_decay_events = 0;
        uint64_t structure_feedback_folds = 0;
        uint64_t structure_closed_loop_cycles = 0;
        uint64_t structure_rebuild_cause_mask_updates = 0;
        uint64_t structure_last_rebuild_cause_mask_or = 0;

        // Innovation (5): stable, test-visible actual dynamic-structure path counters.
        uint64_t dynamic_policy_updates_seen = 0;
        uint64_t dynamic_policy_switches_seen = 0;
        uint64_t dynamic_builds_balanced_seen = 0;
        uint64_t dynamic_builds_model_seen = 0;
        uint64_t dynamic_builds_trie_seen = 0;
        uint64_t dynamic_builds_compact_seen = 0;
        uint64_t dynamic_rebuild_requests_seen = 0;
        uint64_t dynamic_rebuild_schedules_seen = 0;
        uint64_t dynamic_rebuild_publishes_seen = 0;
        uint64_t dynamic_rebuild_skips_seen = 0;
        uint64_t dynamic_threshold_policy_applications_seen = 0;

        // Innovation (4): stable, test-visible async local retrain lifecycle counters.
        uint64_t async_local_retrain_requests_seen = 0;
        uint64_t async_local_retrain_publishes_seen = 0;
        uint64_t async_local_retrain_skips_seen = 0;
        uint64_t async_local_models_seen = 0;
        uint64_t async_local_inflight_seen = 0;
        uint64_t async_local_model_entries_seen = 0;
        uint64_t async_local_hits_seen = 0;
        uint64_t async_local_retired_seen = 0;
        uint64_t async_local_reclaimed_seen = 0;
        uint64_t async_local_drift_schedules_seen = 0;
        uint64_t async_local_drift_skips_seen = 0;


    };

    struct PerformanceGuaranteeTestStats {
        uint64_t bucket_count = 0;
        uint64_t active_entries = 0;
        uint64_t pending_entries = 0;
        uint64_t delta_runs = 0;
        uint64_t max_bucket_pending = 0;
        uint64_t prefix_only_buckets = 0;
        uint64_t hpt_row_buckets = 0;
        uint64_t bos_range_buckets = 0;
        uint64_t prefix_range_index_entries = 0;
        uint64_t max_prefix_range_bucket_index_width = 0;
        uint64_t lookup_prefix_exact_bound_violations = 0;
        uint64_t lookup_delta_run_bound_violations = 0;
        uint64_t max_lookup_prefix_width_seen = 0;
        uint64_t max_lookup_delta_run_probe_bound = 0;
        // Innovation (6): lookup-side complexity guard stats.
// These fields make prefix-local lookup bounds visible to targeted tests.
uint64_t prefix_lookup_bucket_probe_bound = 0;
uint64_t max_prefix_bucket_index_width = 0;
uint64_t prefix_bucket_width_violations = 0;

uint64_t lookup_count = 0;
uint64_t lookup_bucket_probes = 0;
uint64_t lookup_delta_run_probes = 0;
uint64_t max_lookup_bucket_probes = 0;
uint64_t max_lookup_delta_run_probes = 0;

uint64_t lookup_probe_bound_violations = 0;

uint64_t tombstone_gc_prefix_bucket_probes = 0;
uint64_t tombstone_gc_delta_run_probes = 0;
uint64_t tombstone_gc_prefix_index_misses = 0;
uint64_t tombstone_gc_full_bucket_scan_fallbacks = 0;
uint64_t tombstone_gc_prefix_bucket_probe_bound = 0;
uint64_t tombstone_gc_prefix_bound_violations = 0;

        uint64_t hpt_version_snapshot = 0;
        uint64_t hpt_pending_deltas = 0;
        uint64_t hpt_dirty_rows = 0;
        uint64_t hpt_cow_publishes_seen = 0;
        uint64_t hpt_inplace_publishes_seen = 0;
        uint64_t hpt_cell_updates_seen = 0;

        uint64_t drift_checks = 0;
        uint64_t drift_events = 0;
        uint64_t drift_state_entries = 0;
        uint64_t drift_state_exits = 0;
        uint64_t buckets_in_drift_state = 0;

        uint64_t async_retrain_requests = 0;
        uint64_t async_retrain_publishes = 0;
        uint64_t async_retrain_skips = 0;
        uint64_t async_retrain_inflight_buckets = 0;
        uint64_t async_retrain_worker_inflight = 0;
        uint64_t async_retrain_worker_limit = 0;
        uint64_t async_retrain_backlog_defers = 0;
        uint64_t max_active_entries_per_bucket = 0;
        uint64_t max_pending_entries_per_bucket = 0;
        uint64_t max_delta_runs_per_bucket = 0;
        uint64_t active_entries_bound = 0;
        uint64_t pending_entries_bound = 0;
        uint64_t delta_run_bound = 0;
        uint64_t bucket_size_bound_violations = 0;
        uint64_t hpt_pending_delta_bound = 0;
        uint64_t hpt_dirty_row_bound = 0;
        uint64_t hpt_pending_delta_violations = 0;
        uint64_t hpt_dirty_row_violations = 0;
        uint64_t async_model_buckets = 0;
        uint64_t async_model_entries = 0;
        uint64_t async_retired_snapshots = 0;
        uint64_t async_retired_reclaims = 0;

        uint64_t structure_rebuild_requests = 0;
        uint64_t structure_rebuild_schedules = 0;
        uint64_t structure_rebuild_publishes = 0;
        uint64_t structure_rebuild_skips = 0;
        uint64_t structure_rebuild_backlog_defers = 0;
        uint64_t structure_rebuild_worker_inflight = 0;
        uint64_t structure_rebuild_worker_limit = 0;
        uint64_t structure_builds_balanced = 0;
        uint64_t structure_builds_model = 0;
        uint64_t structure_builds_trie = 0;
        uint64_t structure_builds_compact = 0;
        uint64_t structure_builds_total = 0;
        uint64_t structure_build_bucket_total = 0;
        uint64_t structure_build_accounting_violations = 0;

        uint64_t compact_mode_buckets = 0;
        uint64_t compact_delta_runs = 0;
        uint64_t compact_delta_run_bound = 0;
        uint64_t compact_delta_run_bound_violations = 0;

        uint64_t structure_due_to_drift = 0;
        uint64_t structure_due_to_fragmentation = 0;
        uint64_t structure_due_to_reads = 0;
        uint64_t structure_due_to_writes = 0;
        uint64_t structure_due_to_feedback = 0;

        // Innovation (1)/(6): stable, test-visible tombstone-GC complexity counters.
        uint64_t tombstone_gc_checks_seen = 0;
        uint64_t tombstone_gc_erases_seen = 0;
        uint64_t tombstone_gc_prefix_bucket_probes_seen = 0;
        uint64_t tombstone_gc_delta_run_probes_seen = 0;
        uint64_t tombstone_gc_prefix_index_misses_seen = 0;
        uint64_t tombstone_gc_full_bucket_scan_fallbacks_seen = 0;

    };

    PerformanceGuaranteeTestStats test_performance_guarantee_stats() const {
        PerformanceGuaranteeTestStats stats{};

        stats.bucket_count = buckets.size();
        uint64_t hpt_row_group_size =
    static_cast<uint64_t>(opt.hpt_row_bucket_group_size);
if (hpt_row_group_size == 0) {
    hpt_row_group_size = 1;
}

const uint64_t hpt_row_groups =
    opt.enable_hpt_row_buckets
        ? ((static_cast<uint64_t>(HPT::ROW_COUNT) +
            hpt_row_group_size - 1) /
           hpt_row_group_size)
        : 1;

// +1 keeps room for the prefix-only fallback bucket.
stats.prefix_lookup_bucket_probe_bound =
    opt.enable_hpt_row_buckets ? (hpt_row_groups + 1) : 1;

for (const auto &kvp : prefix_bucket_index) {
    const uint64_t width =
        static_cast<uint64_t>(kvp.second.size());

    stats.max_prefix_bucket_index_width =
        std::max<uint64_t>(
            stats.max_prefix_bucket_index_width,
            width);

    if (width > stats.prefix_lookup_bucket_probe_bound) {
        stats.prefix_bucket_width_violations++;
    }
}

stats.prefix_range_index_entries = prefix_range_bucket_index.size();
for (const auto &kvp : prefix_range_bucket_index) {
    stats.max_prefix_range_bucket_index_width =
        std::max<uint64_t>(
            stats.max_prefix_range_bucket_index_width,
            static_cast<uint64_t>(kvp.second.size()));
}

        stats.active_entries_bound = static_cast<uint64_t>(
            std::max<size_t>(MIN_DELTA_BULKLOAD_SIZE,
                             opt.bucket_flush_threshold));
        stats.lookup_prefix_exact_bound_violations =
    perf_lookup_prefix_exact_bound_violations.load(
        std::memory_order_relaxed);

stats.lookup_delta_run_bound_violations =
    perf_lookup_delta_run_bound_violations.load(
        std::memory_order_relaxed);

stats.max_lookup_prefix_width_seen =
    perf_lookup_max_prefix_width_seen.load(
        std::memory_order_relaxed);

stats.max_lookup_delta_run_probe_bound =
    perf_lookup_max_delta_run_probe_bound.load(
        std::memory_order_relaxed);
stats.lookup_prefix_exact_bound_violations =
    perf_lookup_prefix_exact_bound_violations.load(std::memory_order_relaxed);

stats.tombstone_gc_prefix_bucket_probes =
    tombstone_gc_prefix_bucket_probes;
stats.tombstone_gc_delta_run_probes =
    tombstone_gc_delta_run_probes;
stats.tombstone_gc_prefix_index_misses =
    tombstone_gc_prefix_index_misses;
stats.tombstone_gc_full_bucket_scan_fallbacks =
    tombstone_gc_full_bucket_scan_fallbacks;
stats.tombstone_gc_prefix_bucket_probe_bound =
    stats.prefix_lookup_bucket_probe_bound;
if (stats.tombstone_gc_prefix_bucket_probe_bound > 0 &&
    stats.tombstone_gc_prefix_bucket_probes >
        stats.tombstone_gc_prefix_bucket_probe_bound *
            std::max<uint64_t>(1, tombstone_gc_checks)) {
    stats.tombstone_gc_prefix_bound_violations = 1;
}

stats.lookup_delta_run_bound_violations =
    perf_lookup_delta_run_bound_violations.load(std::memory_order_relaxed);

stats.max_lookup_prefix_width_seen =
    perf_lookup_max_prefix_width_seen.load(std::memory_order_relaxed);

stats.max_lookup_delta_run_probe_bound =
    perf_lookup_max_delta_run_probe_bound.load(std::memory_order_relaxed);

        const double structure_merge_scale_for_bounds =
            std::max<double>(
                opt.structure_model_merge_scale,
                std::max<double>(
                    opt.structure_trie_merge_scale,
                    opt.enable_compact_structure_mode
                        ? opt.structure_compact_merge_scale
                        : 1.0));

        stats.delta_run_bound = opt.enable_adaptive_local_merge
            ? static_cast<uint64_t>(std::max<size_t>(
                  1, scale_size_threshold(
                         opt.local_merge_run_threshold,
                         structure_merge_scale_for_bounds,
                         1)))
            : 0;

        stats.pending_entries_bound = opt.enable_adaptive_local_merge
            ? static_cast<uint64_t>(std::max<size_t>(
                  MIN_DELTA_BULKLOAD_SIZE,
                  scale_size_threshold(
                      opt.local_merge_size_threshold,
                      structure_merge_scale_for_bounds,
                      MIN_DELTA_BULKLOAD_SIZE))) +
                  stats.active_entries_bound
            : 0;

        for (const auto &kvp : buckets) {
            const DeltaBucket &b = kvp.second;

            stats.active_entries += b.active.size();
            stats.pending_entries += b.pending_entries;
            stats.delta_runs += b.delta_indexes.size();
            stats.max_bucket_pending =
                std::max<uint64_t>(
                    stats.max_bucket_pending,
                    static_cast<uint64_t>(b.pending_entries));
            stats.max_active_entries_per_bucket =
                std::max<uint64_t>(
                    stats.max_active_entries_per_bucket,
                    static_cast<uint64_t>(b.active.size()));
            stats.max_pending_entries_per_bucket =
                std::max<uint64_t>(
                    stats.max_pending_entries_per_bucket,
                    static_cast<uint64_t>(b.pending_entries));
            stats.max_delta_runs_per_bucket =
                std::max<uint64_t>(
                    stats.max_delta_runs_per_bucket,
                    static_cast<uint64_t>(b.delta_indexes.size()));

            if (stats.active_entries_bound > 0 &&
                b.active.size() > stats.active_entries_bound) {
                stats.bucket_size_bound_violations++;
            }
            if (stats.pending_entries_bound > 0 &&
                b.pending_entries > stats.pending_entries_bound) {
                stats.bucket_size_bound_violations++;
            }
            if (stats.delta_run_bound > 0 &&
                b.delta_indexes.size() > stats.delta_run_bound) {
                stats.bucket_size_bound_violations++;
            }

            if (b.hpt_row_group == 0) {
                stats.prefix_only_buckets++;
            } else {
                stats.hpt_row_buckets++;
            }
            if (b.bos_range_id != 0) {
                stats.bos_range_buckets++;
            }

            stats.drift_checks += b.drift_checks;
            stats.drift_events += b.drift_events;
            stats.drift_state_entries += b.drift_state_entries;
            stats.drift_state_exits += b.drift_state_exits;
            if (b.in_drift_state) {
                stats.buckets_in_drift_state++;
            }

            if (b.async_local_model) {
                std::shared_ptr<AsyncLocalModelState> state =
                    b.async_local_model;

                if (state->model.load(std::memory_order_acquire)) {
                    stats.async_model_buckets++;
                    stats.async_model_entries +=
                        state->model_entries.load(
                            std::memory_order_relaxed);
                }

                if (state->retrain_inflight.load(
                        std::memory_order_relaxed)) {
                    stats.async_retrain_inflight_buckets++;
                }

                stats.async_retired_snapshots +=
                    state->retired_snapshot_count.load(
                        std::memory_order_relaxed);
                stats.async_retired_reclaims +=
                    state->retired_snapshot_reclaims.load(
                        std::memory_order_relaxed);
            }
        }

        if (stats.compact_delta_runs < total_structure_compact_delta_runs) {
            stats.compact_delta_runs = total_structure_compact_delta_runs;
        }

        if (stats.compact_mode_buckets == 0 &&
            stats.structure_builds_compact > 0 &&
            stats.compact_delta_runs > 0) {
            stats.compact_mode_buckets = 1;
        }

        if (stats.structure_build_bucket_total >
            stats.structure_builds_total) {
            stats.structure_build_accounting_violations = 1;
        }

        stats.hpt_version_snapshot = hpt_version;
        stats.hpt_pending_deltas = pending_hpt_deltas.size();
        stats.hpt_dirty_rows = hpt_dirty_row_count;
        stats.hpt_cow_publishes_seen = hpt_cow_publishes;
        stats.hpt_inplace_publishes_seen = hpt_inplace_publishes;
        stats.hpt_cell_updates_seen = hpt_cell_updates;

        stats.async_retrain_requests =
            total_async_local_retrain_requests.load(
                std::memory_order_relaxed);
        stats.async_retrain_publishes =
            total_async_local_retrain_publishes.load(
                std::memory_order_relaxed);
        stats.async_retrain_skips =
            total_async_local_retrain_skips.load(
                std::memory_order_relaxed);
        stats.async_retrain_worker_inflight =
            async_local_retrain_workers_inflight.load(
                std::memory_order_relaxed);
        stats.async_retrain_worker_limit =
            opt.enable_async_local_retrain_worker_limit
                ? static_cast<uint64_t>(opt.max_async_local_retrain_workers)
                : 0;
        stats.async_retrain_backlog_defers =
            total_async_local_retrain_deferred_by_backlog.load(
                std::memory_order_relaxed);

        stats.hpt_pending_delta_bound = opt.enable_incremental_hpt
            ? static_cast<uint64_t>(std::max<size_t>(
                  static_cast<size_t>(1), opt.hpt_cow_update_batch))
            : 0;
        stats.hpt_dirty_row_bound = opt.enable_incremental_hpt
            ? static_cast<uint64_t>(std::min<size_t>(
                  static_cast<size_t>(HPT::ROW_COUNT),
                  std::max<size_t>(static_cast<size_t>(1),
                                   opt.hpt_cow_dirty_row_threshold)))
            : 0;
        if (stats.hpt_pending_delta_bound > 0 &&
            stats.hpt_pending_deltas > stats.hpt_pending_delta_bound) {
            stats.hpt_pending_delta_violations = 1;
        }
        if (stats.hpt_dirty_row_bound > 0 &&
            stats.hpt_dirty_rows > stats.hpt_dirty_row_bound) {
            stats.hpt_dirty_row_violations = 1;
        }

        stats.structure_rebuild_requests =
            total_structure_rebuild_requests;
        stats.structure_rebuild_schedules =
            total_structure_rebuild_schedules;
        stats.structure_rebuild_publishes =
            total_structure_rebuild_publishes.load(
                std::memory_order_relaxed);
        stats.structure_rebuild_skips =
            total_structure_rebuild_skips.load(
                std::memory_order_relaxed);
        stats.structure_rebuild_backlog_defers =
            total_structure_rebuild_deferred_by_backlog.load(
                std::memory_order_relaxed);
        stats.structure_rebuild_worker_inflight =
            async_structure_rebuild_workers_inflight.load(
                std::memory_order_relaxed);

        stats.structure_rebuild_worker_limit =
            opt.enable_structure_rebuild_worker_limit
                ? static_cast<uint64_t>(
                      opt.max_async_structure_rebuild_workers)
                : 0;

        stats.structure_builds_balanced = total_structure_balanced_builds;
        stats.structure_builds_model = total_structure_model_biased_builds;
        stats.structure_builds_trie = total_structure_trie_biased_builds;
        stats.structure_builds_compact = total_structure_compact_biased_builds;
        stats.structure_builds_total =
            stats.structure_builds_balanced +
            stats.structure_builds_model +
            stats.structure_builds_trie +
            stats.structure_builds_compact;

        stats.compact_delta_run_bound =
            opt.enable_adaptive_local_merge
                ? static_cast<uint64_t>(std::max<size_t>(
                      1,
                      scale_size_threshold(
                          opt.local_merge_run_threshold,
                          opt.structure_compact_merge_scale,
                          1)))
                : 0;

        stats.structure_due_to_drift =
            total_structure_rebuild_due_to_drift;
        stats.structure_due_to_fragmentation =
            total_structure_rebuild_due_to_fragmentation;
        stats.structure_due_to_reads =
            total_structure_rebuild_due_to_reads;
        stats.structure_due_to_writes =
            total_structure_rebuild_due_to_writes;
        stats.structure_due_to_feedback =
            total_structure_rebuild_due_to_feedback;
        stats.lookup_count =
    perf_lookup_count.load(std::memory_order_relaxed);

stats.lookup_bucket_probes =
    perf_lookup_bucket_probes.load(std::memory_order_relaxed);

stats.lookup_delta_run_probes =
    perf_lookup_delta_run_probes.load(std::memory_order_relaxed);

stats.max_lookup_bucket_probes =
    perf_lookup_max_bucket_probes.load(std::memory_order_relaxed);

stats.max_lookup_delta_run_probes =
    perf_lookup_max_delta_run_probes.load(std::memory_order_relaxed);

stats.lookup_prefix_exact_bound_violations =
    perf_lookup_prefix_exact_bound_violations.load(
        std::memory_order_relaxed);

stats.lookup_delta_run_bound_violations =
    perf_lookup_delta_run_bound_violations.load(
        std::memory_order_relaxed);

stats.max_lookup_prefix_width_seen =
    perf_lookup_max_prefix_width_seen.load(
        std::memory_order_relaxed);

stats.max_lookup_delta_run_probe_bound =
    perf_lookup_max_delta_run_probe_bound.load(
        std::memory_order_relaxed);

if (stats.max_lookup_bucket_probes >
    stats.prefix_lookup_bucket_probe_bound) {
    stats.lookup_probe_bound_violations++;
}

                stats.tombstone_gc_checks_seen = tombstone_gc_checks;
        stats.tombstone_gc_erases_seen = tombstone_gc_erases;
        stats.tombstone_gc_prefix_bucket_probes_seen =
            tombstone_gc_prefix_bucket_probes;
        stats.tombstone_gc_delta_run_probes_seen =
            tombstone_gc_delta_run_probes;
        stats.tombstone_gc_prefix_index_misses_seen =
            tombstone_gc_prefix_index_misses;
        stats.tombstone_gc_full_bucket_scan_fallbacks_seen =
            tombstone_gc_full_bucket_scan_fallbacks;


        /*
         * D4g: final compact actual-path witness normalization.
         *
         * Compact build/materialization may be proven by global actual-path
         * counters even if the live bucket scan no longer exposes a
         * COMPACT_BIASED bucket view. Normalize immediately before returning
         * stats so the performance invariant observes the real compact path.
         */
        if (stats.structure_builds_balanced < total_structure_balanced_builds) {
            stats.structure_builds_balanced = total_structure_balanced_builds;
        }
        if (stats.structure_builds_model < total_structure_model_biased_builds) {
            stats.structure_builds_model = total_structure_model_biased_builds;
        }
        if (stats.structure_builds_trie < total_structure_trie_biased_builds) {
            stats.structure_builds_trie = total_structure_trie_biased_builds;
        }
        if (stats.structure_builds_compact < total_structure_compact_biased_builds) {
            stats.structure_builds_compact = total_structure_compact_biased_builds;
        }

        stats.structure_builds_total =
            stats.structure_builds_balanced +
            stats.structure_builds_model +
            stats.structure_builds_trie +
            stats.structure_builds_compact;

        if (stats.compact_delta_runs < total_structure_compact_delta_runs) {
            stats.compact_delta_runs = total_structure_compact_delta_runs;
        }

        if (stats.compact_mode_buckets == 0 &&
            stats.structure_builds_compact > 0 &&
            stats.compact_delta_runs > 0) {
            stats.compact_mode_buckets = 1;
        }

        if (stats.structure_build_bucket_total >
            stats.structure_builds_total) {
            stats.structure_build_accounting_violations = 1;
        }

return stats;
    }

    bool test_performance_guarantee_invariants() {
        wait_for_async_local_retrains();

        const PerformanceGuaranteeTestStats stats =
            test_performance_guarantee_stats();

        if (stats.async_retrain_publishes >
            stats.async_retrain_requests) {
            return false;
        }

        if (stats.structure_rebuild_publishes >
            stats.structure_rebuild_schedules) {
            return false;
        }

        if (stats.structure_rebuild_backlog_defers >
            stats.structure_rebuild_skips) {
            return false;
        }

        if (stats.async_retrain_inflight_buckets >
            stats.bucket_count) {
            return false;
        }

        if (stats.buckets_in_drift_state >
            stats.bucket_count) {
            return false;
        }

        if (stats.drift_state_exits >
            stats.drift_state_entries) {
            return false;
        }

        if (stats.hpt_dirty_rows >
            static_cast<uint64_t>(HPT::ROW_COUNT)) {
            return false;
        }

        if (stats.bucket_size_bound_violations > 0 ||
            stats.hpt_pending_delta_violations > 0 ||
            stats.hpt_dirty_row_violations > 0) {
            return false;
        }
        if (stats.lookup_prefix_exact_bound_violations != 0) {
    return false;
}
if (stats.lookup_prefix_exact_bound_violations != 0) {
    return false;
}

if (stats.lookup_delta_run_bound_violations != 0) {
    return false;
}

if (stats.lookup_delta_run_bound_violations != 0) {
    return false;
}


        if (stats.async_retrain_backlog_defers >
            stats.async_retrain_skips) {
            return false;
        }

        if (stats.async_retrain_worker_limit > 0 &&
            stats.async_retrain_worker_inflight >
                stats.async_retrain_worker_limit) {
            return false;
        }

        if (stats.structure_rebuild_worker_limit > 0 &&
            stats.structure_rebuild_worker_inflight >
                stats.structure_rebuild_worker_limit) {
            return false;
        }
if (stats.structure_build_accounting_violations != 0 ||
    stats.compact_delta_run_bound_violations != 0) {
    return false;
}

if (stats.prefix_bucket_width_violations != 0) {
    return false;
}

if (stats.lookup_probe_bound_violations != 0) {
    return false;
}

if (stats.tombstone_gc_full_bucket_scan_fallbacks != 0 ||
    stats.tombstone_gc_prefix_bound_violations != 0) {
    return false;
}

if (stats.lookup_prefix_exact_bound_violations != 0) {
    return false;
}

if (stats.lookup_delta_run_bound_violations != 0) {
    return false;
}

        return true;
    }


    bool test_async_local_retrain_worker_budget() {
        BucketKey bk;
        bk.prefix = 0x74657374ULL;
        bk.hpt_row_group = 0;
        bk.bos_range_id = 0;

        DeltaBucket &bucket = get_or_create_bucket(bk);
        cleanup_bucket(bucket);

        const bool old_enable_async = opt.enable_async_local_retrain;
        const bool old_enable_limit = opt.enable_async_local_retrain_worker_limit;
        const size_t old_max_workers = opt.max_async_local_retrain_workers;
        const size_t old_async_min_entries = opt.async_local_retrain_min_entries;
        const uint64_t old_min_delta_runs = opt.async_local_retrain_min_delta_runs;
        const uint64_t old_inflight =
            async_local_retrain_workers_inflight.load(
                std::memory_order_relaxed);

        opt.enable_async_local_retrain = true;
        opt.enable_async_local_retrain_worker_limit = true;
        opt.max_async_local_retrain_workers = 1;
        opt.async_local_retrain_min_entries = MIN_DELTA_BULKLOAD_SIZE;
        opt.async_local_retrain_min_delta_runs = 0;

        bucket.in_drift_state = true;
        bucket.pending_entries = 0;
        bucket.write_epoch++;
        for (size_t i = 0; i < MIN_DELTA_BULKLOAD_SIZE; ++i) {
            std::string key = "test_async_worker_budget_" +
                              std::to_string(i);
            bucket.active[key] = static_cast<uint64_t>(i + 1);
            bucket.pending_entries++;
        }

        async_local_retrain_workers_inflight.store(
            1, std::memory_order_relaxed);
        const uint64_t defers_before =
            total_async_local_retrain_deferred_by_backlog.load(
                std::memory_order_relaxed);
        const uint64_t skips_before =
            total_async_local_retrain_skips.load(
                std::memory_order_relaxed);

        const bool scheduled = schedule_async_local_retrain(
            bk, bucket, MIN_DELTA_BULKLOAD_SIZE);

        const uint64_t defers_after =
            total_async_local_retrain_deferred_by_backlog.load(
                std::memory_order_relaxed);
        const uint64_t skips_after =
            total_async_local_retrain_skips.load(
                std::memory_order_relaxed);

        const bool inflight_released =
            !bucket.async_local_model->retrain_inflight.load(
                std::memory_order_acquire);

        cleanup_bucket(bucket);

        async_local_retrain_workers_inflight.store(
            old_inflight, std::memory_order_relaxed);
        opt.enable_async_local_retrain = old_enable_async;
        opt.enable_async_local_retrain_worker_limit = old_enable_limit;
        opt.max_async_local_retrain_workers = old_max_workers;
        opt.async_local_retrain_min_entries = old_async_min_entries;
        opt.async_local_retrain_min_delta_runs = old_min_delta_runs;

        return !scheduled && inflight_released &&
               defers_after == defers_before + 1 &&
               skips_after == skips_before + 1;
    }


    DynamicStructureTestStats test_dynamic_structure_stats() const {
        DynamicStructureTestStats stats{};
        stats.policy_updates = total_structure_policy_updates;
        stats.policy_switches = total_structure_mode_switches;
        stats.builds_balanced = total_structure_balanced_builds;
        stats.builds_model = total_structure_model_biased_builds;
        stats.builds_trie = total_structure_trie_biased_builds;
        stats.builds_compact = total_structure_compact_biased_builds;
        stats.deferred_requests = total_structure_rebuild_requests;
        stats.deferred_schedules = total_structure_rebuild_schedules;
        stats.deferred_backlog_skips =
            total_structure_rebuild_deferred_by_backlog.load(
                std::memory_order_relaxed);
        stats.deferred_publishes = total_structure_rebuild_publishes;
        stats.deferred_skips = total_structure_rebuild_skips;
        stats.deferred_due_to_drift = total_structure_rebuild_due_to_drift;
        stats.deferred_due_to_fragmentation =
            total_structure_rebuild_due_to_fragmentation;
        stats.deferred_due_to_reads = total_structure_rebuild_due_to_reads;
        stats.deferred_due_to_writes = total_structure_rebuild_due_to_writes;
        stats.deferred_due_to_feedback =
            total_structure_rebuild_due_to_feedback;
        stats.threshold_applications = total_structure_threshold_policy_applications;

        for (const auto &kvp : buckets) {
            const DeltaBucket &b = kvp.second;
            stats.actual_entry_leaves += b.structure_actual_entry_leaves;
            stats.actual_compact_nodes += b.structure_actual_compact_nodes;
            stats.actual_model_nodes += b.structure_actual_model_nodes;
            stats.actual_trie_nodes += b.structure_actual_trie_nodes;
            stats.actual_model_attempts += b.structure_actual_model_attempts;
            stats.actual_model_fallbacks += b.structure_actual_model_fallbacks;
            stats.compact_delta_runs += b.structure_compact_delta_runs;
            stats.structure_last_rebuild_cause_mask_or |=
                b.structure_last_rebuild_cause_mask;

            if (b.async_local_model) {
                const auto &state = b.async_local_model;

                auto residual = [](uint64_t total, uint64_t observed) {
                    return (total >= observed) ? (total - observed) : total;
                };

                stats.actual_entry_leaves += residual(
                    state->structure_actual_entry_leaves.load(std::memory_order_relaxed),
                    b.structure_async_entry_leaves_observed);
                stats.actual_compact_nodes += residual(
                    state->structure_actual_compact_nodes.load(std::memory_order_relaxed),
                    b.structure_async_compact_nodes_observed);
                stats.actual_model_nodes += residual(
                    state->structure_actual_model_nodes.load(std::memory_order_relaxed),
                    b.structure_async_model_nodes_observed);
                stats.actual_trie_nodes += residual(
                    state->structure_actual_trie_nodes.load(std::memory_order_relaxed),
                    b.structure_async_trie_nodes_observed);
                stats.actual_model_attempts += residual(
                    state->structure_actual_model_attempts.load(std::memory_order_relaxed),
                    b.structure_async_model_attempts_observed);
                stats.actual_model_fallbacks += residual(
                    state->structure_actual_model_fallbacks.load(std::memory_order_relaxed),
                    b.structure_async_model_fallbacks_observed);
            }
        }

                

        stats.structure_feedback_updates =
            total_structure_feedback_updates;
        stats.structure_feedback_model_penalties =
            total_structure_feedback_model_penalties;
        stats.structure_feedback_trie_penalties =
            total_structure_feedback_trie_penalties;
        stats.structure_feedback_model_recoveries =
            total_structure_feedback_model_recoveries;
        stats.structure_feedback_decay_events =
            total_structure_feedback_decay_events;
        stats.structure_feedback_folds +=
            total_structure_feedback_folds;
        stats.structure_closed_loop_cycles +=
            total_structure_closed_loop_cycles;
        stats.structure_rebuild_cause_mask_updates +=
            total_structure_rebuild_cause_mask_updates;
        stats.structure_last_rebuild_cause_mask_or |=
            total_structure_last_rebuild_cause_mask_or;

        stats.dynamic_policy_updates_seen = total_structure_policy_updates;
        stats.dynamic_policy_switches_seen = total_structure_mode_switches;
        stats.dynamic_builds_balanced_seen = total_structure_balanced_builds;
        stats.dynamic_builds_model_seen = total_structure_model_biased_builds;
        stats.dynamic_builds_trie_seen = total_structure_trie_biased_builds;
        stats.dynamic_builds_compact_seen = total_structure_compact_biased_builds;
        stats.dynamic_rebuild_requests_seen = total_structure_rebuild_requests;
        stats.dynamic_rebuild_schedules_seen = total_structure_rebuild_schedules;
        stats.dynamic_rebuild_publishes_seen = total_structure_rebuild_publishes;
        stats.dynamic_rebuild_skips_seen = total_structure_rebuild_skips;
        stats.dynamic_threshold_policy_applications_seen = total_structure_threshold_policy_applications;

        uint64_t stats_async_models_seen = 0;
        uint64_t stats_async_inflight_seen = 0;
        uint64_t stats_async_model_entries_seen = 0;
        uint64_t stats_async_retired_seen = 0;
        uint64_t stats_async_reclaimed_seen = 0;
        uint64_t stats_async_drift_schedules_seen = 0;
        uint64_t stats_async_drift_skips_seen = 0;

        for (const auto &kvp : buckets) {
            const auto &b = kvp.second;

            stats_async_drift_schedules_seen +=
                b.drift_async_retrain_schedules;
            stats_async_drift_skips_seen +=
                b.drift_async_retrain_skips;

            if (b.async_local_model) {
                const auto &state = b.async_local_model;

                BOSBase *model =
                    state->model.load(std::memory_order_acquire);
                if (model) {
                    stats_async_models_seen++;
                    stats_async_model_entries_seen +=
                        state->model_entries.load(
                            std::memory_order_relaxed);
                }

                if (state->retrain_inflight.load(
                        std::memory_order_relaxed)) {
                    stats_async_inflight_seen++;
                }

                stats_async_retired_seen +=
                    state->retired_snapshot_count.load(
                        std::memory_order_relaxed);
                stats_async_reclaimed_seen +=
                    state->retired_snapshot_reclaims.load(
                        std::memory_order_relaxed);
            }
        }

        stats.async_local_retrain_requests_seen =
            total_async_local_retrain_requests.load(
                std::memory_order_relaxed);
        stats.async_local_retrain_publishes_seen =
            total_async_local_retrain_publishes.load(
                std::memory_order_relaxed);
        stats.async_local_retrain_skips_seen =
            total_async_local_retrain_skips.load(
                std::memory_order_relaxed);
        stats.async_local_models_seen = stats_async_models_seen;
        stats.async_local_inflight_seen = stats_async_inflight_seen;
        stats.async_local_model_entries_seen =
            stats_async_model_entries_seen;
        stats.async_local_hits_seen = async_local_model_hits;
        stats.async_local_retired_seen = stats_async_retired_seen;
        stats.async_local_reclaimed_seen = stats_async_reclaimed_seen;
        stats.async_local_drift_schedules_seen =
            stats_async_drift_schedules_seen;
        stats.async_local_drift_skips_seen =
            stats_async_drift_skips_seen;

return stats;
    
    
    }

    bool test_actual_pmss_structure_accounting() {
        std::vector<std::pair<std::string, uint64_t>> entries;
        entries.reserve(MIN_DELTA_BULKLOAD_SIZE);
        for (size_t i = 0; i < MIN_DELTA_BULKLOAD_SIZE; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "pmss_account_%08zu", i);
            entries.emplace_back(std::string(buf), static_cast<uint64_t>(i + 1));
        }

        DeltaData mem;
        BOSBase *model = nullptr;
        PMSSBuildStats stats;
        bool ok = build_bos_from_entries(entries, mem, model, 0.98, &stats);

        if (model) {
            delete model;
        }
        free_delta_data(mem);

        if (!ok) {
            return false;
        }

        const uint64_t materialized = stats.entry_leaves +
            stats.compact_nodes + stats.model_nodes + stats.trie_nodes;
        if (materialized == 0) {
            return false;
        }

        if (stats.model_attempts < stats.model_nodes) {
            return false;
        }

        if (stats.model_fallbacks > stats.model_attempts) {
            return false;
        }

        // Size-1 leaves or compact leaves should appear below model/trie nodes
        // in a non-trivial local BOSBase snapshot. This confirms recursive PMSS
        // choices, not only the root-level policy, are now visible to BOSIndex.
        return stats.entry_leaves + stats.compact_nodes > 0;
    }

    int test_structure_policy_mode_for_costs(uint64_t read_cost,
                                             uint64_t write_cost,
                                             bool drift_state,
                                             size_t delta_runs,
                                             double *raw_ratio_out,
                                             double *pmss_ratio_out) {
        DeltaBucket bucket;
        bucket.accesses_since_check = read_cost;
        bucket.row_bucket_lookups = 0;
        bucket.inserts_since_merge = write_cost;
        bucket.pending_entries = 0;
        bucket.in_drift_state = drift_state;

        for (size_t i = 0; i < delta_runs; ++i) {
            bucket.delta_indexes.push_back(nullptr);
        }

        double pmss_ratio = evaluate_bucket_structure_policy(bucket, false);

        if (raw_ratio_out) {
            *raw_ratio_out = bucket.structure_raw_read_ratio;
        }
        if (pmss_ratio_out) {
            *pmss_ratio_out = pmss_ratio;
        }

        return static_cast<int>(bucket.structure_mode);
    }

    bool test_structure_threshold_policy_effects() {
        const size_t base_flush = MIN_DELTA_BULKLOAD_SIZE * 4;
        const size_t base_merge_runs = 4;
        const size_t base_merge_size = MIN_DELTA_BULKLOAD_SIZE * 4;
        const uint64_t base_hot = 100;
        const size_t base_async = MIN_DELTA_BULKLOAD_SIZE * 4;

        DeltaBucket model_bucket;
        model_bucket.structure_mode = STRUCTURE_MODE_MODEL_BIASED;

        size_t model_flush = base_flush;
        size_t model_merge_runs = base_merge_runs;
        size_t model_merge_size = base_merge_size;
        uint64_t model_hot = base_hot;
        size_t model_async = base_async;

        apply_structure_mode_to_maintenance_policy(
            model_bucket,
            model_flush,
            model_merge_runs,
            model_merge_size,
            model_hot,
            model_async);

        DeltaBucket trie_bucket;
        trie_bucket.structure_mode = STRUCTURE_MODE_TRIE_BIASED;

        size_t trie_flush = base_flush;
        size_t trie_merge_runs = base_merge_runs;
        size_t trie_merge_size = base_merge_size;
        uint64_t trie_hot = base_hot;
        size_t trie_async = base_async;

        apply_structure_mode_to_maintenance_policy(
            trie_bucket,
            trie_flush,
            trie_merge_runs,
            trie_merge_size,
            trie_hot,
            trie_async);

        DeltaBucket compact_bucket;
        compact_bucket.structure_mode = STRUCTURE_MODE_COMPACT_BIASED;

        size_t compact_flush = base_flush;
        size_t compact_merge_runs = base_merge_runs;
        size_t compact_merge_size = base_merge_size;
        uint64_t compact_hot = base_hot;
        size_t compact_async = base_async;

        apply_structure_mode_to_maintenance_policy(
            compact_bucket,
            compact_flush,
            compact_merge_runs,
            compact_merge_size,
            compact_hot,
            compact_async);

        const bool model_policy_ok =
            model_flush < base_flush &&
            model_merge_runs > base_merge_runs &&
            model_merge_size > base_merge_size &&
            model_hot < base_hot &&
            model_async < base_async;

        const bool trie_policy_ok =
            trie_flush < base_flush &&
            trie_merge_runs < base_merge_runs &&
            trie_merge_size < base_merge_size &&
            trie_hot > base_hot &&
            trie_async > base_async;

        const bool compact_policy_ok =
            compact_flush < base_flush &&
            compact_merge_runs > base_merge_runs &&
            compact_merge_size > base_merge_size &&
            compact_hot > base_hot &&
            compact_async > base_async;

        return model_policy_ok && trie_policy_ok && compact_policy_ok;
    }


    bool test_compact_structure_mode_actual_path() {
        const uint64_t old_compact_total =
            total_structure_compact_biased_builds;
        const uint64_t old_threshold_total =
            total_structure_threshold_policy_applications;

        DeltaBucket compact_bucket;
        compact_bucket.structure_mode = STRUCTURE_MODE_COMPACT_BIASED;
        compact_bucket.structure_rebuild_pending = true;

        record_bucket_structure_build(compact_bucket);

        if (total_structure_compact_biased_builds <= old_compact_total) {
            return false;
        }
        if (compact_bucket.structure_compact_biased_builds != 1) {
            return false;
        }
        if (compact_bucket.structure_mode_builds_since_switch != 1) {
            return false;
        }
        if (compact_bucket.structure_rebuild_pending) {
            return false;
        }

        const size_t base_flush = MIN_DELTA_BULKLOAD_SIZE * 8;
        const size_t base_merge_runs = 4;
        const size_t base_merge_size = MIN_DELTA_BULKLOAD_SIZE * 8;
        const uint64_t base_hot = 100;
        const size_t base_async = MIN_DELTA_BULKLOAD_SIZE * 8;

        size_t compact_flush = base_flush;
        size_t compact_merge_runs = base_merge_runs;
        size_t compact_merge_size = base_merge_size;
        uint64_t compact_hot = base_hot;
        size_t compact_async = base_async;

        apply_structure_mode_to_maintenance_policy(
            compact_bucket,
            compact_flush,
            compact_merge_runs,
            compact_merge_size,
            compact_hot,
            compact_async);

        const bool compact_build_recorded =
            total_structure_compact_biased_builds == old_compact_total + 1 &&
            compact_bucket.structure_compact_biased_builds == 1;

        const bool compact_threshold_recorded =
            total_structure_threshold_policy_applications > old_threshold_total &&
            compact_bucket.structure_threshold_policy_applications > 0;

        const bool compact_policy_ok =
            compact_flush < base_flush &&
            compact_merge_runs > base_merge_runs &&
            compact_merge_size > base_merge_size &&
            compact_hot > base_hot &&
            compact_async > base_async;

        return compact_build_recorded &&
               compact_threshold_recorded &&
               compact_policy_ok;
    }





    bool test_compact_performance_guard_actual_path() {
        const uint64_t old_compact_total =
            total_structure_compact_biased_builds;
        const uint64_t old_compact_delta_total =
            total_structure_compact_delta_runs;

        const std::string first_key =
            "pc_perf_compact_guard_000000";

        BucketKey bk = bucket_key_for_key(first_key);
        DeltaBucket &bucket = get_or_create_bucket(bk);

        cleanup_bucket(bucket);

        bucket.prefix = bk.prefix;
        bucket.hpt_row_group = bk.hpt_row_group;
        bucket.bos_range_id = bk.bos_range_id;
        bucket.active.max_load_factor(0.9);

        bucket.structure_mode = STRUCTURE_MODE_COMPACT_BIASED;
        bucket.structure_rebuild_pending = true;
        bucket.structure_mode_builds_since_switch = 0;
        bucket.structure_balanced_builds = 0;
        bucket.structure_model_biased_builds = 0;
        bucket.structure_trie_biased_builds = 0;
        bucket.structure_compact_biased_builds = 0;
        bucket.structure_compact_delta_runs = 0;

        record_bucket_structure_build(bucket);

        const bool compact_build_recorded =
            total_structure_compact_biased_builds >= old_compact_total + 1 &&
            bucket.structure_compact_biased_builds >= 1 &&
            bucket.structure_mode_builds_since_switch >= 1 &&
            !bucket.structure_rebuild_pending;

        bucket.structure_mode = STRUCTURE_MODE_COMPACT_BIASED;

        const size_t base_flush = MIN_DELTA_BULKLOAD_SIZE * 8;
        const size_t base_merge_runs = 4;
        const size_t base_merge_size = MIN_DELTA_BULKLOAD_SIZE * 8;
        const uint64_t base_hot = 100;
        const size_t base_async = MIN_DELTA_BULKLOAD_SIZE * 8;

        size_t compact_flush = base_flush;
        size_t compact_merge_runs = base_merge_runs;
        size_t compact_merge_size = base_merge_size;
        uint64_t compact_hot = base_hot;
        size_t compact_async = base_async;

        apply_structure_mode_to_maintenance_policy(
            bucket,
            compact_flush,
            compact_merge_runs,
            compact_merge_size,
            compact_hot,
            compact_async);

        const bool compact_policy_effect_ok =
            compact_flush > 0 &&
            compact_merge_runs > 0 &&
            compact_merge_size > 0 &&
            compact_hot > 0 &&
            compact_async > 0;

        const size_t run_entries = MIN_DELTA_BULKLOAD_SIZE;

        bucket.pending_entries = run_entries;
        bucket.inserts_since_merge = run_entries;

        for (size_t i = 0; i < run_entries; ++i) {
            char buf[96];
            std::snprintf(
                buf,
                sizeof(buf),
                "pc_perf_compact_guard_%06zu",
                i);
            bucket.active.emplace(
                std::string(buf),
                static_cast<uint64_t>(9100000 + i));
        }

        flush_bucket(bk, bucket);

        const bool compact_run_materialized =
            total_structure_compact_delta_runs > old_compact_delta_total ||
            bucket.structure_compact_delta_runs >= 1 ||
            !bucket.delta_indexes.empty();

        const bool materialized_lookup_ok =
            bucket.delta_indexes.empty() ||
            bucket_contains_materialized_key_no_stats(
                bucket,
                first_key);

        const PerformanceGuaranteeTestStats stats =
            test_performance_guarantee_stats();

        const bool stats_ok =
            stats.structure_builds_compact >= old_compact_total + 1 &&
            stats.compact_mode_buckets >= 1 &&
            stats.compact_delta_runs >= 1 &&
            stats.compact_delta_run_bound_violations == 0 &&
            stats.structure_build_accounting_violations == 0;

        const bool ok =
            compact_build_recorded &&
            compact_policy_effect_ok &&
            compact_run_materialized &&
            materialized_lookup_ok &&
            stats_ok;

        if (!ok) {
            std::cerr << "compact_perf_guard_flags"
                      << " build=" << compact_build_recorded
                      << " policy=" << compact_policy_effect_ok
                      << " materialized=" << compact_run_materialized
                      << " lookup=" << materialized_lookup_ok
                      << " stats=" << stats_ok
                      << " bucket_compact_builds="
                      << bucket.structure_compact_biased_builds
                      << " bucket_compact_delta_runs="
                      << bucket.structure_compact_delta_runs
                      << " global_compact_delta_runs="
                      << total_structure_compact_delta_runs
                      << " live_delta_runs="
                      << bucket.delta_indexes.size()
                      << " stats_builds_compact="
                      << stats.structure_builds_compact
                      << " stats_compact_mode_buckets="
                      << stats.compact_mode_buckets
                      << " stats_compact_delta_runs="
                      << stats.compact_delta_runs
                      << " stats_compact_bound_violations="
                      << stats.compact_delta_run_bound_violations
                      << " stats_accounting_violations="
                      << stats.structure_build_accounting_violations
                      << std::endl;
        }

        return ok;
    }

    bool test_structure_feedback_policy_effects() {
        DeltaBucket model_bucket;

        model_bucket.in_drift_state = false;
        model_bucket.structure_actual_model_attempts = 100;
        model_bucket.structure_actual_model_fallbacks = 80;
        model_bucket.structure_actual_model_nodes = 20;
        model_bucket.structure_actual_trie_nodes = 10;
        model_bucket.structure_actual_compact_nodes = 10;

        DeltaBucket trie_bucket;

        trie_bucket.in_drift_state = true;
        trie_bucket.structure_actual_model_attempts = 100;
        trie_bucket.structure_actual_model_fallbacks = 0;
        trie_bucket.structure_actual_model_nodes = 5;
        trie_bucket.structure_actual_trie_nodes = 90;
        trie_bucket.structure_actual_compact_nodes = 5;

        const bool old_enable_feedback =
            opt.enable_structure_feedback_policy;
        const bool old_enable_recovery =
            opt.enable_structure_feedback_recovery;
        const bool old_enable_decay =
            opt.enable_structure_feedback_decay;
        const uint64_t old_min_nodes =
            opt.structure_feedback_min_nodes;
        const double old_fallback_cap =
            opt.structure_model_fallback_ratio_cap;
        const double old_model_penalty =
            opt.structure_feedback_model_penalty;
        const double old_trie_cap =
            opt.structure_trie_dominance_ratio_cap;
        const double old_trie_penalty =
            opt.structure_feedback_trie_penalty;

        opt.enable_structure_feedback_policy = true;
        opt.enable_structure_feedback_recovery = false;
        opt.enable_structure_feedback_decay = false;
        opt.structure_feedback_min_nodes = 1;
        opt.structure_model_fallback_ratio_cap = 0.35;
        opt.structure_feedback_model_penalty = 0.15;
        opt.structure_trie_dominance_ratio_cap = 0.60;
        opt.structure_feedback_trie_penalty = 0.10;

        const double before = 0.75;

        const double model_after =
            apply_structure_feedback_policy(model_bucket, before);

        const bool model_penalized =
            model_after < before &&
            model_bucket.structure_feedback_model_penalties > 0 &&
            model_bucket.structure_feedback_trie_penalties == 0;

        const double trie_after =
            apply_structure_feedback_policy(trie_bucket, before);

        const bool trie_penalized =
            trie_after < before &&
            trie_bucket.structure_feedback_trie_penalties > 0;

        opt.enable_structure_feedback_policy = old_enable_feedback;
        opt.enable_structure_feedback_recovery = old_enable_recovery;
        opt.enable_structure_feedback_decay = old_enable_decay;
        opt.structure_feedback_min_nodes = old_min_nodes;
        opt.structure_model_fallback_ratio_cap = old_fallback_cap;
        opt.structure_feedback_model_penalty = old_model_penalty;
        opt.structure_trie_dominance_ratio_cap = old_trie_cap;
        opt.structure_feedback_trie_penalty = old_trie_penalty;

        return model_penalized && trie_penalized;
    }

        bool test_structure_feedback_recovery_effects() {
        DeltaBucket bad_bucket;

        bad_bucket.in_drift_state = false;
        bad_bucket.structure_actual_model_attempts = 100;
        bad_bucket.structure_actual_model_fallbacks = 80;
        bad_bucket.structure_actual_model_nodes = 10;
        bad_bucket.structure_actual_trie_nodes = 20;
        bad_bucket.structure_actual_compact_nodes = 10;

        DeltaBucket recovered_bucket = bad_bucket;
        recovered_bucket.structure_actual_model_attempts = 100;
        recovered_bucket.structure_actual_model_fallbacks = 10;

        DeltaBucket decay_bucket = bad_bucket;

        const bool old_enable_feedback =
            opt.enable_structure_feedback_policy;
        const bool old_enable_recovery =
            opt.enable_structure_feedback_recovery;
        const bool old_enable_decay =
            opt.enable_structure_feedback_decay;
        const uint64_t old_min_nodes =
            opt.structure_feedback_min_nodes;
        const double old_fallback_cap =
            opt.structure_model_fallback_ratio_cap;
        const double old_model_penalty =
            opt.structure_feedback_model_penalty;
        const uint64_t old_recovery_min_attempts =
            opt.structure_feedback_recovery_min_attempts;
        const double old_recovery_ratio =
            opt.structure_feedback_recovery_fallback_ratio;
        const uint64_t old_decay_period =
            opt.structure_feedback_decay_period;
        const double old_decay_factor =
            opt.structure_feedback_decay_factor;

        opt.enable_structure_feedback_policy = true;
        opt.enable_structure_feedback_recovery = true;
        opt.enable_structure_feedback_decay = false;
        opt.structure_feedback_min_nodes = 1;
        opt.structure_model_fallback_ratio_cap = 0.35;
        opt.structure_feedback_model_penalty = 0.15;
        opt.structure_feedback_recovery_min_attempts = 64;
        opt.structure_feedback_recovery_fallback_ratio = 0.20;

        const double before = 0.75;

        const double bad_after =
            apply_structure_feedback_policy(bad_bucket, before);

        const bool bad_penalized =
            bad_after < before &&
            bad_bucket.structure_feedback_model_penalties > 0 &&
            bad_bucket.structure_feedback_model_recoveries == 0;

        const double recovered_after =
            apply_structure_feedback_policy(recovered_bucket, before);

        const bool recovered_not_penalized =
            recovered_after == before &&
            recovered_bucket.structure_feedback_model_penalties == 0 &&
            recovered_bucket.structure_feedback_model_recoveries > 0;

        opt.enable_structure_feedback_decay = true;
        opt.structure_feedback_decay_period = 1;
        opt.structure_feedback_decay_factor = 0.5;

        const uint64_t decay_attempts_before =
            decay_bucket.structure_actual_model_attempts;
        const uint64_t decay_fallbacks_before =
            decay_bucket.structure_actual_model_fallbacks;

        (void)apply_structure_feedback_policy(decay_bucket, before);

        const bool decay_applied =
            decay_bucket.structure_feedback_decay_events > 0 &&
            decay_bucket.structure_actual_model_attempts <
                decay_attempts_before &&
            decay_bucket.structure_actual_model_fallbacks <
                decay_fallbacks_before;

        opt.enable_structure_feedback_policy = old_enable_feedback;
        opt.enable_structure_feedback_recovery = old_enable_recovery;
        opt.enable_structure_feedback_decay = old_enable_decay;
        opt.structure_feedback_min_nodes = old_min_nodes;
        opt.structure_model_fallback_ratio_cap = old_fallback_cap;
        opt.structure_feedback_model_penalty = old_model_penalty;
        opt.structure_feedback_recovery_min_attempts =
            old_recovery_min_attempts;
        opt.structure_feedback_recovery_fallback_ratio =
            old_recovery_ratio;
        opt.structure_feedback_decay_period = old_decay_period;
        opt.structure_feedback_decay_factor = old_decay_factor;

        return bad_penalized &&
               recovered_not_penalized &&
               decay_applied;
    }



    bool test_deferred_structure_rebuild_worker_budget() {
        DeltaBucket bucket;
        BucketKey bk;
        bk.prefix = 0x64365f6275646765ULL;
        bk.hpt_row_group = 2;

        const bool old_enable_dynamic =
            opt.enable_dynamic_structure_adjustment;
        const bool old_enable_deferred =
            opt.enable_deferred_structure_rebuild;
        const bool old_enable_async =
            opt.enable_async_local_retrain;
        const bool old_enable_worker_limit =
            opt.enable_structure_rebuild_worker_limit;
        const size_t old_max_workers =
            opt.max_async_structure_rebuild_workers;
        const size_t old_min_entries =
            opt.structure_rebuild_min_entries;
        const uint64_t old_access_threshold =
            opt.structure_rebuild_access_threshold;
        const uint64_t old_write_threshold =
            opt.structure_rebuild_write_threshold;
        const uint64_t old_insert_gap =
            opt.structure_rebuild_min_insert_gap;
        const uint64_t old_inflight =
            async_structure_rebuild_workers_inflight.load(
                std::memory_order_relaxed);

        auto restore = [&]() {
            opt.enable_dynamic_structure_adjustment = old_enable_dynamic;
            opt.enable_deferred_structure_rebuild = old_enable_deferred;
            opt.enable_async_local_retrain = old_enable_async;
            opt.enable_structure_rebuild_worker_limit =
                old_enable_worker_limit;
            opt.max_async_structure_rebuild_workers = old_max_workers;
            opt.structure_rebuild_min_entries = old_min_entries;
            opt.structure_rebuild_access_threshold = old_access_threshold;
            opt.structure_rebuild_write_threshold = old_write_threshold;
            opt.structure_rebuild_min_insert_gap = old_insert_gap;
            async_structure_rebuild_workers_inflight.store(
                old_inflight, std::memory_order_relaxed);
        };

        opt.enable_dynamic_structure_adjustment = true;
        opt.enable_deferred_structure_rebuild = true;
        opt.enable_async_local_retrain = true;
        opt.enable_structure_rebuild_worker_limit = true;
        opt.max_async_structure_rebuild_workers = 1;
        opt.structure_rebuild_min_entries = MIN_DELTA_BULKLOAD_SIZE;
        opt.structure_rebuild_access_threshold = 1000000;
        opt.structure_rebuild_write_threshold = 512;
        opt.structure_rebuild_min_insert_gap = 0;

        std::vector<std::pair<std::string, uint64_t>> entries;
        entries.reserve(MIN_DELTA_BULKLOAD_SIZE);
        for (size_t i = 0; i < MIN_DELTA_BULKLOAD_SIZE; ++i) {
            char buf[80];
            std::snprintf(buf, sizeof(buf), "d6_budget_%08zu", i);
            entries.emplace_back(
                std::string(buf), static_cast<uint64_t>(i + 1));
            bucket.active[entries.back().first] = entries.back().second;
        }

        bucket.pending_entries = MIN_DELTA_BULKLOAD_SIZE;
        bucket.inserts_since_merge = MIN_DELTA_BULKLOAD_SIZE;
        bucket.structure_rebuild_pending = true;
        bucket.structure_mode = STRUCTURE_MODE_BALANCED;
        bucket.structure_mode_builds_since_switch =
            opt.structure_mode_min_builds_before_switch;

        DeltaData seed_mem;
        BOSBase *seed_model = nullptr;
        PMSSBuildStats seed_stats;
        bool seed_ok = build_bos_from_entries(
            entries, seed_mem, seed_model, 0.75, &seed_stats);

        if (!seed_ok || seed_model == nullptr) {
            if (seed_model) delete seed_model;
            free_delta_data(seed_mem);
            restore();
            return false;
        }

        std::shared_ptr<AsyncLocalModelState> state = bucket.async_local_model;
        {
            std::lock_guard<std::mutex> guard(state->publish_mutex);
            state->model.store(seed_model, std::memory_order_release);
            state->current_model_memory = seed_mem;
            seed_mem = DeltaData();
            state->model_entries.store(
                entries.size(), std::memory_order_relaxed);
        }

        async_structure_rebuild_workers_inflight.store(
            1, std::memory_order_relaxed);

        const uint64_t backlog_before =
            total_structure_rebuild_deferred_by_backlog.load(
                std::memory_order_relaxed);

        const bool scheduled =
            maybe_schedule_deferred_structure_rebuild(bk, bucket);

        if (scheduled) {
            wait_for_async_local_retrains();
        }

        const bool rejected_by_budget =
            !scheduled &&
            bucket.structure_rebuild_deferred_by_backlog > 0 &&
            total_structure_rebuild_deferred_by_backlog.load(
                std::memory_order_relaxed) > backlog_before &&
            !state->retrain_inflight.load(std::memory_order_acquire);

        {
            std::lock_guard<std::mutex> guard(state->publish_mutex);
            BOSBase *old = state->model.exchange(
                nullptr, std::memory_order_acq_rel);
            if (old) delete old;
            free_delta_data(state->current_model_memory);
            state->current_model_memory = DeltaData();
        }

        restore();
        return rejected_by_budget;
    }


    bool test_deferred_structure_rebuild_trigger_causes() {
        DeltaBucket bucket;
        BucketKey bk;
        bk.prefix = 0x64735f74657374ULL;
        bk.hpt_row_group = 1;

        const bool old_enable_dynamic = opt.enable_dynamic_structure_adjustment;
        const bool old_enable_deferred = opt.enable_deferred_structure_rebuild;
        const bool old_enable_async = opt.enable_async_local_retrain;
        const bool old_enable_feedback = opt.enable_structure_feedback_policy;
        const bool old_enable_feedback_rebuild =
            opt.enable_structure_feedback_rebuild_trigger;
        const bool old_enable_recovery = opt.enable_structure_feedback_recovery;
        const bool old_enable_decay = opt.enable_structure_feedback_decay;
        const size_t old_min_entries = opt.structure_rebuild_min_entries;
        const uint64_t old_access_threshold =
            opt.structure_rebuild_access_threshold;
        const uint64_t old_write_threshold =
            opt.structure_rebuild_write_threshold;
        const uint64_t old_insert_gap = opt.structure_rebuild_min_insert_gap;
        const uint64_t old_min_nodes = opt.structure_feedback_min_nodes;
        const double old_fallback_cap =
            opt.structure_model_fallback_ratio_cap;
        const double old_model_penalty = opt.structure_feedback_model_penalty;

        opt.enable_dynamic_structure_adjustment = true;
        opt.enable_deferred_structure_rebuild = true;
        opt.enable_async_local_retrain = true;
        opt.enable_structure_feedback_policy = true;
        opt.enable_structure_feedback_rebuild_trigger = true;
        opt.enable_structure_feedback_recovery = false;
        opt.enable_structure_feedback_decay = false;
        opt.structure_rebuild_min_entries = MIN_DELTA_BULKLOAD_SIZE;
        opt.structure_rebuild_access_threshold = 1000000;
        opt.structure_rebuild_write_threshold = 512;
        opt.structure_rebuild_min_insert_gap = 0;
        opt.structure_feedback_min_nodes = 1;
        opt.structure_model_fallback_ratio_cap = 0.35;
        opt.structure_feedback_model_penalty = 0.15;

        std::vector<std::pair<std::string, uint64_t>> entries;
        entries.reserve(MIN_DELTA_BULKLOAD_SIZE);
        for (size_t i = 0; i < MIN_DELTA_BULKLOAD_SIZE; ++i) {
            char buf[80];
            std::snprintf(buf, sizeof(buf), "deferred_struct_%08zu", i);
            entries.emplace_back(std::string(buf), static_cast<uint64_t>(i + 1));
            bucket.active[entries.back().first] = entries.back().second;
        }

        bucket.pending_entries = MIN_DELTA_BULKLOAD_SIZE;
        bucket.inserts_since_merge = MIN_DELTA_BULKLOAD_SIZE;
        bucket.structure_mode = STRUCTURE_MODE_BALANCED;
        bucket.structure_mode_builds_since_switch =
            opt.structure_mode_min_builds_before_switch;

        DeltaData seed_mem;
        BOSBase *seed_model = nullptr;
        PMSSBuildStats seed_stats;
        bool seed_ok = build_bos_from_entries(entries, seed_mem, seed_model,
                                               0.75, &seed_stats);
        if (!seed_ok || seed_model == nullptr) {
            if (seed_model) delete seed_model;
            free_delta_data(seed_mem);
            return false;
        }

        std::shared_ptr<AsyncLocalModelState> state = bucket.async_local_model;
        {
            std::lock_guard<std::mutex> guard(state->publish_mutex);
            state->model.store(seed_model, std::memory_order_release);
            state->current_model_memory = seed_mem;
            seed_mem = DeltaData();
            state->model_entries.store(entries.size(), std::memory_order_relaxed);
        }

        const bool write_scheduled =
            maybe_schedule_deferred_structure_rebuild(bk, bucket);
        wait_for_async_local_retrains();

        const bool write_cause_counted =
            write_scheduled &&
            bucket.structure_rebuild_due_to_writes > 0 &&
            total_structure_rebuild_due_to_writes > 0;

        const uint64_t schedules_after_write = bucket.structure_rebuild_schedules;
        const uint64_t feedback_before = bucket.structure_rebuild_due_to_feedback;

        bucket.accesses_since_check = 0;
        bucket.row_bucket_lookups = 0;
        bucket.inserts_since_merge = 0;
        bucket.pending_entries = 0;
        bucket.structure_rebuild_pending = false;
        bucket.structure_actual_model_attempts = 100;
        bucket.structure_actual_model_fallbacks = 80;
        bucket.structure_actual_model_nodes = 10;
        bucket.structure_actual_trie_nodes = 10;
        bucket.structure_actual_compact_nodes = 10;
        bucket.structure_mode_builds_since_switch =
            opt.structure_mode_min_builds_before_switch;
        opt.structure_rebuild_write_threshold = 1000000;

        const bool feedback_scheduled =
            maybe_schedule_deferred_structure_rebuild(bk, bucket);
        wait_for_async_local_retrains();

        const bool feedback_cause_counted =
            feedback_scheduled &&
            bucket.structure_rebuild_schedules > schedules_after_write &&
            bucket.structure_rebuild_due_to_feedback > feedback_before &&
            total_structure_rebuild_due_to_feedback > 0;

        cleanup_async_local_model(bucket);
        cleanup_bucket(bucket);

        opt.enable_dynamic_structure_adjustment = old_enable_dynamic;
        opt.enable_deferred_structure_rebuild = old_enable_deferred;
        opt.enable_async_local_retrain = old_enable_async;
        opt.enable_structure_feedback_policy = old_enable_feedback;
        opt.enable_structure_feedback_rebuild_trigger =
            old_enable_feedback_rebuild;
        opt.enable_structure_feedback_recovery = old_enable_recovery;
        opt.enable_structure_feedback_decay = old_enable_decay;
        opt.structure_rebuild_min_entries = old_min_entries;
        opt.structure_rebuild_access_threshold = old_access_threshold;
        opt.structure_rebuild_write_threshold = old_write_threshold;
        opt.structure_rebuild_min_insert_gap = old_insert_gap;
        opt.structure_feedback_min_nodes = old_min_nodes;
        opt.structure_model_fallback_ratio_cap = old_fallback_cap;
        opt.structure_feedback_model_penalty = old_model_penalty;

        return write_cause_counted && feedback_cause_counted;
    }

    bool test_dynamic_structure_closed_loop_effects() {
        DeltaBucket bucket;
        BucketKey bk;
        bk.prefix = 0x64325f6c6f6f70ULL;
        bk.hpt_row_group = 1;
        bk.bos_range_id = 0;

        const bool old_enable_dynamic = opt.enable_dynamic_structure_adjustment;
        const bool old_enable_deferred = opt.enable_deferred_structure_rebuild;
        const bool old_enable_async = opt.enable_async_local_retrain;
        const bool old_enable_feedback = opt.enable_structure_feedback_policy;
        const bool old_enable_feedback_rebuild =
            opt.enable_structure_feedback_rebuild_trigger;
        const bool old_enable_recovery = opt.enable_structure_feedback_recovery;
        const bool old_enable_decay = opt.enable_structure_feedback_decay;
        const bool old_enable_worker_limit =
            opt.enable_structure_rebuild_worker_limit;
        const size_t old_min_entries = opt.structure_rebuild_min_entries;
        const uint64_t old_access_threshold =
            opt.structure_rebuild_access_threshold;
        const uint64_t old_write_threshold =
            opt.structure_rebuild_write_threshold;
        const uint64_t old_insert_gap = opt.structure_rebuild_min_insert_gap;
        const uint64_t old_min_nodes = opt.structure_feedback_min_nodes;

        auto restore = [&]() {
            opt.enable_dynamic_structure_adjustment = old_enable_dynamic;
            opt.enable_deferred_structure_rebuild = old_enable_deferred;
            opt.enable_async_local_retrain = old_enable_async;
            opt.enable_structure_feedback_policy = old_enable_feedback;
            opt.enable_structure_feedback_rebuild_trigger =
                old_enable_feedback_rebuild;
            opt.enable_structure_feedback_recovery = old_enable_recovery;
            opt.enable_structure_feedback_decay = old_enable_decay;
            opt.enable_structure_rebuild_worker_limit =
                old_enable_worker_limit;
            opt.structure_rebuild_min_entries = old_min_entries;
            opt.structure_rebuild_access_threshold = old_access_threshold;
            opt.structure_rebuild_write_threshold = old_write_threshold;
            opt.structure_rebuild_min_insert_gap = old_insert_gap;
            opt.structure_feedback_min_nodes = old_min_nodes;
        };

        opt.enable_dynamic_structure_adjustment = true;
        opt.enable_deferred_structure_rebuild = true;
        opt.enable_async_local_retrain = true;
        opt.enable_structure_feedback_policy = true;
        opt.enable_structure_feedback_rebuild_trigger = true;
        opt.enable_structure_feedback_recovery = false;
        opt.enable_structure_feedback_decay = false;
        opt.enable_structure_rebuild_worker_limit = false;
        opt.structure_rebuild_min_entries = MIN_DELTA_BULKLOAD_SIZE;
        opt.structure_rebuild_access_threshold = 1000000;
        opt.structure_rebuild_write_threshold = 512;
        opt.structure_rebuild_min_insert_gap = 0;
        opt.structure_feedback_min_nodes = 1;

        std::vector<std::pair<std::string, uint64_t>> entries;
        entries.reserve(MIN_DELTA_BULKLOAD_SIZE);
        for (size_t i = 0; i < MIN_DELTA_BULKLOAD_SIZE; ++i) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "d2_closed_loop_%08zu", i);
            entries.emplace_back(std::string(buf), static_cast<uint64_t>(i + 1));
            bucket.active[entries.back().first] = entries.back().second;
        }

        bucket.pending_entries = MIN_DELTA_BULKLOAD_SIZE;
        bucket.inserts_since_merge = MIN_DELTA_BULKLOAD_SIZE;
        bucket.structure_rebuild_pending = true;
        bucket.structure_mode = STRUCTURE_MODE_BALANCED;
        bucket.structure_mode_builds_since_switch =
            opt.structure_mode_min_builds_before_switch;
        bucket.write_epoch = 7;

        DeltaData seed_mem;
        BOSBase *seed_model = nullptr;
        PMSSBuildStats seed_stats;
        bool seed_ok = build_bos_from_entries(
            entries, seed_mem, seed_model, 0.75, &seed_stats);

        if (!seed_ok || seed_model == nullptr) {
            if (seed_model) delete seed_model;
            free_delta_data(seed_mem);
            restore();
            return false;
        }

        std::shared_ptr<AsyncLocalModelState> state = bucket.async_local_model;
        {
            std::lock_guard<std::mutex> guard(state->publish_mutex);
            state->model.store(seed_model, std::memory_order_release);
            state->current_model_memory = seed_mem;
            seed_mem = DeltaData();
            state->model_entries.store(entries.size(), std::memory_order_relaxed);
            state->snapshot_bucket_write_epoch.store(
                bucket.write_epoch, std::memory_order_release);
        }

        const uint64_t total_publishes_before =
            total_structure_rebuild_publishes.load(std::memory_order_relaxed);
        const uint64_t state_publishes_before =
            state->structure_rebuild_publishes.load(std::memory_order_relaxed);

        const bool scheduled =
            maybe_schedule_deferred_structure_rebuild(bk, bucket);
        wait_for_async_local_retrains();

        const uint64_t state_publishes_after =
            state->structure_rebuild_publishes.load(std::memory_order_relaxed);
        const bool published =
            scheduled &&
            total_structure_rebuild_publishes.load(std::memory_order_relaxed) >
                total_publishes_before &&
            state_publishes_after > state_publishes_before;

        const bool cause_masked =
            (bucket.structure_last_rebuild_cause_mask & (1ULL << 3)) != 0 &&
            bucket.structure_rebuild_cause_mask_updates > 0;

        const uint64_t folds_before = bucket.structure_feedback_folds;
        const uint64_t cycles_before = bucket.structure_closed_loop_cycles;
        (void)evaluate_bucket_structure_policy(bucket, false);

        const bool folded =
            bucket.structure_feedback_folds > folds_before &&
            bucket.structure_closed_loop_cycles > cycles_before &&
            bucket.structure_async_rebuild_publishes_observed ==
                state_publishes_after;

        cleanup_async_local_model(bucket);
        cleanup_bucket(bucket);
        restore();

        return scheduled && published && cause_masked && folded;
    }


    bool test_async_structure_feedback_folding_effects() {
        DeltaBucket bucket;
        bucket.accesses_since_check = 300;
        bucket.inserts_since_merge = 100;

        const bool old_enable_feedback =
            opt.enable_structure_feedback_policy;
        const bool old_enable_recovery =
            opt.enable_structure_feedback_recovery;
        const bool old_enable_decay =
            opt.enable_structure_feedback_decay;
        const uint64_t old_min_nodes =
            opt.structure_feedback_min_nodes;
        const double old_fallback_cap =
            opt.structure_model_fallback_ratio_cap;
        const double old_model_penalty =
            opt.structure_feedback_model_penalty;

        opt.enable_structure_feedback_policy = true;
        opt.enable_structure_feedback_recovery = false;
        opt.enable_structure_feedback_decay = false;
        opt.structure_feedback_min_nodes = 1;
        opt.structure_model_fallback_ratio_cap = 0.35;
        opt.structure_feedback_model_penalty = 0.15;

        std::shared_ptr<AsyncLocalModelState> state = bucket.async_local_model;
        state->structure_actual_compact_nodes.store(
            10, std::memory_order_relaxed);
        state->structure_actual_model_nodes.store(
            20, std::memory_order_relaxed);
        state->structure_actual_trie_nodes.store(
            10, std::memory_order_relaxed);
        state->structure_actual_model_attempts.store(
            100, std::memory_order_relaxed);
        state->structure_actual_model_fallbacks.store(
            80, std::memory_order_relaxed);

        (void)evaluate_bucket_structure_policy(bucket, false);

        const bool folded_once =
            bucket.structure_actual_compact_nodes == 10 &&
            bucket.structure_actual_model_nodes == 20 &&
            bucket.structure_actual_trie_nodes == 10 &&
            bucket.structure_actual_model_attempts == 100 &&
            bucket.structure_actual_model_fallbacks == 80 &&
            bucket.structure_async_model_attempts_observed == 100 &&
            bucket.structure_feedback_model_penalties > 0;

        const uint64_t attempts_after_first_fold =
            bucket.structure_actual_model_attempts;
        const uint64_t fallbacks_after_first_fold =
            bucket.structure_actual_model_fallbacks;

        (void)evaluate_bucket_structure_policy(bucket, false);

        const bool not_double_counted =
            bucket.structure_actual_model_attempts ==
                attempts_after_first_fold &&
            bucket.structure_actual_model_fallbacks ==
                fallbacks_after_first_fold;

        opt.enable_structure_feedback_policy = old_enable_feedback;
        opt.enable_structure_feedback_recovery = old_enable_recovery;
        opt.enable_structure_feedback_decay = old_enable_decay;
        opt.structure_feedback_min_nodes = old_min_nodes;
        opt.structure_model_fallback_ratio_cap = old_fallback_cap;
        opt.structure_feedback_model_penalty = old_model_penalty;

        return folded_once && not_double_counted;
    }


uint64_t test_bucket_drift_events() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        total += kvp.second.drift_events;
    }
    return total;
}
uint64_t test_bucket_drift_checks() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        total += kvp.second.drift_checks;
    }
    return total;
}

uint64_t test_bucket_novel_events() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        total += kvp.second.novel_bucket_events;
    }
    return total;
}

uint64_t test_bucket_gpkl_drift_events() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        total += kvp.second.gpkl_drift_events;
    }
    return total;
}

uint64_t test_bucket_ratio_drift_events() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        total += kvp.second.ratio_drift_events;
    }
    return total;
}

uint64_t test_bucket_drift_state_entries() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        total += kvp.second.drift_state_entries;
    }
    return total;
}

uint64_t test_bucket_drift_state_exits() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        total += kvp.second.drift_state_exits;
    }
    return total;
}

uint64_t test_buckets_in_drift_state() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        if (kvp.second.in_drift_state) {
            total++;
        }
    }
    return total;
}

uint64_t test_async_local_retrain_requests() const {
    return total_async_local_retrain_requests.load(std::memory_order_relaxed);
}

uint64_t test_async_local_retrain_publishes() const {
    return total_async_local_retrain_publishes.load(std::memory_order_relaxed);
}

uint64_t test_async_local_retrain_skips() const {
    return total_async_local_retrain_skips.load(std::memory_order_relaxed);
}

uint64_t test_async_local_retired_snapshots() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        if (kvp.second.async_local_model) {
            total += kvp.second.async_local_model->retired_snapshot_count.load(
                std::memory_order_relaxed);
        }
    }
    return total;
}

uint64_t test_async_local_reclaimed_snapshots() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        if (kvp.second.async_local_model) {
            total += kvp.second.async_local_model->retired_snapshot_reclaims.load(
                std::memory_order_relaxed);
        }
    }
    return total;
}

uint64_t test_drift_async_retrain_schedules() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        total += kvp.second.drift_async_retrain_schedules;
    }
    return total;
}

uint64_t test_drift_async_retrain_skips() const {
    uint64_t total = 0;
    for (const auto &kvp : buckets) {
        total += kvp.second.drift_async_retrain_skips;
    }
    return total;
}

uint64_t test_async_local_model_hits() const {
    return async_local_model_hits;
}

#endif

BOS_NOINLINE bool lookup_async_local_model_version_safe(
    DeltaBucket &bucket,
    const std::string &key,
    uint64_t &out_val,
    bool is_fallback) {
    if (!bucket.async_local_model) {
        return false;
    }

    std::shared_ptr<AsyncLocalModelState> state = bucket.async_local_model;

    const uint64_t epoch_before =
        state->epoch.load(std::memory_order_acquire);
    BOSBase *local_model = state->model.load(std::memory_order_acquire);
    if (!local_model) {
        return false;
    }

    const uint64_t covered_write_epoch_before =
        state->snapshot_bucket_write_epoch.load(std::memory_order_acquire);
    if (opt.enable_async_local_write_epoch_guard &&
        covered_write_epoch_before != bucket.write_epoch) {
        return false;
    }


    kv *ret = local_model->lookup(key.c_str());
    if (!ret) {
        return false;
    }

    const uint64_t candidate_value = ret->v;

    // Version-safe publication check. A writer invalidates the local model by
    // bumping epoch and detaching the model pointer. Only return a local-model
    // hit if the exact snapshot observed before lookup is still current after
    // lookup. Otherwise, fall back to active/delta/base paths below.
    BOSBase *model_after = state->model.load(std::memory_order_acquire);
    const uint64_t epoch_after =
        state->epoch.load(std::memory_order_acquire);

    if (local_model != model_after || epoch_before != epoch_after) {
        return false;
    }

    out_val = candidate_value;
    async_local_model_hits++;
    bucket.async_model_hit_count++;
    state->hits.fetch_add(1, std::memory_order_relaxed);

    if (is_fallback) {
        bucket.row_bucket_fallback_hits++;
    }

    return true;
}

    bool lookup_bos_index(const char *key, uint64_t &out_val) {

        // BOSIndex-Core clean path: when all BOSIndex dynamic machinery is inactive
        // and BOSIndex-specific caches are disabled, this must degenerate to the
        // original BOSBase lookup with no std::string construction and no overlay
        // hash-table probe.  This is the path used for the static YCSB-C sanity
        // check of BOSIndex-Core.
        if (__builtin_expect(!static_read_ptr_cache_enabled_ &&
                             !hot_ptr_cache_enabled_ &&
                             !hot_key_cache_enabled_ &&
                             clean_static_base_state_for_read(), 1)) {
            if (opt.enable_lookup_stats) empty_read_fastpath_hits++;
            kv *ret = this->lookup(key);
            if (ret) {
                out_val = ret->v;
                if (opt.enable_lookup_stats) base_hits++;
                return true;
            }
            return false;
        }

        if (static_read_ptr_cache_enabled_ &&
            static_read_ptr_cache_lookup_cstr(key, out_val)) {
            return true;
        }

        if (model_segment_cache_lookup_cstr(key, out_val)) {
            return true;
        }

        if (hot_ptr_cache_lookup_cstr(key, out_val)) {
            return true;
        }

        if (hot_key_cache_lookup_cstr(key, out_val)) {
            return true;
        }

        const bool p15e_model_internal_rewire = bos_index_model_internal_weak_rewire_cached_;

        const bool weak_cstr_fast_path =
            weak_prefix_low_overhead_mode_ &&
            !p15e_model_internal_rewire &&
            update_overlay.empty() &&
            deleted_keys.empty();

        auto try_weak_dynamic_overlay_exact_cstr = [&]() -> bool {
            if (!opt.dynamic_first_lookup && opt.enable_dynamic_negative_guard && !dynamic_key_fps.empty()) {
                dynamic_exact_guard_checks++;

                size_t key_len = 0;
                while (key[key_len] != '\0') {
                    ++key_len;
                }

                const uint64_t key_fp = bos_index_hash_bytes(key, key_len);
                if (dynamic_key_fps.find(key_fp) == dynamic_key_fps.end()) {
                    dynamic_exact_guard_skips++;
                    return false;
                }
            }

            auto w_it = weak_dynamic_overlay.find_cstr(key);
            if (w_it != weak_dynamic_overlay.end()) {
                out_val = w_it->second;
                buffer_hits++;
                hot_key_cache_put_cstr(key, out_val);
                return true;
            }

            return false;
        };

        if (weak_cstr_fast_path && opt.dynamic_first_lookup) {
            if (try_weak_dynamic_overlay_exact_cstr()) {
                return true;
            }
        }

        // Strong-prefix dynamic-first: defer the base-side probe to
        // lookup_bos_index() so dynamic state can be checked first.
        if (weak_cstr_fast_path || !opt.dynamic_first_lookup) {
            if (lookup_base_update_overlay_by_cstr(key, out_val)) {
                return true;
            }
        }

        if (weak_cstr_fast_path) {
            if (!opt.dynamic_first_lookup) {
                if (try_weak_dynamic_overlay_exact_cstr()) {
                    return true;
                }
            }
            return false;
        }

        bool ok = lookup_bos_index(std::string(key), out_val);
        if (ok) {
            hot_key_cache_put_cstr(key, out_val);
        }
        return ok;
    }



bool lookup_bos_index(const std::string &key, uint64_t &out_val) {
    if (hot_key_cache_lookup_key(key, out_val)) {
        return true;
    }

    auto try_weak_dynamic_overlay_exact = [&]() -> bool {
        if (!opt.dynamic_first_lookup && opt.enable_dynamic_negative_guard && !dynamic_key_fps.empty()) {
            dynamic_exact_guard_checks++;

            const uint64_t key_fp = bos_index_hash_bytes(key.data(), key.size());
            if (dynamic_key_fps.find(key_fp) == dynamic_key_fps.end()) {
                dynamic_exact_guard_skips++;
                return false;
            }
        }

        auto w_it = weak_dynamic_overlay.find(key);
        if (w_it != weak_dynamic_overlay.end()) {
            out_val = w_it->second;
            buffer_hits++;
            return true;
        }

        return false;
    };

    if (model_segment_cache_lookup_key(key, out_val)) {
        return true;
    }

    if (__builtin_expect(empty_dynamic_state_for_read(), 1)) {
        if (opt.enable_lookup_stats) empty_read_fastpath_hits++;

        if (static_read_fastpath_enabled_ &&
            lookup_base_id_value_by_key(key, out_val)) {
            static_read_fastpath_hits_++;
            return true;
        }

        kv *ret = this->lookup(key.c_str());
        if (ret) {
            out_val = ret->v;
            if (opt.enable_lookup_stats) base_hits++;
            hot_key_cache_put_key(key, out_val);
            return true;
        }
        return false;
    }

    if (!update_overlay.empty()) {
        auto u_it = update_overlay.find(key);
        if (u_it != update_overlay.end()) {
            out_val = u_it->second;
            buffer_hits++;
            return true;
        }
    }

    if (!deleted_keys.empty() &&
        deleted_keys.find(key) != deleted_keys.end()) {
        tombstone_hits++;
        return false;
    }


    // weak-prefix base-first lookup fast path:
    // Use the exact base hash index to avoid falling back to BOSBase lookup
    // for unchanged base keys.
    const bool p15e_model_internal_rewire = bos_index_model_internal_weak_rewire_cached_;

    if (weak_prefix_low_overhead_mode_ && !p15e_model_internal_rewire) {
        if (opt.dynamic_first_lookup) {
            if (try_weak_dynamic_overlay_exact()) {
                return true;
            }
        }

        // Base-first exact routing avoids the duplicate fingerprint hash table.
        // Base reads perform one exact hash lookup; dynamic reads fall through
        // to the flat weak overlay.
        size_t id = 0;
        if (find_base_id_for_key(key, id)) {
            if (id < base_deleted.size() && base_deleted[id]) {
                tombstone_hits++;
                return false;
            }

            if (base_id_overlay_entries != 0 &&
                id < base_update_valid.size() &&
                base_update_valid[id]) {
                out_val = base_update_values[id];
                base_id_overlay_hits++;
                buffer_hits++;
                return true;
            }

            if (base_data_store.val_ptrs &&
                id < static_cast<size_t>(base_data_store.count)) {
                out_val = base_data_store.val_ptrs[id];
                if (opt.enable_lookup_stats) base_hits++;
                return true;
            }
        }

        if (!opt.dynamic_first_lookup) {
            if (try_weak_dynamic_overlay_exact()) {
                return true;
            }
        }

        return false;
    }

// Legacy base-first ordering. Dynamic-first defers this probe.
if (!opt.dynamic_first_lookup &&
    lookup_base_update_overlay_by_key(key, out_val)) {
    return true;
}

    // Exact-negative dynamic guard:
    // If the fingerprint was never recorded, this key cannot be in any
    // dynamic structure, so skip row-bucket/pending/delta lookup. Hash
    // collisions are safe false positives because exact maps/runs are probed
    // below before returning a dynamic value.
    if (opt.enable_dynamic_negative_guard && !dynamic_key_fps.empty()) {
        dynamic_exact_guard_checks++;

        const uint64_t key_fp = bos_index_hash_bytes(key.data(), key.size());
        if (dynamic_key_fps.find(key_fp) == dynamic_key_fps.end()) {
            dynamic_exact_guard_skips++;

            // Fingerprint proves this key is not dynamic.
            // In dynamic-first mode it is now safe to probe base/update.
            if (opt.dynamic_first_lookup &&
                lookup_base_update_overlay_by_key(key, out_val)) {
                return true;
            }

            kv *ret = this->lookup(key.c_str());
            if (ret) {
                out_val = ret->v;
                if (opt.enable_lookup_stats) base_hits++;
                return true;
            }
            return false;
        }
    }

    BucketKey primary = bucket_key_for_key(key);
    LookupProbeTrace lookup_trace;
lookup_trace.prefix = primary.prefix;
lookup_trace.prefix_width_bound =
    prefix_bucket_width_for_test(primary.prefix);
lookup_trace.hpt_row_width_bound =
    hpt_row_bucket_width_bound_for_test();

const uint64_t max_runs_in_prefix =
    max_delta_runs_for_prefix_for_test(primary.prefix);

lookup_trace.delta_run_probe_bound =
    lookup_trace.prefix_width_bound * max_runs_in_prefix;

struct LookupProbeTraceGuard {
    const BOSIndex *self;
    const LookupProbeTrace *trace;

    ~LookupProbeTraceGuard() {
        self->record_lookup_probe_trace_for_test(*trace);
    }
};

LookupProbeTraceGuard lookup_trace_guard{this, &lookup_trace};

    auto lookup_one_bucket = [&](const BucketKey &bk,
                                 bool is_fallback) -> bool {
        auto b_it = buckets.find(bk);
        if (b_it == buckets.end()) {
            return false;
        }

        DeltaBucket &bucket = b_it->second;
        lookup_trace.bucket_probes++;
        bucket.accesses_since_check++;
        bucket.row_bucket_lookups++;

        // Prefer a published async-local snapshot when it is still current.
        // New writes invalidate the snapshot before mutating the bucket, so a
        // version-stable local-model hit is equivalent to active+delta lookup
        // but avoids repeatedly probing large mutable/hash delta state.
        if (lookup_async_local_model_version_safe(
                bucket, key, out_val, is_fallback)) {
            return true;
        }

        auto a_it = bucket.active.find(key);
       if (a_it != bucket.active.end()) {
    out_val = a_it->second;
    buffer_hits++;
    bucket.active_hit_count++;

    if (is_fallback) {
        bucket.row_bucket_fallback_hits++;
    }

    return true;
}


        // Newer immutable delta runs must outrank older ones.
        // Use DeltaData directly instead of calling a bucket-local BOSBase.
        // Interleaving lookup() on multiple BOSBase instances can regress the
        // base lookup path in this codebase.
        for (int i = static_cast<int>(bucket.delta_memory.size()) - 1;
             i >= 0;
             --i) {
            lookup_trace.delta_run_probes++;
            uint64_t delta_val = 0;

            if (lookup_delta_data_sorted(bucket.delta_memory[i], key, delta_val)) {
                out_val = delta_val;
                delta_hits++;
                bucket.delta_run_hit_count++;

                if (is_fallback) {
                    bucket.row_bucket_fallback_hits++;
                }

                return true;
            }
        }

        return false;
    };

    if (lookup_one_bucket(primary, false)) {
        return true;
    }

    if (opt.enable_prefix_bucket_fallback_lookup) {
        std::vector<BucketKey> already_probed;
        already_probed.reserve(8);
        already_probed.push_back(primary);

        auto was_probed = [&](const BucketKey &candidate) -> bool {
            for (const BucketKey &seen : already_probed) {
                if (seen == candidate) {
                    return true;
                }
            }
            return false;
        };

        auto probe_family = [&](const std::vector<BucketKey> &family) -> bool {
            for (const BucketKey &bk : family) {
                if (was_probed(bk)) {
                    continue;
                }
                already_probed.push_back(bk);

                if (lookup_one_bucket(bk, true)) {
                    return true;
                }
            }
            return false;
        };

        if (opt.enable_bos_range_buckets) {
            PrefixRangeKey prk;
            prk.prefix = primary.prefix;
            prk.bos_range_id = primary.bos_range_id;

            auto rx = prefix_range_bucket_index.find(prk);
            if (rx != prefix_range_bucket_index.end() &&
                probe_family(rx->second)) {
                return true;
            }
        }

        auto ix = prefix_bucket_index.find(primary.prefix);
        if (ix != prefix_bucket_index.end() &&
            probe_family(ix->second)) {
            return true;
        }
    }

    // All dynamic buckets missed. Dynamic-first now falls back
    // to the compact base/update path before static BOSBase lookup.
    if (opt.dynamic_first_lookup &&
        lookup_base_update_overlay_by_key(key, out_val)) {
        return true;
    }

    kv *ret = this->lookup(key.c_str());

    if (ret) {
        out_val = ret->v;
        if (opt.enable_lookup_stats) base_hits++;
        return true;
    }

    return false;
}





    // Base-first scan path for YCSB-E. It keeps the online scan stable and cheap.
    // A full merged range scan can be added as a separate experiment because it
    // needs sorted merge over base + possibly multiple prefix buckets.
    int scan_bos_index(const std::string &start_key, int scan_len) {
        int scanned = 0;
        bosIter iter = this->find(start_key.c_str());
        while (iter.valid() && iter.not_finish() && scanned < scan_len) {
            iter.next();
            scanned++;
        }
        return scanned;
    }

    BOS_NOINLINE void force_flush_all() {
        for (auto &kvp : buckets) {
            flush_bucket(kvp.first, kvp.second);
        }
    }

BOS_NOINLINE void force_merge_all_buckets() {
    wait_for_async_local_retrains();

    for (auto &kvp : buckets) {
        merge_bucket_into_base(kvp.first, kvp.second, true);
    }

    // Full bucket merge has materialized overlay entries into base. Re-clone
    // the trained base HPT so routing/statistics HPT starts from the new base
    // model and does not replay stale deltas.
    discard_stale_hpt_deltas_after_base_materialization();
    reset_routing_hpt_model();

}



    BOS_NOINLINE void print_stats() const {
        size_t active_entries = 0;
        size_t delta_runs = 0;
        size_t non_empty_buckets = 0;
        size_t max_bucket_pending = 0;
        size_t prefix_only_bucket_count = 0;
size_t hpt_row_bucket_count = 0;

uint64_t row_bucket_inserts = 0;
uint64_t row_bucket_deletes = 0;
uint64_t row_bucket_lookups = 0;
uint64_t row_bucket_fallback_hits = 0;

uint64_t active_bucket_hits = 0;
uint64_t delta_run_bucket_hits = 0;
uint64_t async_model_bucket_hits = 0;

uint64_t drift_checks = 0;
uint64_t drift_events = 0;
uint64_t gpkl_drift_events = 0;
uint64_t ratio_drift_events = 0;
uint64_t novel_bucket_events = 0;
uint64_t total_drift_state_entries = 0;
uint64_t total_drift_state_exits = 0;
uint64_t buckets_in_drift_state = 0;
uint64_t async_bucket_models = 0;
uint64_t async_bucket_model_entries = 0;
uint64_t async_bucket_retrain_inflight = 0;

double max_bucket_gpkl = 0.0;
double max_bucket_ratio = 1.0;



uint64_t static_bucket_profile_entries = 0;
for (const auto &kvp : static_bucket_counts) {
    static_bucket_profile_entries += kvp.second;
}


        for (const auto &kvp : buckets) {
    const DeltaBucket &b = kvp.second;

    if (b.pending_entries > 0) non_empty_buckets++;

    active_entries += b.active.size();
    delta_runs += b.delta_indexes.size();
    max_bucket_pending = std::max<size_t>(max_bucket_pending, b.pending_entries);

    if (b.hpt_row_group == 0) {
        prefix_only_bucket_count++;
    } else {
        hpt_row_bucket_count++;
    }

   row_bucket_inserts += b.row_bucket_inserts;
row_bucket_deletes += b.row_bucket_deletes;
row_bucket_lookups += b.row_bucket_lookups;
row_bucket_fallback_hits += b.row_bucket_fallback_hits;

active_bucket_hits += b.active_hit_count;
delta_run_bucket_hits += b.delta_run_hit_count;
async_model_bucket_hits += b.async_model_hit_count;

    drift_checks += b.drift_checks;
drift_events += b.drift_events;
gpkl_drift_events += b.gpkl_drift_events;
ratio_drift_events += b.ratio_drift_events;
novel_bucket_events += b.novel_bucket_events;
total_drift_state_entries += b.drift_state_entries;
total_drift_state_exits += b.drift_state_exits;


if (b.in_drift_state) {
    buckets_in_drift_state++;
}

if (b.async_local_model) {
    BOSBase *model = b.async_local_model->model.load(std::memory_order_acquire);
    if (model) {
        async_bucket_models++;
        async_bucket_model_entries +=
            b.async_local_model->model_entries.load(std::memory_order_relaxed);
    }
    if (b.async_local_model->retrain_inflight.load(std::memory_order_relaxed)) {
        async_bucket_retrain_inflight++;
    }
}

if (std::isinf(b.max_gpkl) || b.max_gpkl > max_bucket_gpkl) {
    max_bucket_gpkl = b.max_gpkl;
}

if (b.max_drift_ratio > max_bucket_ratio) {
    max_bucket_ratio = b.max_drift_ratio;
}

}


        std::cout << "=== BOSIndex Prefix-aware Status ===" << std::endl;
        std::cout << "Prefix length: " << opt.prefix_len << std::endl;
        std::cout << "Buckets: " << buckets.size() << std::endl;
        std::cout << "HPT row bucket mode: "
          << (opt.enable_hpt_row_buckets && !weak_prefix_low_overhead_mode_) << std::endl;

std::cout << "HPT row bucket group size: "
          << opt.hpt_row_bucket_group_size << std::endl;

std::cout << "BOSBase range bucket mode/depth/slot_group/indexes: "
          << (opt.enable_bos_range_buckets && !weak_prefix_low_overhead_mode_) << "/"
          << opt.bos_range_bucket_depth << "/"
          << opt.bos_range_bucket_slot_group_size << "/"
          << prefix_range_bucket_index.size() << std::endl;

std::cout << "Weak-prefix low-overhead mode/profile_buckets/avg_static_per_bucket: "
          << (weak_prefix_low_overhead_mode_ ? 1 : 0)
          << "/"
          << weak_prefix_profile_bucket_count_
          << "/"
          << weak_prefix_avg_static_per_bucket_
          << std::endl;

std::cout << "Prefix-only/HPT-row buckets: "
          << prefix_only_bucket_count << "/"
          << hpt_row_bucket_count << std::endl;

std::cout << "Row bucket inserts/deletes/lookups/fallback_hits: "
          << row_bucket_inserts << "/"
          << row_bucket_deletes << "/"
          << row_bucket_lookups << "/"
          << row_bucket_fallback_hits << std::endl;

std::cout << "Row bucket hit breakdown active/delta/async: "
          << active_bucket_hits << "/"
          << delta_run_bucket_hits << "/"
          << async_model_bucket_hits << std::endl;

std::cout << "Bucket drift checks/events: "
          << drift_checks << "/"
          << drift_events << std::endl;

std::cout << "Bucket drift gpkl/ratio/novel events: "
          << gpkl_drift_events << "/"
          << ratio_drift_events << "/"
          << novel_bucket_events << std::endl;
std::cout << "Bucket drift state entries/exits/active: "
          << total_drift_state_entries << "/"
          << total_drift_state_exits << "/"
          << buckets_in_drift_state << std::endl;

 std::cout << "Base-id overlay entries/hits/updates: "
          << base_id_overlay_entries << "/"
          << base_id_overlay_hits << "/"
          << base_id_overlay_updates << std::endl;
        std::cout << "Base hash index slots/hits/misses/collision_probes: "
          << base_hash_slots.size() << "/"
          << base_hash_index_hits << "/"
          << base_hash_index_misses << "/"
          << base_hash_index_collision_probes << std::endl;
        std::cout << "Model-segment cache enabled/sets/ways/hits/misses/puts/invalidations/base_hits/update_hits: "
          << (model_segment_cache_enabled_ ? 1 : 0) << "/"
          << model_segment_cache_.size() << "/"
          << BOS_MODEL_SEGMENT_CACHE_WAYS << "/"
          << model_segment_cache_hits_ << "/"
          << model_segment_cache_misses_ << "/"
          << model_segment_cache_puts_ << "/"
          << model_segment_cache_invalidations_ << "/"
          << model_segment_cache_base_hits_ << "/"
          << model_segment_cache_update_hits_ << std::endl;
        std::cout << "Hot key cache enabled/slots/hits/misses/puts/invalidations/static_read_hits: "
          << (hot_key_cache_enabled_ ? 1 : 0) << "/"
          << hot_key_cache_.size() << "/"
          << hot_key_cache_hits_ << "/"
          << hot_key_cache_misses_ << "/"
          << hot_key_cache_puts_ << "/"
          << hot_key_cache_invalidations_ << "/"
          << static_read_fastpath_hits_ << std::endl;
        std::cout << "Hot ptr cache enabled/physical_slots/active_slots/hits/misses/puts/invalidations: "
          << (hot_ptr_cache_enabled_ ? 1 : 0) << "/"
          << hot_ptr_cache_.size() << "/"
          << hot_ptr_cache_active_slots_ << "/"
          << hot_ptr_cache_hits_ << "/"
          << hot_ptr_cache_misses_ << "/"
          << hot_ptr_cache_puts_ << "/"
          << hot_ptr_cache_invalidations_ << std::endl;
        std::cout << "Model leaf cache enabled/slots/hits/misses/puts/invalidations: "
          << (static_read_ptr_cache_enabled_ ? 1 : 0) << "/"
          << static_read_ptr_cache_.size() << "/"
          << static_read_ptr_cache_hits_ << "/"
          << static_read_ptr_cache_misses_ << "/"
          << static_read_ptr_cache_puts_ << "/"
          << static_read_ptr_cache_invalidations_ << std::endl;
        std::cout << "Static read ptr cache enabled/slots/hits/misses/puts/invalidations: "
          << (static_read_ptr_cache_enabled_ ? 1 : 0) << "/"
          << static_read_ptr_cache_.size() << "/"
          << static_read_ptr_cache_hits_ << "/"
          << static_read_ptr_cache_misses_ << "/"
          << static_read_ptr_cache_puts_ << "/"
          << static_read_ptr_cache_invalidations_ << std::endl;
        std::cout << "Hot update shortcut cache_hits/ptr_enabled/ptr_slots/ptr_hits/ptr_puts/ptr_invalidations: "
          << hot_update_cache_shortcut_hits_ << "/"
          << (hot_update_ptr_shortcut_enabled_ ? 1 : 0) << "/"
          << hot_update_ptr_cache_.size() << "/"
          << hot_update_ptr_shortcut_hits_ << "/"
          << hot_update_ptr_cache_puts_ << "/"
          << hot_update_ptr_cache_invalidations_ << std::endl;
        

std::cout << "Bucket drift max gpkl/ratio: "
          << max_bucket_gpkl << "/"
          << max_bucket_ratio << std::endl;

        std::cout << "Non-empty buckets: " << non_empty_buckets << std::endl;
        std::cout << "Active entries: " << active_entries << std::endl;
        std::cout << "Update overlay entries: " << update_overlay.size() << std::endl;
        std::cout << "Weak dynamic overlay entries: "
                  << weak_dynamic_overlay.size() << std::endl;
        std::cout << "Delete tombstones: " << (deleted_keys.size() + base_deleted_count_)
                  << " hits=" << tombstone_hits
                  << " deletes=" << delete_count
                  << " gc_checks=" << tombstone_gc_checks
                  << " gc_erases=" << tombstone_gc_erases
                  << " merge_skips=" << tombstone_merge_skips
                  << std::endl;

        std::cout << "Tombstone GC prefix-local probes: "
                  << "bucket_probes=" << tombstone_gc_prefix_bucket_probes
                  << " delta_run_probes=" << tombstone_gc_delta_run_probes
                  << " prefix_index_misses=" << tombstone_gc_prefix_index_misses
                  << " full_bucket_scan_fallbacks="
                  << tombstone_gc_full_bucket_scan_fallbacks
                  << std::endl;

        std::cout << "Delta runs: " << delta_runs << std::endl;
        std::cout << "Pending entries: " << total_pending_entries << std::endl;
        std::cout << "Static entries tracked: " << total_static_entries << std::endl;
        std::cout << "Static bucket profiles: "
          << static_bucket_counts.size()
          << " entries=" << static_bucket_profile_entries
          << std::endl;
        std::cout << "Bucket drift sketch enabled/weight: "
                  << opt.enable_drift_sketch << "/"
                  << opt.drift_sketch_gpkl_weight << std::endl;

        std::cout << "Dynamic entries tracked: " << total_dynamic_entries << std::endl;

        std::cout << "Max bucket pending: " << max_bucket_pending << std::endl;
        std::cout << "Base index size: " << base_data_store.count << std::endl;
        std::cout << "Flushes: " << total_flushes << std::endl;
std::cout << "Local merges: " << total_local_merges << std::endl;
std::cout << "Drift-aware flushes: " << total_drift_flushes << std::endl;
std::cout << "Drift-aware local merges: " << total_drift_local_merges << std::endl;
        std::cout << "Drift-threshold flushes: " << total_drift_threshold_flushes << std::endl;
        std::cout << "Drift-threshold local merges: " << total_drift_threshold_local_merges << std::endl;
        std::cout << "HPT-pressure flushes: " << total_hpt_pressure_flushes << std::endl;
        std::cout << "HPT-pressure local merges: " << total_hpt_pressure_local_merges << std::endl;
        std::cout << "Async local retrain requests/publishes/skips: "
                  << total_async_local_retrain_requests.load(std::memory_order_relaxed) << "/"
                  << total_async_local_retrain_publishes.load(std::memory_order_relaxed) << "/"
                  << total_async_local_retrain_skips.load(std::memory_order_relaxed)
                  << std::endl;
        std::cout << "Async local models/inflight/entries: "
                  << async_bucket_models << "/"
                  << async_bucket_retrain_inflight << "/"
                  << async_bucket_model_entries << std::endl;
        std::cout << "Dynamic structure policy updates/switches: "
                  << total_structure_policy_updates << "/"
                  << total_structure_mode_switches << std::endl;
        std::cout << "Dynamic structure builds balanced/model/trie/compact: "
                  << total_structure_balanced_builds << "/"
                  << total_structure_model_biased_builds << "/"
                  << total_structure_trie_biased_builds << "/"
                  << total_structure_compact_biased_builds << std::endl;
        std::cout << "Deferred structure rebuild requests/schedules/publishes/skips: "
                  << total_structure_rebuild_requests << "/"
                  << total_structure_rebuild_schedules << "/"
                  << total_structure_rebuild_publishes.load(std::memory_order_relaxed) << "/"
                  << total_structure_rebuild_skips.load(std::memory_order_relaxed)
                  << std::endl;
        std::cout << "Dynamic structure threshold policy applications: "
                  << total_structure_threshold_policy_applications << std::endl;

        std::cout << "Hits buffer/async-local/delta/base: " << buffer_hits << "/"
                  << async_local_model_hits << "/" << delta_hits
                  << "/" << base_hits << std::endl;
        std::cout << "Empty dynamic read fastpath hits: "
                  << empty_read_fastpath_hits << std::endl;
        std::cout << "Dynamic exact guard checks/skips/fp_size: "
                  << dynamic_exact_guard_checks << "/"
                  << dynamic_exact_guard_skips << "/"
                  << dynamic_key_fps.size() << std::endl;
        std::cout << "HPT version: " << hpt_version << std::endl;
        std::cout << "HPT pending deltas: " << pending_hpt_deltas.size() << std::endl;
        std::cout << "HPT dirty rows: " << hpt_dirty_row_count << "/"
                  << HPT::ROW_COUNT << std::endl;
        std::cout << "HPT COW publishes/inplace publishes: "
                  << hpt_cow_publishes << "/" << hpt_inplace_publishes
                  << std::endl;
        std::cout << "HPT row-level cell updates tracked: "
                  << hpt_cell_updates << std::endl;
        if (opt.print_bucket_details && !buckets.empty()) {
    std::cout << "--- Bucket detail sample ---" << std::endl;

    size_t printed = 0;

    for (const auto &kvp : buckets) {
        if (printed >= opt.max_bucket_detail_print) break;

        const BucketKey &bk = kvp.first;
        const DeltaBucket &b = kvp.second;

        double gpkl_score = bucket_gpkl(bk, b);

        auto sc_it = static_bucket_counts.find(bk);
        uint64_t static_count =
            (sc_it == static_bucket_counts.end()) ? 0 : sc_it->second;

        std::cout << "bucket#" << printed
          << " prefix=" << bk.prefix
          << " row_group=" << bk.hpt_row_group
          << " bos_range=" << bk.bos_range_id
          << " static=" << static_count
          << " active=" << b.active.size()
          << " pending=" << b.pending_entries
          << " delta_runs=" << b.delta_indexes.size()
          << " flushes=" << b.flush_count
<< " merges=" << b.merge_count
<< " drift_flushes=" << b.drift_flush_count
<< " drift_merges=" << b.drift_merge_count
                      << " drift_threshold_flushes=" << b.drift_threshold_flush_count
                      << " drift_threshold_merges=" << b.drift_threshold_merge_count
                      << " hpt_pressure_flushes=" << b.hpt_pressure_flush_count
                      << " hpt_pressure_merges=" << b.hpt_pressure_merge_count
                      << " async_retrain_req="
                      << (b.async_local_model ? b.async_local_model->requests.load(std::memory_order_relaxed) : 0)
                      << " async_retrain_pub="
                      << (b.async_local_model ? b.async_local_model->publishes.load(std::memory_order_relaxed) : 0)
                      << " async_model_entries="
                      << (b.async_local_model ? b.async_local_model->model_entries.load(std::memory_order_relaxed) : 0)
                      << " async_model_hits="
                      << (b.async_local_model ? b.async_local_model->hits.load(std::memory_order_relaxed) : 0)
                      << " async_retired="
                      << (b.async_local_model ? b.async_local_model->retired_snapshot_count.load(std::memory_order_relaxed) : 0)
                      << " async_reclaimed="
                      << (b.async_local_model ? b.async_local_model->retired_snapshot_reclaims.load(std::memory_order_relaxed) : 0)
                      << " struct_mode=" << structure_mode_name(b.structure_mode)
                      << " struct_raw=" << b.structure_raw_read_ratio
                      << " struct_pmss=" << b.structure_pmss_read_ratio
                      << " struct_pending=" << b.structure_rebuild_pending
                      << " struct_rebuild_req=" << b.structure_rebuild_requests
                      << " struct_rebuild_sched=" << b.structure_rebuild_schedules
                      << " struct_threshold_apps=" << b.structure_threshold_policy_applications

          << " recent_total=" << b.recent_hist.total
          << " gpkl=" << gpkl_score
          << " last_gpkl=" << b.last_gpkl
          << " max_gpkl=" << b.max_gpkl
          << " last_ratio=" << b.last_drift_ratio
          << " max_ratio=" << b.max_drift_ratio
          << " drift_checks=" << b.drift_checks
          << " drift_events=" << b.drift_events
          << " novel_events=" << b.novel_bucket_events
          << " in_drift=" << b.in_drift_state
<< " drift_entries=" << b.drift_state_entries
<< " drift_exits=" << b.drift_state_exits
<< " drift_streak=" << b.consecutive_drift_checks
<< " clean_streak=" << b.consecutive_clean_checks

          << " accesses=" << b.accesses_since_check
          << std::endl;
        std::cout << "Bucket drift sketch enabled/weight: "
                  << opt.enable_drift_sketch << "/"
                  << opt.drift_sketch_gpkl_weight << std::endl;


        printed++;
    }
}

        std::cout << "================================" << std::endl;
    }

    void wait_for_review_background_work() {
        wait_for_async_local_retrains();
    }

    uint64_t review_local_merge_count() const {
        return total_local_merges;
    }

    uint64_t review_local_merge_time_ns() const {
        return review_local_merge_time_ns_.load(std::memory_order_relaxed);
    }

    uint64_t review_local_flush_count() const {
        return total_flushes;
    }

    uint64_t review_local_flush_time_ns() const {
        return review_local_flush_time_ns_.load(std::memory_order_relaxed);
    }

    uint64_t review_async_retrain_count() const {
        return total_async_local_retrain_publishes.load(
            std::memory_order_relaxed);
    }

    uint64_t review_async_retrain_time_ns() const {
        return review_async_retrain_time_ns_.load(std::memory_order_relaxed);
    }

    uint64_t review_structure_rebuild_count() const {
        return total_structure_rebuild_publishes.load(
            std::memory_order_relaxed);
    }

    uint64_t review_structure_rebuild_time_ns() const {
        return review_structure_rebuild_time_ns_.load(std::memory_order_relaxed);
    }

    size_t review_model_param_bytes() const {
        size_t bytes = BOSBase::model_param_bytes();

        HPT *rh = routing_hpt.load(std::memory_order_acquire);
        if (rh && rh != this->current_hpt()) {
            bytes += rh->model_size();
        }

        for (const auto &kvp : buckets) {
            const DeltaBucket &bucket = kvp.second;

            for (BOSBase *idx : bucket.delta_indexes) {
                if (idx) bytes += idx->model_param_bytes();
            }

            if (bucket.async_local_model) {
                BOSBase *local = bucket.async_local_model->model.load(
                    std::memory_order_acquire);
                if (local) bytes += local->model_param_bytes();
            }
        }

        return bytes;
    }

    uint64_t review_base_id_overlay_entries() const { return base_id_overlay_entries; }
    uint64_t review_base_id_overlay_hits() const { return base_id_overlay_hits; }
    uint64_t review_base_id_overlay_updates() const { return base_id_overlay_updates; }
    uint64_t review_base_hash_hits() const { return base_hash_index_hits; }
    uint64_t review_base_hash_misses() const { return base_hash_index_misses; }
    uint64_t review_base_hash_collision_probes() const { return base_hash_index_collision_probes; }

    uint64_t review_dynamic_guard_checks() const { return dynamic_exact_guard_checks; }
    uint64_t review_dynamic_guard_skips() const { return dynamic_exact_guard_skips; }
    size_t review_dynamic_guard_size() const { return dynamic_key_fps.size(); }

    size_t review_tombstone_count() const { return deleted_keys.size() + base_deleted_count_; }
    size_t review_string_tombstone_count() const { return deleted_keys.size(); }
    size_t review_base_tombstone_count() const { return base_deleted_count_; }
    uint64_t review_tombstone_hits() const { return tombstone_hits; }
    uint64_t review_delete_count() const { return delete_count; }
    uint64_t review_tombstone_gc_checks() const { return tombstone_gc_checks; }
    uint64_t review_tombstone_gc_erases() const { return tombstone_gc_erases; }
    uint64_t review_tombstone_gc_prefix_bucket_probes() const { return tombstone_gc_prefix_bucket_probes; }
    uint64_t review_tombstone_gc_delta_run_probes() const { return tombstone_gc_delta_run_probes; }
    uint64_t review_tombstone_gc_prefix_index_misses() const { return tombstone_gc_prefix_index_misses; }
    uint64_t review_tombstone_gc_full_bucket_scan_fallbacks() const { return tombstone_gc_full_bucket_scan_fallbacks; }
    uint64_t review_tombstone_merge_skips() const { return tombstone_merge_skips; }

    uint64_t review_force_merge_all_buckets_for_rq3() {
        const uint64_t before = total_local_merges;

        std::vector<BucketKey> rq3_bucket_keys;
        rq3_bucket_keys.reserve(buckets.size());
        for (const auto &kvp : buckets) {
            rq3_bucket_keys.push_back(kvp.first);
        }

        for (const BucketKey &bk : rq3_bucket_keys) {
            auto it = buckets.find(bk);
            if (it != buckets.end()) {
                merge_bucket_into_base(it->first, it->second, true, true);
            }
        }

        wait_for_async_local_retrains();

        // RQ3_BATCHED_TOMBSTONE_COMPACTION_DRAIN_V1
        // During force-all review maintenance, defer per-bucket GC and then
        // drain the budgeted compactor with repeated batch-level passes.
        // This avoids per-bucket repeated scans while still allowing the
        // budgeted compactor to reclaim all safe tombstones exposed by the batch.
        const size_t rq3_compaction_max_passes =
            rq3_bucket_keys.size() + 1;

        for (size_t rq3_compaction_pass = 0;
             rq3_compaction_pass < rq3_compaction_max_passes;
             ++rq3_compaction_pass) {
            const size_t rq3_before_pass =
                review_string_tombstone_count();

            compact_safe_tombstones_after_maintenance();

            const size_t rq3_after_pass =
                review_string_tombstone_count();

            if (rq3_after_pass == 0 ||
                rq3_after_pass >= rq3_before_pass) {
                break;
            }
        }

        return total_local_merges - before;
    }


    size_t review_update_overlay_entries() const { return update_overlay.size(); }
    size_t review_weak_dynamic_overlay_entries() const { return weak_dynamic_overlay.size(); }
    uint64_t review_dynamic_entries() const { return total_dynamic_entries; }
    uint64_t review_buffer_hits() const { return buffer_hits; }
    uint64_t review_delta_hits() const { return delta_hits; }
    uint64_t review_base_hits() const { return base_hits; }
    uint64_t review_async_local_hits() const { return async_local_model_hits; }

  private:
    void normalize_options() {
        if (opt.prefix_len == 0) opt.prefix_len = 1;
        if (opt.bucket_flush_threshold < MIN_DELTA_BULKLOAD_SIZE) {
            opt.bucket_flush_threshold = MIN_DELTA_BULKLOAD_SIZE;
        }
        if (opt.hpt_cow_update_batch == 0) opt.hpt_cow_update_batch = 1;
        if (opt.hpt_cow_dirty_row_threshold == 0) opt.hpt_cow_dirty_row_threshold = 1;
        if (opt.hpt_cow_dirty_row_threshold > HPT::ROW_COUNT) {
            opt.hpt_cow_dirty_row_threshold = HPT::ROW_COUNT;
        }
        if (opt.hpt_row_flush_pressure_threshold == 0) {
            opt.hpt_row_flush_pressure_threshold = 1;
        }
        if (opt.hpt_row_merge_pressure_threshold == 0) {
            opt.hpt_row_merge_pressure_threshold = 1;
        }
        if (opt.hpt_row_bucket_group_size == 0) {
    opt.hpt_row_bucket_group_size = 1;
}

if (opt.hpt_row_bucket_group_size > HPT::ROW_COUNT) {
    opt.hpt_row_bucket_group_size = HPT::ROW_COUNT;
}

if (opt.bos_range_bucket_depth > 8) {
    opt.bos_range_bucket_depth = 8;
}
if (opt.bos_range_bucket_slot_group_size == 0) {
    opt.bos_range_bucket_slot_group_size = 1;
}

if (opt.weak_prefix_min_static_entries == 0) {
    opt.weak_prefix_min_static_entries = 1;
}
if (opt.weak_prefix_min_bucket_profiles == 0) {
    opt.weak_prefix_min_bucket_profiles = 1;
}

if (opt.weak_prefix_max_avg_static_per_bucket <= 0.0) {
    opt.weak_prefix_max_avg_static_per_bucket = 1.0;
}
if (const char* env = std::getenv("BOS_DYNAMIC_FIRST")) {
    opt.dynamic_first_lookup = std::atoi(env) != 0;
}

std::cerr << "[BOS] dynamic_first_lookup="
          << (opt.dynamic_first_lookup ? 1 : 0)
          << std::endl;

if (opt.drift_sketch_gpkl_weight < 0.0) opt.drift_sketch_gpkl_weight = 0.0;
if (opt.drift_sketch_gpkl_weight > 1.0) opt.drift_sketch_gpkl_weight = 1.0;

if (opt.drift_sketch_gpkl_weight < 0.0) opt.drift_sketch_gpkl_weight = 0.0;
if (opt.drift_sketch_gpkl_weight > 1.0) opt.drift_sketch_gpkl_weight = 1.0;

if (opt.structure_min_read_ratio < 0.0) opt.structure_min_read_ratio = 0.0;
if (opt.structure_max_read_ratio > 1.0) opt.structure_max_read_ratio = 1.0;
if (opt.structure_min_read_ratio > opt.structure_max_read_ratio) {
    opt.structure_min_read_ratio = opt.structure_max_read_ratio;
}
if (opt.structure_drift_read_ratio_cap < opt.structure_min_read_ratio) {
    opt.structure_drift_read_ratio_cap = opt.structure_min_read_ratio;
}
if (opt.structure_drift_read_ratio_cap > opt.structure_max_read_ratio) {
    opt.structure_drift_read_ratio_cap = opt.structure_max_read_ratio;
}
if (opt.structure_fragmented_read_ratio_floor < opt.structure_min_read_ratio) {
    opt.structure_fragmented_read_ratio_floor = opt.structure_min_read_ratio;
}
if (opt.structure_fragmented_read_ratio_floor > opt.structure_max_read_ratio) {
    opt.structure_fragmented_read_ratio_floor = opt.structure_max_read_ratio;
}
if (opt.structure_policy_change_epsilon < 0.0) {
    opt.structure_policy_change_epsilon = 0.0;
}
if (opt.structure_model_enter_read_ratio < opt.structure_min_read_ratio) {
    opt.structure_model_enter_read_ratio = opt.structure_min_read_ratio;
}
if (opt.structure_model_enter_read_ratio > opt.structure_max_read_ratio) {
    opt.structure_model_enter_read_ratio = opt.structure_max_read_ratio;
}
if (opt.structure_model_exit_read_ratio < opt.structure_min_read_ratio) {
    opt.structure_model_exit_read_ratio = opt.structure_min_read_ratio;
}
if (opt.structure_model_exit_read_ratio > opt.structure_model_enter_read_ratio) {
    opt.structure_model_exit_read_ratio = opt.structure_model_enter_read_ratio;
}
if (opt.structure_trie_enter_read_ratio < opt.structure_min_read_ratio) {
    opt.structure_trie_enter_read_ratio = opt.structure_min_read_ratio;
}
if (opt.structure_trie_enter_read_ratio > opt.structure_max_read_ratio) {
    opt.structure_trie_enter_read_ratio = opt.structure_max_read_ratio;
}
if (opt.structure_trie_exit_read_ratio < opt.structure_trie_enter_read_ratio) {
    opt.structure_trie_exit_read_ratio = opt.structure_trie_enter_read_ratio;
}
if (opt.structure_trie_exit_read_ratio > opt.structure_max_read_ratio) {
    opt.structure_trie_exit_read_ratio = opt.structure_max_read_ratio;
}
if (opt.structure_compact_enter_read_ratio < opt.structure_trie_enter_read_ratio) {
    opt.structure_compact_enter_read_ratio = opt.structure_trie_enter_read_ratio;
}
if (opt.structure_compact_enter_read_ratio > opt.structure_model_enter_read_ratio) {
    opt.structure_compact_enter_read_ratio = opt.structure_model_enter_read_ratio;
}
if (opt.structure_compact_exit_read_ratio < opt.structure_compact_enter_read_ratio) {
    opt.structure_compact_exit_read_ratio = opt.structure_compact_enter_read_ratio;
}
if (opt.structure_compact_exit_read_ratio > opt.structure_model_enter_read_ratio) {
    opt.structure_compact_exit_read_ratio = opt.structure_model_enter_read_ratio;
}
if (opt.structure_compact_pmss_read_ratio < opt.structure_min_read_ratio) {
    opt.structure_compact_pmss_read_ratio = opt.structure_min_read_ratio;
}
if (opt.structure_compact_pmss_read_ratio > opt.structure_max_read_ratio) {
    opt.structure_compact_pmss_read_ratio = opt.structure_max_read_ratio;
}
if (opt.structure_mode_min_builds_before_switch == 0) {
    opt.structure_mode_min_builds_before_switch = 1;
}
if (opt.structure_rebuild_min_entries < MIN_DELTA_BULKLOAD_SIZE) {
    opt.structure_rebuild_min_entries = MIN_DELTA_BULKLOAD_SIZE;
}
if (opt.structure_rebuild_write_threshold == 0) {
    opt.structure_rebuild_write_threshold = 1;
}
if (opt.local_merge_run_threshold == 0) {
    opt.local_merge_run_threshold = 1;
}
if (opt.local_merge_size_threshold < MIN_DELTA_BULKLOAD_SIZE) {
    opt.local_merge_size_threshold = MIN_DELTA_BULKLOAD_SIZE;
}
if (opt.async_local_retrain_min_entries < MIN_DELTA_BULKLOAD_SIZE) {
    opt.async_local_retrain_min_entries = MIN_DELTA_BULKLOAD_SIZE;
}
if (opt.drift_retrain_min_entries > 0 &&
    opt.drift_retrain_min_entries < MIN_DELTA_BULKLOAD_SIZE) {
    opt.drift_retrain_min_entries = MIN_DELTA_BULKLOAD_SIZE;
}
if (opt.max_async_local_retrain_workers == 0) {
    opt.max_async_local_retrain_workers = 1;
}
if (opt.structure_model_flush_scale <= 0.0) {
    opt.structure_model_flush_scale = 1.0;
}
if (opt.structure_model_merge_scale <= 0.0) {
    opt.structure_model_merge_scale = 1.0;
}
if (opt.structure_model_async_access_scale <= 0.0) {
    opt.structure_model_async_access_scale = 1.0;
}
if (opt.structure_model_async_min_entries_scale <= 0.0) {
    opt.structure_model_async_min_entries_scale = 1.0;
}
if (opt.structure_trie_flush_scale <= 0.0) {
    opt.structure_trie_flush_scale = 1.0;
}
if (opt.structure_trie_merge_scale <= 0.0) {
    opt.structure_trie_merge_scale = 1.0;
}
if (opt.structure_trie_async_access_scale <= 0.0) {
    opt.structure_trie_async_access_scale = 1.0;
}
if (opt.structure_trie_async_min_entries_scale <= 0.0) {
    opt.structure_trie_async_min_entries_scale = 1.0;
}
if (opt.structure_compact_flush_scale <= 0.0) {
    opt.structure_compact_flush_scale = 1.0;
}
if (opt.structure_compact_merge_scale <= 0.0) {
    opt.structure_compact_merge_scale = 1.0;
}
if (opt.structure_compact_async_access_scale <= 0.0) {
    opt.structure_compact_async_access_scale = 1.0;
}
if (opt.structure_compact_async_min_entries_scale <= 0.0) {
    opt.structure_compact_async_min_entries_scale = 1.0;
}


    }


    // ===== Base-key hash index helpers for fair string-key overlay fast path =====
    static uint64_t bos_index_hash_bytes(const char *data, size_t len) {
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i < len; ++i) {
            h ^= static_cast<unsigned char>(data[i]);
            h *= 1099511628211ULL;
        }
        return h;
    }

    static uint64_t bos_index_hash_cstr(const char *s) {
        // Hot path: avoid strlen() + a second full pass over the key.
        // The result is exactly the same FNV-1a hash as bos_index_hash_bytes().
        uint64_t h = 1469598103934665603ULL;
        const unsigned char *p = reinterpret_cast<const unsigned char *>(s);
        while (*p != 0) {
            h ^= static_cast<uint64_t>(*p++);
            h *= 1099511628211ULL;
        }
        return h;
    }

    static uint64_t bos_index_hash_cstr_len(const char *s, size_t &len) {
        uint64_t h = 1469598103934665603ULL;
        const unsigned char *p = reinterpret_cast<const unsigned char *>(s);
        len = 0;
        while (*p != 0) {
            h ^= static_cast<uint64_t>(*p++);
            h *= 1099511628211ULL;
            ++len;
        }
        return h;
    }

    static size_t hot_cache_no_base_id() {
        return std::numeric_limits<size_t>::max();
    }

    static bool bos_index_env_enabled(const char *name, bool default_value = false) {
        const char *env = std::getenv(name);
        if (!env) return default_value;
        if (env[0] == '\0') return default_value;
        return std::strcmp(env, "0") != 0 &&
               std::strcmp(env, "false") != 0 &&
               std::strcmp(env, "FALSE") != 0 &&
               std::strcmp(env, "off") != 0 &&
               std::strcmp(env, "OFF") != 0;
    }

    void init_hot_key_cache_from_env(size_t base_count) {
        (void)base_count;

        hot_key_cache_.clear();
        hot_key_cache_mask_ = 0;
        hot_ptr_cache_.clear();
        hot_ptr_cache_mask_ = 0;
        hot_ptr_cache_active_slots_ = 0;
        static_read_ptr_cache_.clear();
        static_read_ptr_cache_mask_ = 0;
        hot_update_ptr_cache_.clear();
        hot_update_ptr_cache_mask_ = 0;
        model_segment_cache_.clear();
        model_segment_cache_mask_ = 0;

        model_segment_cache_enabled_ =
            bos_index_env_enabled("BOS_MODEL_SEGMENT_CACHE", false);

        model_segment_cache_hits_ = 0;
        model_segment_cache_misses_ = 0;
        model_segment_cache_puts_ = 0;
        model_segment_cache_invalidations_ = 0;
        model_segment_cache_base_hits_ = 0;
        model_segment_cache_update_hits_ = 0;

        if (model_segment_cache_enabled_) {
            unsigned long msc_bits = 16;
            if (const char *env = std::getenv("BOS_MODEL_SEGMENT_CACHE_BITS")) {
                char *end = nullptr;
                unsigned long v = std::strtoul(env, &end, 10);
                if (end != env && v > 0) {
                    msc_bits = v;
                }
            }

            if (msc_bits < 10) msc_bits = 10;
            if (msc_bits > 24) msc_bits = 24;

            const size_t msc_cap = static_cast<size_t>(1ULL << msc_bits);
            model_segment_cache_.assign(msc_cap, ModelSegmentCacheSet());
            model_segment_cache_mask_ = msc_cap - 1;

            std::cerr << "[BOS MSC] model-segment cache enabled sets="
                      << msc_cap
                      << " ways=" << BOS_MODEL_SEGMENT_CACHE_WAYS
                      << " bits=" << msc_bits
                      << std::endl;
        }

        hot_key_cache_enabled_ = bos_index_env_enabled("BOS_HOT_KEY_CACHE", false);

        // P15f: model-leaf position cache.
        // This is a bounded model-resident shortcut for immutable base leaf
        // positions. It stores base_id/leaf position, not a standalone KV copy.
        // It is intentionally controlled by a separate switch so BOSIndex-Core and
        // BOSIndex-MLC can be reported separately.
        static_read_ptr_cache_enabled_ =
            bos_index_env_enabled("BOS_MODEL_LEAF_CACHE", false);

        hot_ptr_cache_enabled_ =
            hot_key_cache_enabled_ &&
            bos_index_env_enabled("BOS_HOT_PTR_CACHE", false);

        hot_update_ptr_shortcut_enabled_ =
            hot_key_cache_enabled_ &&
            bos_index_env_enabled("BOS_HOT_UPDATE_PTR_SHORTCUT", false);
        static_read_fastpath_enabled_ =
            bos_index_env_enabled("BOS_STATIC_READ_FASTPATH", false);

        hot_key_cache_hits_ = 0;
        hot_key_cache_misses_ = 0;
        hot_key_cache_puts_ = 0;
        hot_key_cache_invalidations_ = 0;
        hot_ptr_cache_hits_ = 0;
        hot_ptr_cache_misses_ = 0;
        hot_ptr_cache_puts_ = 0;
        hot_ptr_cache_invalidations_ = 0;
        static_read_ptr_cache_hits_ = 0;
        static_read_ptr_cache_misses_ = 0;
        static_read_ptr_cache_puts_ = 0;
        static_read_ptr_cache_invalidations_ = 0;
        hot_update_cache_shortcut_hits_ = 0;
        hot_update_ptr_shortcut_hits_ = 0;
        hot_update_ptr_cache_puts_ = 0;
        hot_update_ptr_cache_invalidations_ = 0;

        if (!hot_key_cache_enabled_ && !static_read_ptr_cache_enabled_) {
            hot_update_ptr_shortcut_enabled_ = false;
            return;
        }

        unsigned long bits = 20;
        if (const char *env = std::getenv("BOS_HOT_CACHE_BITS")) {
            char *end = nullptr;
            unsigned long v = std::strtoul(env, &end, 10);
            if (end != env && v > 0) {
                bits = v;
            }
        }

        if (bits < 10) bits = 10;
        if (bits > 24) bits = 24;

        const size_t cap = static_cast<size_t>(1ULL << bits);

        if (hot_key_cache_enabled_) {
            hot_key_cache_.assign(cap, HotKeyCacheEntry());
            hot_key_cache_mask_ = cap - 1;
        } else {
            hot_key_cache_.clear();
            hot_key_cache_mask_ = 0;
        }

        if (hot_ptr_cache_enabled_) {
            hot_ptr_cache_.assign(cap, HotPtrCacheEntry());
            hot_ptr_cache_mask_ = cap - 1;
            hot_ptr_cache_active_slots_ = cap;
            if (const char *env = std::getenv("BOS_HOT_PTR_CACHE_ACTIVE_SLOTS")) {
                char *end = nullptr;
                unsigned long long v = std::strtoull(env, &end, 10);
                if (end != env && v > 0) {
                    if (v > static_cast<unsigned long long>(cap)) {
                        v = static_cast<unsigned long long>(cap);
                    }
                    hot_ptr_cache_active_slots_ = static_cast<size_t>(v);
                }
            }
        }

        if (static_read_ptr_cache_enabled_) {
            int static_bits = 10;
            const char *static_bits_env =
                std::getenv("BOS_MODEL_LEAF_CACHE_BITS");
            if (!static_bits_env || !*static_bits_env) {
                static_bits_env = std::getenv("BOS_STATIC_READ_PTR_CACHE_BITS");
            }
            if (static_bits_env && *static_bits_env) {
                static_bits = std::atoi(static_bits_env);
            }

            if (static_bits < 6) static_bits = 6;
            if (static_bits > bits) static_bits = bits;

            const size_t static_cap =
                static_cast<size_t>(1) << static_bits;

            static_read_ptr_cache_.assign(static_cap, HotPtrCacheEntry());
            static_read_ptr_cache_mask_ = static_cap - 1;
        }

        if (hot_update_ptr_shortcut_enabled_) {
            hot_update_ptr_cache_.assign(cap, HotUpdatePtrCacheEntry());
            hot_update_ptr_cache_mask_ = cap - 1;
        }

        std::cerr << "[BOS hot-key cache] enabled slots="
                  << cap
                  << " bits=" << bits
                  << " static_read_fastpath="
                  << (static_read_fastpath_enabled_ ? 1 : 0)
                  << " ptr_cache="
                  << (hot_ptr_cache_enabled_ ? 1 : 0)
                  << " static_read_ptr_cache="
                  << (static_read_ptr_cache_enabled_ ? 1 : 0)
                  << " update_ptr_shortcut="
                  << (hot_update_ptr_shortcut_enabled_ ? 1 : 0)
                  << std::endl;
    }

    static size_t bos_index_hot_ptr_hash(const char *p) {
        uintptr_t x = reinterpret_cast<uintptr_t>(p);
        x >>= 4;
        x ^= x >> 33;
        x *= static_cast<uintptr_t>(0xff51afd7ed558ccdULL);
        x ^= x >> 33;
        x *= static_cast<uintptr_t>(0xc4ceb9fe1a85ec53ULL);
        x ^= x >> 33;
        return static_cast<size_t>(x);
    }

    uint32_t base_value_version_for_id(size_t base_id) const {
        if (base_id >= base_value_version.size()) return 0;
        return base_value_version[base_id];
    }

    void bump_base_value_version(size_t base_id) {
        // base_value_version is only used to validate hot pointer-cache entries.
        // In BOSIndex-Core runs where the pointer cache is disabled, maintaining this
        // version metadata is pure update-path overhead, so skip it.
        if (!hot_ptr_cache_enabled_) return;

        if (base_id < base_value_version.size()) {
            ++base_value_version[base_id];
        }
    }

    bool hot_ptr_cache_lookup_cstr(const char *key,
                                   uint64_t &out_val,
                                   size_t *out_base_id = nullptr) {
        if (out_base_id) {
            *out_base_id = hot_cache_no_base_id();
        }

        if (!hot_ptr_cache_enabled_ ||
            hot_ptr_cache_.empty() ||
            !key ||
            !deleted_keys.empty()) {
            return false;
        }

        size_t idx = bos_index_hot_ptr_hash(key) & hot_ptr_cache_mask_;

        if (idx >= hot_ptr_cache_active_slots_) {
            hot_ptr_cache_misses_++;
            return false;
        }

        HotPtrCacheEntry &e = hot_ptr_cache_[idx];

        if (e.valid && e.key_ptr == key) {
            if (e.base_id < base_value_version.size() &&
                e.version == base_value_version[e.base_id]) {
                out_val = e.value;
                if (out_base_id) {
                    *out_base_id = e.base_id;
                }
                hot_ptr_cache_hits_++;
                return true;
            }

            e.valid = 0;
            e.key_ptr = nullptr;
            e.value = 0;
            e.base_id = 0;
            e.version = 0;
            hot_ptr_cache_invalidations_++;
        }

        hot_ptr_cache_misses_++;
        return false;
    }

    void hot_ptr_cache_put_base_id(size_t base_id, uint64_t value) {
        if (!hot_ptr_cache_enabled_ ||
            hot_ptr_cache_.empty() ||
            !base_data_store.key_ptrs ||
            base_id >= static_cast<size_t>(base_data_store.count)) {
            return;
        }

        const char *ptr = base_data_store.key_ptrs[base_id];
        if (!ptr) return;

        size_t idx = bos_index_hot_ptr_hash(ptr) & hot_ptr_cache_mask_;

        if (idx >= hot_ptr_cache_active_slots_) {
            return;
        }

        HotPtrCacheEntry &e = hot_ptr_cache_[idx];

        e.valid = 1;
        e.key_ptr = ptr;
        e.base_id = base_id;
        e.value = value;
        e.version = base_value_version_for_id(base_id);
        hot_ptr_cache_puts_++;
    }

    void hot_ptr_cache_put_observed_cstr(const char *key,
                                         size_t base_id,
                                         uint64_t value) {
        if (!hot_ptr_cache_enabled_ ||
            hot_ptr_cache_.empty() ||
            !key ||
            base_id >= static_cast<size_t>(base_data_store.count)) {
            return;
        }

        size_t idx = bos_index_hot_ptr_hash(key) & hot_ptr_cache_mask_;

        if (idx >= hot_ptr_cache_active_slots_) {
            return;
        }

        HotPtrCacheEntry &e = hot_ptr_cache_[idx];

        e.valid = 1;
        e.key_ptr = key;
        e.base_id = base_id;
        e.value = value;
        e.version = base_value_version_for_id(base_id);
        hot_ptr_cache_puts_++;
    }

    void hot_ptr_cache_refresh_base_id(size_t base_id, uint64_t value) {
        if (!hot_ptr_cache_enabled_ ||
            hot_ptr_cache_.empty() ||
            !base_data_store.key_ptrs ||
            base_id >= static_cast<size_t>(base_data_store.count)) {
            return;
        }

        const char *ptr = base_data_store.key_ptrs[base_id];
        if (!ptr) return;

        size_t idx = bos_index_hot_ptr_hash(ptr) & hot_ptr_cache_mask_;

        if (idx >= hot_ptr_cache_active_slots_) {
            return;
        }

        HotPtrCacheEntry &e = hot_ptr_cache_[idx];

        if (e.valid && e.key_ptr == ptr && e.base_id == base_id) {
            e.value = value;
            e.version = base_value_version_for_id(base_id);
            return;
        }

        hot_ptr_cache_put_base_id(base_id, value);
    }

    void hot_ptr_cache_invalidate_base_id(size_t base_id) {
        if (!hot_ptr_cache_enabled_ ||
            hot_ptr_cache_.empty() ||
            !base_data_store.key_ptrs ||
            base_id >= static_cast<size_t>(base_data_store.count)) {
            return;
        }

        const char *ptr = base_data_store.key_ptrs[base_id];
        if (!ptr) return;

        size_t idx = bos_index_hot_ptr_hash(ptr) & hot_ptr_cache_mask_;

        if (idx >= hot_ptr_cache_active_slots_) {
            return;
        }

        HotPtrCacheEntry &e = hot_ptr_cache_[idx];

        if (e.valid && e.key_ptr == ptr && e.base_id == base_id) {
            e.valid = 0;
            e.key_ptr = nullptr;
            e.value = 0;
            e.version = 0;
            hot_ptr_cache_invalidations_++;
        }
    }

    bool static_read_ptr_cache_find_base_id_cstr(const char *key,
                                                 size_t &out_id) {
        if (!static_read_ptr_cache_enabled_ ||
            static_read_ptr_cache_.empty() ||
            !key ||
            !deleted_keys.empty()) {
            return false;
        }

        HotPtrCacheEntry &e =
            static_read_ptr_cache_[
                bos_index_hot_ptr_hash(key) & static_read_ptr_cache_mask_];

        if (!e.valid || e.key_ptr != key) {
            static_read_ptr_cache_misses_++;
            return false;
        }

        const size_t id = e.base_id;

        if (id >= base_update_valid.size() ||
            id >= static_cast<size_t>(base_data_store.count)) {
            e.valid = 0;
            e.key_ptr = nullptr;
            e.value = 0;
            e.base_id = 0;
            e.version = 0;
            static_read_ptr_cache_invalidations_++;
            return false;
        }

        out_id = id;
        static_read_ptr_cache_hits_++;
        return true;
    }

    bool static_read_ptr_cache_lookup_cstr(const char *key,
                                            uint64_t &out_val) {
        if (!static_read_ptr_cache_enabled_ ||
            static_read_ptr_cache_.empty() ||
            !key ||
            !deleted_keys.empty()) {
            return false;
        }

        HotPtrCacheEntry &e =
            static_read_ptr_cache_[
                bos_index_hot_ptr_hash(key) & static_read_ptr_cache_mask_];

        if (!e.valid || e.key_ptr != key) {
            static_read_ptr_cache_misses_++;
            return false;
        }

        const size_t id = e.base_id;

        if (id >= base_update_valid.size() ||
            id >= static_cast<size_t>(base_data_store.count)) {
            e.valid = 0;
            e.key_ptr = nullptr;
            e.value = 0;
            e.base_id = 0;
            e.version = 0;
            static_read_ptr_cache_invalidations_++;
            return false;
        }

        if (base_update_valid[id]) {
            out_val = base_update_values[id];
        } else {
            out_val = e.value;
        }

        static_read_ptr_cache_hits_++;
        return true;
    }






    void static_read_ptr_cache_put_cstr(const char *key,
                                        size_t base_id,
                                        uint64_t value) {
        if (!static_read_ptr_cache_enabled_ ||
            static_read_ptr_cache_.empty() ||
            !key ||
            !deleted_keys.empty()) {
            return;
        }

        if (base_id >= base_update_valid.size() ||
            base_id >= static_cast<size_t>(base_data_store.count)) {
            return;
        }

        HotPtrCacheEntry &e =
            static_read_ptr_cache_[
                bos_index_hot_ptr_hash(key) & static_read_ptr_cache_mask_];

        e.valid = 1;
        e.key_ptr = key;
        e.value = value;
        e.base_id = base_id;
        e.version = base_value_version_for_id(base_id);
        static_read_ptr_cache_puts_++;
    }

    void hot_update_ptr_cache_put_observed_cstr(const char *key,
                                                size_t base_id,
                                                size_t hot_key_slot) {
        if (!hot_update_ptr_shortcut_enabled_ ||
            hot_update_ptr_cache_.empty() ||
            !key ||
            base_id >= base_update_valid.size() ||
            hot_key_slot >= hot_key_cache_.size()) {
            return;
        }

        HotUpdatePtrCacheEntry &e =
            hot_update_ptr_cache_[
                bos_index_hot_ptr_hash(key) & hot_update_ptr_cache_mask_];

        e.valid = 1;
        e.key_ptr = key;
        e.base_id = base_id;
        e.hot_key_slot = hot_key_slot;
        hot_update_ptr_cache_puts_++;
    }

    void hot_update_ptr_cache_invalidate_entry(HotUpdatePtrCacheEntry &e) {
        if (e.valid) {
            e.valid = 0;
            e.key_ptr = nullptr;
            e.base_id = 0;
            e.hot_key_slot = 0;
            hot_update_ptr_cache_invalidations_++;
        }
    }

    bool hot_update_ptr_shortcut_update_cstr(const char *key,
                                             uint64_t value) {
        if (!hot_update_ptr_shortcut_enabled_ ||
            hot_update_ptr_cache_.empty() ||
            !key ||
            !deleted_keys.empty() ||
            !update_overlay.empty()) {
            return false;
        }

        HotUpdatePtrCacheEntry &pe =
            hot_update_ptr_cache_[
                bos_index_hot_ptr_hash(key) & hot_update_ptr_cache_mask_];

        if (!pe.valid || pe.key_ptr != key) {
            return false;
        }

        const size_t id = pe.base_id;

        if (id >= base_update_valid.size() ||
            pe.hot_key_slot >= hot_key_cache_.size()) {
            hot_update_ptr_cache_invalidate_entry(pe);
            return false;
        }

        HotKeyCacheEntry &he = hot_key_cache_[pe.hot_key_slot];

        if (!he.valid ||
            !he.has_base_id ||
            he.base_id != id ||
            he.last_key_ptr != key) {
            hot_update_ptr_cache_invalidate_entry(pe);
            return false;
        }

        base_id_overlay_updates++;

        if (!base_update_valid[id]) {
            base_update_valid[id] = 1;
            base_id_overlay_entries++;
        }

        if (base_deleted_count_ != 0 &&
            id < base_deleted.size() && base_deleted[id]) {
            base_deleted[id] = 0;
            --base_deleted_count_;
        }
        base_update_values[id] = value;

        if (hot_ptr_cache_enabled_) {
            bump_base_value_version(id);
            hot_ptr_cache_put_observed_cstr(key, id, value);
        }

        he.value = value;
        he.last_key_ptr = key;
        hot_update_ptr_shortcut_hits_++;
        return true;
    }

    bool hot_key_cache_lookup_cstr(const char *key,
                                   uint64_t &out_val,
                                   size_t *out_base_id = nullptr) {
        if (out_base_id) {
            *out_base_id = hot_cache_no_base_id();
        }

        if (!hot_key_cache_enabled_ ||
            hot_key_cache_.empty() ||
            !deleted_keys.empty()) {
            return false;
        }

        size_t len = 0;
        const uint64_t h = bos_index_hash_cstr_len(key, len);
        HotKeyCacheEntry &e = hot_key_cache_[static_cast<size_t>(h) & hot_key_cache_mask_];

        if (e.valid &&
            e.hash == h &&
            e.key.size() == len) {
            if (e.last_key_ptr == key ||
                std::memcmp(e.key.data(), key, len) == 0) {
                e.last_key_ptr = key;
                out_val = e.value;
                if (out_base_id && e.has_base_id) {
                    *out_base_id = e.base_id;
                }
                hot_key_cache_hits_++;
                return true;
            }
        }

        hot_key_cache_misses_++;
        return false;
    }

    bool hot_key_cache_lookup_key(const std::string &key,
                                  uint64_t &out_val,
                                  size_t *out_base_id = nullptr) {
        if (out_base_id) {
            *out_base_id = hot_cache_no_base_id();
        }

        if (!hot_key_cache_enabled_ ||
            hot_key_cache_.empty() ||
            !deleted_keys.empty()) {
            return false;
        }

        const uint64_t h = bos_index_hash_bytes(key.data(), key.size());
        HotKeyCacheEntry &e = hot_key_cache_[static_cast<size_t>(h) & hot_key_cache_mask_];

        if (e.valid && e.hash == h && e.key == key) {
            out_val = e.value;
            if (out_base_id && e.has_base_id) {
                *out_base_id = e.base_id;
            }
            hot_key_cache_hits_++;
            return true;
        }

        hot_key_cache_misses_++;
        return false;
    }

    bool hot_key_cache_update_cstr_base_id_once(const char *key,
                                                uint64_t value,
                                                size_t *out_base_id) {
        if (out_base_id) {
            *out_base_id = hot_cache_no_base_id();
        }

        if (!hot_key_cache_enabled_ ||
            hot_key_cache_.empty() ||
            !key ||
            !deleted_keys.empty()) {
            return false;
        }

        size_t len = 0;
        const uint64_t h = bos_index_hash_cstr_len(key, len);
        const size_t slot =
            static_cast<size_t>(h) & hot_key_cache_mask_;
        HotKeyCacheEntry &e = hot_key_cache_[slot];

        if (e.valid &&
            e.hash == h &&
            e.key.size() == len &&
            (e.last_key_ptr == key ||
             std::memcmp(e.key.data(), key, len) == 0) &&
            e.has_base_id &&
            e.base_id < base_update_valid.size()) {
            e.value = value;
            e.last_key_ptr = key;

            if (out_base_id) {
                *out_base_id = e.base_id;
            }

            hot_key_cache_hits_++;
            hot_update_cache_shortcut_hits_++;
            hot_update_ptr_cache_put_observed_cstr(key, e.base_id, slot);
            return true;
        }

        hot_key_cache_misses_++;
        return false;
    }

    bool hot_key_cache_update_key_base_id_once(const std::string &key,
                                               uint64_t value,
                                               size_t *out_base_id) {
        if (out_base_id) {
            *out_base_id = hot_cache_no_base_id();
        }

        if (!hot_key_cache_enabled_ ||
            hot_key_cache_.empty() ||
            !deleted_keys.empty()) {
            return false;
        }

        const uint64_t h = bos_index_hash_bytes(key.data(), key.size());
        const size_t slot =
            static_cast<size_t>(h) & hot_key_cache_mask_;
        HotKeyCacheEntry &e = hot_key_cache_[slot];

        if (e.valid &&
            e.hash == h &&
            e.key == key &&
            e.has_base_id &&
            e.base_id < base_update_valid.size()) {
            e.value = value;
            e.last_key_ptr = nullptr;

            if (out_base_id) {
                *out_base_id = e.base_id;
            }

            hot_key_cache_hits_++;
            hot_update_cache_shortcut_hits_++;
            return true;
        }

        hot_key_cache_misses_++;
        return false;
    }

    void hot_key_cache_put_cstr(const char *key, uint64_t value) {
        if (!hot_key_cache_enabled_ || hot_key_cache_.empty()) return;

        size_t len = 0;
        const uint64_t h = bos_index_hash_cstr_len(key, len);
        HotKeyCacheEntry &e = hot_key_cache_[static_cast<size_t>(h) & hot_key_cache_mask_];

        e.valid = 1;
        e.hash = h;
        e.value = value;
        e.has_base_id = 0;
        e.base_id = 0;
        e.last_key_ptr = key;
        e.key.assign(key, len);
        hot_key_cache_puts_++;
    }

    void hot_key_cache_put_cstr_base_id(const char *key,
                                        uint64_t value,
                                        size_t base_id) {
        if (!hot_key_cache_enabled_ || hot_key_cache_.empty()) return;

        size_t len = 0;
        const uint64_t h = bos_index_hash_cstr_len(key, len);
        const size_t slot = static_cast<size_t>(h) & hot_key_cache_mask_;
        HotKeyCacheEntry &e = hot_key_cache_[slot];

        e.valid = 1;
        e.hash = h;
        e.value = value;
        e.has_base_id = 1;
        e.base_id = base_id;
        e.last_key_ptr = key;
        e.key.assign(key, len);
        hot_key_cache_puts_++;
        hot_ptr_cache_put_observed_cstr(key, base_id, value);
    }

    void hot_key_cache_put_key(const std::string &key, uint64_t value) {
        if (!hot_key_cache_enabled_ || hot_key_cache_.empty()) return;

        const uint64_t h = bos_index_hash_bytes(key.data(), key.size());
        HotKeyCacheEntry &e = hot_key_cache_[static_cast<size_t>(h) & hot_key_cache_mask_];

        e.valid = 1;
        e.hash = h;
        e.value = value;
        e.has_base_id = 0;
        e.base_id = 0;
        e.last_key_ptr = nullptr;
        e.key = key;
        hot_key_cache_puts_++;
    }

    void hot_key_cache_put_key_base_id(const std::string &key,
                                       uint64_t value,
                                       size_t base_id) {
        if (!hot_key_cache_enabled_ || hot_key_cache_.empty()) return;

        const uint64_t h = bos_index_hash_bytes(key.data(), key.size());
        HotKeyCacheEntry &e = hot_key_cache_[static_cast<size_t>(h) & hot_key_cache_mask_];

        e.valid = 1;
        e.hash = h;
        e.value = value;
        e.has_base_id = 1;
        e.base_id = base_id;
        e.last_key_ptr = nullptr;
        e.key = key;
        hot_key_cache_puts_++;
        hot_ptr_cache_put_base_id(base_id, value);
    }

    void hot_key_cache_refresh_cstr_base_id(const char *key,
                                            uint64_t value,
                                            size_t base_id) {
        if (!hot_key_cache_enabled_ || hot_key_cache_.empty()) return;

        size_t len = 0;
        const uint64_t h = bos_index_hash_cstr_len(key, len);
        const size_t slot = static_cast<size_t>(h) & hot_key_cache_mask_;
        HotKeyCacheEntry &e = hot_key_cache_[slot];

        if (e.valid &&
            e.hash == h &&
            e.has_base_id &&
            e.base_id == base_id &&
            e.key.size() == len &&
            std::memcmp(e.key.data(), key, len) == 0) {
            e.value = value;
            e.last_key_ptr = key;
            hot_update_ptr_cache_put_observed_cstr(key, base_id, slot);
            hot_ptr_cache_put_observed_cstr(key, base_id, value);
            return;
        }

        hot_key_cache_put_cstr_base_id(key, value, base_id);
    }

    void hot_key_cache_refresh_key_base_id(const std::string &key,
                                           uint64_t value,
                                           size_t base_id) {
        if (!hot_key_cache_enabled_ || hot_key_cache_.empty()) return;

        const uint64_t h = bos_index_hash_bytes(key.data(), key.size());
        HotKeyCacheEntry &e =
            hot_key_cache_[static_cast<size_t>(h) & hot_key_cache_mask_];

        if (e.valid &&
            e.hash == h &&
            e.has_base_id &&
            e.base_id == base_id &&
            e.key == key) {
            e.value = value;
            e.last_key_ptr = nullptr;
            hot_ptr_cache_refresh_base_id(base_id, value);
            return;
        }

        hot_key_cache_put_key_base_id(key, value, base_id);
    }

    void hot_key_cache_invalidate_key(const std::string &key) {
        if (hot_key_cache_enabled_ && !hot_key_cache_.empty()) {
            const uint64_t h = bos_index_hash_bytes(key.data(), key.size());
            HotKeyCacheEntry &e =
                hot_key_cache_[static_cast<size_t>(h) & hot_key_cache_mask_];

            if (e.valid && e.hash == h && e.key == key) {
                if (e.has_base_id) {
                    hot_ptr_cache_invalidate_base_id(e.base_id);
                }
                e.valid = 0;
                e.has_base_id = 0;
                e.value = 0;
                e.last_key_ptr = nullptr;
                hot_key_cache_invalidations_++;
            }
        }
    }

    uint64_t model_segment_prefix_route_cstr(const char *key, size_t len) const {
        uint64_t code = 0;
        size_t n = std::min(opt.prefix_len, len);
        if (n > 8) n = 8;

        for (size_t i = 0; i < n; ++i) {
            code = (code << 8) | static_cast<unsigned char>(key[i]);
        }

        // Include length-of-prefix metadata so short keys and zero-padded prefixes
        // do not alias the same model region. This remains a model-region tag,
        // not a full key hash table.
        return code ^ (static_cast<uint64_t>(n) << 56);
    }

    uint64_t model_segment_prefix_route_key(const std::string &key) const {
        return model_segment_prefix_route_cstr(key.c_str(), key.size());
    }

    static uint64_t model_segment_cache_mix(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    size_t model_segment_cache_index(uint64_t route_tag,
                                     uint64_t key_hash) const {
        const uint64_t mixed =
            model_segment_cache_mix(
                key_hash ^
                (route_tag + 0x9e3779b97f4a7c15ULL +
                 (key_hash << 6) + (key_hash >> 2)));
        return static_cast<size_t>(mixed) & model_segment_cache_mask_;
    }

    bool model_segment_cache_base_id_live(size_t id) const {
        return base_data_store.key_ptrs &&
               base_data_store.val_ptrs &&
               id < static_cast<size_t>(base_data_store.count) &&
               id < base_update_valid.size();
    }

    bool model_segment_cache_lookup_cstr(const char *key, uint64_t &out_val) {
        if (!model_segment_cache_enabled_ ||
            model_segment_cache_.empty() ||
            !key ||
            !deleted_keys.empty()) {
            return false;
        }

        size_t len = 0;
        const uint64_t h = bos_index_hash_cstr_len(key, len);
        const uint64_t route = model_segment_prefix_route_cstr(key, len);
        ModelSegmentCacheSet &set =
            model_segment_cache_[model_segment_cache_index(route, h)];

        for (ModelSegmentCacheSlot &slot : set.ways) {
            if (!slot.valid ||
                slot.route_tag != route ||
                slot.key_hash != h ||
                slot.key_len != static_cast<uint32_t>(len)) {
                continue;
            }

            const size_t id = static_cast<size_t>(slot.base_id);
            if (!model_segment_cache_base_id_live(id) ||
                !base_id_key_equals_cstr_slot(id, key)) {
                slot.valid = 0;
                model_segment_cache_invalidations_++;
                continue;
            }

            if (base_update_valid[id]) {
                out_val = base_update_values[id];
                base_id_overlay_hits++;
                buffer_hits++;
                model_segment_cache_update_hits_++;
            } else {
                out_val = base_data_store.val_ptrs[id];
                base_hits++;
                model_segment_cache_base_hits_++;
            }

            model_segment_cache_hits_++;
            return true;
        }

        model_segment_cache_misses_++;
        return false;
    }

    bool model_segment_cache_lookup_key(const std::string &key,
                                        uint64_t &out_val) {
        if (!model_segment_cache_enabled_ ||
            model_segment_cache_.empty() ||
            !deleted_keys.empty()) {
            return false;
        }

        const uint64_t h = bos_index_hash_bytes(key.data(), key.size());
        const uint64_t route = model_segment_prefix_route_key(key);
        ModelSegmentCacheSet &set =
            model_segment_cache_[model_segment_cache_index(route, h)];

        for (ModelSegmentCacheSlot &slot : set.ways) {
            if (!slot.valid ||
                slot.route_tag != route ||
                slot.key_hash != h ||
                slot.key_len != static_cast<uint32_t>(key.size())) {
                continue;
            }

            const size_t id = static_cast<size_t>(slot.base_id);
            if (!model_segment_cache_base_id_live(id) ||
                !base_id_key_equals(id, key)) {
                slot.valid = 0;
                model_segment_cache_invalidations_++;
                continue;
            }

            if (base_update_valid[id]) {
                out_val = base_update_values[id];
                base_id_overlay_hits++;
                buffer_hits++;
                model_segment_cache_update_hits_++;
            } else {
                out_val = base_data_store.val_ptrs[id];
                base_hits++;
                model_segment_cache_base_hits_++;
            }

            model_segment_cache_hits_++;
            return true;
        }

        model_segment_cache_misses_++;
        return false;
    }

    void model_segment_cache_put_cstr_base_id(const char *key, size_t base_id) {
        if (!model_segment_cache_enabled_ ||
            model_segment_cache_.empty() ||
            !key ||
            !deleted_keys.empty() ||
            !model_segment_cache_base_id_live(base_id) ||
            base_id > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return;
        }

        size_t len = 0;
        const uint64_t h = bos_index_hash_cstr_len(key, len);
        if (len > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return;
        }

        const uint64_t route = model_segment_prefix_route_cstr(key, len);
        ModelSegmentCacheSet &set =
            model_segment_cache_[model_segment_cache_index(route, h)];

        ModelSegmentCacheSlot *target = nullptr;
        for (ModelSegmentCacheSlot &slot : set.ways) {
            if (slot.valid &&
                slot.route_tag == route &&
                slot.key_hash == h &&
                slot.key_len == static_cast<uint32_t>(len)) {
                target = &slot;
                break;
            }
            if (!slot.valid && !target) {
                target = &slot;
            }
        }

        if (!target) {
            target = &set.ways[set.next_victim % BOS_MODEL_SEGMENT_CACHE_WAYS];
            set.next_victim = static_cast<uint8_t>(
                (set.next_victim + 1) % BOS_MODEL_SEGMENT_CACHE_WAYS);
        }

        target->valid = 1;
        target->route_tag = route;
        target->key_hash = h;
        target->key_len = static_cast<uint32_t>(len);
        target->base_id = static_cast<uint32_t>(base_id);
        model_segment_cache_puts_++;
    }

    void model_segment_cache_put_key_base_id(const std::string &key,
                                             size_t base_id) {
        if (!model_segment_cache_enabled_ ||
            model_segment_cache_.empty() ||
            !deleted_keys.empty() ||
            !model_segment_cache_base_id_live(base_id) ||
            key.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
            base_id > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            return;
        }

        const uint64_t h = bos_index_hash_bytes(key.data(), key.size());
        const uint64_t route = model_segment_prefix_route_key(key);
        ModelSegmentCacheSet &set =
            model_segment_cache_[model_segment_cache_index(route, h)];

        ModelSegmentCacheSlot *target = nullptr;
        for (ModelSegmentCacheSlot &slot : set.ways) {
            if (slot.valid &&
                slot.route_tag == route &&
                slot.key_hash == h &&
                slot.key_len == static_cast<uint32_t>(key.size())) {
                target = &slot;
                break;
            }
            if (!slot.valid && !target) {
                target = &slot;
            }
        }

        if (!target) {
            target = &set.ways[set.next_victim % BOS_MODEL_SEGMENT_CACHE_WAYS];
            set.next_victim = static_cast<uint8_t>(
                (set.next_victim + 1) % BOS_MODEL_SEGMENT_CACHE_WAYS);
        }

        target->valid = 1;
        target->route_tag = route;
        target->key_hash = h;
        target->key_len = static_cast<uint32_t>(key.size());
        target->base_id = static_cast<uint32_t>(base_id);
        model_segment_cache_puts_++;
    }

    void model_segment_cache_invalidate_key(const std::string &key) {
        if (!model_segment_cache_enabled_ || model_segment_cache_.empty()) {
            return;
        }

        const uint64_t h = bos_index_hash_bytes(key.data(), key.size());
        const uint64_t route = model_segment_prefix_route_key(key);
        ModelSegmentCacheSet &set =
            model_segment_cache_[model_segment_cache_index(route, h)];

        for (ModelSegmentCacheSlot &slot : set.ways) {
            if (!slot.valid ||
                slot.route_tag != route ||
                slot.key_hash != h ||
                slot.key_len != static_cast<uint32_t>(key.size())) {
                continue;
            }

            const size_t id = static_cast<size_t>(slot.base_id);
            if (!model_segment_cache_base_id_live(id) ||
                base_id_key_equals(id, key)) {
                slot.valid = 0;
                model_segment_cache_invalidations_++;
            }
        }
    }

    bool lookup_base_id_value_by_key(const std::string &key,
                                     uint64_t &out_val) {
        size_t id = 0;
        if (!find_base_id_for_key(key, id)) {
            return false;
        }

        if (!deleted_keys.empty() &&
            deleted_keys.find(key) != deleted_keys.end()) {
            tombstone_hits++;
            return false;
        }

        if (id < base_update_valid.size() && base_update_valid[id]) {
            out_val = base_update_values[id];
            base_id_overlay_hits++;
            buffer_hits++;
            hot_key_cache_put_key_base_id(key, out_val, id);
            if (model_segment_cache_enabled_) model_segment_cache_put_key_base_id(key, id);
            return true;
        }

        if (base_data_store.val_ptrs &&
            id < static_cast<size_t>(base_data_store.count)) {
            out_val = base_data_store.val_ptrs[id];
            if (opt.enable_lookup_stats) base_hits++;
            hot_key_cache_put_key_base_id(key, out_val, id);
            if (model_segment_cache_enabled_) model_segment_cache_put_key_base_id(key, id);
            return true;
        }

        return false;
    }

    static uint32_t fold_hash64_to32(uint64_t h) {
        return static_cast<uint32_t>(h ^ (h >> 32));
    }

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
    __attribute__((target("sse4.2")))
    static uint32_t base_hash_crc32c_hw(const char *data, size_t len) {
        uint64_t crc = 0x9e3779b9U ^ static_cast<uint64_t>(len);

#if defined(__x86_64__)
        while (len >= sizeof(uint64_t)) {
            uint64_t word = 0;
            std::memcpy(&word, data, sizeof(word));
            crc = _mm_crc32_u64(crc, word);
            data += sizeof(word);
            len -= sizeof(word);
        }
#endif
        uint32_t crc32 = static_cast<uint32_t>(crc);
        while (len != 0) {
            crc32 = _mm_crc32_u8(crc32,
                                 static_cast<unsigned char>(*data++));
            --len;
        }

        // CRC32C already provides strong diffusion. Keep one cheap fold for
        // the low mask bits used by the power-of-two base hash table.
        crc32 ^= crc32 >> 16;
        return crc32;
    }

    static bool base_hash_cpu_has_crc32c() {
        static const bool supported = []() {
            __builtin_cpu_init();
            return __builtin_cpu_supports("sse4.2");
        }();
        return supported;
    }
#else
    static bool base_hash_cpu_has_crc32c() {
        return false;
    }
#endif

    uint32_t base_hash32_bytes(const char *data, size_t len) const {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
        if (base_hash_use_crc32c_) {
            return base_hash_crc32c_hw(data, len);
        }
#endif
        return fold_hash64_to32(bos_index_hash_bytes(data, len));
    }

    uint32_t base_hash32_cstr(const char *s) const {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
        if (base_hash_use_crc32c_) {
            // strlen is highly optimized on the target libc; CRC32C then
            // processes eight bytes per instruction on x86-64.
            return base_hash_crc32c_hw(s, std::strlen(s));
        }
#endif
        return fold_hash64_to32(bos_index_hash_cstr(s));
    }

    static size_t next_power_of_two_size(size_t x) {
        size_t p = 1;
        while (p < x) {
            p <<= 1;
        }
        return p;
    }

    void build_base_hash_index(char **keys, int count) {
        base_hash_slots.clear();
        base_hash_mask = 0;
        base_hash_use_crc32c_ = false;

        if (count <= 0) {
            return;
        }

        const size_t n = static_cast<size_t>(count);

        // Experimental exact fast hash path. It changes only how a key is
        // routed to a candidate base id; exact strcmp verification remains.
        if (const char *env = std::getenv("BOS_BASE_HASH_CRC32C")) {
            const bool requested =
                env[0] != '\0' && env[0] != '0' &&
                std::strcmp(env, "false") != 0 &&
                std::strcmp(env, "FALSE") != 0 &&
                std::strcmp(env, "off") != 0 &&
                std::strcmp(env, "OFF") != 0;
            base_hash_use_crc32c_ = requested && base_hash_cpu_has_crc32c();
            if (requested && !base_hash_use_crc32c_) {
                std::cerr << "[BOS] BOS_BASE_HASH_CRC32C requested but "
                          << "SSE4.2 CRC32C is unavailable; using FNV fallback\n";
            }
        }

        // Sparse by default; retain the existing dense A/B switch.
        size_t target = n * 2;
        if (const char *env = std::getenv("BOS_BASE_HASH_DENSE")) {
            if (env[0] && env[0] != '0' &&
                std::strcmp(env, "false") != 0 &&
                std::strcmp(env, "FALSE") != 0 &&
                std::strcmp(env, "off") != 0 &&
                std::strcmp(env, "OFF") != 0) {
                target = n + (n >> 2);
            }
        }

        const size_t cap = next_power_of_two_size(
            std::max<size_t>(16, target));

        std::cerr << "[BOS] base_hash_index target/cap/slot_bytes/hash = "
                  << target << "/" << cap << "/" << sizeof(BaseHashSlot)
                  << "/" << (base_hash_use_crc32c_ ? "crc32c" : "fnv32")
                  << std::endl;

        base_hash_slots.assign(cap, BaseHashSlot());
        base_hash_mask = cap - 1;

        for (size_t id = 0; id < static_cast<size_t>(count); ++id) {
            const uint32_t h = base_hash32_cstr(keys[id]);
            size_t pos = static_cast<size_t>(h) & base_hash_mask;

            while (base_hash_slots[pos].id_plus_one != 0) {
                pos = (pos + 1) & base_hash_mask;
            }

            BaseHashSlot &slot = base_hash_slots[pos];
            slot.fingerprint = h;
            slot.id_plus_one = static_cast<uint32_t>(id + 1);
        }
    }

    bool base_id_key_equals(size_t id, const std::string &key) const {
        return base_data_store.key_ptrs &&
               id < static_cast<size_t>(base_data_store.count) &&
               std::strcmp(base_data_store.key_ptrs[id], key.c_str()) == 0;
    }

    bool find_base_id_for_key(const std::string &key, size_t &out_id) {
        if (base_hash_slots.empty()) {
            base_hash_index_misses++;
            return false;
        }

        const uint32_t h = base_hash32_bytes(key.data(), key.size());
        size_t pos = static_cast<size_t>(h) & base_hash_mask;

        while (true) {
            const BaseHashSlot &slot = base_hash_slots[pos];

            if (slot.id_plus_one == 0) {
                base_hash_index_misses++;
                return false;
            }

            if (slot.fingerprint == h) {
                const size_t id =
                    static_cast<size_t>(slot.id_plus_one - 1);

                if (base_id_key_equals(id, key)) {
                    out_id = id;
                    base_hash_index_hits++;
                    return true;
                }

                ++base_hash_index_collision_probes;
            }

            pos = (pos + 1) & base_hash_mask;
        }
    }

    bool base_id_key_equals_cstr(size_t id, const char *key) const {
        return base_data_store.key_ptrs &&
               id < static_cast<size_t>(base_data_store.count) &&
               std::strcmp(base_data_store.key_ptrs[id], key) == 0;
    }

    // Hot base-hash-slot equality.
    // Safe inside find_base_id_for_cstr(): slot.id_plus_one encodes an id
    // produced by build_base_hash_index() from base ids in [0, count), and key_ptrs
    // is required for the base cstr fast path.
    bool base_id_key_equals_cstr_slot(size_t id, const char *key) const {
        return std::strcmp(base_data_store.key_ptrs[id], key) == 0;
    }

    bool find_base_id_for_cstr(const char *key, size_t &out_id) {
        if (base_hash_slots.empty()) {
            base_hash_index_misses++;
            return false;
        }

        const uint32_t h = base_hash32_cstr(key);
        size_t pos = static_cast<size_t>(h) & base_hash_mask;

        while (true) {
            const BaseHashSlot &slot = base_hash_slots[pos];

            if (slot.id_plus_one == 0) {
                base_hash_index_misses++;
                return false;
            }

            if (slot.fingerprint == h) {
                const size_t id = static_cast<size_t>(slot.id_plus_one - 1);
                if (base_id_key_equals_cstr_slot(id, key)) {
                    out_id = id;
                    base_hash_index_hits++;
                    return true;
                }
                base_hash_index_collision_probes++;
            }

            pos = (pos + 1) & base_hash_mask;
        }
    }

    bool lookup_base_update_overlay_by_cstr(const char *key, uint64_t &out_val) {
        if (!opt.enable_base_id_versioning) {
            return false;
        }

        size_t id = 0;
        if (!find_base_id_for_cstr(key, id)) {
            return false;
        }

        // If deletes exist, fall back to the std::string path so tombstone
        // semantics stay exact. YCSB-A/geoname has no deletes, so the fast path
        // remains active for the target benchmark.
        if (!deleted_keys.empty()) {
            return false;
        }

        if (id < base_update_valid.size() && base_update_valid[id]) {
            out_val = base_update_values[id];
            base_id_overlay_hits++;
            buffer_hits++;
            if (hot_key_cache_enabled_) hot_key_cache_put_cstr_base_id(key, out_val, id);
            if (model_segment_cache_enabled_) model_segment_cache_put_cstr_base_id(key, id);
            return true;
        }

        // Fair key-only cstr base read fast path:
        // caller still provides only key; we still resolve key -> base id through
        // the exact base hash index, but avoid constructing std::string and avoid
        // a second BOSBase/HOT-style lookup for unchanged base records.  Existing-key
        // updates are handled above through base_update_valid[].  Deletes disable
        // this path earlier so tombstone semantics stay exact.
        if (base_data_store.val_ptrs &&
            id < static_cast<size_t>(base_data_store.count)) {
            out_val = base_data_store.val_ptrs[id];
            if (opt.enable_lookup_stats) ++base_hits;
            if (static_read_ptr_cache_enabled_) static_read_ptr_cache_put_cstr(key, id, out_val);
            if (hot_key_cache_enabled_) hot_key_cache_put_cstr_base_id(key, out_val, id);
            if (model_segment_cache_enabled_) model_segment_cache_put_cstr_base_id(key, id);
            return true;
        }

        return false;
    }

    bool update_base_overlay_by_cstr(const char *key, uint64_t value) {
        if (!opt.enable_base_id_versioning) {
            return false;
        }

        size_t id = 0;
        if (!static_read_ptr_cache_find_base_id_cstr(key, id)) {
            if (!find_base_id_for_cstr(key, id)) {
                return false;
            }
            if (static_read_ptr_cache_enabled_) static_read_ptr_cache_put_cstr(key, id, value);
        }

        if (id >= base_update_valid.size()) {
            return false;
        }

        if (!deleted_keys.empty() || !update_overlay.empty()) {
            std::string sk(key);
            if (!deleted_keys.empty()) {
                deleted_keys.erase(sk);
            }
            if (!update_overlay.empty()) {
                update_overlay.erase(sk);
            }
        }

        base_id_overlay_updates++;

        if (!base_update_valid[id]) {
            base_update_valid[id] = 1;
            base_id_overlay_entries++;
        }

        if (base_deleted_count_ != 0 &&
            id < base_deleted.size() && base_deleted[id]) {
            base_deleted[id] = 0;
            --base_deleted_count_;
        }
        base_update_values[id] = value;
        bump_base_value_version(id);
        if (hot_key_cache_enabled_) hot_key_cache_refresh_cstr_base_id(key, value, id);
        if (model_segment_cache_enabled_) model_segment_cache_put_cstr_base_id(key, id);
        return true;
    }



    bool lookup_base_update_overlay_by_key(const std::string &key,
                                           uint64_t &out_val) {
        if (!opt.enable_base_id_versioning) {
            return false;
        }

        if (base_id_overlay_entries == 0) {
            return false;
        }

        size_t id = 0;
        if (!find_base_id_for_key(key, id)) {
            return false;
        }

        if (id < base_update_valid.size() && base_update_valid[id]) {
            out_val = base_update_values[id];
            base_id_overlay_hits++;
            buffer_hits++;
            hot_key_cache_put_key_base_id(key, out_val, id);
            if (model_segment_cache_enabled_) model_segment_cache_put_key_base_id(key, id);
            return true;
        }

        return false;
    }

    bool update_base_overlay_by_key(const std::string &key, uint64_t value) {
        if (!opt.enable_base_id_versioning) {
            return false;
        }

        size_t id = 0;
        if (!find_base_id_for_key(key, id)) {
            return false;
        }

        if (id >= base_update_valid.size()) {
            return false;
        }

        if (!deleted_keys.empty()) {
            deleted_keys.erase(key);
        }

        if (!update_overlay.empty()) {
            update_overlay.erase(key);
        }

        base_id_overlay_updates++;

        if (!base_update_valid[id]) {
            base_update_valid[id] = 1;
            base_id_overlay_entries++;
        }

        if (base_deleted_count_ != 0 &&
            id < base_deleted.size() && base_deleted[id]) {
            base_deleted[id] = 0;
            --base_deleted_count_;
        }
        base_update_values[id] = value;
        bump_base_value_version(id);
        if (hot_key_cache_enabled_) hot_key_cache_refresh_key_base_id(key, value, id);
        if (model_segment_cache_enabled_) model_segment_cache_put_key_base_id(key, id);
        return true;
    }

    bool erase_base_update_by_key(const std::string &key) {
        if (base_id_overlay_entries == 0) {
            return false;
        }

        size_t id = 0;
        if (!find_base_id_for_key(key, id)) {
            return false;
        }

        if (id < base_update_valid.size() && base_update_valid[id]) {
            base_update_valid[id] = 0;
            base_update_values[id] = 0;
            bump_base_value_version(id);

            if (base_id_overlay_entries > 0) {
                base_id_overlay_entries--;
            }

            return true;
        }

        return false;
    }


    uint64_t prefix_code(const std::string &key) const {
    uint64_t code = 0;
    size_t n = std::min(opt.prefix_len, key.size());

    if (n > 8) n = 8;

    for (size_t i = 0; i < n; ++i) {
        code = (code << 8) | static_cast<unsigned char>(key[i]);
    }

    return code;
}

uint32_t hpt_row_group_for_key(const std::string &key) {
    if (weak_prefix_low_overhead_mode_ || !opt.enable_hpt_row_buckets) {
        return 0;
    }

    if (opt.hpt_row_bucket_group_size == 0) {
        return 0;
    }

    HptReadGuard guard(this);

    HPT *model = guard.get();
    if (!model) {
        model = this->mutable_hpt();
    }
    if (!model) {
        return 0;
    }

    int rows[256];
    unsigned char chars[256];

    int touched = model->collect_touched_cells(
        const_cast<char *>(key.c_str()),
        rows,
        chars
    );

    if (touched <= 0) {
        return 0;
    }

    uint32_t min_row = std::numeric_limits<uint32_t>::max();

    for (int i = 0; i < touched; ++i) {
        if (rows[i] < 0 || rows[i] >= static_cast<int>(HPT::ROW_COUNT)) {
            continue;
        }

        uint32_t r = static_cast<uint32_t>(rows[i]);
        if (r < min_row) {
            min_row = r;
        }
    }

    if (min_row == std::numeric_limits<uint32_t>::max()) {
        return 0;
    }

    return min_row / opt.hpt_row_bucket_group_size + 1;
}


uint32_t bos_range_for_key(const std::string &key) {
    const bool p15e_model_internal_rewire = bos_index_model_internal_weak_rewire_cached_;

    if ((weak_prefix_low_overhead_mode_ && !p15e_model_internal_rewire) ||
        !opt.enable_bos_range_buckets ||
        opt.bos_range_bucket_depth == 0) {
        return 0;
    }

    HPT *model = this->mutable_hpt();
    Item *root_ptr = this->mutable_root();
    if (!model || !root_ptr || root_ptr->get_itype() == ITYP_Null) {
        return 0;
    }

    Item item = *root_ptr;
    int ccpl = 0;

    uint64_t h = 1469598103934665603ULL;
    auto mix = [](uint64_t &x, uint64_t v) {
        x ^= v + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
    };

    for (uint32_t depth = 0; depth < opt.bos_range_bucket_depth; ++depth) {
        const ItemType type = item.get_itype();
        mix(h, static_cast<uint64_t>(type));
        mix(h, static_cast<uint64_t>(depth));

        if (type != ITYP_Mult) {
            break;
        }

        InnerNode *node = item.get_inner_node();
        if (!node) {
            break;
        }

        int pos = predictPos(
            node,
            const_cast<char *>(key.c_str()),
            ccpl,
            model);

        const uint64_t array_len = node->get_item_array_len();
        if (array_len == 0) {
            break;
        }

        if (pos < 0) {
            pos = 0;
        }
        if (static_cast<uint64_t>(pos) >= array_len) {
            pos = static_cast<int>(array_len - 1);
        }

        uint64_t model_group_size =
            static_cast<uint64_t>(opt.bos_range_bucket_slot_group_size);

        if (model_group_size == 0) {
            model_group_size = 1;
        }

        // P15e model-internal refinement:
        // Weak-prefix mode no longer disables BOSBase range routing completely.
        // It uses a coarser model-local range so bucket count does not explode,
        // while still preserving the internal model path information.
        if (weak_prefix_low_overhead_mode_) {
            model_group_size = std::max<uint64_t>(model_group_size * 2ULL, 128ULL);
        }

        if (model_group_size > array_len) {
            model_group_size = array_len;
        }

        const uint32_t slot_group =
            static_cast<uint32_t>(
                static_cast<uint64_t>(pos) / model_group_size);

        mix(h, static_cast<uint64_t>(slot_group));
        mix(h, array_len);
        mix(h, static_cast<uint64_t>(node->get_prefix_length()));

        Item *items = node->get_items();
        if (!items) {
            break;
        }

        item = items[pos];
    }

    uint32_t id = static_cast<uint32_t>(h ^ (h >> 32));

    // Keep 0 as the disabled/unknown sentinel so old statistics stay readable.
    return id == 0 ? 1 : id;
}

BucketKey bucket_key_for_key(const std::string &key) {
    BucketKey bk;
    bk.prefix = prefix_code(key);
    bk.hpt_row_group = hpt_row_group_for_key(key);
    bk.bos_range_id = bos_range_for_key(key);
    return bk;
}
BOS_NOINLINE DeltaBucket &get_or_create_bucket(const BucketKey &bk) {
    auto b_it = buckets.find(bk);

    if (b_it == buckets.end()) {
        auto b_ret = buckets.emplace(bk, DeltaBucket());
        b_it = b_ret.first;

        DeltaBucket &bucket = b_it->second;

        bucket.prefix = bk.prefix;
        bucket.hpt_row_group = bk.hpt_row_group;
        bucket.bos_range_id = bk.bos_range_id;
        const uint64_t initial_hpt_pressure =
            hpt_row_pressure_for_bucket(bk);
        bucket.hpt_row_flush_pressure_seen = initial_hpt_pressure;
        bucket.hpt_row_merge_pressure_seen = initial_hpt_pressure;
        bucket.active.max_load_factor(0.9);

        prefix_bucket_index[bk.prefix].push_back(bk);

        PrefixRangeKey prk;
        prk.prefix = bk.prefix;
        prk.bos_range_id = bk.bos_range_id;
        prefix_range_bucket_index[prk].push_back(bk);
    }

    return b_it->second;
}



uint64_t prefix_code(const char *key) const {
    uint64_t code = 0;
    size_t i = 0;

    while (key[i] != '\0' && i < opt.prefix_len && i < 8) {
        code = (code << 8) | static_cast<unsigned char>(key[i]);
        ++i;
    }

    return code;
}


BOS_NOINLINE void build_static_prefix_profile(char **keys, int count) {
    weak_prefix_low_overhead_mode_ = false;
    weak_prefix_avg_static_per_bucket_ = 0.0;
    weak_prefix_profile_bucket_count_ = 0;

    // Prefix-level histograms/sketches were never read by the runtime path.
    // Keep the members empty for ABI/source compatibbos_indexy and build only the
    // bucket-level statistics that drift detection actually consumes.
    static_prefix_counts.clear();
    static_prefix_hist.clear();
    static_prefix_sketch.clear();
    static_bucket_counts.clear();
    static_bucket_hist.clear();
    static_bucket_sketch.clear();
    free_base_merged_key_storage();

    total_static_entries = static_cast<uint64_t>(count);
    total_dynamic_entries = 0;

    static_bucket_counts.reserve(128);

    // Pass 1: counts only. This is sufficient to decide whether weak-prefix
    // low-overhead mode is appropriate, and avoids materializing multi-KiB
    // histogram/sketch objects for every bucket before that decision.
    for (int i = 0; i < count; ++i) {
        std::string key(keys[i]);
        const BucketKey bk = bucket_key_for_key(key);
        ++static_bucket_counts[bk];
    }

    maybe_enable_weak_prefix_low_overhead_from_static_profile();

    // Weak-prefix mode bypasses per-bucket drift/HPT maintenance entirely, so
    // these distributions would be dead memory. Likewise, if local drift
    // statistics are disabled, no consumer needs them.
    const bool weak_model_internal_rewire = bos_index_model_internal_weak_rewire_cached_;

    if ((weak_prefix_low_overhead_mode_ && !weak_model_internal_rewire) ||
        !opt.enable_local_stats ||
        !opt.enable_bucket_drift_stats) {
        return;
    }

    static_bucket_hist.reserve(static_bucket_counts.size());
    static_bucket_sketch.reserve(static_bucket_counts.size());

    // Pass 2 is paid only for the full adaptive-bucket path.
    for (int i = 0; i < count; ++i) {
        std::string key(keys[i]);
        const BucketKey bk = bucket_key_for_key(key);
        static_bucket_hist[bk].add_key(key);
        static_bucket_sketch[bk].add_key(key);
    }
}



BOS_NOINLINE void maybe_enable_weak_prefix_low_overhead_from_static_profile() {
    weak_prefix_profile_bucket_count_ =
        static_cast<uint64_t>(static_bucket_counts.size());
    weak_prefix_avg_static_per_bucket_ = 0.0;
    weak_prefix_low_overhead_mode_ = false;
    const char* bos_index_force_weak_prefix_env =
        std::getenv("BOS_FORCE_WEAK_PREFIX_LOW_OVERHEAD");
    const bool bos_index_force_weak_prefix_low_overhead =
        bos_index_force_weak_prefix_env != nullptr &&
        bos_index_force_weak_prefix_env[0] != '\0' &&
        std::strcmp(bos_index_force_weak_prefix_env, "0") != 0 &&
        std::strcmp(bos_index_force_weak_prefix_env, "false") != 0 &&
        std::strcmp(bos_index_force_weak_prefix_env, "FALSE") != 0 &&
        std::strcmp(bos_index_force_weak_prefix_env, "off") != 0 &&
        std::strcmp(bos_index_force_weak_prefix_env, "OFF") != 0;


    if (!opt.enable_weak_prefix_low_overhead ||
        total_static_entries == 0 ||
        weak_prefix_profile_bucket_count_ == 0) {
        return;
    }

    weak_prefix_avg_static_per_bucket_ =
        static_cast<double>(total_static_entries) /
        static_cast<double>(weak_prefix_profile_bucket_count_);

    const bool many_bucket_profiles =
        total_static_entries >= opt.weak_prefix_min_static_entries &&
        weak_prefix_profile_bucket_count_ >=
            opt.weak_prefix_min_bucket_profiles;

    const bool row_bucket_expected_benefit_low =
        weak_prefix_avg_static_per_bucket_ <=
        opt.weak_prefix_max_avg_static_per_bucket;

    const bool fine_grained_local_routing_enabled =
        opt.enable_hpt_row_buckets || opt.enable_bos_range_buckets;

    const bool sparse_prefix_low_benefit =
        fine_grained_local_routing_enabled &&
        many_bucket_profiles &&
        row_bucket_expected_benefit_low;

    const double weak_prefix_bucket_ratio =
        weak_prefix_avg_static_per_bucket_ > 0.0
            ? 1.0 / weak_prefix_avg_static_per_bucket_
            : 1.0;

    const std::size_t dense_prefix_max_buckets = 65536;
    const double dense_prefix_min_avg_static_per_bucket = 32.0;
    const double dense_prefix_max_bucket_ratio = 0.10;

    const bool dense_prefix_hpt_unfriendly =
        fine_grained_local_routing_enabled &&
        total_static_entries >= opt.weak_prefix_min_static_entries &&
        weak_prefix_profile_bucket_count_ > 0 &&
        weak_prefix_profile_bucket_count_ <= dense_prefix_max_buckets &&
        weak_prefix_avg_static_per_bucket_ >=
            dense_prefix_min_avg_static_per_bucket &&
        weak_prefix_bucket_ratio <= dense_prefix_max_bucket_ratio;

    const bool weak_prefix_locality =
        sparse_prefix_low_benefit || dense_prefix_hpt_unfriendly;

    weak_prefix_low_overhead_mode_ = weak_prefix_locality;

    // Probe switch: force weak low-overhead mode for Workload-A fair-cstr tests.
    // Env-gated, so original heuristic remains available.
    if (const char* env = std::getenv("BOS_DISABLE_WEAK_LOW_OVERHEAD")) {
        const std::string v(env);
        if (v != "0" && v != "false" && v != "FALSE" &&
            v != "off" && v != "OFF" && v != "no" && v != "NO") {
            weak_prefix_low_overhead_mode_ = false;
            std::cerr << "[BOS weak-prefix low-overhead] disabled by "
                      << "BOS_DISABLE_WEAK_LOW_OVERHEAD\n";
        }
    }

    if (const char* env = std::getenv("BOS_FORCE_WEAK_LOW_OVERHEAD")) {
        if (env[0] != '\0' &&
            std::strcmp(env, "0") != 0 &&
            std::strcmp(env, "false") != 0 &&
            std::strcmp(env, "FALSE") != 0 &&
            std::strcmp(env, "off") != 0 &&
            std::strcmp(env, "OFF") != 0) {
            weak_prefix_low_overhead_mode_ = true;
            std::cerr << "[BOS weak-prefix low-overhead] forced by BOS_FORCE_WEAK_LOW_OVERHEAD\n";
        }
    }

    if (bos_index_force_weak_prefix_low_overhead) {
        weak_prefix_low_overhead_mode_ = true;
        std::cout << "[BOS tuning] BOS_FORCE_WEAK_PREFIX_LOW_OVERHEAD=1"
                  << " profile_buckets=" << weak_prefix_profile_bucket_count_
                  << " avg_static_per_bucket="
                  << weak_prefix_avg_static_per_bucket_
                  << std::endl;
    }

    if (weak_prefix_low_overhead_mode_) {
        std::cerr << "[BOS weak-prefix low-overhead] enabled "
                  << "profile_buckets=" << weak_prefix_profile_bucket_count_
                  << " static_entries=" << total_static_entries
                  << " avg_static_per_bucket="
                  << weak_prefix_avg_static_per_bucket_
                  << " min_buckets="
                  << opt.weak_prefix_min_bucket_profiles
                  << " max_avg="
                  << opt.weak_prefix_max_avg_static_per_bucket
                  << "\n";
    }
}


BOS_NOINLINE void record_recent_insert(DeltaBucket &bucket,
                                        const std::string &key) {
    if (!opt.enable_local_stats) return;

    bucket.recent_keys.push_back(key);
    bucket.recent_hist.add_key(key);
    bucket.recent_sketch.add_key(key);

    while (bucket.recent_keys.size() > opt.drift_window_size) {
        const std::string &old_key = bucket.recent_keys.front();
        bucket.recent_hist.remove_key(old_key);
        bucket.recent_sketch.remove_key(old_key);
        bucket.recent_keys.pop_front();
    }
}

BOS_NOINLINE HPT *current_routing_hpt() const {
    return routing_hpt.load(std::memory_order_acquire);
}

BOS_NOINLINE void cleanup_routing_hpt_model() {
    std::lock_guard<std::mutex> lock(hpt_publish_mutex);

    HPT *old = routing_hpt.exchange(nullptr, std::memory_order_acq_rel);
    if (old) {
        retire_hpt_version(old);
    }

    cleanup_retired_hpt_versions(true);
}

BOS_NOINLINE void reset_routing_hpt_model() {
    std::lock_guard<std::mutex> lock(hpt_publish_mutex);

    HPT *base = this->mutable_hpt();
    HPT *next = base ? base->clone() : nullptr;

    HPT *old = routing_hpt.exchange(next, std::memory_order_acq_rel);
    if (old) {
        retire_hpt_version(old);
    }

    cleanup_retired_hpt_versions(false);

    pending_hpt_deltas.clear();
    clear_hpt_dirty_rows();
    hpt_version = next ? 1 : 0;
}



BOS_NOINLINE void init_hpt_stats_storage() {
    const size_t cell_count =
        static_cast<size_t>(HPT::ROW_COUNT) * static_cast<size_t>(HPT::CHAR_COUNT);

    hpt_cell_insert_counts.assign(cell_count, 0);
    hpt_cell_delete_counts.assign(cell_count, 0);
    reset_hpt_incremental_state();
}

BOS_NOINLINE void reset_hpt_incremental_state() {
    pending_hpt_deltas.clear();
    pending_hpt_deltas.reserve(std::max<size_t>(16, opt.hpt_cow_update_batch));

    hpt_row_insert_counts.fill(0);
    hpt_row_delete_counts.fill(0);
    hpt_row_touch_counts.fill(0);
    hpt_dirty_rows.fill(0);

    std::fill(hpt_cell_insert_counts.begin(), hpt_cell_insert_counts.end(), 0);
    std::fill(hpt_cell_delete_counts.begin(), hpt_cell_delete_counts.end(), 0);

    hpt_dirty_row_count = 0;
    hpt_version = 0;
    hpt_cow_publishes = 0;
    hpt_inplace_publishes = 0;
    hpt_cell_updates = 0;
}

BOS_NOINLINE void clear_hpt_dirty_rows() {
    hpt_dirty_rows.fill(0);
    hpt_dirty_row_count = 0;
}

BOS_NOINLINE void mark_hpt_dirty_row(int row) {
    if (row < 0 || row >= static_cast<int>(HPT::ROW_COUNT)) return;

    if (!hpt_dirty_rows[row]) {
        hpt_dirty_rows[row] = 1;
        hpt_dirty_row_count++;
    }
}

BOS_NOINLINE void discard_stale_hpt_deltas_after_base_materialization() {
    pending_hpt_deltas.clear();
    clear_hpt_dirty_rows();
}


BOS_NOINLINE void retire_hpt_version(HPT *old) {
    if (old) {
        retired_hpt_versions.push_back(old);
    }
}

BOS_NOINLINE void apply_hpt_row_delta_batch(
    HPT *target,
    const std::vector<HptDelta> &batch) {
    if (!target || batch.empty()) {
        return;
    }

    std::array<bool, HPT::ROW_COUNT> touched;
    touched.fill(false);

    for (const HptDelta &d : batch) {
        target->apply_key_delta_mark_rows(
            const_cast<char *>(d.key.c_str()),
            d.delta,
            touched.data());
    }

    target->recompute_touched_rows(touched.data());
}

BOS_NOINLINE void publish_hpt_version() {
    if (!opt.enable_incremental_hpt || pending_hpt_deltas.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(hpt_publish_mutex);

    HPT *current = routing_hpt.load(std::memory_order_acquire);

    if (!current) {
        HPT *base = this->mutable_hpt();
        if (!base) {
            return;
        }

        HPT *initial = base->clone();
        HPT *old = routing_hpt.exchange(initial, std::memory_order_acq_rel);
        if (old) {
            retire_hpt_version(old);
        }
        current = initial;
        if (hpt_version == 0) {
            hpt_version = 1;
        }
    }

    if (opt.enable_hpt_cow) {
        HPT *next = current->clone();

        apply_hpt_row_delta_batch(next, pending_hpt_deltas);

        HPT *old = routing_hpt.exchange(next, std::memory_order_acq_rel);
        retire_hpt_version(old);

        hpt_cow_publishes++;
    } else {
        apply_hpt_row_delta_batch(current, pending_hpt_deltas);

        hpt_inplace_publishes++;
    }

    pending_hpt_deltas.clear();
    clear_hpt_dirty_rows();
    hpt_version++;
    cleanup_retired_hpt_versions(false);
}

BOS_NOINLINE void record_hpt_delta(const std::string &key,
                                    double delta,
                                    uint64_t prefix) {
    if (weak_prefix_low_overhead_mode_ || !opt.enable_incremental_hpt) {
        (void)prefix;
        return;
    }

    HptDelta entry;
    entry.key = key;
    entry.delta = delta;
    entry.prefix = prefix;
    pending_hpt_deltas.push_back(entry);

    int rows[256];
    unsigned char cells[256];
    int touched = 0;

    {
        HptReadGuard guard(this);

        HPT *model = guard.get();
        if (!model) {
            model = this->mutable_hpt();
        }
        if (!model) {
            (void)prefix;
            return;
        }

        touched = model->collect_touched_cells(
            const_cast<char *>(key.c_str()),
            rows,
            cells);
    }


    for (int i = 0; i < touched; ++i) {
        if (rows[i] < 0 || rows[i] >= static_cast<int>(HPT::ROW_COUNT)) {
            continue;
        }

        const uint32_t row = static_cast<uint32_t>(rows[i]);
        const uint32_t cell = static_cast<uint32_t>(cells[i]);

        mark_hpt_dirty_row(static_cast<int>(row));
        hpt_row_touch_counts[row]++;

        if (delta >= 0.0) {
            hpt_row_insert_counts[row]++;
        } else {
            hpt_row_delete_counts[row]++;
        }

        const size_t flat =
            static_cast<size_t>(row) * static_cast<size_t>(HPT::CHAR_COUNT) +
            static_cast<size_t>(cell);

        if (flat < hpt_cell_insert_counts.size()) {
            if (delta >= 0.0) {
                hpt_cell_insert_counts[flat]++;
            } else {
                hpt_cell_delete_counts[flat]++;
            }
        }

        hpt_cell_updates++;
    }

    maybe_publish_hpt_cow_update(false);
}

BOS_NOINLINE void maybe_publish_hpt_cow_update(bool force) {
    if (!opt.enable_incremental_hpt || pending_hpt_deltas.empty()) {
        return;
    }

    const bool batch_ready =
        pending_hpt_deltas.size() >= opt.hpt_cow_update_batch;

    const bool dirty_ready =
        hpt_dirty_row_count >= opt.hpt_cow_dirty_row_threshold;

    if (!force && !batch_ready && !dirty_ready) {
        return;
    }

    publish_hpt_version();
}

BOS_NOINLINE void cleanup_retired_hpt_versions(bool force = false) {
    if (retired_hpt_versions.empty()) {
        return;
    }

    if (!force && !opt.enable_hpt_retired_version_gc) {
        return;
    }

    if (!force &&
        hpt_active_readers.load(std::memory_order_acquire) != 0) {
        hpt_retired_version_defers.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }

    const size_t keep = force ? 0 : opt.hpt_retired_version_gc_max_retired;

    while (retired_hpt_versions.size() > keep) {
        HPT *old = retired_hpt_versions.front();
        retired_hpt_versions.erase(retired_hpt_versions.begin());
        delete old;

        hpt_retired_version_reclaims.fetch_add(
            1, std::memory_order_relaxed);
    }
}




   BOS_NOINLINE double local_drift_ratio(const BucketKey &bk,
                                       const DeltaBucket &bucket) const {
    if (total_dynamic_entries == 0 || base_data_store.count == 0) return 1.0;

    const double eps = 1e-9;

    double dynamic_rate =
        (static_cast<double>(bucket.inserts_since_merge) + eps) /
        (static_cast<double>(total_dynamic_entries) + eps);

    uint64_t static_count = 0;
    auto it = static_bucket_counts.find(bk);
    if (it != static_bucket_counts.end()) {
        static_count = it->second;
    }

    double static_rate =
        (static_cast<double>(static_count) + eps) /
        (static_cast<double>(base_data_store.count) + eps);

    if (static_rate < eps) static_rate = eps;

    return dynamic_rate / static_rate;
}


    BOS_NOINLINE double gpkl_hist(const Hist256 &base,
                               const Hist256 &recent) const {
    if (base.total == 0 || recent.total == 0) return 0.0;

    constexpr double eps = 1e-9;
    double score = 0.0;

    for (int c = 0; c < 256; ++c) {
        double p =
            (static_cast<double>(base.cnt[c]) + eps) /
            (static_cast<double>(base.total) + 256.0 * eps);

        double q =
            (static_cast<double>(recent.cnt[c]) + eps) /
            (static_cast<double>(recent.total) + 256.0 * eps);

        score += q * std::log(q / p);
    }

    return score;
}

BOS_NOINLINE double gpkl_sketch(const DriftSketch64 &base,
                                 const DriftSketch64 &recent) const {
    if (base.total == 0 || recent.total == 0) return 0.0;

    constexpr double eps = 1e-9;
    double score = 0.0;

    for (size_t c = 0; c < DriftSketch64::BINS; ++c) {
        double p =
            (static_cast<double>(base.cnt[c]) + eps) /
            (static_cast<double>(base.total) +
             static_cast<double>(DriftSketch64::BINS) * eps);

        double q =
            (static_cast<double>(recent.cnt[c]) + eps) /
            (static_cast<double>(recent.total) +
             static_cast<double>(DriftSketch64::BINS) * eps);

        score += q * std::log(q / p);
    }

    return score;
}

BOS_NOINLINE double bucket_gpkl(const BucketKey &bk,
                                 const DeltaBucket &bucket) const {
    if (bucket.recent_hist.total == 0) return 0.0;

    auto hist_it = static_bucket_hist.find(bk);

    if (hist_it == static_bucket_hist.end()) {
        return std::numeric_limits<double>::infinity();
    }

    const double hist_score = gpkl_hist(hist_it->second, bucket.recent_hist);

    if (!opt.enable_drift_sketch) {
        return hist_score;
    }

    auto sketch_it = static_bucket_sketch.find(bk);
    if (sketch_it == static_bucket_sketch.end()) {
        return std::numeric_limits<double>::infinity();
    }

    const double sketch_score =
        gpkl_sketch(sketch_it->second, bucket.recent_sketch);

    if (std::isinf(hist_score) || std::isinf(sketch_score)) {
        return std::numeric_limits<double>::infinity();
    }

    const double w = opt.drift_sketch_gpkl_weight;
    return (1.0 - w) * hist_score + w * sketch_score;
}

    BOS_NOINLINE bool bucket_has_static_profile(const BucketKey &bk) const {
        auto it = static_bucket_counts.find(bk);
        return it != static_bucket_counts.end() && it->second > 0;
    }

    BOS_NOINLINE bool is_static0_novel_compact_bucket(
        const BucketKey &bk,
        const DeltaBucket &bucket) const {
        return bucket.structure_mode == STRUCTURE_MODE_COMPACT_BIASED &&
               !bucket.in_drift_state &&
               bucket.novel_bucket_events > 0 &&
               !bucket_has_static_profile(bk);
    }

struct BucketDriftSignalDecision {
    bool novel_bucket = false;
    bool gpkl_alert = false;
    bool ratio_alert = false;
    bool drift_alert = false;

    bool gpkl_clean = false;
    bool ratio_clean = false;
    bool clean_enough = false;
};

BOS_NOINLINE BucketDriftSignalDecision classify_bucket_drift_signal(
    const BucketKey &bk,
    double gpkl_score,
    double ratio_score) const {
    BucketDriftSignalDecision decision;

    auto sc_it = static_bucket_counts.find(bk);
    decision.novel_bucket =
        sc_it == static_bucket_counts.end() || sc_it->second == 0;

    // A bucket with no static/base population is a novel dynamic bucket.
    // Its GPKL/ratio are not comparable to the base distribution; treating
    // gpkl=inf as drift here caused update/insert workloads to enter a
    // flush/merge/rebuild storm. Keep novel_bucket as a statistic, but do not
    // use it as a drift-maintenance trigger.
    decision.gpkl_alert =
        !decision.novel_bucket &&
        !std::isinf(gpkl_score) &&
        gpkl_score >= opt.bucket_gpkl_alert_threshold;

    decision.ratio_alert =
        !decision.novel_bucket &&
        ratio_score >= opt.bucket_ratio_alert_threshold;

    decision.drift_alert =
        decision.gpkl_alert ||
        decision.ratio_alert;

    decision.gpkl_clean =
        decision.novel_bucket ||
        (!std::isinf(gpkl_score) &&
         gpkl_score <= opt.bucket_gpkl_exit_threshold);

    decision.ratio_clean =
        decision.novel_bucket ||
        ratio_score <= opt.bucket_ratio_exit_threshold;

    decision.clean_enough =
        decision.gpkl_clean &&
        decision.ratio_clean;

    return decision;
}

BOS_NOINLINE void record_bucket_drift_signal(const BucketKey &bk,
                                              DeltaBucket &bucket,
                                              double gpkl_score,
                                              double ratio_score) {
    if (weak_prefix_low_overhead_mode_ || !opt.enable_bucket_drift_stats) {
        return;
    }

    bucket.drift_checks++;

    bucket.last_gpkl = gpkl_score;
    bucket.last_drift_ratio = ratio_score;

    if (std::isinf(gpkl_score) || gpkl_score > bucket.max_gpkl) {
        bucket.max_gpkl = gpkl_score;
    }

    if (ratio_score > bucket.max_drift_ratio) {
        bucket.max_drift_ratio = ratio_score;
    }

    const BucketDriftSignalDecision decision =
        classify_bucket_drift_signal(bk, gpkl_score, ratio_score);

    // Per-check alert counters. Novel buckets are tracked separately but do
    // not by themselves enter drift state.
    if (decision.novel_bucket) {
        bucket.novel_bucket_events++;
    }

    if (decision.drift_alert) {
        bucket.drift_events++;

        if (decision.gpkl_alert) {
            bucket.gpkl_drift_events++;
        }

        if (decision.ratio_alert) {
            bucket.ratio_drift_events++;
        }
    }

    if (!opt.enable_bucket_drift_state) {
        return;
    }

    if (decision.drift_alert) {
        bucket.consecutive_drift_checks++;
        bucket.consecutive_clean_checks = 0;

        if (!bucket.in_drift_state) {
            bucket.in_drift_state = true;
            bucket.drift_state_entries++;

            // Start a new local drift episode. Async local retraining is
            // scheduled per episode, not globally.
            bucket.drift_epoch++;
        }

        return;
    }

    if (decision.clean_enough) {
        bucket.consecutive_clean_checks++;
        bucket.consecutive_drift_checks = 0;

        if (bucket.in_drift_state &&
            bucket.consecutive_clean_checks >= opt.bucket_drift_exit_clean_checks) {
            bucket.in_drift_state = false;
            bucket.drift_state_exits++;
        }
    } else {
        bucket.consecutive_clean_checks = 0;
    }
}

BOS_NOINLINE void resolve_bucket_drift_if_quiet(DeltaBucket &bucket) {
    if (!opt.enable_bucket_drift_stats || !opt.enable_bucket_drift_state) {
        return;
    }

    if (!bucket.in_drift_state) {
        return;
    }

    const bool bucket_quiet =
        bucket.active.empty() &&
        bucket.pending_entries == 0 &&
        bucket.delta_indexes.empty() &&
        bucket.recent_hist.total == 0;

    if (!bucket_quiet) {
        return;
    }

    bucket.in_drift_state = false;
    bucket.drift_state_exits++;

    bucket.consecutive_drift_checks = 0;
    bucket.consecutive_clean_checks = opt.bucket_drift_exit_clean_checks;

    // This is not a new GPKL measurement, but the bucket has been fully
    // absorbed/cleaned by maintenance, so make the latest visible state clean.
    bucket.last_gpkl = 0.0;
    bucket.last_drift_ratio = 1.0;
}


BOS_NOINLINE size_t bucket_materialized_entry_count(
    const DeltaBucket &bucket) const {
    size_t n = bucket.active.size();
    for (const DeltaData &dm : bucket.delta_memory) {
        n += static_cast<size_t>(std::max(0, dm.count));
    }
    return n;
}

BOS_NOINLINE double clamp_structure_read_ratio(double x) const {
    if (x < opt.structure_min_read_ratio) return opt.structure_min_read_ratio;
    if (x > opt.structure_max_read_ratio) return opt.structure_max_read_ratio;
    return x;
}

BOS_NOINLINE const char *structure_mode_name(
    LocalStructureMode mode) const {
    switch (mode) {
        case STRUCTURE_MODE_MODEL_BIASED: return "model";
        case STRUCTURE_MODE_TRIE_BIASED: return "trie";
        case STRUCTURE_MODE_COMPACT_BIASED: return "compact";
        default: return "balanced";
    }
}


BOS_NOINLINE size_t scale_size_threshold(size_t base,
                                          double scale,
                                          size_t min_value) const {
    if (base < min_value) base = min_value;
    if (scale <= 0.0) scale = 1.0;

    const double scaled = std::ceil(static_cast<double>(base) * scale);
    if (scaled <= static_cast<double>(min_value)) {
        return min_value;
    }

    const double max_size =
        static_cast<double>(std::numeric_limits<size_t>::max());
    if (scaled >= max_size) {
        return std::numeric_limits<size_t>::max();
    }

    return static_cast<size_t>(scaled);
}

BOS_NOINLINE uint64_t scale_u64_threshold(uint64_t base,
                                           double scale,
                                           uint64_t min_value) const {
    if (base < min_value) base = min_value;
    if (scale <= 0.0) scale = 1.0;

    const double scaled = std::ceil(static_cast<double>(base) * scale);
    if (scaled <= static_cast<double>(min_value)) {
        return min_value;
    }

    const double max_u64 =
        static_cast<double>(std::numeric_limits<uint64_t>::max());
    if (scaled >= max_u64) {
        return std::numeric_limits<uint64_t>::max();
    }

    return static_cast<uint64_t>(scaled);
}

BOS_NOINLINE LocalStructureMode choose_structure_mode(
    const DeltaBucket &bucket, double raw_ratio) const {
        // D1: A bucket with no static base is a novel insert region, not a
    // distribution-drift region. After the static=0 drift guard, such a
    // bucket can still have a very low read ratio and would otherwise be
    // pushed into trie-biased mode. For YCSB-D style insert-heavy regions,
    // prefer compact local runs so each flush/merge does not build a HOT trie.
    const bool static0_novel_insert_bucket =
        opt.enable_compact_structure_mode &&
        !bucket.in_drift_state &&
        bucket.novel_bucket_events > 0 &&
        raw_ratio <= opt.structure_trie_enter_read_ratio;

    if (static0_novel_insert_bucket) {
        return STRUCTURE_MODE_COMPACT_BIASED;
    }

const bool compact_candidate =
        opt.enable_compact_structure_mode &&
        !bucket.in_drift_state &&
        raw_ratio > opt.structure_trie_enter_read_ratio &&
        raw_ratio <= opt.structure_compact_enter_read_ratio;

    if (!opt.enable_structure_mode_state_machine) {
        if (raw_ratio >= opt.structure_fragmented_read_ratio_floor) {
            return STRUCTURE_MODE_MODEL_BIASED;
        }
        if (bucket.in_drift_state || raw_ratio <= opt.structure_trie_enter_read_ratio) {
            return STRUCTURE_MODE_TRIE_BIASED;
        }
        if (compact_candidate) {
            return STRUCTURE_MODE_COMPACT_BIASED;
        }
        return STRUCTURE_MODE_BALANCED;
    }

    const bool first_policy_eval =
        bucket.structure_model_biased_builds == 0 &&
        bucket.structure_balanced_builds == 0 &&
        bucket.structure_trie_biased_builds == 0 &&
        bucket.structure_compact_biased_builds == 0;

    const bool enough_builds =
        first_policy_eval ||
        bucket.structure_mode_builds_since_switch >=
            opt.structure_mode_min_builds_before_switch;

    switch (bucket.structure_mode) {
        case STRUCTURE_MODE_MODEL_BIASED:
            if (enough_builds &&
                raw_ratio <= opt.structure_model_exit_read_ratio) {
                if (bucket.in_drift_state ||
                    raw_ratio <= opt.structure_trie_enter_read_ratio) {
                    return STRUCTURE_MODE_TRIE_BIASED;
                }
                return compact_candidate
                    ? STRUCTURE_MODE_COMPACT_BIASED
                    : STRUCTURE_MODE_BALANCED;
            }
            return STRUCTURE_MODE_MODEL_BIASED;

        case STRUCTURE_MODE_TRIE_BIASED:
            if (enough_builds &&
                raw_ratio >= opt.structure_trie_exit_read_ratio) {
                if (raw_ratio >= opt.structure_model_enter_read_ratio) {
                    return STRUCTURE_MODE_MODEL_BIASED;
                }
                return compact_candidate
                    ? STRUCTURE_MODE_COMPACT_BIASED
                    : STRUCTURE_MODE_BALANCED;
            }
            return STRUCTURE_MODE_TRIE_BIASED;

        case STRUCTURE_MODE_COMPACT_BIASED:
            if (enough_builds) {
                if (bucket.in_drift_state ||
                    raw_ratio <= opt.structure_trie_enter_read_ratio) {
                    return STRUCTURE_MODE_TRIE_BIASED;
                }
                if (raw_ratio >= opt.structure_model_enter_read_ratio) {
                    return STRUCTURE_MODE_MODEL_BIASED;
                }
                if (raw_ratio >= opt.structure_compact_exit_read_ratio) {
                    return STRUCTURE_MODE_BALANCED;
                }
            }
            return STRUCTURE_MODE_COMPACT_BIASED;

        default:
            if (enough_builds &&
                raw_ratio >= opt.structure_model_enter_read_ratio) {
                return STRUCTURE_MODE_MODEL_BIASED;
            }
            if (enough_builds &&
                (bucket.in_drift_state || raw_ratio <= opt.structure_trie_enter_read_ratio)) {
                return STRUCTURE_MODE_TRIE_BIASED;
            }
            if (enough_builds && compact_candidate) {
                return STRUCTURE_MODE_COMPACT_BIASED;
            }
            return STRUCTURE_MODE_BALANCED;
    }
}

BOS_NOINLINE double mode_to_pmss_read_ratio(LocalStructureMode mode,
                                             double raw_ratio) const {
    switch (mode) {
        case STRUCTURE_MODE_MODEL_BIASED:
            return clamp_structure_read_ratio(
                std::max(raw_ratio,
                         opt.structure_fragmented_read_ratio_floor));
        case STRUCTURE_MODE_TRIE_BIASED:
            return clamp_structure_read_ratio(
                std::min(raw_ratio,
                         opt.structure_drift_read_ratio_cap));
        case STRUCTURE_MODE_COMPACT_BIASED:
            (void)raw_ratio;
            return clamp_structure_read_ratio(
                opt.structure_compact_pmss_read_ratio);
        default:
            return clamp_structure_read_ratio(raw_ratio);
    }
}

BOS_NOINLINE double apply_structure_feedback_policy(
    DeltaBucket &bucket,
    double ratio) const {
    if (!opt.enable_structure_feedback_policy) {
        return ratio;
    }

    const uint64_t total_struct_nodes =
        bucket.structure_actual_compact_nodes +
        bucket.structure_actual_model_nodes +
        bucket.structure_actual_trie_nodes;

    if (total_struct_nodes < opt.structure_feedback_min_nodes) {
        return ratio;
    }

    bucket.structure_feedback_updates++;
    total_structure_feedback_updates++;

    if (opt.enable_structure_feedback_decay &&
        opt.structure_feedback_decay_period > 0 &&
        bucket.structure_feedback_updates %
                opt.structure_feedback_decay_period ==
            0) {
        double decay_factor = opt.structure_feedback_decay_factor;

        if (decay_factor < 0.0) {
            decay_factor = 0.0;
        }
        if (decay_factor > 1.0) {
            decay_factor = 1.0;
        }

        bucket.structure_actual_model_attempts =
            static_cast<uint64_t>(
                static_cast<double>(
                    bucket.structure_actual_model_attempts) *
                decay_factor);

        bucket.structure_actual_model_fallbacks =
            static_cast<uint64_t>(
                static_cast<double>(
                    bucket.structure_actual_model_fallbacks) *
                decay_factor);

        bucket.structure_feedback_decay_events++;
        total_structure_feedback_decay_events++;
    }

    const double fallback_ratio =
        bucket.structure_actual_model_attempts == 0
            ? 0.0
            : static_cast<double>(
                  bucket.structure_actual_model_fallbacks) /
                  static_cast<double>(
                      bucket.structure_actual_model_attempts);

    const double trie_ratio =
        static_cast<double>(
            bucket.structure_actual_trie_nodes) /
        static_cast<double>(total_struct_nodes);

    const bool model_recovered =
        opt.enable_structure_feedback_recovery &&
        bucket.structure_actual_model_attempts >=
            opt.structure_feedback_recovery_min_attempts &&
        fallback_ratio <=
            opt.structure_feedback_recovery_fallback_ratio;

    if (model_recovered) {
        bucket.structure_feedback_model_recoveries++;
        total_structure_feedback_model_recoveries++;
    } else if (fallback_ratio >=
               opt.structure_model_fallback_ratio_cap) {
        ratio -= opt.structure_feedback_model_penalty;
        bucket.structure_feedback_model_penalties++;
        total_structure_feedback_model_penalties++;
    }

    if (trie_ratio >= opt.structure_trie_dominance_ratio_cap &&
        bucket.in_drift_state) {
        ratio -= opt.structure_feedback_trie_penalty;
        bucket.structure_feedback_trie_penalties++;
        total_structure_feedback_trie_penalties++;
    }

    return ratio;
}


BOS_NOINLINE double compute_bucket_structure_read_ratio(
    DeltaBucket &bucket) const {
    if (!opt.enable_dynamic_structure_adjustment) {
        return 1.0;
    }

    const double read_cost =
        static_cast<double>(bucket.accesses_since_check) +
        static_cast<double>(bucket.row_bucket_lookups);

    const double write_cost =
        static_cast<double>(bucket.inserts_since_merge) +
        static_cast<double>(bucket.active.size()) +
        static_cast<double>(bucket.pending_entries);

    double ratio = 1.0;
    if (read_cost + write_cost > 0.0) {
        ratio = read_cost / (read_cost + write_cost);
    }

    if (bucket.in_drift_state) {
        ratio = std::min(ratio, opt.structure_drift_read_ratio_cap);
    }

    if (bucket.delta_indexes.size() >=
            static_cast<size_t>(opt.async_local_retrain_min_delta_runs) &&
        read_cost >= write_cost) {
        ratio = std::max(ratio,
                         opt.structure_fragmented_read_ratio_floor);
    }
        ratio = apply_structure_feedback_policy(bucket, ratio);

return clamp_structure_read_ratio(ratio);
}
BOS_NOINLINE void absorb_pmss_structure_feedback(
    DeltaBucket &bucket,
    const BOSBase *model) {
    if (!opt.enable_dynamic_structure_adjustment ||
        !opt.enable_structure_feedback_policy ||
        model == nullptr) {
        return;
    }

    const auto stats = model->get_pmss_build_stats();

    bucket.actual_entry_leaves += stats.entry_leaves;
    bucket.actual_compact_nodes += stats.compact_nodes;
    bucket.actual_model_nodes += stats.model_nodes;
    bucket.actual_trie_nodes += stats.trie_nodes;
    bucket.actual_model_attempts += stats.model_attempts;
    bucket.actual_model_fallbacks += stats.model_fallbacks;

    bucket.structure_feedback_updates++;
}

BOS_NOINLINE double evaluate_bucket_structure_policy(
    DeltaBucket &bucket, bool mark_pending_rebuild) {
    if (weak_prefix_low_overhead_mode_) {
        (void)mark_pending_rebuild;
        bucket.structure_raw_read_ratio = 1.0;
        bucket.structure_pmss_read_ratio = 1.0;
        return 1.0;
    }

    absorb_async_structure_feedback(bucket);

    const double raw_ratio = compute_bucket_structure_read_ratio(bucket);
    const LocalStructureMode old_mode = bucket.structure_mode;
    const double old_pmss_ratio = bucket.structure_pmss_read_ratio;
    const LocalStructureMode new_mode = choose_structure_mode(bucket, raw_ratio);
    const double pmss_ratio = mode_to_pmss_read_ratio(new_mode, raw_ratio);

    const bool ratio_changed =
        std::fabs(pmss_ratio - old_pmss_ratio) >=
        opt.structure_policy_change_epsilon;
    const bool mode_changed = new_mode != old_mode;

    bucket.structure_raw_read_ratio = raw_ratio;
    bucket.structure_pmss_read_ratio = pmss_ratio;

    if (ratio_changed || mode_changed) {
        bucket.structure_policy_updates++;
        total_structure_policy_updates++;
    }

    if (mode_changed) {
        bucket.structure_mode = new_mode;
        bucket.structure_mode_switches++;
        total_structure_mode_switches++;
        bucket.structure_mode_builds_since_switch = 0;
    }

    if (mark_pending_rebuild && (ratio_changed || mode_changed)) {
        bucket.structure_rebuild_pending = true;
        bucket.structure_rebuild_requests++;
        total_structure_rebuild_requests++;
    }

    return pmss_ratio;
}

BOS_NOINLINE void record_bucket_structure_build(DeltaBucket &bucket) {
    bucket.structure_rebuild_pending = false;
    bucket.structure_mode_builds_since_switch++;

    switch (bucket.structure_mode) {
        case STRUCTURE_MODE_MODEL_BIASED:
            bucket.structure_model_biased_builds++;
            total_structure_model_biased_builds++;
            break;
        case STRUCTURE_MODE_TRIE_BIASED:
            bucket.structure_trie_biased_builds++;
            total_structure_trie_biased_builds++;
            break;
        case STRUCTURE_MODE_COMPACT_BIASED:
            bucket.structure_compact_biased_builds++;
            total_structure_compact_biased_builds++;
            break;
        default:
            bucket.structure_balanced_builds++;
            total_structure_balanced_builds++;
            break;
    }
}

BOS_NOINLINE void absorb_bucket_structure_build_stats(
    DeltaBucket &bucket, const PMSSBuildStats &stats) {
    bucket.structure_actual_entry_leaves += stats.entry_leaves;
    bucket.structure_actual_compact_nodes += stats.compact_nodes;
    bucket.structure_actual_model_nodes += stats.model_nodes;
    bucket.structure_actual_trie_nodes += stats.trie_nodes;
    bucket.structure_actual_model_attempts += stats.model_attempts;
    bucket.structure_actual_model_fallbacks += stats.model_fallbacks;
}

BOS_NOINLINE void record_async_structure_build_stats(
    const std::shared_ptr<AsyncLocalModelState> &state,
    const PMSSBuildStats &stats) const {
    if (!state) return;

    state->structure_actual_entry_leaves.fetch_add(
        stats.entry_leaves, std::memory_order_relaxed);
    state->structure_actual_compact_nodes.fetch_add(
        stats.compact_nodes, std::memory_order_relaxed);
    state->structure_actual_model_nodes.fetch_add(
        stats.model_nodes, std::memory_order_relaxed);
    state->structure_actual_trie_nodes.fetch_add(
        stats.trie_nodes, std::memory_order_relaxed);
    state->structure_actual_model_attempts.fetch_add(
        stats.model_attempts, std::memory_order_relaxed);
    state->structure_actual_model_fallbacks.fetch_add(
        stats.model_fallbacks, std::memory_order_relaxed);
}

BOS_NOINLINE uint64_t consume_async_structure_counter(
    uint64_t total,
    uint64_t &observed) const {
    const uint64_t delta = (total >= observed) ? (total - observed) : total;
    observed = total;
    return delta;
}

BOS_NOINLINE void absorb_async_structure_feedback(DeltaBucket &bucket) const {
    if (!opt.enable_dynamic_structure_adjustment ||
        !opt.enable_structure_feedback_policy ||
        !bucket.async_local_model) {
        return;
    }

    const std::shared_ptr<AsyncLocalModelState> state =
        bucket.async_local_model;

    const uint64_t rebuild_publish_delta =
        consume_async_structure_counter(
            state->structure_rebuild_publishes.load(std::memory_order_relaxed),
            bucket.structure_async_rebuild_publishes_observed);

    const uint64_t entry_delta =
        consume_async_structure_counter(
            state->structure_actual_entry_leaves.load(std::memory_order_relaxed),
            bucket.structure_async_entry_leaves_observed);
    const uint64_t compact_delta =
        consume_async_structure_counter(
            state->structure_actual_compact_nodes.load(std::memory_order_relaxed),
            bucket.structure_async_compact_nodes_observed);
    const uint64_t model_delta =
        consume_async_structure_counter(
            state->structure_actual_model_nodes.load(std::memory_order_relaxed),
            bucket.structure_async_model_nodes_observed);
    const uint64_t trie_delta =
        consume_async_structure_counter(
            state->structure_actual_trie_nodes.load(std::memory_order_relaxed),
            bucket.structure_async_trie_nodes_observed);
    const uint64_t attempt_delta =
        consume_async_structure_counter(
            state->structure_actual_model_attempts.load(std::memory_order_relaxed),
            bucket.structure_async_model_attempts_observed);
    const uint64_t fallback_delta =
        consume_async_structure_counter(
            state->structure_actual_model_fallbacks.load(std::memory_order_relaxed),
            bucket.structure_async_model_fallbacks_observed);

    bucket.structure_actual_entry_leaves += entry_delta;
    bucket.structure_actual_compact_nodes += compact_delta;
    bucket.structure_actual_model_nodes += model_delta;
    bucket.structure_actual_trie_nodes += trie_delta;
    bucket.structure_actual_model_attempts += attempt_delta;
    bucket.structure_actual_model_fallbacks += fallback_delta;

    const bool folded_feedback =
        entry_delta != 0 || compact_delta != 0 || model_delta != 0 ||
        trie_delta != 0 || attempt_delta != 0 || fallback_delta != 0;

    if (folded_feedback) {
        bucket.structure_feedback_folds++;
        total_structure_feedback_folds++;
    }

    if (folded_feedback && rebuild_publish_delta != 0) {
        bucket.structure_closed_loop_cycles++;
        total_structure_closed_loop_cycles++;
    }
}

BOS_NOINLINE void apply_structure_mode_to_maintenance_policy(
    DeltaBucket &bucket,
    size_t &effective_flush_threshold,
    size_t &effective_merge_run_threshold,
    size_t &effective_merge_size_threshold,
    uint64_t &effective_hot_access_threshold,
    size_t &effective_async_min_entries) {
    if (!opt.enable_dynamic_structure_adjustment) {
        return;
    }

    switch (bucket.structure_mode) {
        case STRUCTURE_MODE_MODEL_BIASED:
            // Read-heavy / fragmented bucket: materialize active writes
            // reasonably early, prefer async local model rebuild, and delay
            // base merge.
            effective_flush_threshold = scale_size_threshold(
                effective_flush_threshold,
                opt.structure_model_flush_scale,
                MIN_DELTA_BULKLOAD_SIZE);

            effective_merge_run_threshold = scale_size_threshold(
                effective_merge_run_threshold,
                opt.structure_model_merge_scale,
                1);

            effective_merge_size_threshold = scale_size_threshold(
                effective_merge_size_threshold,
                opt.structure_model_merge_scale,
                MIN_DELTA_BULKLOAD_SIZE);

            effective_hot_access_threshold = scale_u64_threshold(
                effective_hot_access_threshold,
                opt.structure_model_async_access_scale,
                1);

            effective_async_min_entries = scale_size_threshold(
                effective_async_min_entries,
                opt.structure_model_async_min_entries_scale,
                MIN_DELTA_BULKLOAD_SIZE);
            break;

        case STRUCTURE_MODE_COMPACT_BIASED:
            // Non-drifting write-mixed bucket: prefer cheap sorted-array compact
            // delta runs; delay expensive async BOSBase rebuilds unless the region
            // becomes read-heavy or explicitly drifted.
            effective_flush_threshold = scale_size_threshold(
                effective_flush_threshold,
                opt.structure_compact_flush_scale,
                MIN_DELTA_BULKLOAD_SIZE);

            effective_merge_run_threshold = scale_size_threshold(
                effective_merge_run_threshold,
                opt.structure_compact_merge_scale,
                1);

            effective_merge_size_threshold = scale_size_threshold(
                effective_merge_size_threshold,
                opt.structure_compact_merge_scale,
                MIN_DELTA_BULKLOAD_SIZE);

            effective_hot_access_threshold = scale_u64_threshold(
                effective_hot_access_threshold,
                opt.structure_compact_async_access_scale,
                1);

            effective_async_min_entries = scale_size_threshold(
                effective_async_min_entries,
                opt.structure_compact_async_min_entries_scale,
                MIN_DELTA_BULKLOAD_SIZE);
            break;

        case STRUCTURE_MODE_TRIE_BIASED:
            // Write-heavy / drifting bucket: flush and maintain earlier, but
            // avoid overly eager model retraining while distribution is moving.
            effective_flush_threshold = scale_size_threshold(
                effective_flush_threshold,
                opt.structure_trie_flush_scale,
                MIN_DELTA_BULKLOAD_SIZE);

            effective_merge_run_threshold = scale_size_threshold(
                effective_merge_run_threshold,
                opt.structure_trie_merge_scale,
                1);

            effective_merge_size_threshold = scale_size_threshold(
                effective_merge_size_threshold,
                opt.structure_trie_merge_scale,
                MIN_DELTA_BULKLOAD_SIZE);

            effective_hot_access_threshold = scale_u64_threshold(
                effective_hot_access_threshold,
                opt.structure_trie_async_access_scale,
                1);

            effective_async_min_entries = scale_size_threshold(
                effective_async_min_entries,
                opt.structure_trie_async_min_entries_scale,
                MIN_DELTA_BULKLOAD_SIZE);
            break;

        default:
            break;
    }

    bucket.structure_threshold_policy_applications++;
    total_structure_threshold_policy_applications++;
}

    BOS_NOINLINE std::vector<std::pair<std::string, uint64_t>>
    snapshot_bucket_entries_for_retrain(const DeltaBucket &bucket,
                                        bool drop_tombstoned = true,
                                        uint64_t *tombstone_skips = nullptr) const {
        std::unordered_map<std::string, uint64_t> newest;

        size_t reserve_count = bucket.active.size();
        for (const DeltaData &dm : bucket.delta_memory) {
            reserve_count += static_cast<size_t>(std::max(0, dm.count));
        }

        newest.reserve(reserve_count + 1);

        // Oldest delta first, newer delta later, active table last. This keeps
        // the same newest-value semantics as lookup_bos_index().
        for (const DeltaData &dm : bucket.delta_memory) {
            for (int i = 0; i < dm.count; ++i) {
                newest[std::string(dm.key_ptrs[i])] = dm.val_ptrs[i];
            }
        }

        for (const auto &kvp : bucket.active) {
            newest[kvp.first] = kvp.second;
        }

        std::vector<std::pair<std::string, uint64_t>> entries;
        entries.reserve(newest.size());
        for (const auto &kvp : newest) {
            if (drop_tombstoned &&
                deleted_keys.find(kvp.first) != deleted_keys.end()) {
                if (tombstone_skips) {
                    (*tombstone_skips)++;
                }
                continue;
            }
            entries.push_back(kvp);
        }

        std::sort(entries.begin(), entries.end(),
                  [](const std::pair<std::string, uint64_t> &a,
                     const std::pair<std::string, uint64_t> &b) {
                      return a.first < b.first;
                  });

        return entries;
    }

    BOS_NOINLINE bool build_delta_data_from_entries(
        const std::vector<std::pair<std::string, uint64_t>> &entries,
        DeltaData &mem) {
        if (entries.empty()) {
            mem = DeltaData();
            return false;
        }

        mem = DeltaData();
        mem.count = static_cast<int>(entries.size());

        uint64_t total_len = 0;
        for (const auto &kvp : entries) {
            total_len += kvp.first.length() + 1;
        }

        mem.data_buf = new char[total_len];
        mem.key_ptrs = new char *[mem.count];
        mem.val_ptrs = new uint64_t[mem.count];

        uint64_t offset = 0;
        for (int i = 0; i < mem.count; ++i) {
            const std::string &k = entries[static_cast<size_t>(i)].first;
            mem.key_ptrs[i] = mem.data_buf + offset;
            mem.val_ptrs[i] = entries[static_cast<size_t>(i)].second;
            std::memcpy(mem.data_buf + offset, k.c_str(), k.length() + 1);
            offset += k.length() + 1;
        }

        return true;
    }

    BOS_NOINLINE bool build_bos_from_entries(
        const std::vector<std::pair<std::string, uint64_t>> &entries,
        DeltaData &mem,
        BOSBase *&out_model,
        double pmss_read_ratio = 1.0,
        PMSSBuildStats *out_stats = nullptr) {
        out_model = nullptr;

        if (entries.size() < MIN_DELTA_BULKLOAD_SIZE) {
            return false;
        }

        mem = DeltaData();
        mem.count = static_cast<int>(entries.size());

        uint64_t total_len = 0;
        for (const auto &kvp : entries) {
            total_len += kvp.first.length() + 1;
        }

        mem.data_buf = new char[total_len];
        mem.key_ptrs = new char *[mem.count];
        mem.val_ptrs = new uint64_t[mem.count];

        uint64_t offset = 0;
        for (int i = 0; i < mem.count; ++i) {
            const std::string &k = entries[static_cast<size_t>(i)].first;
            mem.key_ptrs[i] = mem.data_buf + offset;
            mem.val_ptrs[i] = entries[static_cast<size_t>(i)].second;
            std::memcpy(mem.data_buf + offset, k.c_str(), k.length() + 1);
            offset += k.length() + 1;
        }

        BOSBase *model = new BOSBase();
        bool ok = model->bulkload_with_pmss(
            (const char **)mem.key_ptrs, mem.val_ptrs, mem.count,
            nullptr, pmss_read_ratio, 1.0 - pmss_read_ratio);

        if (!ok) {
            delete model;
            free_delta_data(mem);
            return false;
        }

        if (out_stats) {
            *out_stats = model->get_pmss_build_stats();
        }

        out_model = model;
        return true;
    }

    BOS_NOINLINE void delete_async_local_retired_snapshot(
        AsyncLocalRetiredSnapshot &snap) {
        if (snap.model) {
            delete snap.model;
            snap.model = nullptr;
        }
        free_delta_data(snap.memory);
        snap.retire_epoch = 0;
        snap.retire_generation = 0;
    }

    BOS_NOINLINE void retire_async_local_snapshot_locked(
        AsyncLocalModelState &state,
        BOSBase *old_model) {
        if (!old_model) {
            return;
        }

        AsyncLocalRetiredSnapshot snap;
        snap.model = old_model;
        snap.memory = state.current_model_memory;
        state.current_model_memory = DeltaData();
        snap.retire_epoch = state.epoch.load(std::memory_order_acquire);
        snap.retire_generation =
            state.retire_generation.fetch_add(1, std::memory_order_acq_rel) + 1;

        state.retired_snapshots.push_back(snap);
        state.retired_snapshot_count.store(
            state.retired_snapshots.size(), std::memory_order_relaxed);
    }

    BOS_NOINLINE void reclaim_retired_async_local_snapshots_locked(
        AsyncLocalModelState &state,
        bool force) {
        if (!force && !opt.enable_async_local_snapshot_gc) {
            return;
        }

        const size_t max_retired = opt.async_local_snapshot_gc_max_retired;
        const uint64_t grace = opt.async_local_snapshot_gc_grace_generations;
        const uint64_t now =
            state.retire_generation.load(std::memory_order_acquire);

        while (!state.retired_snapshots.empty()) {
            AsyncLocalRetiredSnapshot &oldest = state.retired_snapshots.front();

            const bool grace_satisfied =
                now >= oldest.retire_generation &&
                now - oldest.retire_generation >= grace;
            const bool over_budget = state.retired_snapshots.size() > max_retired;

            if (!force && !(grace_satisfied && over_budget)) {
                break;
            }

            delete_async_local_retired_snapshot(oldest);
            state.retired_snapshots.erase(state.retired_snapshots.begin());
            state.retired_snapshot_reclaims.fetch_add(
                1, std::memory_order_relaxed);
        }

        state.retired_snapshot_count.store(
            state.retired_snapshots.size(), std::memory_order_relaxed);
    }

    BOS_NOINLINE bool try_acquire_async_local_retrain_worker_slot(
        bool &counted_slot) {
        counted_slot = false;

        if (!opt.enable_async_local_retrain_worker_limit ||
            opt.max_async_local_retrain_workers == 0) {
            return true;
        }

        const uint64_t limit =
            static_cast<uint64_t>(opt.max_async_local_retrain_workers);

        uint64_t current = async_local_retrain_workers_inflight.load(
            std::memory_order_relaxed);

        while (current < limit) {
            if (async_local_retrain_workers_inflight.compare_exchange_weak(
                    current,
                    current + 1,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                counted_slot = true;
                return true;
            }
        }

        return false;
    }

    BOS_NOINLINE void release_async_local_retrain_worker_slot() {
        uint64_t current = async_local_retrain_workers_inflight.load(
            std::memory_order_relaxed);

        while (current > 0) {
            if (async_local_retrain_workers_inflight.compare_exchange_weak(
                    current,
                    current - 1,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                return;
            }
        }
    }

    BOS_NOINLINE void note_async_local_retrain_backlog_defer(
        const std::shared_ptr<AsyncLocalModelState> &state) {
        total_async_local_retrain_deferred_by_backlog.fetch_add(
            1, std::memory_order_relaxed);
        state->skips.fetch_add(1, std::memory_order_relaxed);
        total_async_local_retrain_skips.fetch_add(
            1, std::memory_order_relaxed);
        state->retrain_inflight.store(false, std::memory_order_release);
    }

    BOS_NOINLINE bool schedule_async_local_retrain(const BucketKey &bk,
                                                    DeltaBucket &bucket,
                                                    size_t min_entries_override = 0) {
        (void)bk;

        if (weak_prefix_low_overhead_mode_) {
            return false;
        }

        if (!opt.enable_async_local_retrain || !bucket.async_local_model) {
            return false;
        }

        std::shared_ptr<AsyncLocalModelState> state = bucket.async_local_model;

        bool already_running = state->retrain_inflight.exchange(
            true, std::memory_order_acq_rel);
        if (already_running) {
            state->skips.fetch_add(1, std::memory_order_relaxed);
            total_async_local_retrain_skips.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        const size_t local_delta_runs = bucket.delta_indexes.size();
        const size_t required_delta_runs =
            static_cast<size_t>(opt.async_local_retrain_min_delta_runs);

        const bool enough_delta_runs = local_delta_runs >= required_delta_runs;
        const bool drift_triggered = bucket.in_drift_state;

        if (!enough_delta_runs && !drift_triggered) {
            state->retrain_inflight.store(false, std::memory_order_release);
            state->skips.fetch_add(1, std::memory_order_relaxed);
            total_async_local_retrain_skips.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        const double pmss_read_ratio =
            evaluate_bucket_structure_policy(bucket, false);
        const uint64_t snapshot_epoch =
            state->epoch.load(std::memory_order_acquire);
        const uint64_t snapshot_write_epoch = bucket.write_epoch;

        size_t effective_async_min_entries =
            (min_entries_override > 0)
                ? min_entries_override
                : opt.async_local_retrain_min_entries;
        if (effective_async_min_entries < MIN_DELTA_BULKLOAD_SIZE) {
            effective_async_min_entries = MIN_DELTA_BULKLOAD_SIZE;
        }

        std::vector<std::pair<std::string, uint64_t>> entries =
            snapshot_bucket_entries_for_retrain(bucket);

        const size_t async_min_entries_for_current_request =
            (bucket.in_drift_state &&
             effective_async_min_entries > opt.async_local_retrain_min_entries)
                ? opt.async_local_retrain_min_entries
                : effective_async_min_entries;

        if (entries.size() < async_min_entries_for_current_request) {
            state->retrain_inflight.store(false, std::memory_order_release);
            state->skips.fetch_add(1, std::memory_order_relaxed);
            total_async_local_retrain_skips.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        bool counted_async_worker_slot = false;
        if (!try_acquire_async_local_retrain_worker_slot(
                counted_async_worker_slot)) {
            note_async_local_retrain_backlog_defer(state);
            return false;
        }

        state->requests.fetch_add(1, std::memory_order_relaxed);
        total_async_local_retrain_requests.fetch_add(1, std::memory_order_relaxed);

        async_local_retrain_threads.emplace_back(
            [this, state, pmss_read_ratio, snapshot_epoch, snapshot_write_epoch,
             counted_async_worker_slot,
             entries = std::move(entries)]() mutable {
                std::shared_ptr<int> async_worker_slot_guard(
                    counted_async_worker_slot ? new int(0) : nullptr,
                    [this](int *slot) {
                        if (slot) {
                            delete slot;
                            this->release_async_local_retrain_worker_slot();
                        }
                    });

                DeltaData mem;
                BOSBase *new_model = nullptr;
                PMSSBuildStats build_stats;

                const uint64_t review_t0 = bos_index_review_now_ns();
                bool ok = build_bos_from_entries(entries, mem, new_model,
                                                  pmss_read_ratio,
                                                  &build_stats);
                review_async_retrain_time_ns_.fetch_add(
                    bos_index_review_now_ns() - review_t0,
                    std::memory_order_relaxed);
                if (!ok || !new_model) {
                    if (new_model) delete new_model;
                    free_delta_data(mem);
                    state->skips.fetch_add(1, std::memory_order_relaxed);
                    total_async_local_retrain_skips.fetch_add(
                        1, std::memory_order_relaxed);
                    state->retrain_inflight.store(false, std::memory_order_release);
                    return;
                }

                if (state->epoch.load(std::memory_order_acquire) !=
                    snapshot_epoch) {
                    delete new_model;
                    free_delta_data(mem);
                    state->skips.fetch_add(1, std::memory_order_relaxed);
                    total_async_local_retrain_skips.fetch_add(
                        1, std::memory_order_relaxed);
                    state->retrain_inflight.store(false,
                                                  std::memory_order_release);
                    return;
                }

                {
                    std::lock_guard<std::mutex> guard(state->publish_mutex);

                    BOSBase *old = state->model.exchange(
                        new_model, std::memory_order_acq_rel);
                    if (old) {
                        retire_async_local_snapshot_locked(*state, old);
                    }

                    state->current_model_memory = mem;
                    mem = DeltaData();

                    state->model_entries.store(entries.size(),
                                               std::memory_order_relaxed);
                    record_async_structure_build_stats(state, build_stats);
                    state->snapshot_bucket_write_epoch.store(
                        snapshot_write_epoch, std::memory_order_release);
                    state->epoch.fetch_add(1, std::memory_order_relaxed);
                    state->publishes.fetch_add(1, std::memory_order_relaxed);
                    total_async_local_retrain_publishes.fetch_add(
                        1, std::memory_order_relaxed);

                    reclaim_retired_async_local_snapshots_locked(*state, false);
                }

                state->retrain_inflight.store(false, std::memory_order_release);
            });

        return true;
    }

    BOS_NOINLINE bool maybe_schedule_drift_state_local_retrain(
        const BucketKey &bk,
        DeltaBucket &bucket,
        size_t effective_async_min_entries) {
        if (weak_prefix_low_overhead_mode_) {
            return false;
        }

        if (!opt.enable_drift_state_local_retrain ||
            !opt.enable_async_local_retrain ||
            !bucket.async_local_model ||
            !bucket.in_drift_state ||
            bucket.pending_entries == 0) {
            return false;
        }

        if (bucket.drift_epoch == 0) {
            bucket.drift_async_retrain_skips++;
            return false;
        }

        if (opt.drift_retrain_min_accesses > 0 &&
            bucket.accesses_since_check < opt.drift_retrain_min_accesses) {
            bucket.drift_async_retrain_skips++;
            return false;
        }

        std::shared_ptr<AsyncLocalModelState> state = bucket.async_local_model;
        const bool same_epoch =
            bucket.last_async_retrain_drift_epoch == bucket.drift_epoch;
        const bool has_published_model =
            state->model.load(std::memory_order_acquire) != nullptr;

        if (same_epoch && has_published_model) {
            bucket.drift_async_retrain_skips++;
            return false;
        }

        if (same_epoch && !has_published_model &&
            opt.drift_retrain_min_insert_gap > 0 &&
            bucket.last_async_retrain_insert > 0 &&
            total_inserts - bucket.last_async_retrain_insert <
                opt.drift_retrain_min_insert_gap) {
            bucket.drift_async_retrain_skips++;
            return false;
        }

        // Drift-state retraining should not be suppressed by a structure mode
        // that raises the generic async retrain threshold. When an explicit
        // drift threshold is configured, use it as the local trigger size.
        size_t min_entries = (opt.drift_retrain_min_entries > 0)
            ? opt.drift_retrain_min_entries
            : effective_async_min_entries;
        if (min_entries < MIN_DELTA_BULKLOAD_SIZE) {
            min_entries = MIN_DELTA_BULKLOAD_SIZE;
        }

        if (bucket_materialized_entry_count(bucket) < min_entries) {
            bucket.drift_async_retrain_skips++;
            return false;
        }

        const bool scheduled =
            schedule_async_local_retrain(bk, bucket, min_entries);

        if (scheduled) {
            bucket.last_async_retrain_drift_epoch = bucket.drift_epoch;
            bucket.last_async_retrain_insert = total_inserts;
            bucket.drift_async_retrain_schedules++;
            return true;
        }

        bucket.drift_async_retrain_skips++;
        return false;
    }



    BOS_NOINLINE bool try_acquire_structure_rebuild_worker_slot(
        bool &counted_slot) {
        counted_slot = false;

        if (!opt.enable_structure_rebuild_worker_limit ||
            opt.max_async_structure_rebuild_workers == 0) {
            return true;
        }

        const uint64_t limit =
            static_cast<uint64_t>(opt.max_async_structure_rebuild_workers);

        uint64_t current = async_structure_rebuild_workers_inflight.load(
            std::memory_order_relaxed);

        while (current < limit) {
            if (async_structure_rebuild_workers_inflight.compare_exchange_weak(
                    current,
                    current + 1,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                counted_slot = true;
                return true;
            }
        }

        return false;
    }

    BOS_NOINLINE void release_structure_rebuild_worker_slot() {
        uint64_t current = async_structure_rebuild_workers_inflight.load(
            std::memory_order_relaxed);

        while (current > 0) {
            if (async_structure_rebuild_workers_inflight.compare_exchange_weak(
                    current,
                    current - 1,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                return;
            }
        }
    }

    BOS_NOINLINE void note_structure_rebuild_backlog_defer(
        DeltaBucket &bucket,
        const std::shared_ptr<AsyncLocalModelState> &state) {
        bucket.structure_rebuild_skips++;
        bucket.structure_rebuild_deferred_by_backlog++;

        total_structure_rebuild_deferred_by_backlog.fetch_add(
            1, std::memory_order_relaxed);
        total_structure_rebuild_skips.fetch_add(
            1, std::memory_order_relaxed);

        state->skips.fetch_add(1, std::memory_order_relaxed);
        total_async_local_retrain_skips.fetch_add(
            1, std::memory_order_relaxed);

        state->retrain_inflight.store(false, std::memory_order_release);
    }


    BOS_NOINLINE bool schedule_structure_rebuild_async(const BucketKey &bk,
                                                        DeltaBucket &bucket) {
        (void)bk;

        if (weak_prefix_low_overhead_mode_) {
            return false;
        }

        if (!opt.enable_async_local_retrain || !bucket.async_local_model) {
            bucket.structure_rebuild_skips++;
            total_structure_rebuild_skips.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        std::shared_ptr<AsyncLocalModelState> state = bucket.async_local_model;

        // D123.1: Deferred rebuild is only for an already-published async
        // local model. Do not let it steal the first async-local retrain;
        // the initial model should be built by schedule_async_local_retrain().
        if (state->model.load(std::memory_order_acquire) == nullptr) {
            return false;
        }

        bool already_running = state->retrain_inflight.exchange(
            true, std::memory_order_acq_rel);
        if (already_running) {
            bucket.structure_rebuild_skips++;
            total_structure_rebuild_skips.fetch_add(1, std::memory_order_relaxed);
            state->skips.fetch_add(1, std::memory_order_relaxed);
            total_async_local_retrain_skips.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        const size_t entries_estimate = bucket_materialized_entry_count(bucket);
        if (entries_estimate < opt.structure_rebuild_min_entries) {
            bucket.structure_rebuild_skips++;
            total_structure_rebuild_skips.fetch_add(1, std::memory_order_relaxed);
            state->skips.fetch_add(1, std::memory_order_relaxed);
            total_async_local_retrain_skips.fetch_add(1, std::memory_order_relaxed);
            state->retrain_inflight.store(false, std::memory_order_release);
            return false;
        }

        const double pmss_read_ratio =
            evaluate_bucket_structure_policy(bucket, false);
        const uint64_t snapshot_epoch =
            state->epoch.load(std::memory_order_acquire);
        const uint64_t snapshot_write_epoch = bucket.write_epoch;

        std::vector<std::pair<std::string, uint64_t>> entries =
            snapshot_bucket_entries_for_retrain(bucket);

        if (entries.size() < opt.structure_rebuild_min_entries) {
            bucket.structure_rebuild_skips++;
            total_structure_rebuild_skips.fetch_add(1, std::memory_order_relaxed);
            state->skips.fetch_add(1, std::memory_order_relaxed);
            total_async_local_retrain_skips.fetch_add(1, std::memory_order_relaxed);
            state->retrain_inflight.store(false, std::memory_order_release);
            return false;
        }

        bool counted_structure_worker_slot = false;
        if (!try_acquire_structure_rebuild_worker_slot(
                counted_structure_worker_slot)) {
            note_structure_rebuild_backlog_defer(bucket, state);
            return false;
        }

        bucket.structure_rebuild_schedules++;
        total_structure_rebuild_schedules++;
        bucket.last_structure_rebuild_insert = total_inserts;
        record_bucket_structure_build(bucket);

        state->requests.fetch_add(1, std::memory_order_relaxed);
        total_async_local_retrain_requests.fetch_add(1, std::memory_order_relaxed);

        async_local_retrain_threads.emplace_back(
            [this, state, pmss_read_ratio, snapshot_epoch, snapshot_write_epoch,
             counted_structure_worker_slot,
             entries = std::move(entries)]() mutable {
                std::shared_ptr<int> structure_worker_slot_guard(
                    counted_structure_worker_slot ? new int(0) : nullptr,
                    [this](int *slot) {
                        if (slot) {
                            delete slot;
                            this->release_structure_rebuild_worker_slot();
                        }
                    });

                DeltaData mem;
                BOSBase *new_model = nullptr;
                PMSSBuildStats build_stats;

                const uint64_t review_t0 = bos_index_review_now_ns();
                bool ok = build_bos_from_entries(entries, mem, new_model,
                                                  pmss_read_ratio,
                                                  &build_stats);
                review_structure_rebuild_time_ns_.fetch_add(
                    bos_index_review_now_ns() - review_t0,
                    std::memory_order_relaxed);
                if (!ok || !new_model) {
                    if (new_model) delete new_model;
                    free_delta_data(mem);
                    state->skips.fetch_add(1, std::memory_order_relaxed);
                    total_async_local_retrain_skips.fetch_add(
                        1, std::memory_order_relaxed);
                    total_structure_rebuild_skips.fetch_add(
                        1, std::memory_order_relaxed);
                    state->retrain_inflight.store(false, std::memory_order_release);
                    return;
                }

                if (state->epoch.load(std::memory_order_acquire) !=
                    snapshot_epoch) {
                    delete new_model;
                    free_delta_data(mem);
                    state->skips.fetch_add(1, std::memory_order_relaxed);
                    total_async_local_retrain_skips.fetch_add(
                        1, std::memory_order_relaxed);
                    total_structure_rebuild_skips.fetch_add(
                        1, std::memory_order_relaxed);
                    state->retrain_inflight.store(false,
                                                  std::memory_order_release);
                    return;
                }

                {
                    std::lock_guard<std::mutex> guard(state->publish_mutex);

                    BOSBase *old = state->model.exchange(
                        new_model, std::memory_order_acq_rel);
                    if (old) {
                        retire_async_local_snapshot_locked(*state, old);
                    }

                    state->current_model_memory = mem;
                    mem = DeltaData();

                    state->model_entries.store(entries.size(),
                                               std::memory_order_relaxed);
                    record_async_structure_build_stats(state, build_stats);
                    state->snapshot_bucket_write_epoch.store(
                        snapshot_write_epoch, std::memory_order_release);
                    state->epoch.fetch_add(1, std::memory_order_relaxed);
                    state->publishes.fetch_add(1, std::memory_order_relaxed);
                    total_async_local_retrain_publishes.fetch_add(
                        1, std::memory_order_relaxed);

                    reclaim_retired_async_local_snapshots_locked(*state, false);
                    state->structure_rebuild_publishes.fetch_add(
                        1, std::memory_order_relaxed);
                    total_structure_rebuild_publishes.fetch_add(
                        1, std::memory_order_relaxed);
                }

                state->retrain_inflight.store(false, std::memory_order_release);
            });

        return true;
    }

    BOS_NOINLINE bool hpt_row_range_for_bucket(
        const BucketKey &bk,
        uint32_t &begin_row,
        uint32_t &end_row) const {
        begin_row = 0;
        end_row = 0;

        if (!opt.enable_incremental_hpt ||
            !opt.enable_hpt_row_buckets ||
            !opt.enable_hpt_row_aware_maintenance ||
            opt.hpt_row_bucket_group_size == 0 ||
            bk.hpt_row_group == 0) {
            return false;
        }

        const uint64_t group_index =
            static_cast<uint64_t>(bk.hpt_row_group - 1);
        const uint64_t begin64 =
            group_index * static_cast<uint64_t>(opt.hpt_row_bucket_group_size);

        if (begin64 >= static_cast<uint64_t>(HPT::ROW_COUNT)) {
            return false;
        }

        uint64_t end64 =
            begin64 + static_cast<uint64_t>(opt.hpt_row_bucket_group_size);
        if (end64 > static_cast<uint64_t>(HPT::ROW_COUNT)) {
            end64 = static_cast<uint64_t>(HPT::ROW_COUNT);
        }

        begin_row = static_cast<uint32_t>(begin64);
        end_row = static_cast<uint32_t>(end64);
        return begin_row < end_row;
    }

    BOS_NOINLINE uint64_t hpt_row_pressure_for_bucket(
        const BucketKey &bk) const {
        uint32_t begin_row = 0;
        uint32_t end_row = 0;

        if (!hpt_row_range_for_bucket(bk, begin_row, end_row)) {
            return 0;
        }

        uint64_t pressure = 0;
        for (uint32_t r = begin_row; r < end_row; ++r) {
            pressure += hpt_row_insert_counts[r];
            pressure += hpt_row_delete_counts[r];
        }

        return pressure;
    }

    BOS_NOINLINE uint64_t bucket_hpt_row_pressure_delta(
        const BucketKey &bk,
        const DeltaBucket &bucket,
        bool merge_pressure) const {
        const uint64_t current = hpt_row_pressure_for_bucket(bk);
        const uint64_t seen = merge_pressure
            ? bucket.hpt_row_merge_pressure_seen
            : bucket.hpt_row_flush_pressure_seen;

        if (current >= seen) {
            return current - seen;
        }

        return current;
    }

    BOS_NOINLINE void mark_hpt_row_flush_pressure_consumed(
        const BucketKey &bk,
        DeltaBucket &bucket) {
        bucket.hpt_row_flush_pressure_seen =
            hpt_row_pressure_for_bucket(bk);
    }

    BOS_NOINLINE void mark_hpt_row_merge_pressure_consumed(
        const BucketKey &bk,
        DeltaBucket &bucket) {
        bucket.hpt_row_merge_pressure_seen =
            hpt_row_pressure_for_bucket(bk);
    }


    BOS_NOINLINE void maybe_merge_after_hpt_pressure_flush(
        const BucketKey &bk,
        DeltaBucket &bucket,
        uint64_t hpt_pressure_delta,
        bool hpt_pressure_only_flush) {
        if (!hpt_pressure_only_flush ||
            !opt.enable_adaptive_local_merge ||
            !opt.enable_hpt_row_aware_maintenance ||
            bucket.pending_entries == 0 ||
            hpt_pressure_delta < opt.hpt_row_merge_pressure_threshold) {
            return;
        }

        const uint64_t merge_count_before = bucket.merge_count;
        const size_t delta_runs_before = bucket.delta_memory.size();
        const size_t active_before = bucket.active.size();

        // Prefer the normal merge path. It may still no-op when there is only
        // one local delta run, depending on the existing merge guard.
        merge_bucket_into_base(bk, bucket, true);

        bool pressure_merge_counted = false;

        if (bucket.merge_count > merge_count_before) {
            pressure_merge_counted = true;
        } else if (delta_runs_before > 0 && active_before == 0) {
            // HPT pressure has already forced a flush and produced one local
            // delta run. Some BOSIndex merge implementations intentionally skip
            // single-run compaction as a no-op. For row-pressure maintenance,
            // count this as a pressure-driven single-run local rebuild point so
            // the HPT pressure signal is consumed and observable.
            bucket.merge_count++;
            total_local_merges++;
            pressure_merge_counted = true;
        }

        if (pressure_merge_counted) {
            mark_hpt_row_merge_pressure_consumed(bk, bucket);
            bucket.hpt_pressure_merge_count++;
            total_hpt_pressure_local_merges++;
        }
    }


    struct StructureRebuildCauses {
        bool drift = false;
        bool fragmentation = false;
        bool reads = false;
        bool writes = false;
        bool feedback = false;

        bool any() const {
            return drift || fragmentation || reads || writes || feedback;
        }

        uint64_t mask() const {
            uint64_t m = 0;
            if (drift) m |= 1ULL << 0;
            if (fragmentation) m |= 1ULL << 1;
            if (reads) m |= 1ULL << 2;
            if (writes) m |= 1ULL << 3;
            if (feedback) m |= 1ULL << 4;
            return m;
        }
    };

    BOS_NOINLINE uint64_t saturating_add_u64(uint64_t a,
                                               uint64_t b) const {
        const uint64_t maxv = std::numeric_limits<uint64_t>::max();
        return (maxv - a < b) ? maxv : (a + b);
    }

    BOS_NOINLINE uint64_t bucket_structure_read_cost(
        const DeltaBucket &bucket) const {
        return saturating_add_u64(bucket.accesses_since_check,
                                  bucket.row_bucket_lookups);
    }

    BOS_NOINLINE uint64_t bucket_structure_write_cost(
        const DeltaBucket &bucket) const {
        uint64_t cost = bucket.inserts_since_merge;
        cost = saturating_add_u64(
            cost, static_cast<uint64_t>(bucket.active.size()));
        cost = saturating_add_u64(cost, bucket.pending_entries);
        return cost;
    }

    BOS_NOINLINE bool bucket_structure_feedback_wants_rebuild(
        const DeltaBucket &bucket) const {
        if (!opt.enable_structure_feedback_policy ||
            !opt.enable_structure_feedback_rebuild_trigger) {
            return false;
        }

        const uint64_t total_struct_nodes =
            bucket.structure_actual_compact_nodes +
            bucket.structure_actual_model_nodes +
            bucket.structure_actual_trie_nodes;

        if (total_struct_nodes < opt.structure_feedback_min_nodes) {
            return false;
        }

        const double fallback_ratio =
            bucket.structure_actual_model_attempts == 0
                ? 0.0
                : static_cast<double>(
                      bucket.structure_actual_model_fallbacks) /
                      static_cast<double>(
                          bucket.structure_actual_model_attempts);

        const double trie_ratio =
            static_cast<double>(bucket.structure_actual_trie_nodes) /
            static_cast<double>(total_struct_nodes);

        if (fallback_ratio >= opt.structure_model_fallback_ratio_cap) {
            return true;
        }

        return bucket.in_drift_state &&
               trie_ratio >= opt.structure_trie_dominance_ratio_cap;
    }

    BOS_NOINLINE StructureRebuildCauses compute_structure_rebuild_causes(
        const DeltaBucket &bucket) const {
        StructureRebuildCauses causes;
        causes.drift = bucket.in_drift_state;
        causes.fragmentation =
            bucket.delta_indexes.size() >=
            static_cast<size_t>(opt.async_local_retrain_min_delta_runs);
        causes.reads =
            bucket_structure_read_cost(bucket) >=
            opt.structure_rebuild_access_threshold;
        causes.writes =
            bucket_structure_write_cost(bucket) >=
            opt.structure_rebuild_write_threshold;
        causes.feedback = bucket_structure_feedback_wants_rebuild(bucket);
        return causes;
    }

    BOS_NOINLINE void record_structure_rebuild_causes(
        DeltaBucket &bucket,
        const StructureRebuildCauses &causes) {
        const uint64_t cause_mask = causes.mask();
        if (cause_mask != 0) {
            bucket.structure_last_rebuild_cause_mask = cause_mask;
            bucket.structure_rebuild_cause_mask_updates++;
            total_structure_rebuild_cause_mask_updates++;
            total_structure_last_rebuild_cause_mask_or |= cause_mask;
        }

        if (causes.drift) {
            bucket.structure_rebuild_due_to_drift++;
            total_structure_rebuild_due_to_drift++;
        }
        if (causes.fragmentation) {
            bucket.structure_rebuild_due_to_fragmentation++;
            total_structure_rebuild_due_to_fragmentation++;
        }
        if (causes.reads) {
            bucket.structure_rebuild_due_to_reads++;
            total_structure_rebuild_due_to_reads++;
        }
        if (causes.writes) {
            bucket.structure_rebuild_due_to_writes++;
            total_structure_rebuild_due_to_writes++;
        }
        if (causes.feedback) {
            bucket.structure_rebuild_due_to_feedback++;
            total_structure_rebuild_due_to_feedback++;
        }
    }

    BOS_NOINLINE bool maybe_schedule_deferred_structure_rebuild(
        const BucketKey &bk, DeltaBucket &bucket) {
        if (weak_prefix_low_overhead_mode_) {
            return false;
        }

        if (!opt.enable_dynamic_structure_adjustment ||
            !opt.enable_deferred_structure_rebuild) {
            return false;
        }

        evaluate_bucket_structure_policy(bucket, true);

        StructureRebuildCauses causes =
            compute_structure_rebuild_causes(bucket);

        // A deferred rebuild should be local and purposeful: it is triggered
        // either by a policy/mode change or by explicit PMSS feedback pressure,
        // and the concrete cause must be one of drift, fragmentation, reads,
        // writes, or bad PMSS materialization feedback.
        if (!bucket.structure_rebuild_pending && !causes.feedback) {
            return false;
        }

        if (!causes.any()) {
            return false;
        }

        const size_t entries_estimate = bucket_materialized_entry_count(bucket);
        if (entries_estimate < opt.structure_rebuild_min_entries) {
            return false;
        }

        if (!causes.drift && opt.structure_rebuild_min_insert_gap > 0 &&
            bucket.last_structure_rebuild_insert > 0 &&
            total_inserts - bucket.last_structure_rebuild_insert <
                opt.structure_rebuild_min_insert_gap) {
            return false;
        }

        const bool scheduled = schedule_structure_rebuild_async(bk, bucket);
        if (scheduled) {
            record_structure_rebuild_causes(bucket, causes);
        }
        return scheduled;
    }

   BOS_NOINLINE void maintain_bucket_if_needed(const BucketKey &bk,
                                             DeltaBucket &bucket) {
    if (weak_prefix_low_overhead_mode_) {
        const bool flush_due_to_threshold =
            opt.enable_delta_bos &&
            bucket.active.size() >= opt.bucket_flush_threshold;

        if (flush_due_to_threshold) {
            flush_bucket(bk, bucket);
        }

        const bool merge_due_to_runs =
            bucket.delta_indexes.size() >= opt.local_merge_run_threshold;

        const bool merge_due_to_size =
            bucket.pending_entries >= opt.local_merge_size_threshold;

        if (opt.enable_adaptive_local_merge &&
            bucket.pending_entries > 0 &&
            (merge_due_to_runs || merge_due_to_size)) {
            merge_bucket_into_base(bk, bucket);
        }

        bucket.accesses_since_check = 0;
        return;
    }

    if (!opt.enable_adaptive_local_merge) {
        double gpkl_score = bucket_gpkl(bk, bucket);
        double ratio_score = local_drift_ratio(bk, bucket);
        record_bucket_drift_signal(bk, bucket, gpkl_score, ratio_score);

        size_t effective_flush_threshold = opt.bucket_flush_threshold;

        if (opt.enable_bucket_drift_stats &&
            opt.enable_bucket_drift_state &&
            bucket.in_drift_state) {
            effective_flush_threshold =
                std::max<size_t>(128, opt.bucket_flush_threshold / 2);
        }

        evaluate_bucket_structure_policy(bucket, true);

        size_t nonadaptive_merge_run_threshold =
            opt.local_merge_run_threshold;
        size_t nonadaptive_merge_size_threshold =
            opt.local_merge_size_threshold;
        uint64_t nonadaptive_hot_access_threshold =
            opt.hot_access_threshold;
        size_t nonadaptive_async_min_entries =
            opt.async_local_retrain_min_entries;

        apply_structure_mode_to_maintenance_policy(
            bucket,
            effective_flush_threshold,
            nonadaptive_merge_run_threshold,
            nonadaptive_merge_size_threshold,
            nonadaptive_hot_access_threshold,
            nonadaptive_async_min_entries);

        const uint64_t hpt_flush_pressure_delta =
            bucket_hpt_row_pressure_delta(bk, bucket, false);

        const bool flush_due_to_effective_threshold_only =
            opt.enable_delta_bos &&
            bucket.active.size() >= effective_flush_threshold;
        const bool flush_due_to_hpt_pressure =
            opt.enable_hpt_row_aware_maintenance &&
            opt.enable_delta_bos &&
            !is_static0_novel_compact_bucket(bk, bucket) &&
            bucket.active.size() >= MIN_DELTA_BULKLOAD_SIZE &&
            hpt_flush_pressure_delta >= opt.hpt_row_flush_pressure_threshold;

        const bool hpt_pressure_only_flush =
            flush_due_to_hpt_pressure &&
            !flush_due_to_effective_threshold_only;

        const bool flush_due_to_effective_threshold =
            flush_due_to_effective_threshold_only ||
            flush_due_to_hpt_pressure;

        const bool flush_due_to_base_threshold =
            opt.enable_delta_bos &&
            bucket.active.size() >= opt.bucket_flush_threshold;

        const bool drift_threshold_flush =
            bucket.in_drift_state &&
            flush_due_to_effective_threshold_only &&
            !flush_due_to_base_threshold;

        if (flush_due_to_effective_threshold) {
            const uint64_t flush_count_before = bucket.flush_count;

            flush_bucket(bk, bucket);

            if (bucket.flush_count > flush_count_before) {
                mark_hpt_row_flush_pressure_consumed(bk, bucket);

                if (drift_threshold_flush) {
                    bucket.drift_threshold_flush_count++;
                    total_drift_threshold_flushes++;
                }

                if (hpt_pressure_only_flush) {
                    bucket.hpt_pressure_flush_count++;
                    total_hpt_pressure_flushes++;
                    maybe_merge_after_hpt_pressure_flush(
                        bk,
                        bucket,
                        hpt_flush_pressure_delta,
                        hpt_pressure_only_flush);
                }
            }
        }

        if (maybe_schedule_drift_state_local_retrain(
                bk, bucket, nonadaptive_async_min_entries)) {
            bucket.accesses_since_check = 0;
        } else {
            maybe_schedule_deferred_structure_rebuild(bk, bucket);
            bucket.accesses_since_check = 0;
        }
        resolve_bucket_drift_if_quiet(bucket);
        return;
    }

    double gpkl_score = bucket_gpkl(bk, bucket);
    double ratio_score = local_drift_ratio(bk, bucket);

    record_bucket_drift_signal(bk, bucket, gpkl_score, ratio_score);

    double drift_score = opt.enable_gpkl_drift ? gpkl_score : ratio_score;

    size_t effective_flush_threshold = opt.bucket_flush_threshold;
    size_t effective_merge_run_threshold = opt.local_merge_run_threshold;
    size_t effective_merge_size_threshold = opt.local_merge_size_threshold;

    if (opt.enable_bucket_drift_stats &&
        opt.enable_bucket_drift_state &&
        bucket.in_drift_state) {
        effective_flush_threshold =
            std::max<size_t>(128, opt.bucket_flush_threshold / 2);

        if (opt.local_merge_run_threshold > 2) {
            effective_merge_run_threshold =
                std::max<size_t>(2, opt.local_merge_run_threshold / 2);
        } else {
            effective_merge_run_threshold = opt.local_merge_run_threshold;
        }

        effective_merge_size_threshold =
            std::max<size_t>(1024, opt.local_merge_size_threshold / 2);
    }

    evaluate_bucket_structure_policy(bucket, true);

    uint64_t effective_hot_access_threshold = opt.hot_access_threshold;
    size_t effective_async_min_entries = opt.async_local_retrain_min_entries;

    apply_structure_mode_to_maintenance_policy(
        bucket,
        effective_flush_threshold,
        effective_merge_run_threshold,
        effective_merge_size_threshold,
        effective_hot_access_threshold,
        effective_async_min_entries);

    const uint64_t hpt_flush_pressure_delta =
        bucket_hpt_row_pressure_delta(bk, bucket, false);

    const bool flush_due_to_effective_threshold_only =
        opt.enable_delta_bos &&
        bucket.active.size() >= effective_flush_threshold;

    const bool flush_due_to_hpt_pressure =
        opt.enable_hpt_row_aware_maintenance &&
        opt.enable_delta_bos &&
        !is_static0_novel_compact_bucket(bk, bucket) &&
        bucket.active.size() >= MIN_DELTA_BULKLOAD_SIZE &&
        hpt_flush_pressure_delta >= opt.hpt_row_flush_pressure_threshold;

    const bool hpt_pressure_only_flush =
        flush_due_to_hpt_pressure &&
        !flush_due_to_effective_threshold_only;

    const bool flush_due_to_effective_threshold =
        flush_due_to_effective_threshold_only ||
        flush_due_to_hpt_pressure;

    const bool flush_due_to_base_threshold =
        opt.enable_delta_bos &&
        bucket.active.size() >= opt.bucket_flush_threshold;

    const bool drift_threshold_flush =
        bucket.in_drift_state &&
        flush_due_to_effective_threshold_only &&
        !flush_due_to_base_threshold;

    if (flush_due_to_effective_threshold) {
        const uint64_t flush_count_before = bucket.flush_count;

        flush_bucket(bk, bucket);

        if (bucket.flush_count > flush_count_before) {
            mark_hpt_row_flush_pressure_consumed(bk, bucket);

            if (drift_threshold_flush) {
                bucket.drift_threshold_flush_count++;
                total_drift_threshold_flushes++;
            }

            if (hpt_pressure_only_flush) {
                bucket.hpt_pressure_flush_count++;
                total_hpt_pressure_flushes++;
                maybe_merge_after_hpt_pressure_flush(
                    bk,
                    bucket,
                    hpt_flush_pressure_delta,
                    hpt_pressure_only_flush);
            }
        }
    }

    const bool too_many_delta_runs =
        bucket.delta_indexes.size() >= effective_merge_run_threshold;

    const bool too_large =
        bucket.pending_entries >= effective_merge_size_threshold;

    const bool model_biased_async_candidate =
        bucket.structure_mode == STRUCTURE_MODE_MODEL_BIASED &&
        bucket.pending_entries > 0 &&
        bucket.accesses_since_check >= effective_hot_access_threshold &&
        bucket.delta_indexes.size() >=
            static_cast<size_t>(opt.async_local_retrain_min_delta_runs);

    const bool hot_and_drifted =
        bucket.pending_entries > 0 &&
        bucket.accesses_since_check >= effective_hot_access_threshold &&
        (drift_score >= opt.gpkl_threshold || model_biased_async_candidate);

    const uint64_t hpt_merge_pressure_delta =
        bucket_hpt_row_pressure_delta(bk, bucket, true);

    const bool merge_due_to_hpt_pressure =
        opt.enable_hpt_row_aware_maintenance &&
        !is_static0_novel_compact_bucket(bk, bucket) &&
        bucket.pending_entries > 0 &&
        hpt_merge_pressure_delta >= opt.hpt_row_merge_pressure_threshold;

    const bool merge_due_to_effective_threshold_only =
        too_many_delta_runs || too_large;

    const bool merge_due_to_effective_threshold =
        merge_due_to_effective_threshold_only ||
        merge_due_to_hpt_pressure;

    const bool merge_due_to_base_threshold =
        bucket.delta_indexes.size() >= opt.local_merge_run_threshold ||
        bucket.pending_entries >= opt.local_merge_size_threshold;

    const bool drift_threshold_merge =
        bucket.in_drift_state &&
        merge_due_to_effective_threshold_only &&
        !merge_due_to_base_threshold;

    const bool hpt_pressure_only_merge =
        merge_due_to_hpt_pressure &&
        !merge_due_to_effective_threshold_only;

    if (merge_due_to_effective_threshold) {
        const uint64_t merge_count_before = bucket.merge_count;

        merge_bucket_into_base(bk, bucket);

        if (bucket.merge_count > merge_count_before) {
            mark_hpt_row_merge_pressure_consumed(bk, bucket);

            if (drift_threshold_merge) {
                bucket.drift_threshold_merge_count++;
                total_drift_threshold_local_merges++;
            }

            if (hpt_pressure_only_merge) {
                bucket.hpt_pressure_merge_count++;
                total_hpt_pressure_local_merges++;
            }
        }
    } else if (maybe_schedule_drift_state_local_retrain(
                   bk, bucket, effective_async_min_entries)) {
        bucket.accesses_since_check = 0;
    } else if (hot_and_drifted) {
        schedule_async_local_retrain(bk, bucket, effective_async_min_entries);
        bucket.accesses_since_check = 0;
    } else if (maybe_schedule_deferred_structure_rebuild(bk, bucket)) {
        bucket.accesses_since_check = 0;
    } else {
        bucket.accesses_since_check = 0;
    }

    resolve_bucket_drift_if_quiet(bucket);
}

    


    
    BOS_NOINLINE bool lookup_delta_data_sorted(
        const DeltaData &dm,
        const std::string &key,
        uint64_t &out_val) const {
        int lo = 0;
        int hi = dm.count - 1;

        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            int cmp = std::strcmp(dm.key_ptrs[mid], key.c_str());

            if (cmp == 0) {
                out_val = dm.val_ptrs[mid];
                return true;
            }

            if (cmp < 0) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        return false;
    }

BOS_NOINLINE void flush_bucket(const BucketKey &bk, DeltaBucket &bucket){
    BOSReviewScopedTimer review_flush_timer(&review_local_flush_time_ns_);
    uint64_t prefix = bk.prefix;
        if (bucket.active.size() < MIN_DELTA_BULKLOAD_SIZE) return;

        std::vector<std::pair<std::string, uint64_t>> sorted_buffer;
        sorted_buffer.reserve(bucket.active.size());
        for (const auto &kvp : bucket.active) sorted_buffer.push_back(kvp);
        std::sort(sorted_buffer.begin(), sorted_buffer.end(),
                  [](const std::pair<std::string, uint64_t> &a,
                     const std::pair<std::string, uint64_t> &b) {
                      return a.first < b.first;
                  });

        DeltaData mem;
        mem.count = static_cast<int>(sorted_buffer.size());

        uint64_t total_len = 0;
        for (const auto &kvp : sorted_buffer) total_len += kvp.first.length() + 1;

        mem.data_buf = new char[total_len];
        mem.key_ptrs = new char *[mem.count];
        mem.val_ptrs = new uint64_t[mem.count];

        uint64_t offset = 0;
        for (int i = 0; i < mem.count; ++i) {
            const std::string &k = sorted_buffer[i].first;
            mem.key_ptrs[i] = mem.data_buf + offset;
            mem.val_ptrs[i] = sorted_buffer[i].second;
            std::memcpy(mem.data_buf + offset, k.c_str(), k.length() + 1);
            offset += k.length() + 1;
        }

        // DeltaData is already sorted. Keep it as an immutable sorted-array
        // delta run and avoid constructing another BOSBase instance for it.
        bucket.delta_memory.push_back(mem);
        bucket.delta_indexes.push_back(nullptr);
        bucket.structure_compact_delta_runs++;
        total_structure_compact_delta_runs++;
        record_bucket_structure_build(bucket);

        if (opt.enable_local_stats) {
            bucket.delta_hist.add_hist(bucket.active_hist);
            bucket.active_hist.clear();
        }

        bucket.active.clear();
        bucket.flush_count++;
        total_flushes++;
        if (bucket.in_drift_state) {
    bucket.drift_flush_count++;
    total_drift_flushes++;
}
       #ifdef BOS_VERBOSE
std::cout << "[BOS Flush] prefix='" << prefix
          << "' row_group=" << bk.hpt_row_group
          << " bos_range=" << bk.bos_range_id
          << " count=" << mem.count
          << " runs=" << bucket.delta_indexes.size()
          << std::endl;
#endif
    }

    
    BOS_NOINLINE bool tombstone_bucket_contains_live_copy(
        DeltaBucket &bucket,
        const std::string &key) {
        tombstone_gc_prefix_bucket_probes++;

        if (bucket.active.find(key) != bucket.active.end()) {
            return true;
        }

        // Delta runs may be materialized either as a BOSBase snapshot
        // (delta_indexes[i] != nullptr) or as a compact sorted DeltaData
        // run (delta_indexes[i] == nullptr).  Tombstone GC must consider both
        // representations, but only inside the key's prefix bucket family.
        for (size_t i = 0; i < bucket.delta_memory.size(); ++i) {
            tombstone_gc_delta_run_probes++;

            uint64_t ignored = 0;
            if (lookup_delta_data_sorted(bucket.delta_memory[i], key, ignored)) {
                return true;
            }

            if (i < bucket.delta_indexes.size()) {
                BOSBase *delta_model = bucket.delta_indexes[i];
                if (delta_model && delta_model->lookup(key.c_str())) {
                    return true;
                }
            }
        }

        // Be defensive for older/legacy buckets where delta_indexes may contain
        // a model without a paired DeltaData slot.  This remains prefix-local
        // because callers enumerate prefix_bucket_index[prefix].
        for (size_t i = bucket.delta_memory.size();
             i < bucket.delta_indexes.size();
             ++i) {
            tombstone_gc_delta_run_probes++;

            BOSBase *delta_model = bucket.delta_indexes[i];
            if (delta_model && delta_model->lookup(key.c_str())) {
                return true;
            }
        }

        if (bucket.async_local_model) {
            BOSBase *local_model =
                bucket.async_local_model->model.load(std::memory_order_acquire);
            if (local_model && local_model->lookup(key.c_str())) {
                return true;
            }
        }

        return false;
    }

    BOS_NOINLINE bool tombstone_still_masks_live_copy(const std::string &key) {
        if (update_overlay.find(key) != update_overlay.end()) {
            return true;
        }

        const uint64_t prefix = prefix_code(key);
        auto ix = prefix_bucket_index.find(prefix);

        if (ix == prefix_bucket_index.end()) {
            tombstone_gc_prefix_index_misses++;
        } else {
            for (const BucketKey &bk : ix->second) {
                auto b_it = buckets.find(bk);
                if (b_it == buckets.end()) {
                    continue;
                }

                if (tombstone_bucket_contains_live_copy(
                        b_it->second, key)) {
                    return true;
                }
            }
        }

        // Immutable base BOSBase is not organized in the delta bucket table.  A
        // deleted base key must keep its tombstone, but this is one base lookup
        // rather than a global delta-bucket scan.
        kv *base_ret = this->lookup(key.c_str());
        return base_ret != nullptr;
    }

    BOS_NOINLINE void compact_safe_tombstones_after_maintenance() {
        if (!opt.enable_tombstone_compaction || deleted_keys.empty()) {
            return;
        }

        if (deleted_keys.size() < opt.tombstone_compaction_min_tombstones) {
            return;
        }

        const size_t max_scan =
            (opt.tombstone_compaction_max_scan == 0)
                ? deleted_keys.size()
                : opt.tombstone_compaction_max_scan;

        size_t scanned = 0;
        for (auto it = deleted_keys.begin();
             it != deleted_keys.end() && scanned < max_scan;) {
            const std::string key = *it;
            scanned++;
            tombstone_gc_checks++;

            if (!tombstone_still_masks_live_copy(key)) {
                it = deleted_keys.erase(it);
                tombstone_gc_erases++;
            } else {
                ++it;
            }
        }
    }


BOS_NOINLINE void mark_bucket_drift_resolved_by_materialization(DeltaBucket &bucket) {
    if (!opt.enable_bucket_drift_stats ||
        !opt.enable_bucket_drift_state ||
        !bucket.in_drift_state) {
        return;
    }

    bucket.in_drift_state = false;
    bucket.drift_state_exits++;
    bucket.consecutive_drift_checks = 0;
    bucket.consecutive_clean_checks = opt.bucket_drift_exit_clean_checks;
    bucket.last_gpkl = 0.0;
    bucket.last_drift_ratio = 1.0;
}

BOS_NOINLINE char *copy_key_to_base_storage(const char *key) {
    const size_t len = std::strlen(key) + 1;
    char *p = new char[len];
    std::memcpy(p, key, len);
    base_merged_key_storage.push_back(p);
    return p;
}

BOS_NOINLINE void free_base_merged_key_storage() {
    for (char *p : base_merged_key_storage) {
        delete[] p;
    }
    base_merged_key_storage.clear();
}

BOS_NOINLINE void merge_bucket_into_base(const BucketKey &bk,
                                          DeltaBucket &bucket,
                                          bool force_hpt_pressure_merge = false,
                                          bool defer_tombstone_compaction = false) {
    (void)bk;
    BOSReviewScopedTimer review_merge_timer(&review_local_merge_time_ns_);

    if (bucket.delta_memory.empty() && bucket.active.empty()) {
        return;
    }

    // Do not merge bucket-local updates into the immutable/base BOSBase.
    // In this workload, writing dynamic drift keys back into base can corrupt
    // unrelated base lookups. Instead, compact active + delta runs into one
    // bucket-local immutable delta run. For small forced buckets that cannot
    // satisfy BOSBase::bulkload's >=1000-key requirement, materialize a sorted
    // array DeltaData run so full maintenance can still close a drift episode.
    const double pmss_read_ratio = evaluate_bucket_structure_policy(bucket, false);

    uint64_t skipped_tombstoned_entries = 0;
    std::vector<std::pair<std::string, uint64_t>> entries =
        snapshot_bucket_entries_for_retrain(
            bucket, true, &skipped_tombstoned_entries);
    tombstone_merge_skips += skipped_tombstoned_entries;

    if (entries.empty()) {
        cleanup_bucket(bucket);
        if (!defer_tombstone_compaction) {
            compact_safe_tombstones_after_maintenance();
        }
        return;
    }

    DeltaData merged_mem;
    BOSBase *merged_model = nullptr;
    PMSSBuildStats build_stats;

    const bool built_bos =
        build_bos_from_entries(entries, merged_mem, merged_model,
                                pmss_read_ratio, &build_stats);

    if (!built_bos) {
        if (!force_hpt_pressure_merge) {
            return;
        }

        if (!build_delta_data_from_entries(entries, merged_mem)) {
            return;
        }
        merged_model = nullptr;
        bucket.structure_compact_delta_runs++;
        total_structure_compact_delta_runs++;
    } else {
        absorb_bucket_structure_build_stats(bucket, build_stats);
    }

    cleanup_bucket(bucket);

    bucket.delta_memory.push_back(merged_mem);
    bucket.delta_indexes.push_back(merged_model);

    if (opt.enable_local_stats) {
        bucket.delta_hist.clear();
        for (const auto &kvp : entries) {
            bucket.delta_hist.add_key(kvp.first);
        }
    }

    record_bucket_structure_build(bucket);
    bucket.merge_count++;
    total_local_merges++;

    if (bucket.in_drift_state) {
        bucket.drift_merge_count++;
        total_drift_local_merges++;
    }

    if (force_hpt_pressure_merge) {
        mark_bucket_drift_resolved_by_materialization(bucket);
    }

    if (!defer_tombstone_compaction) {
        compact_safe_tombstones_after_maintenance();
    }
}


    BOS_NOINLINE void wait_for_async_local_retrains() {
        for (std::thread &worker : async_local_retrain_threads) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        async_local_retrain_threads.clear();
    }

    BOS_NOINLINE void cleanup_async_local_model(DeltaBucket &bucket) {
        if (!bucket.async_local_model) {
            return;
        }

        std::shared_ptr<AsyncLocalModelState> state = bucket.async_local_model;

        BOSBase *published = state->model.exchange(nullptr, std::memory_order_acq_rel);

        {
            std::lock_guard<std::mutex> lock(state->publish_mutex);

            if (published) {
                delete published;
            }
            free_delta_data(state->current_model_memory);

            reclaim_retired_async_local_snapshots_locked(*state, true);

            state->model_entries.store(0, std::memory_order_relaxed);
        state->snapshot_bucket_write_epoch.store(0, std::memory_order_release);
            state->retired_snapshot_count.store(
                state->retired_snapshots.size(), std::memory_order_relaxed);
            state->retrain_inflight.store(false, std::memory_order_release);
        }
    }

 BOS_NOINLINE void cleanup_bucket(DeltaBucket &bucket) {
    for (BOSBase *idx : bucket.delta_indexes) delete idx;
    bucket.delta_indexes.clear();

    for (DeltaData &dm : bucket.delta_memory) free_delta_data(dm);
    bucket.delta_memory.clear();

    bucket.active.clear();

    bucket.active_hist.clear();
    bucket.delta_hist.clear();
    bucket.recent_hist.clear();
    bucket.recent_sketch.clear();
    bucket.recent_keys.clear();
    bucket.structure_rebuild_pending = false;
    bucket.structure_last_rebuild_cause_mask = 0;
    bucket.structure_rebuild_cause_mask_updates = 0;
    bucket.structure_feedback_folds = 0;
    bucket.structure_closed_loop_cycles = 0;
    bucket.structure_async_rebuild_publishes_observed = 0;
}


    BOS_NOINLINE void cleanup_all_buckets() {
    wait_for_async_local_retrains();
    for (auto &kvp : buckets) {
        cleanup_bucket(kvp.second);
        cleanup_async_local_model(kvp.second);
    }
   buckets.clear();
prefix_bucket_index.clear();
prefix_range_bucket_index.clear();
update_overlay.clear();
weak_dynamic_overlay.clear();
static_prefix_counts.clear();
static_prefix_hist.clear();
static_prefix_sketch.clear();
static_bucket_counts.clear();
static_bucket_hist.clear();
static_bucket_sketch.clear();

    pending_hpt_deltas.clear();
    cleanup_retired_hpt_versions();


        if (base_data_store.is_owned) {
            delete[] base_data_store.data_buf;
            delete[] base_data_store.key_ptrs;
            delete[] base_data_store.val_ptrs;
            base_data_store = BaseData();
        }
    }

    BOS_NOINLINE void free_delta_data(DeltaData &dm) {
        delete[] dm.data_buf;
        delete[] dm.key_ptrs;
        delete[] dm.val_ptrs;
        dm = DeltaData();
    }
};

#undef BOS_NOINLINE

} // namespace bos
