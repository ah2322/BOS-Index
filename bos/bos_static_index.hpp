#pragma once

#include "bos_compact_node.hpp"
#include "bos_hot_adapter.hpp"
#include "bos_iterator.hpp"
#include "bos_key_value.hpp"
#include "bos_string_model.hpp"
#include "bos_index_node.hpp"

#include <cmath>
#include <stack>
#include <chrono>

namespace bos {

class BOSBase {
  private:
    // For bulk load, the index needs at least 1000 strings to train the model
    static const int min_bulk_load_size = 1000;

    // Whether the index has been bulk loaded
    bool hasBeenBuild = false;

    // The Global String Model: Hash-enhanced Prefix Table
    HPT *hpt = nullptr;

    // The Structural Decision Tree
    PMSS *pmss = nullptr;

    // The root node of the index.
    Item root;

    // Review metrics: separate model-training time from the total bulk-load time.
    double last_hpt_train_ms = 0.0;
    double last_pmss_build_ms = 0.0;

    // Review metrics: separate model-training time from the total bulk-load time.

  public:
    BOSBase() = default;
    ~BOSBase() = default;

    bool bulkload(const char **_keys, const uint64_t *_vals, const int _len,
                  HPT *_hpt = NULL) {
        RT_ASSERT(hasBeenBuild == false);
        return _bulkload((const str *)_keys, _vals, _len, _hpt, 1.0, 0.0);
    }

    // BOSIndex dynamic local structure adjustment: build a BOSBase instance with a
    // per-bucket PMSS read/write bias. Existing callers use bulkload(); BOSIndex
    // calls this only for newly materialized local delta/async models, so old
    // structures stay untouched and adjustment is progressive.
    bool bulkload_with_pmss(const char **_keys, const uint64_t *_vals,
                            const int _len, HPT *_hpt,
                            double pmss_read_ratio,
                            double pmss_write_ratio) {
        RT_ASSERT(hasBeenBuild == false);
        return _bulkload((const str *)_keys, _vals, _len, _hpt,
                         pmss_read_ratio, pmss_write_ratio);
    }

    void destroy() {
        RT_ASSERT(hasBeenBuild);
        return _destroy();
    }

    kv *lookup(const char *_key) {
        RT_ASSERT(hasBeenBuild);
        return _lookup((const str)_key);
    }

    bool insert(const char *_key, const uint64_t _val) {
        RT_ASSERT(hasBeenBuild);
        return _insert((const str)_key, (const val)_val);
    }

    /**
     * If update, return the kv_entry's old value
     * If insert, return 0
     */
    val upsert(const char *_key, const uint64_t _val) {
        RT_ASSERT(hasBeenBuild);
        return _upsert((const str)_key, (const val)_val);
    }

    bool remove(const char *_key) {
        RT_ASSERT(hasBeenBuild);
        return _remove((const str)_key);
    }

    bosIter find(const char *_key) const {
        RT_ASSERT(hasBeenBuild);
        return _find((const str)_key);
    }

    bosIter begin() const {
        RT_ASSERT(hasBeenBuild);
        return _begin();
    }

    
    size_t model_size_bytes() {
        return hpt ? hpt->model_size() : 0;
    }

    uint64_t hpt_parameter_count() {
        return hpt ? hpt->model_parameter_count() : 0;
    }

    uint64_t local_linear_parameter_count() {
        PMSSBuildStats s = get_pmss_build_stats();
        return static_cast<uint64_t>(s.model_nodes) * 2ULL;  // 每个线性模型节点 k,b 两个参数
    }

    uint64_t model_parameter_count() {
        return hpt_parameter_count() + local_linear_parameter_count();
    }

PMSSBuildStats get_pmss_build_stats() const {
        return pmss ? pmss->getBuildStats() : PMSSBuildStats();
    }

    double model_train_time_ms() const {
        return last_hpt_train_ms;
    }

    double pmss_build_time_ms() const {
        return last_pmss_build_ms;
    }

    size_t model_param_bytes() const {
        size_t bytes = 0;
        if (hpt) bytes += hpt->model_size();
        if (pmss) bytes += sizeof(PMSS);
        return bytes;
    }




protected:
    HPT *mutable_hpt() {
        return hpt;
    }

    const HPT *current_hpt() const {
        return hpt;
    }

    HPT *replace_hpt(HPT *new_hpt) {
        HPT *old_hpt = hpt;
        hpt = new_hpt;
        return old_hpt;
    }

    PMSS *mutable_pmss() {
        return pmss;
    }

    Item *mutable_root() {
        return &root;
    }

    bool has_been_built() const {
        return hasBeenBuild;
    }


  private:
    bool _bulkload(const str *_keys, const uint64_t *_vals, const int _len,
                   HPT *_hpt = NULL,
                   double pmss_read_ratio = 1.0,
                   double pmss_write_ratio = 0.0) {
        // Check the input is sorted and unique
        if (_len < min_bulk_load_size) {
            std::cerr << "[Bulk Load]: For bulk load, the index needs at least "
                      << min_bulk_load_size << " strings!" << std::endl;
            return false;
        }

        for (int i = 1; i < _len; ++i) {
          
            if (ustrcmp(_keys[i], _keys[i - 1]) == 0) {
                std::cerr << "[Bulk Load]: The input strings are not unique!"
                          << std::endl;
                return false;
            }
        }

        last_hpt_train_ms = 0.0;
        last_pmss_build_ms = 0.0;

        // Train the Hash-enhanced Prefix Table
        auto hpt_train_start = std::chrono::steady_clock::now();
        if (_hpt) {
            hpt = _hpt;
        } else {
            hpt = new HPT();
            hpt->train(_keys, _len);
        }
        auto hpt_train_end = std::chrono::steady_clock::now();
        last_hpt_train_ms = std::chrono::duration<double, std::milli>(
            hpt_train_end - hpt_train_start).count();

        // Init the Performance Model for Structure Selection. BOSIndex may pass
        // per-local-region read/write bias to dynamically choose model-based
        // nodes vs HOT subtries while compact nodes remain selected by size.
        pmss = new PMSS(pmss_read_ratio, pmss_write_ratio);

        // Bulk load the root
        KVS2 kvs = {(const str *)_keys, (const val *)_vals};

        auto pmss_build_start = std::chrono::steady_clock::now();
        root = pmss_bulk(kvs, 0, _len, 0, hpt, pmss);
        auto pmss_build_end = std::chrono::steady_clock::now();
        last_pmss_build_ms = std::chrono::duration<double, std::milli>(
            pmss_build_end - pmss_build_start).count();

        hasBeenBuild = true;
        return true;
    }

    void _destroy() {
        delete hpt;
        delete pmss;

        KVS1 kvs;

        // Delete the main structure of the index
        root.recursive_extract(kvs);

        // Delete the key-value entry of the index
        kvs.self_delete();

        // --- 关键修复：重置构建状态 ---
        hasBeenBuild = false;
    }

    kv *_lookup(const str _key) {
        int ccpl = 0;
        Item item = root;

        while (1) {
            switch (item.get_itype()) {
            case ITYP_Trie: {
                return trie_search(item, _key);
            };
            case ITYP_Sing: {
                return sing_search(item, _key, ccpl);
            };
            case ITYP_CNod: {
                return cnod_search(item, _key);
            }
            case ITYP_Null: {
                return NULL;
            };
            }

            // Recursively locate the position
            item = *item.locate(_key, ccpl, hpt);
        }

        return NULL;
    }

    bool _insert(const str _key, const val _val) {
        int ccpl = 0;
        Item *item = &root;
        PathStack stack(hpt, pmss);
        bool result;

        while (1) {
            switch (item->get_itype()) {
            case ITYP_Trie: {
                result = trie_insert(*item, _key, _val);
                goto RET;
            }
            case ITYP_Sing: {
                result = sing_insert(*item, _key, _val, ccpl);
                goto RET;
            }
            case ITYP_CNod: {
                result = cnod_insert(*item, _key, _val, hpt, pmss);
                goto RET;
            }
            case ITYP_Null: {
                item->set_entry(new_kv(_key, _val));
                result = true;
                goto RET;
            }
            }

            // Record the path
            stack.record_path(item, ccpl);

            // Recursively locate the position
            item = item->locate(_key, ccpl, hpt);
        }

    RET:

        if (result == true) {
            stack.change_num(1);
        }

        return result;
    }

    bool _remove(const str _key) {
        int ccpl = 0;
        Item *item = &root;
        PathStack stack(hpt, pmss);
        bool result;

        while (1) {
            switch (item->get_itype()) {
            case ITYP_Trie: {
                result = trie_remove(*item, _key);
                goto RET;
            }
            case ITYP_Sing: {
                result = sing_remove(*item, _key, ccpl);
                goto RET;
            }
            case ITYP_CNod: {
                result = cnod_remove(*item, _key, hpt, pmss);
                goto RET;
            }
            case ITYP_Null: {
                result = false;
                goto RET;
            }
            }

            // Record the path
            stack.record_path(item, ccpl);

            // Recursively locate the position
            item = item->locate(_key, ccpl, hpt);
        }

    RET:

        if (result == true) {
            stack.change_num(-1);
        }

        return result;
    }

    val _upsert(const str _key, const val _val) {
        int ccpl = 0;
        Item *item = &root;
        PathStack stack(hpt, pmss);
        val result;

        while (1) {
            switch (item->get_itype()) {
            case ITYP_Trie: {
                result = trie_upsert(*item, _key, _val);
                goto RET;
            }
            case ITYP_Sing: {
                result = sing_upsert(*item, _key, _val, ccpl);
                goto RET;
            }
            case ITYP_CNod: {
                result = cnod_upsert(*item, _key, _val, hpt, pmss);
                goto RET;
            }
            case ITYP_Null: {
                item->set_entry(new_kv(_key, _val));
                result = 0;
                goto RET;
            }
            }

            // Record the path
            stack.record_path(item, ccpl);

            // Recursively locate the position
            item = item->locate(_key, ccpl, hpt);
        }

    RET:

        if (result == 0) {
            stack.change_num(1);
        }

        return result;
    }

    bosIter _find(const str _key) const {
        int pos, ccpl = 0;
        Item item = root;

        bosIter iter;

        while (1) {
            switch (item.get_itype()) {
            case ITYP_Trie: {
                trie_find(item, _key, iter);
                return iter;
            };
            case ITYP_Sing: {
                sing_find(item, iter);
                return iter;
            };
            case ITYP_CNod: {
                cnod_find(item, _key, iter);
                return iter;
            }
            case ITYP_Null: {
                iter.set_invalid();
                return iter;
            };
            }

            // Recursively locate the position (and record the path)
            item = *recordPath_find(item, _key, ccpl, hpt, iter);
        }

        iter.set_invalid();
        return iter;
    }

    bosIter _begin() const {
        bosIter iter;
        iter.FIRST(root);
        return iter;
    }
};
}; // namespace bos
