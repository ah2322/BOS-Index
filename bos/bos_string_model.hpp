#pragma once

#include "bos_common.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <xmmintrin.h>
#include <unordered_map>
#include <shared_mutex>
#include <atomic>
#include <deque>

namespace bos {
// ========================================
// Prefix-aware Incremental Delta Layer
// ========================================
/**
 * Hash-enhanced Prefix Table
 */
class HPT {
  public:
    // Attenuation factor, should be in (0, 1]
    static constexpr double AF = 0.5;

    // Position hash bits length
    static constexpr int PS_HASH_LEN = 5;

    // Front char hash bits length
    static constexpr int FC_HASH_LEN = 5;

    // Position hash mask and front char hash mask
    static constexpr uint32_t PS_MASK = (1 << PS_HASH_LEN) - 1;
    static constexpr uint32_t FC_MASK = (1 << FC_HASH_LEN) - 1;

    // Position array length and front char array length
    static constexpr uint32_t PS_SZ = PS_MASK + 1;
    static constexpr uint32_t FC_SZ = FC_MASK + 1;

    // Flattened HPT row count: PS_SZ * FC_SZ.
    // BOSIndex uses this for row-level dirty tracking and COW publication.
    static constexpr uint32_t ROW_COUNT = PS_SZ * FC_SZ;
    static constexpr uint32_t CHAR_COUNT = MAX_CH;

    // Units in a table line
    class UNI {
      public:
        double CDF; // Cumulative Distribution Function
        double PRO;

      public:
        UNI() : CDF(0), PRO(0) {}
        ~UNI() = default;
    };

    // Hash-enhanced Prefix Table
    UNI *m[PS_SZ][FC_SZ];

    // 新增：保存原始频率，支持 HPT 行级增量更新
    double *freq[PS_SZ][FC_SZ];

    // 训练时的全局公共前缀长度
    int train_gcpl = 0;

    // 缓存衰减权重，避免增量更新时反复 pow()
    double cached_weight[256];

  public:
    HPT() {
        // The table cannot be too large!
        // ST_ASSERT((PS_HASH_LEN + FC_HASH_LEN) <= 12);

        cached_weight[0] = 1.0;
        for (int i = 1; i < 256; ++i) {
            cached_weight[i] = cached_weight[i - 1] * AF;
        }

        for (int i = 0; i < PS_SZ; ++i) {
            for (int j = 0; j < FC_SZ; ++j) {
                m[i][j] = new UNI[MAX_CH];
                freq[i][j] = new double[MAX_CH];

                for (int c = 0; c < MAX_CH; ++c) {
                    freq[i][j][c] = 0.0;
                }
            }
        }
    }

    ~HPT() { destroy(); }

    void destroy() {
        for (int i = 0; i < PS_SZ; ++i) {
            for (int j = 0; j < FC_SZ; ++j) {
                delete[] m[i][j];
                delete[] freq[i][j];

                m[i][j] = nullptr;
                freq[i][j] = nullptr;
            }
        }
    }

    /**
     * Return the byte size of a UNI.
     */
    size_t unit_size() { return sizeof(UNI); }

    /**
     * Return the byte size of the model.
     */
    size_t model_size() const {
        return sizeof(UNI) * PS_SZ * FC_SZ * MAX_CH +
               sizeof(double) * PS_SZ * FC_SZ * MAX_CH;
    }

    uint64_t model_parameter_count() const {
        // CDF + PRO + freq
        return static_cast<uint64_t>(PS_SZ) *
               static_cast<uint64_t>(FC_SZ) *
               static_cast<uint64_t>(MAX_CH) * 3ULL;
    }

    void clear_freq_and_model() {
        for (int x = 0; x < PS_SZ; ++x) {
            for (int y = 0; y < FC_SZ; ++y) {
                for (int c = 0; c < MAX_CH; ++c) {
                    freq[x][y][c] = 0.0;
                    m[x][y][c].CDF = 0.0;
                    m[x][y][c].PRO = 0.0;
                }
            }
        }
    }

    void recompute_row(int ps, int fc) {
        double row_sum = 0.0;

        for (int c = 0; c < MAX_CH; ++c) {
            row_sum += freq[ps][fc][c];
        }

        if (row_sum <= 0.0) {
            for (int c = 0; c < MAX_CH; ++c) {
                m[ps][fc][c].CDF = 0.0;
                m[ps][fc][c].PRO = 0.0;
            }
            return;
        }

        for (int c = 0; c < MAX_CH; ++c) {
            m[ps][fc][c].CDF = freq[ps][fc][c] / row_sum;
            m[ps][fc][c].PRO = m[ps][fc][c].CDF;
        }

        double acc = m[ps][fc][0].CDF;
        m[ps][fc][0].CDF = 0.0;

        for (int c = 1; c < MAX_CH; ++c) {
            double tmp = m[ps][fc][c].CDF;
            m[ps][fc][c].CDF = acc;
            acc += tmp;
        }
    }

    void recompute_all_rows() {
        for (int x = 0; x < PS_SZ; ++x) {
            for (int y = 0; y < FC_SZ; ++y) {
                recompute_row(x, y);
            }
        }
    }

    static constexpr uint32_t row_id(uint32_t ps, uint32_t fc) {
        return ps * FC_SZ + fc;
    }

    void apply_key_delta_mark_rows(const str key, double delta, bool *touched_rows) {
        if (!key || !touched_rows) return;

        int len = ustrlen(key);

        for (int b = train_gcpl; b < len && (b - train_gcpl) < 256; ++b) {
            unsigned char curr = static_cast<unsigned char>(key[b]);

            if (curr >= MAX_CH) {
                continue;
            }

            uint32_t ps = static_cast<uint32_t>(b) & PS_MASK;
            uint32_t fc = b == 0
                              ? 0
                              : (static_cast<unsigned char>(key[b - 1]) & FC_MASK);

            double w = cached_weight[b - train_gcpl];

            freq[ps][fc][curr] += delta * w;

            if (freq[ps][fc][curr] < 0.0) {
                freq[ps][fc][curr] = 0.0;
            }

            touched_rows[row_id(ps, fc)] = true;
        }
    }

    void recompute_touched_rows(const bool *touched_rows) {
        if (!touched_rows) return;

        for (uint32_t ps = 0; ps < PS_SZ; ++ps) {
            for (uint32_t fc = 0; fc < FC_SZ; ++fc) {
                if (touched_rows[row_id(ps, fc)]) {
                    recompute_row(static_cast<int>(ps), static_cast<int>(fc));
                }
            }
        }
    }


    template <typename RowOut, typename CellOut>
    int collect_touched_cells(const str key,
                              RowOut *rows,
                              CellOut *cells,
                              int max_cells = 256) const {
        if (!key || !rows || max_cells <= 0) return 0;

        int out = 0;
        int len = ustrlen(key);

        for (int b = train_gcpl; b < len && (b - train_gcpl) < 256; ++b) {
            unsigned char curr = static_cast<unsigned char>(key[b]);

            if (curr >= MAX_CH) {
                continue;
            }

            uint32_t ps = static_cast<uint32_t>(b) & PS_MASK;
            uint32_t fc = b == 0
                              ? 0
                              : (static_cast<unsigned char>(key[b - 1]) & FC_MASK);

            uint32_t row = row_id(ps, fc);

            rows[out] = static_cast<RowOut>(row);
            if (cells) {
                cells[out] = static_cast<CellOut>(curr);
            }

            ++out;
            if (out >= max_cells) break;
        }

        return out;
    }

    template <typename RowOut, typename CellOut>
    int collect_touched_cells(const std::string &key,
                              RowOut *rows,
                              CellOut *cells,
                              int max_cells = 256) const {
        return collect_touched_cells(const_cast<char *>(key.c_str()),
                                     rows,
                                     cells,
                                     max_cells);
    }


    /**
     * Train the HPT.
     * @param keys The input keys.
     * @param len The input data size.
     *
     * @return true when success, false otherwise.
     */
    bool train(const str *keys, const int len) {
        if (len <= 0) return false;

        clear_freq_and_model();

        unsigned char dst_ch;

        // Global common prefix length
        const int gcpl = (len > 1) ? ucpl(keys[0], keys[len - 1]) : 0;
        train_gcpl = gcpl;

        // Recording the pairs
        for (int i = 0; i < len; ++i) {
            // We only consider the distinguishing prefix
            int max_len = 0;

            if (len == 1) {
                max_len = ustrlen(keys[0]);
            } else if (i == 0) {
                max_len = ucpl(keys[0], keys[1]) + 1;
            } else if (i == len - 1) {
                max_len = ucpl(keys[len - 1], keys[len - 2]) + 1;
            } else {
                max_len = std::max<int>(ucpl(keys[i], keys[i - 1]),
                                        ucpl(keys[i], keys[i + 1])) +
                          1;
            }

            int end_len = std::min<int>(ustrlen(keys[i]), max_len);

            // Record the occurrence frequency in table
            for (int b = gcpl; b < end_len && (b - gcpl) < 256; ++b) {
                dst_ch = static_cast<unsigned char>(keys[i][b]);

                if (dst_ch >= MAX_CH) {
                    continue;
                }

                int _ps = b & PS_MASK;
                int _fc = b == 0
                              ? 0
                              : (static_cast<unsigned char>(keys[i][b - 1]) &
                                 FC_MASK);

                freq[_ps][_fc][dst_ch] += cached_weight[b - gcpl];
            }
        }

        // Generate the cdf distribution from the frequency
        recompute_all_rows();

        return true;
    }

    /**
     * 新增：HPT 行级增量更新。
     *
     * 说明：
     * - 这是单线程 in-place 版本。
     * - 每插入一个 key，只更新它经过的 HPT rows。
     * - 它不做 Copy-on-Write，也不做并发版本回收。
     */
    void observe_key_incremental(const str key, double delta) {
        if (!key) return;

        bool touched[PS_SZ][FC_SZ];

        for (int x = 0; x < PS_SZ; ++x) {
            for (int y = 0; y < FC_SZ; ++y) {
                touched[x][y] = false;
            }
        }

        int len = ustrlen(key);

        for (int b = train_gcpl; b < len && (b - train_gcpl) < 256; ++b) {
            unsigned char curr = static_cast<unsigned char>(key[b]);

            if (curr >= MAX_CH) {
                continue;
            }

            int ps = b & PS_MASK;
            int fc = b == 0
                         ? 0
                         : (static_cast<unsigned char>(key[b - 1]) & FC_MASK);

            double w = cached_weight[b - train_gcpl];

            freq[ps][fc][curr] += delta * w;

            if (freq[ps][fc][curr] < 0.0) {
                freq[ps][fc][curr] = 0.0;
            }

            touched[ps][fc] = true;
        }

        for (int x = 0; x < PS_SZ; ++x) {
            for (int y = 0; y < FC_SZ; ++y) {
                if (touched[x][y]) {
                    recompute_row(x, y);
                }
            }
        }
    }

    /**
     * 新增：给后续 Copy-on-Write HPT 用。
     */
    HPT *clone() const {
        HPT *copy = new HPT();

        copy->train_gcpl = this->train_gcpl;

        for (int i = 0; i < 256; ++i) {
            copy->cached_weight[i] = this->cached_weight[i];
        }

        for (int x = 0; x < PS_SZ; ++x) {
            for (int y = 0; y < FC_SZ; ++y) {
                for (int c = 0; c < MAX_CH; ++c) {
                    copy->freq[x][y][c] = this->freq[x][y][c];
                    copy->m[x][y][c] = this->m[x][y][c];
                }
            }
        }

        return copy;
    }

    /**
     * Get the position in the BOSBase node's item array.
     *
     * @param key The input key.
     * @param size The item array's length.
     * @param gcpl The group's partial key length. Calculation will skip the
     * common prefix.
     * @param ssl Second Skip Length. Provided by one byte index.
     * @param k The local linear model's slope.
     * @param b The local linear model's intercept.
     *
     * @return a integer which stands for key's position in the node array.
     */
    inline int getPos(const str key, const int size, int gcpl, double k = 1,
                      double b = 0) const {
        double ps = size * k;
        double c = size * b;

        for (int i = gcpl; key[i] && ps >= 1; ++i) {
            int context =
                (i == 0) ? 0 : (static_cast<unsigned char>(key[i - 1]) & FC_MASK);

            unsigned char curr = static_cast<unsigned char>(key[i]);
            if (curr >= MAX_CH) continue;

            const auto &uni = m[i & PS_MASK][context][curr];
            c += ps * uni.CDF;
            ps *= uni.PRO;
        }

        return static_cast<int>(c);
    }

    inline int getPos_woGCPL(const str key, const int size, double k = 1,
                             double b = 0) const {
        double pro = size * k;
        double cdf = size * b;

        unsigned char first = static_cast<unsigned char>(key[0]);
        if (first < MAX_CH) {
            const auto &uni = m[0][0][first];
            cdf += pro * uni.CDF;
            pro *= uni.PRO;
        }

        for (int i = 1; key[i] && pro >= 1; ++i) {
            int context = static_cast<unsigned char>(key[i - 1]) & FC_MASK;

            unsigned char curr = static_cast<unsigned char>(key[i]);
            if (curr >= MAX_CH) continue;

            const auto &uni0 = m[i & PS_MASK][context][curr];
            cdf += pro * uni0.CDF;
            pro *= uni0.PRO;
        }

        return static_cast<int>(cdf);
    }

    /**
     * Return a CDF value of key which is NOT processed by the local model.
     */
    inline double getCdf(const str key, int gcpl) const {
        double pro = 1;
        double cdf = 0;
        static constexpr double min_double = 1. / (1UL << 52);

        for (int i = gcpl; key[i] && pro >= min_double; ++i) {
            int context =
                (i == 0) ? 0 : (static_cast<unsigned char>(key[i - 1]) & FC_MASK);

            unsigned char curr = static_cast<unsigned char>(key[i]);
            if (curr >= MAX_CH) continue;

            const auto &uni = m[i & PS_MASK][context][curr];

            cdf += pro * uni.CDF;
            pro *= uni.PRO;
        }

        return cdf;
    }
};

}; // namespace bos
