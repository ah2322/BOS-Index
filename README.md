# BOS-Index

An updatable learned index for dynamic string keys.

## Build

    make

## Test

    make check

Expected: BOS correctness test passed
 
## Configuration

BOS_FORCE_WEAK_PREFIX_LOW_OVERHEAD=1 enables the forced weak-prefix low-overhead mode.


## Source layout

- `bos/bos_index.hpp`: main BOS-Index API.
- `bos/bos_static_index.hpp`: BOSBase index implementation.
- `bos/bos_string_model.hpp`: HPT string prediction model.
- `bos/bos_structure_selector.hpp`:  node-type selection.
- `bos/bos_index_node.hpp`: index-node construction and search.
- `bos/bos_hot_adapter.hpp` and `bos/hot_src/`: HOT adapter and bundled third-party implementation.
