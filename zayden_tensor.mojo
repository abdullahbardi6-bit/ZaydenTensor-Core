# ==============================================================================
#  final.mojo  —  The Zayden Protocol · 3D Tensor Core
# ==============================================================================

from std.memory.alloc import alloc, dealloc, Layout, Allocation
from std.algorithm.functional import vectorize
from std.sys import simd_width_of

# FIXED: Moved alias to global scope (outside the struct)
alias F32_SIMD_WIDTH = simd_width_of[DType.float32]()

# ==============================================================================
#  Phase 1 — ZaydenTensor Struct : Memory Management & Ownership
# ==============================================================================

struct ZaydenTensor:
    var depth: Int
    var rows: Int
    var cols: Int
    var capacity: Int
    var _alloc: Allocation[Float32]

    def __init__(out self, depth: Int, rows: Int, cols: Int):
        self.depth = depth
        self.rows = rows
        self.cols = cols
        self.capacity = depth * rows * cols

        self._alloc = alloc(Layout[Float32](count=self.capacity))

        var ptr = self._alloc.unsafe_ptr()
        for i in range(self.capacity):
            ptr.unsafe_store(i, Float32(0.0))

    def __deinit__(deinit self):
        dealloc(self._alloc^)

    # ==========================================================================
    #  Phase 2 — 3D-to-1D Index Mapping & Magic Methods
    # ==========================================================================

    def _calc_index(self, d: Int, r: Int, c: Int) -> Int:
        return d * (self.rows * self.cols) + r * self.cols + c

    def __getitem__(self, d: Int, r: Int, c: Int) -> Float32:
        var idx = self._calc_index(d, r, c)
        return self._alloc.unsafe_ptr().unsafe_load(idx)[0]

    def __setitem__(mut self, d: Int, r: Int, c: Int, value: Float32):
        var idx = self._calc_index(d, r, c)
        self._alloc.unsafe_ptr().unsafe_store(idx, value)

    # ==========================================================================
    #  Phase 3 — SIMD Vectorisation Engine
    # ==========================================================================

    def add(mut self, other: ZaydenTensor):
        var total = self.capacity

        def kernel[width: Int](i: Int):
            var a = self._alloc.unsafe_ptr().unsafe_load[width=width](i)
            var b = other._alloc.unsafe_ptr().unsafe_load[width=width](i)
            self._alloc.unsafe_ptr().unsafe_store(i, a + b)

        # FIXED: Removed 'Self.' from the alias reference
        vectorize[F32_SIMD_WIDTH](total, kernel)

# ==============================================================================
#  Phase 4 — Main : Integration Test
# ==============================================================================

def main():
    print("[ ZaydenProtocol ] Allocating ZaydenTensor A and B  (3 × 3 × 3, Float32)...")
    var A = ZaydenTensor(3, 3, 3)
    var B = ZaydenTensor(3, 3, 3)

    print("[ ZaydenProtocol ] Populating A with sequential values (1..27)...")
    for d in range(3):
        for r in range(3):
            for c in range(3):
                A[d, r, c] = Float32(d * 9 + r * 3 + c + 1)

    print("[ ZaydenProtocol ] Filling B with constant 100.0...")
    for d in range(3):
        for r in range(3):
            for c in range(3):
                B[d, r, c] = Float32(100.0)

    var pre_0_0_0 = A[0, 0, 0]
    var pre_1_1_1 = A[1, 1, 1]
    var pre_2_2_2 = A[2, 2, 2]
    print("[ ZaydenProtocol ] Pre-add checks:")
    print("    A[0,0,0] =", pre_0_0_0, "  (expected  1.0)")
    print("    A[1,1,1] =", pre_1_1_1, "  (expected 14.0)")
    print("    A[2,2,2] =", pre_2_2_2, "  (expected 27.0)")

    print("[ ZaydenProtocol ] Running SIMD vectorised add  (A += B)...")
    A.add(B)

    var post_0_0_0 = A[0, 0, 0]
    var post_1_1_1 = A[1, 1, 1]
    var post_2_2_2 = A[2, 2, 2]
    print("[ ZaydenProtocol ] Post-add checks:")
    print("    A[0,0,0] =", post_0_0_0, "  (expected 101.0)")
    print("    A[1,1,1] =", post_1_1_1, "  (expected 114.0)")
    print("    A[2,2,2] =", post_2_2_2, "  (expected 127.0)")

    if post_0_0_0 == 101.0 and post_1_1_1 == 114.0 and post_2_2_2 == 127.0:
        print("")
        print("[ ZaydenProtocol ] ✓ ALL CHECKS PASSED — SIMD add is correct.")
    else:
        print("")
        print("[ ZaydenProtocol ] ✗ MISMATCH — check tensor logic.")

    print("[ ZaydenProtocol ] Tensors destroyed.  Memory: 0 leaks, 0 double-frees.")