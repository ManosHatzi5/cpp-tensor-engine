# Cache Locality, Loop Interchange, and Tiling Rationale

## 1. Hardware Fundamentals: The Memory Wall & Cache Hierarchy
Modern x86-64 CPUs operate at clock frequencies exceeding $3.5\text{ GHz}$, yielding sub-nanosecond instruction cycle latencies. Conversely, fetching operands from main DRAM incurs a high latency penalty of roughly $50\text{--}100\text{ ns}$ (~$200\text{--}300$ clock cycles).

To bridge this "Memory Wall", CPU architectures implement hierarchical caches:
* **L1 Data Cache (L1d):** $\approx 32\text{--}48\text{ KB}$ per core, $\approx 1\text{ ns}$ latency ($4\text{--}5$ cycles).
* **L2 Cache:** $\approx 512\text{ KB -- } 1.25\text{ MB}$ per core, $\approx 3\text{--}4\text{ ns}$ latency ($12\text{--}14$ cycles).
* **L3 Cache (LLC):** $\approx 16\text{--}64\text{ MB}$ shared, $\approx 10\text{--}20\text{ ns}$ latency ($40\text{--}60$ cycles).

### Cache Lines and Spatial Locality
Main memory transfers data to the cache in quantized $64\text{-byte}$ blocks designated as **Cache Lines**. A single cache line hosts:
$$\frac{64\text{ bytes}}{4\text{ bytes per IEEE 754 float}} = 16\text{ single-precision floats}$$

When accessing address $p$, the CPU fetches the surrounding $64\text{-byte}$ aligned boundary.
* **Spatial Locality (Unit Stride):** Accessing sequential elements $p[0], p[1], \dots, p[15]$ incurs one cache miss followed by 15 consecutive $L1$ cache hits.
* **Strided Access (Cache Thrashing):** Traversing a column of a row-major matrix of dimension $N = 1024$ steps by $N \times 4\text{ bytes} = 4096\text{ bytes}$ per iteration. Each step fetches a new $64\text{-byte}$ cache line, reads a single $4\text{-byte}$ float, and discards the remaining 15 floats, reducing memory bus utilization to $6.25\%$.

---

## 2. Loop Permutation: $i\text{-}j\text{-}k$ vs. $i\text{-}k\text{-}j$

Matrix multiplication evaluates the bilinear form:
$$C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} B_{k, j}$$

Our contiguous row-major memory mapping function is defined as:
$$\text{Offset}(r, c, \text{cols}) = r \cdot \text{cols} + c$$

### The Naive $i \to j \to k$ Loop (Dot Product Mode)
```cpp
for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
        for (size_t k = 0; k < K; ++k) {
            C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
    }
}