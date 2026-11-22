# Exclusive Prefix Scan

Blelloch work-efficient scan, 3-phase multi-block, exclusive (out[0]=0).

## Phases
Phase A: each block Blelloch-scans 512 elems, writes block sum.
Phase B: 1 block scans blockSums (exclusive).
Phase C: each block adds scanned block sum to local results.

## Blelloch
Up-sweep: tree reduce. Down-sweep: set last=0, distribute prefix.
Shared mem for both sweeps, __syncthreads() each step.

## Complexity
Work: O(n) - each element touched O(1) times per phase, vs O(n log n) for naive Hillis-Steele.
Depth / Span: O(log n) - log steps per block + log steps for blockSums.
Global round-trips: 3 - read x (A), read/write blockSums (B), read y + write y (C). No per-step global sync.
Note: Phase B is single-block, covers up to 512 blocks (N up to 262k). For 10M, recurse Phase B (scan blockSums in multiple blocks) or launch Phase B twice.

## Build / Run
nvcc -O3 -arch=sm_80 scan.cu -o scan
./scan 10000000
