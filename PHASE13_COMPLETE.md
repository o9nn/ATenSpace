# Phase 13 Progress Summary

## Status: ✅ Complete

**Started**: April 2026  
**Completed**: April 2026  
**Focus**: PLN Implication completeness — `PLNImplicationStep`, `PLNImplicationChainStep`, multi-hop forward chains, `makePLNCompletePipeline`

---

## ✅ Completed Work

### 1. PLN Implication Step (`InferencePipeline.h`)

`PLNImplicationStep` evaluates and creates `IMPLICATION_LINK` atoms using the PLN material-implication formula:

```
strength   = 1 - sA + sA × sB        (P(A→B) = P(¬A) + P(A∧B))
confidence = min(cA, cB)             (limited-evidence principle)
```

**Phase A** — Evaluates existing `IMPLICATION_LINK(A, B)` atoms whose confidence is `0` (unset), computing the TV from the antecedent and consequent truth values.

**Phase B** — Creates new `IMPLICATION_LINK(A, B)` for ordered pairs of non-link atoms where:
- antecedent strength ≥ `minAntecedentStrength` (default `0.5`)
- both atoms have confidence > 0
- the computed implication strength ≥ `minImplicationStrength` (default `0.5`)
- no `IMPLICATION_LINK(A, B)` already exists (directed: A→B ≠ B→A)

```cpp
PLNImplicationStep step(/*minAntecedentStrength=*/0.5f,
                        /*minImplicationStrength=*/0.5f);
bool changed = step.execute(workingSet, space);
// → creates IMPLICATION_LINK(A, B) for qualifying pairs
```

---

### 2. PLN Implication Chain Step (`InferencePipeline.h`)

`PLNImplicationChainStep` computes the **transitive closure** of implication chains in a single step:

```
Given: A→B, B→C, C→D
Derives: A→C, A→D, B→D   (in one step, up to maxDepth hops)
```

While `PLNDeductionStep` applies the deduction rule to every *pair* of directly-connected links (single hop), `PLNImplicationChainStep` follows full chains of any depth up to `maxDepth`:

```cpp
PLNImplicationChainStep step(/*maxDepth=*/3,
                             /*minChainConfidence=*/0.0f);
```

Key properties:
- Only follows `IMPLICATION_LINK` atoms with `confidence > 0`.
- TV of derived chain link is computed via repeated PLN deduction: `strength = sAB × sBC × ...`
- Already-existing links are never duplicated (directed pair tracking).
- BFS from every source atom in the working set.

```cpp
// Three-hop chain: A→B, B→C, C→D
step.execute(workingSet, space);
// → adds A→C (tv from deduce(AB, BC))
//        A→D (tv from deduce(A→C, CD))
//        B→D (tv from deduce(BC, CD))
```

---

### 3. Fluent API on `InferencePipeline`

```cpp
pipeline.plnImplication(minAntecedentStrength, minImplicationStrength)
         .plnImplicationChain(maxDepth, minChainConfidence);
```

---

### 4. `makePLNCompletePipeline` Factory Helper

Creates an 8-step complete PLN pipeline:

```
PLNDeduction → PLNConjunction → PLNDisjunction → PLNSimilarity
             → PLNImplication → PLNImplicationChain
             → PLNRevision → TVThreshold
```

```cpp
auto pipeline = makePLNCompletePipeline(space,
    /*tvThreshold=*/0.0f,
    /*minConfidence=*/0.0f,
    /*minStrength=*/0.3f,
    /*minSimilarity=*/0.5f,
    /*minAntStrength=*/0.5f,
    /*minImpStrength=*/0.5f,
    /*chainMaxDepth=*/3);

auto result = pipeline.run({A, B, ab, bc});
```

`makePLNFullPipeline` (Phase 12, 6 steps) is **unchanged** — fully backward compatible.

Python:
```python
p = at.make_pln_complete_pipeline(
    space,
    min_antecedent_strength=0.5,
    min_implication_strength=0.5,
    chain_max_depth=3,
)
result = p.run([A, B, ab, bc])
```

---

### 5. Python Bindings (`python_bindings.cpp`)

```python
# Step classes
at.PLNImplicationStep(min_antecedent_strength=0.5, min_implication_strength=0.5)
at.PLNImplicationChainStep(max_depth=3, min_chain_confidence=0.0)

# Fluent pipeline methods
p.pln_implication(min_antecedent_strength=0.5, min_implication_strength=0.5)
p.pln_implication_chain(max_depth=3, min_chain_confidence=0.0)

# Factory
at.make_pln_complete_pipeline(space, tv_threshold=0.0, min_confidence=0.0,
                               min_strength=0.3, min_similarity=0.5,
                               min_antecedent_strength=0.5,
                               min_implication_strength=0.5,
                               chain_max_depth=3)
```

---

### 6. Module Version

Python module version updated from `0.12.0` → **`0.13.0`**.

---

### 7. Build System Fix (`CMakeLists.txt`)

Fixed a pre-existing include-ordering bug where the legacy `aten/src` directory was added as a regular `-I` path before PyTorch's system headers, causing `ATen/core/TensorMethods.h` not-found errors with modern PyTorch (2.x+). The fix promotes torch includes to come first in the system-include search order, then adds `aten/src` as a trailing system include.

---

### 8. CI Updates (`build.yml`)

`atomspace_test_phase13` added to the "Verify key binaries" step.

---

## 📊 Test Coverage

### C++ Tests (`test_phase13.cpp`) — 23 tests

| Group | Tests |
|---|---|
| PLNImplicationStep | 7 (creates link, TV formula, antecedent threshold, imp-strength threshold, evaluate existing, no duplicates, directionality) |
| PLNImplicationChainStep | 7 (two-hop, three-hop, max-depth, no duplicates, TV deduction formula, c=0 filtered, no-op) |
| InferencePipeline Phase 13 | 6 (fluent methods ×2, step count, step names, backward-compat, chaining) |
| End-to-end | 3 (complete pipeline, implication via run, chain via run) |

### Python Tests (`tests/python/test_phase13.py`) — 34 tests

| Class | Tests |
|---|---|
| `TestPLNImplicationStep` | 8 |
| `TestPLNImplicationChainStep` | 8 |
| `TestInferencePipelinePhase13` | 8 |
| `TestModuleVersion` | 1 (expects `0.13.0`) |

---

## 📊 Code Statistics

| File | Lines Changed | Description |
|---|---|---|
| `InferencePipeline.h` | +250 | PLNImplicationStep, PLNImplicationChainStep; fluent methods; `makePLNCompletePipeline` |
| `python_bindings.cpp` | +60 | Phase 13 step bindings, fluent methods, factory helper, version bump |
| `test_phase13.cpp` | 370 new | 23 C++ tests |
| `tests/python/test_phase13.py` | 380 new | 34 Python tests |
| `CMakeLists.txt` | +9 (fix +4 +4 phase13) | Include ordering fix; `atomspace_test_phase13` target |
| `.github/workflows/build.yml` | +1 | Verify phase13 binary |

**Total Phase 13**: ~1,070 lines of new/modified code + 23 C++ tests + 34 Python tests

---

## 🚀 How to Build and Test

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu

cmake -S aten/src/ATen/atomspace -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
      -DBUILD_PYTHON_BINDINGS=OFF

cmake --build build --parallel 4

# Run Phase 13 tests
./build/atomspace_test_phase13

# Run Phase 11 / 12 tests (still passing)
./build/atomspace_test_phase11
./build/atomspace_test_phase12
```

---

## 🎯 Remaining Phase 13 Items (future)

1. **VariableBinding query callback** — Expose a proper Python callable interface for `PatternMatcher.query()` that passes `VariableBinding` objects
2. **Negation-as-failure at query level** — `AbsentLink`/`NotExistsLink` patterns checking for the absence of atoms in the AtomSpace
3. **Distributed AtomSpace** — Sharding and replication for large knowledge graphs
4. **Safetensors support** — Direct weight loading from HuggingFace `.safetensors` files
5. **TensorLogicEngine vectorised PLN** — Batch-inference using tensor operations for large working sets

---

**Phase 13 Status**: ✅ Complete  
**Last Updated**: April 2026
