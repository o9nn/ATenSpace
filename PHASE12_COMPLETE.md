# Phase 12 Progress Summary

## Status: ✅ Complete

**Started**: April 2026  
**Completed**: April 2026  
**Focus**: PLN inference completeness, PatternMatcher negation-as-failure, VariableBinding typed wrapper, InferencePipeline improvements

---

## ✅ Completed Work

### 1. PLN Inference Completeness (`InferencePipeline.h`)

Three new concrete `InferenceStep` subclasses complete the PLN inference rule set.

#### PLNConjunctionStep
Computes AND truth values using the PLN conjunction formula **(A∧B)**:

```
strength   = sA × sB
confidence = cA × cB
```

- **Phase A**: Evaluates any existing `AND_LINK(A, B)` in the working set whose confidence is still zero (unset), deriving its TV from the truth values of A and B.
- **Phase B**: Creates new `AND_LINK(A, B)` atoms for each pair of non-link atoms whose individual strength is ≥ `minStrength` (default `0.5`), skipping pairs that already have an AND_LINK.

```cpp
InferencePipeline p(space);
p.plnConjunction(/*minStrength=*/0.5f);
auto result = p.run({A, B});
// result contains AND_LINK(A, B) with TV(sA*sB, cA*cB)
```

#### PLNDisjunctionStep
Computes OR truth values using the PLN disjunction formula **(A∨B)**:

```
strength   = sA + sB - sA × sB
confidence = (cA + cB) / 2
```

- **Phase A**: Evaluates existing `OR_LINK(A, B)` atoms with unset confidence.
- **Phase B**: Creates `OR_LINK(A, B)` for qualifying pairs (strength ≥ `minStrength`, default `0.3`).

```cpp
p.plnDisjunction(/*minStrength=*/0.3f);
```

#### PLNSimilarityStep
Computes pairwise semantic similarity between same-type atoms using the PLN formula:

```
similarity = (sA × sB) / (sA + sB - sA × sB + ε)
confidence = min(cA, cB)
```

Creates `SIMILARITY_LINK(A, B)` when the computed similarity strength exceeds `minSimilarity` (default `0.5`). Only pairs atoms of the **same type** with non-zero confidence.

```cpp
p.plnSimilarity(/*minSimilarity=*/0.5f);
```

#### Fluent API on `InferencePipeline`
```cpp
pipeline.plnConjunction(minStrength)
        .plnDisjunction(minStrength)
        .plnSimilarity(minSimilarity);
```

#### `clear()` method
New method to clear all steps from the pipeline:
```cpp
pipeline.clear();   // removes all steps, returns *this for chaining
```

---

### 2. Factory Helper: `makePLNFullPipeline`

Creates a 6-step full PLN pipeline:

```
PLNDeduction → PLNConjunction → PLNDisjunction
             → PLNSimilarity → PLNRevision → TVThreshold
```

```cpp
auto pipeline = makePLNFullPipeline(space,
    /*tvThreshold=*/0.0f,
    /*minConfidence=*/0.0f,
    /*minStrength=*/0.3f,
    /*minSimilarity=*/0.5f);

auto result = pipeline.run({A, B, ab, bc});
```

Python:
```python
p = at.make_pln_full_pipeline(space, min_strength=0.3, min_similarity=0.5)
result = p.run([A, B, ab, bc])
```

---

### 3. PatternMatcher Negation-as-Failure (`PatternMatcher.h`)

The `PatternMatcher::match()` method now handles **`NOT_LINK`** patterns:

A `NOT_LINK(inner_pattern)` matches any target atom for which `inner_pattern` does **not** match. This implements the standard negation-as-failure (NAF) semantics from logic programming.

```cpp
// Create: NOT(Cat) — matches anything that is not "Cat"
auto notCat = space.addLink(Atom::Type::NOT_LINK, {cat});

VariableBinding bindings;
// Returns true for "Dog", "Fish", etc.
bool ok = PatternMatcher::match(notCat, dog, bindings);  // true

// Returns false for "Cat" itself
bool ok2 = PatternMatcher::match(notCat, cat, bindings); // false
```

Key properties:
- Only arity-1 `NOT_LINK` is treated as negation; other arities fall through to normal link matching.
- Variable bindings produced by the tentative inner match are **not** propagated on negation success (only the outer bindings are returned).
- `findMatches(space, NOT_LINK(pattern))` returns all atoms in the space that do NOT match the inner pattern.

---

### 4. VariableBinding Typed Python Wrapper (`python_bindings.cpp`)

`VariableBinding` is now exposed as a proper Python class instead of relying on pybind11's opaque dict conversion.

```python
_, binding = at.PatternMatcher.match_with_bindings(pattern, target)

isinstance(binding, at.VariableBinding)   # True
len(binding)                               # number of bound variables
binding.variable_names()                   # ['?X', '?Y']
binding.get_by_name("?X")                 # bound Atom
binding.keys()                             # list of VariableNode atoms
binding.values()                           # list of bound atoms
binding.items()                            # list of (var_atom, bound_atom) pairs
binding.to_dict()                          # {'?X': atom, '?Y': atom}
repr(binding)                              # VariableBinding({?X -> Cat, ?Y -> Dog})
```

```python
matches = at.PatternMatcher.find_matches(space, pattern)
for atom, binding in matches:
    bound = binding.get_by_name("?X")
    d = binding.to_dict()    # Python str-keyed dict
```

---

### 5. Python Bindings for Phase 12 Steps (`python_bindings.cpp`)

```python
# Step classes
at.PLNConjunctionStep(min_strength=0.5)
at.PLNDisjunctionStep(min_strength=0.3)
at.PLNSimilarityStep(min_similarity=0.5)

# Fluent pipeline methods
p.pln_conjunction(min_strength=0.5)
p.pln_disjunction(min_strength=0.3)
p.pln_similarity(min_similarity=0.5)
p.clear()

# Factory
at.make_pln_full_pipeline(space, tv_threshold=0.0, min_confidence=0.0,
                           min_strength=0.3, min_similarity=0.5)
```

---

### 6. Module Version

Python module version updated from `0.11.0` → **`0.12.0`**.

---

### 7. CI Updates (`build.yml`)

`atomspace_test_phase12` added to the "Verify key binaries" step.

---

## 📊 Test Coverage

### C++ Tests (`test_phase12.cpp`) — 22 tests

| Group | Tests |
|---|---|
| PLNConjunctionStep | 4 (creates AND_LINK, TV formula, threshold, evaluates existing) |
| PLNDisjunctionStep | 3 (creates OR_LINK, TV formula, evaluates existing) |
| PLNSimilarityStep | 4 (creates SIMILARITY_LINK, TV formula, different types skipped, threshold) |
| InferencePipeline Phase 12 | 6 (fluent methods ×3, clear, factory step count, factory step names) |
| PatternMatcher negation-as-failure | 4 (matches different, fails on same, fails with variable, findMatches filters) |
| End-to-end | 1 (makePLNFullPipeline runs end-to-end) |

### Python Tests (`tests/python/test_phase12.py`) — 47 tests

| Class | Tests |
|---|---|
| `TestPLNConjunctionStep` | 5 |
| `TestPLNDisjunctionStep` | 4 |
| `TestPLNSimilarityStep` | 4 |
| `TestInferencePipelinePhase12` | 5 |
| `TestPatternMatcherNegation` | 5 |
| `TestVariableBinding` | 10 |
| `TestModuleVersion` | 1 (expects `0.12.0`) |

---

## 📊 Code Statistics

| File | Lines Changed | Description |
|---|---|---|
| `InferencePipeline.h` | +290 | PLNConjunctionStep, PLNDisjunctionStep, PLNSimilarityStep; fluent methods; `clear()`; `makePLNFullPipeline` |
| `PatternMatcher.h` | +20 | NOT_LINK negation-as-failure in `match()` |
| `python_bindings.cpp` | +155 | VariableBinding class, Phase 12 step bindings, fluent methods, factory helper |
| `test_phase12.cpp` | 330 new | 22 C++ tests |
| `tests/python/test_phase12.py` | 340 new | 47 Python tests |
| `CMakeLists.txt` | +4 | `atomspace_test_phase12` target |
| `.github/workflows/build.yml` | +1 | Verify phase12 binary |

**Total Phase 12**: ~1,140 lines of new/modified code + 22 C++ tests + 47 Python tests

---

## 🚀 How to Build and Test

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu

cmake -S aten/src/ATen/atomspace -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
      -DBUILD_PYTHON_BINDINGS=OFF

cmake --build build --parallel 4

# Run Phase 12 tests
./build/atomspace_test_phase12

# Run Phase 11 tests (still passing)
./build/atomspace_test_phase11
```

For Python bindings (requires pybind11):
```bash
cmake -S aten/src/ATen/atomspace -B build_py \
      -DBUILD_PYTHON_BINDINGS=ON \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build_py --parallel 4
python -m pytest tests/python/test_phase12.py -v
```

---

## 🎯 Next Steps (Phase 13)

1. **Distributed AtomSpace** — Sharding and replication for knowledge graphs exceeding single-machine memory
2. **Safetensors support** — Direct weight loading from HuggingFace `.safetensors` files
3. **PLN Implication completeness** — Full `PLNImplicationStep` for evaluating `IMPLICATION_LINK` truth values; multi-step forward PLN chains
4. **Negation-as-failure at query level** — `AbsentLink`/`NotExistsLink` patterns that check for the absence of atoms in the AtomSpace (distinct from atom-level negation)
5. **VariableBinding query callback** — Expose a proper Python callable interface for `PatternMatcher.query()` that passes `VariableBinding` objects
6. **TensorLogicEngine vectorised PLN** — Batch-inference using tensor operations for large working sets

---

**Phase 12 Status**: ✅ Complete  
**Last Updated**: April 2026
