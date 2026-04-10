"""
test_phase12.py - Phase 12 Python tests

Covers:
  - PLNConjunctionStep
  - PLNDisjunctionStep
  - PLNSimilarityStep
  - InferencePipeline Phase 12 fluent methods
  - make_pln_full_pipeline factory helper
  - InferencePipeline.clear()
  - PatternMatcher negation-as-failure (NOT_LINK)
  - VariableBinding typed wrapper class
  - Module version 0.12.0
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    import atomspace as at
except ImportError:
    import importlib.util
    _candidates = [
        os.path.join(os.path.dirname(__file__), "../../build_py"),
        os.path.join(os.path.dirname(__file__), "../../build"),
    ]
    _loaded = False
    for _bd in _candidates:
        for _f in os.listdir(_bd) if os.path.isdir(_bd) else []:
            if _f.startswith("atomspace") and _f.endswith(".so"):
                _spec = importlib.util.spec_from_file_location(
                    "atomspace", os.path.join(_bd, _f))
                at = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(at)
                _loaded = True
                break
        if _loaded:
            break
    if not _loaded:
        raise ImportError("Cannot find atomspace module")


# ---------------------------------------------------------------------------
# PLNConjunctionStep
# ---------------------------------------------------------------------------

class TestPLNConjunctionStep(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def _make_node(self, name, strength, confidence):
        n = at.create_concept_node(self.space, name)
        n.set_truth_value(at.TruthValue.create(strength, confidence))
        return n

    def test_creates_and_link(self):
        """PLNConjunctionStep creates AND_LINK for two high-strength atoms"""
        A = self._make_node("ConjA1", 0.8, 0.9)
        B = self._make_node("ConjB1", 0.7, 0.8)
        step = at.PLNConjunctionStep(min_strength=0.5)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B])
        types = [a.get_type() for a in result.atoms]
        self.assertIn(at.AtomType.AND_LINK, types)

    def test_and_link_tv_strength(self):
        """PLNConjunctionStep AND_LINK strength = sA * sB"""
        sA, sB = 0.8, 0.6
        A = self._make_node("ConjA2", sA, 0.9)
        B = self._make_node("ConjB2", sB, 0.8)
        step = at.PLNConjunctionStep(min_strength=0.3)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B])
        for atom in result.atoms:
            if atom.get_type() == at.AtomType.AND_LINK:
                tv = atom.get_truth_value()
                s = at.TruthValue.get_strength(tv)
                self.assertAlmostEqual(s, sA * sB, places=3)

    def test_no_and_link_below_threshold(self):
        """PLNConjunctionStep does not create AND_LINK when both atoms below threshold"""
        A = self._make_node("ConjA3", 0.2, 0.9)
        B = self._make_node("ConjB3", 0.3, 0.8)
        step = at.PLNConjunctionStep(min_strength=0.5)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B])
        types = [a.get_type() for a in result.atoms]
        self.assertNotIn(at.AtomType.AND_LINK, types)

    def test_evaluates_existing_and_link(self):
        """PLNConjunctionStep fills TV of existing AND_LINK whose confidence is 0"""
        A = self._make_node("ConjA4", 0.9, 0.8)
        B = self._make_node("ConjB4", 0.6, 0.7)
        and_link = self.space.add_link(at.AtomType.AND_LINK, [A, B])
        # TV should be default (c=0) before step runs
        self.assertAlmostEqual(
            at.TruthValue.get_confidence(and_link.get_truth_value()), 0.0, places=5)
        step = at.PLNConjunctionStep(min_strength=0.5)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        p.run([A, B, and_link])
        s = at.TruthValue.get_strength(and_link.get_truth_value())
        self.assertAlmostEqual(s, 0.9 * 0.6, places=3)

    def test_fluent_api(self):
        """InferencePipeline.pln_conjunction appends PLNConjunctionStep"""
        p = at.InferencePipeline(self.space)
        p.pln_conjunction(0.4)
        self.assertEqual(p.size(), 1)
        self.assertIn("PLNConjunction", p.step_names())


# ---------------------------------------------------------------------------
# PLNDisjunctionStep
# ---------------------------------------------------------------------------

class TestPLNDisjunctionStep(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def _make_node(self, name, strength, confidence):
        n = at.create_concept_node(self.space, name)
        n.set_truth_value(at.TruthValue.create(strength, confidence))
        return n

    def test_creates_or_link(self):
        """PLNDisjunctionStep creates OR_LINK for qualifying atoms"""
        A = self._make_node("DisjA1", 0.6, 0.8)
        B = self._make_node("DisjB1", 0.5, 0.7)
        p = at.InferencePipeline(self.space)
        p.pln_disjunction(0.3)
        result = p.run([A, B])
        types = [a.get_type() for a in result.atoms]
        self.assertIn(at.AtomType.OR_LINK, types)

    def test_or_link_tv_strength(self):
        """PLNDisjunctionStep OR_LINK strength = sA + sB - sA*sB"""
        sA, sB = 0.6, 0.5
        A = self._make_node("DisjA2", sA, 0.8)
        B = self._make_node("DisjB2", sB, 0.7)
        p = at.InferencePipeline(self.space)
        p.pln_disjunction(0.3)
        result = p.run([A, B])
        for atom in result.atoms:
            if atom.get_type() == at.AtomType.OR_LINK:
                s = at.TruthValue.get_strength(atom.get_truth_value())
                self.assertAlmostEqual(s, sA + sB - sA * sB, places=3)

    def test_evaluates_existing_or_link(self):
        """PLNDisjunctionStep fills TV of existing OR_LINK with unset confidence"""
        sA, sB = 0.7, 0.4
        A = self._make_node("DisjA3", sA, 0.9)
        B = self._make_node("DisjB3", sB, 0.6)
        or_link = self.space.add_link(at.AtomType.OR_LINK, [A, B])
        self.assertAlmostEqual(
            at.TruthValue.get_confidence(or_link.get_truth_value()), 0.0, places=5)
        p = at.InferencePipeline(self.space)
        p.pln_disjunction(0.3)
        p.run([A, B, or_link])
        s = at.TruthValue.get_strength(or_link.get_truth_value())
        self.assertAlmostEqual(s, sA + sB - sA * sB, places=3)

    def test_fluent_api(self):
        """InferencePipeline.pln_disjunction appends PLNDisjunctionStep"""
        p = at.InferencePipeline(self.space)
        p.pln_disjunction()
        self.assertEqual(p.size(), 1)
        self.assertIn("PLNDisjunction", p.step_names())


# ---------------------------------------------------------------------------
# PLNSimilarityStep
# ---------------------------------------------------------------------------

class TestPLNSimilarityStep(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def _make_node(self, name, strength, confidence):
        n = at.create_concept_node(self.space, name)
        n.set_truth_value(at.TruthValue.create(strength, confidence))
        return n

    def test_creates_similarity_link(self):
        """PLNSimilarityStep creates SIMILARITY_LINK for same-type similar atoms"""
        A = self._make_node("SimA1", 0.9, 0.8)
        B = self._make_node("SimB1", 0.8, 0.7)
        p = at.InferencePipeline(self.space)
        p.pln_similarity(0.3)
        result = p.run([A, B])
        types = [a.get_type() for a in result.atoms]
        self.assertIn(at.AtomType.SIMILARITY_LINK, types)

    def test_similarity_tv_formula(self):
        """PLNSimilarityStep: sim_strength = sA*sB / (sA+sB-sA*sB)"""
        sA, sB = 0.8, 0.8
        A = self._make_node("SimA2", sA, 0.9)
        B = self._make_node("SimB2", sB, 0.9)
        p = at.InferencePipeline(self.space)
        p.pln_similarity(0.3)
        result = p.run([A, B])
        for atom in result.atoms:
            if atom.get_type() == at.AtomType.SIMILARITY_LINK:
                s = at.TruthValue.get_strength(atom.get_truth_value())
                expected = (sA * sB) / (sA + sB - sA * sB + 1e-6)
                self.assertAlmostEqual(s, expected, places=2)

    def test_skips_high_threshold(self):
        """PLNSimilarityStep skips pairs below threshold"""
        A = self._make_node("SimA3", 0.01, 0.9)
        B = self._make_node("SimB3", 0.99, 0.9)
        p = at.InferencePipeline(self.space)
        p.pln_similarity(0.9)   # high threshold
        result = p.run([A, B])
        types = [a.get_type() for a in result.atoms]
        self.assertNotIn(at.AtomType.SIMILARITY_LINK, types)

    def test_fluent_api(self):
        """InferencePipeline.pln_similarity appends PLNSimilarityStep"""
        p = at.InferencePipeline(self.space)
        p.pln_similarity(0.5)
        self.assertEqual(p.size(), 1)
        self.assertIn("PLNSimilarity", p.step_names())


# ---------------------------------------------------------------------------
# InferencePipeline Phase 12 API
# ---------------------------------------------------------------------------

class TestInferencePipelinePhase12(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def test_clear(self):
        """InferencePipeline.clear() removes all steps"""
        p = at.InferencePipeline(self.space)
        p.pln_deduction().pln_revision().pln_conjunction()
        self.assertEqual(p.size(), 3)
        p.clear()
        self.assertEqual(p.size(), 0)

    def test_make_pln_full_pipeline_step_count(self):
        """make_pln_full_pipeline creates 6-step pipeline"""
        p = at.make_pln_full_pipeline(self.space)
        self.assertEqual(p.size(), 6)

    def test_make_pln_full_pipeline_step_names(self):
        """make_pln_full_pipeline has correct step order"""
        p = at.make_pln_full_pipeline(self.space)
        names = p.step_names()
        self.assertEqual(names[0], "PLNDeduction")
        self.assertEqual(names[1], "PLNConjunction")
        self.assertEqual(names[2], "PLNDisjunction")
        self.assertEqual(names[3], "PLNSimilarity")
        self.assertEqual(names[4], "PLNRevision")
        self.assertEqual(names[5], "TVThreshold")

    def test_full_pipeline_derives_atoms(self):
        """make_pln_full_pipeline end-to-end produces more atoms than seeds"""
        A = at.create_concept_node(self.space, "FpA")
        B = at.create_concept_node(self.space, "FpB")
        C = at.create_concept_node(self.space, "FpC")
        A.set_truth_value(at.TruthValue.create(0.9, 0.8))
        B.set_truth_value(at.TruthValue.create(0.8, 0.7))
        C.set_truth_value(at.TruthValue.create(0.7, 0.6))
        ab = self.space.add_link(at.AtomType.INHERITANCE_LINK, [A, B])
        bc = self.space.add_link(at.AtomType.INHERITANCE_LINK, [B, C])
        ab.set_truth_value(at.TruthValue.create(0.85, 0.75))
        bc.set_truth_value(at.TruthValue.create(0.75, 0.65))

        p = at.make_pln_full_pipeline(self.space, min_strength=0.5,
                                       min_similarity=0.3)
        result = p.run([A, B, C, ab, bc])
        self.assertGreater(len(result.atoms), 5)

    def test_fluent_chain(self):
        """Fluent chain pln_conjunction().pln_disjunction().pln_similarity() works"""
        p = at.InferencePipeline(self.space)
        p.pln_conjunction(0.4).pln_disjunction(0.3).pln_similarity(0.5)
        self.assertEqual(p.size(), 3)


# ---------------------------------------------------------------------------
# PatternMatcher negation-as-failure
# ---------------------------------------------------------------------------

class TestPatternMatcherNegation(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def test_not_link_matches_different_atom(self):
        """PatternMatcher: NOT_LINK(A) matches B when B != A"""
        A = at.create_concept_node(self.space, "NegTestA")
        B = at.create_concept_node(self.space, "NegTestB")
        not_a = self.space.add_link(at.AtomType.NOT_LINK, [A])
        result = at.PatternMatcher.match(not_a, B)
        self.assertTrue(result)

    def test_not_link_does_not_match_same_atom(self):
        """PatternMatcher: NOT_LINK(A) does not match A itself"""
        A = at.create_concept_node(self.space, "NegTestC")
        A2 = at.create_concept_node(self.space, "NegTestC")  # same name
        not_a = self.space.add_link(at.AtomType.NOT_LINK, [A])
        result = at.PatternMatcher.match(not_a, A2)
        self.assertFalse(result)

    def test_not_link_with_variable_never_matches(self):
        """PatternMatcher: NOT_LINK(?X) never matches because ?X binds everything"""
        var = at.create_variable_node(self.space, "?V")
        B = at.create_concept_node(self.space, "NegTestD")
        not_var = self.space.add_link(at.AtomType.NOT_LINK, [var])
        result = at.PatternMatcher.match(not_var, B)
        self.assertFalse(result)

    def test_not_link_find_matches_filters_out_target(self):
        """PatternMatcher.find_matches with NOT_LINK excludes the negated atom"""
        cat  = at.create_concept_node(self.space, "NegCat")
        dog  = at.create_concept_node(self.space, "NegDog")
        fish = at.create_concept_node(self.space, "NegFish")
        not_cat = self.space.add_link(at.AtomType.NOT_LINK, [cat])

        matches = at.PatternMatcher.find_matches(self.space, not_cat)
        matched_nodes = [
            a.get_name() for (a, _) in matches if a.is_node()
        ]
        self.assertNotIn("NegCat", matched_nodes)
        self.assertIn("NegDog", matched_nodes)
        self.assertIn("NegFish", matched_nodes)

    def test_not_link_in_link_outgoing(self):
        """PatternMatcher: NOT_LINK nested inside link pattern works"""
        A = at.create_concept_node(self.space, "InnerA")
        B = at.create_concept_node(self.space, "InnerB")
        not_b = self.space.add_link(at.AtomType.NOT_LINK, [B])
        # Pattern: InheritanceLink(A, NOT(B)) should match InheritanceLink(A, X)
        # where X != B
        C = at.create_concept_node(self.space, "InnerC")
        pattern = self.space.add_link(at.AtomType.INHERITANCE_LINK, [A, not_b])
        # Target where second element is C (not B)
        target = self.space.add_link(at.AtomType.INHERITANCE_LINK, [A, C])
        bindings = {}
        ok = at.PatternMatcher.match(pattern, target)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# VariableBinding typed wrapper
# ---------------------------------------------------------------------------

class TestVariableBinding(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def _get_binding(self):
        """Helper: get a VariableBinding from a match."""
        var_x = at.create_variable_node(self.space, "?X")
        animal = at.create_concept_node(self.space, "VBAnimal")
        cat = at.create_concept_node(self.space, "VBCat")
        pattern = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [var_x, animal])
        target = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [cat, animal])
        _, binding = at.PatternMatcher.match_with_bindings(pattern, target)
        return binding, var_x

    def test_variable_binding_is_typed(self):
        """VariableBinding is an instance of at.VariableBinding"""
        binding, _ = self._get_binding()
        self.assertIsInstance(binding, at.VariableBinding)

    def test_variable_binding_len(self):
        """VariableBinding.__len__ returns number of bound variables"""
        binding, _ = self._get_binding()
        self.assertEqual(len(binding), 1)

    def test_variable_binding_get_by_name(self):
        """VariableBinding.get_by_name returns bound atom for variable name"""
        binding, _ = self._get_binding()
        result = binding.get_by_name("?X")
        self.assertIsNotNone(result)
        self.assertEqual(result.get_name(), "VBCat")

    def test_variable_binding_variable_names(self):
        """VariableBinding.variable_names returns list with variable name"""
        binding, _ = self._get_binding()
        names = binding.variable_names()
        self.assertIn("?X", names)

    def test_variable_binding_to_dict(self):
        """VariableBinding.to_dict returns dict with string keys"""
        binding, _ = self._get_binding()
        d = binding.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("?X", d)
        self.assertEqual(d["?X"].get_name(), "VBCat")

    def test_variable_binding_keys(self):
        """VariableBinding.keys() returns list of atom handles"""
        binding, var_x = self._get_binding()
        keys = binding.keys()
        self.assertEqual(len(keys), 1)

    def test_variable_binding_values(self):
        """VariableBinding.values() returns list of bound atoms"""
        binding, _ = self._get_binding()
        vals = binding.values()
        self.assertEqual(len(vals), 1)
        self.assertEqual(vals[0].get_name(), "VBCat")

    def test_variable_binding_repr(self):
        """VariableBinding.__repr__ contains -> separator"""
        binding, _ = self._get_binding()
        r = repr(binding)
        self.assertIn("->", r)
        self.assertIn("VariableBinding", r)

    def test_find_matches_returns_variable_binding(self):
        """PatternMatcher.find_matches returns VariableBinding in each pair"""
        var_x = at.create_variable_node(self.space, "?FMX")
        animal = at.create_concept_node(self.space, "FMAnimal")
        cat = at.create_concept_node(self.space, "FMCat")
        self.space.add_link(at.AtomType.INHERITANCE_LINK, [cat, animal])
        pattern = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [var_x, animal])
        matches = at.PatternMatcher.find_matches(self.space, pattern)
        self.assertGreater(len(matches), 0)
        for (atom, binding) in matches:
            self.assertIsInstance(binding, at.VariableBinding)

    def test_empty_binding_for_ground_match(self):
        """VariableBinding is empty when pattern has no variables"""
        cat    = at.create_concept_node(self.space, "GrCat")
        animal = at.create_concept_node(self.space, "GrAnimal")
        target = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [cat, animal])
        ok, binding = at.PatternMatcher.match_with_bindings(target, target)
        self.assertTrue(ok)
        self.assertEqual(len(binding), 0)


# ---------------------------------------------------------------------------
# Module version
# ---------------------------------------------------------------------------

class TestModuleVersion(unittest.TestCase):
    def test_version(self):
        """Module version is 0.12.0"""
        self.assertEqual(at.__version__, "0.12.0")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
