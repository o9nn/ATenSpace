#!/usr/bin/env python3
"""
ATenSpace Python Bindings - Phase 11 Tests

Tests for:
  - PLNDeductionStep (deduction rule: A→B, B→C ⊢ A→C)
  - PLNRevisionStep  (merge duplicate truth values)
  - PLNAbductionStep (abduction: B, A→B ⊢ A)
  - PLNInductionStep (induction: count instances → MemberLinks)
  - InferencePipeline fluent PLN methods
  - make_pln_reasoning_pipeline factory
  - PatternMatcher static helpers (find_matches, substitute, unify, etc.)
  - Pattern class (has_variables, get_variables)
  - Module version 0.11.0
"""

import unittest
import torch
import sys

try:
    import atenspace as at
except ImportError:
    print("ERROR: atenspace module not found. Please build and install first:")
    print("  pip install -e .")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tv(s, c):
    return torch.tensor([s, c])


def _strength(tv):
    return tv[0].item()


def _confidence(tv):
    return tv[1].item()


# ---------------------------------------------------------------------------
# PLNDeductionStep
# ---------------------------------------------------------------------------

class TestPLNDeductionStep(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def _inh(self, a, b, s=0.9, c=0.8):
        link = self.space.add_link(at.AtomType.INHERITANCE_LINK, [a, b])
        link.set_truth_value(_tv(s, c))
        return link

    def test_step_exists(self):
        """PLNDeductionStep can be instantiated"""
        step = at.PLNDeductionStep()
        self.assertIsNotNone(step)

    def test_step_with_min_confidence(self):
        """PLNDeductionStep accepts min_confidence argument"""
        step = at.PLNDeductionStep(min_confidence=0.5)
        self.assertIsNotNone(step)

    def test_deduction_via_pipeline(self):
        """InferencePipeline.pln_deduction() derives A→C from A→B + B→C"""
        A = at.create_concept_node(self.space, "A")
        B = at.create_concept_node(self.space, "B")
        C = at.create_concept_node(self.space, "C")
        ab = self._inh(A, B)
        bc = self._inh(B, C)

        p = at.InferencePipeline(self.space)
        p.pln_deduction()
        result = p.run([ab, bc])

        # Should have derived A→C
        ac_found = any(
            a.is_link() and
            a.get_type() == at.AtomType.INHERITANCE_LINK and
            len(a.get_outgoing_set()) == 2 and
            a.get_outgoing_set()[0].get_name() == "A" and
            a.get_outgoing_set()[1].get_name() == "C"
            for a in result.atoms
        )
        self.assertTrue(ac_found, "A→C should be derived by deduction")

    def test_deduction_no_chain(self):
        """PLNDeductionStep produces nothing when links do not chain"""
        A = at.create_concept_node(self.space, "A")
        B = at.create_concept_node(self.space, "B")
        X = at.create_concept_node(self.space, "X")
        Y = at.create_concept_node(self.space, "Y")
        ab = self._inh(A, B)
        xy = self._inh(X, Y)

        p = at.InferencePipeline(self.space)
        p.pln_deduction()
        result = p.run([ab, xy])
        # No new links should appear
        self.assertEqual(len(result.atoms), 2)

    def test_step_via_add_step(self):
        """PLNDeductionStep can be appended via add_step"""
        A = at.create_concept_node(self.space, "P")
        B = at.create_concept_node(self.space, "Q")
        C = at.create_concept_node(self.space, "R")
        ab = self._inh(A, B)
        bc = self._inh(B, C)

        p = at.InferencePipeline(self.space)
        p.add_step(at.PLNDeductionStep())
        result = p.run([ab, bc])
        self.assertGreaterEqual(len(result.atoms), 2)


# ---------------------------------------------------------------------------
# PLNRevisionStep
# ---------------------------------------------------------------------------

class TestPLNRevisionStep(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def test_step_exists(self):
        """PLNRevisionStep can be instantiated"""
        step = at.PLNRevisionStep()
        self.assertIsNotNone(step)

    def test_revision_deduplicates(self):
        """PLNRevisionStep removes duplicated atom handles from working set"""
        node = at.create_concept_node(self.space, "Node")
        node.set_truth_value(_tv(0.6, 0.5))

        p = at.InferencePipeline(self.space)
        p.pln_revision()
        # Pass the same handle twice — revision should merge them
        result = p.run([node, node])
        self.assertEqual(len(result.atoms), 1)

    def test_revision_via_add_step(self):
        """PLNRevisionStep can be added via add_step"""
        p = at.InferencePipeline(self.space)
        p.add_step(at.PLNRevisionStep())
        self.assertEqual(p.size(), 1)


# ---------------------------------------------------------------------------
# PLNAbductionStep
# ---------------------------------------------------------------------------

class TestPLNAbductionStep(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def test_step_exists(self):
        """PLNAbductionStep can be instantiated with defaults"""
        step = at.PLNAbductionStep()
        self.assertIsNotNone(step)

    def test_step_with_args(self):
        """PLNAbductionStep accepts min_observation_strength and min_confidence"""
        step = at.PLNAbductionStep(min_observation_strength=0.8,
                                   min_confidence=0.1)
        self.assertIsNotNone(step)

    def test_abduction_infers_cause(self):
        """PLNAbductionStep adds antecedent atom when observation is strong"""
        raining     = at.create_concept_node(self.space, "Raining")
        wet_ground  = at.create_concept_node(self.space, "WetGround")

        rule = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [raining, wet_ground])
        rule.set_truth_value(_tv(0.95, 0.9))
        wet_ground.set_truth_value(_tv(0.9, 0.85))

        p = at.InferencePipeline(self.space)
        p.pln_abduction(min_observation_strength=0.7)
        result = p.run([wet_ground, rule])

        names = [a.get_name() for a in result.atoms if a.is_node()]
        self.assertIn("Raining", names)

    def test_abduction_weak_observation(self):
        """PLNAbductionStep does not add atoms for weak observations"""
        A = at.create_concept_node(self.space, "A")
        B = at.create_concept_node(self.space, "B")
        rule = self.space.add_link(at.AtomType.INHERITANCE_LINK, [A, B])
        rule.set_truth_value(_tv(0.9, 0.8))
        B.set_truth_value(_tv(0.3, 0.9))  # strength 0.3 < threshold 0.7

        p = at.InferencePipeline(self.space)
        p.pln_abduction(min_observation_strength=0.7)
        result = p.run([B, rule])
        # No new atom should be added (A already present as outgoing atom ref)
        self.assertLessEqual(len(result.atoms), 2)


# ---------------------------------------------------------------------------
# PLNInductionStep
# ---------------------------------------------------------------------------

class TestPLNInductionStep(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def test_step_exists(self):
        """PLNInductionStep can be instantiated"""
        step = at.PLNInductionStep()
        self.assertIsNotNone(step)

    def test_induction_emits_member_links(self):
        """PLNInductionStep emits MemberLinks for instances"""
        spot = at.create_concept_node(self.space, "Spot")
        fido = at.create_concept_node(self.space, "Fido")
        dog  = at.create_concept_node(self.space, "Dog")

        l1 = self.space.add_link(at.AtomType.INHERITANCE_LINK, [spot, dog])
        l2 = self.space.add_link(at.AtomType.INHERITANCE_LINK, [fido, dog])

        p = at.InferencePipeline(self.space)
        p.pln_induction()
        result = p.run([l1, l2])

        member_links = [a for a in result.atoms
                        if a.is_link() and
                        a.get_type() == at.AtomType.MEMBER_LINK]
        self.assertEqual(len(member_links), 2)

    def test_induction_induced_tv_confidence(self):
        """PLNInductionStep gives induced TV with positive confidence"""
        cat    = at.create_concept_node(self.space, "Cat")
        animal = at.create_concept_node(self.space, "Animal")
        l = self.space.add_link(at.AtomType.INHERITANCE_LINK, [cat, animal])

        p = at.InferencePipeline(self.space)
        p.pln_induction()
        result = p.run([l])

        member_links = [a for a in result.atoms
                        if a.is_link() and
                        a.get_type() == at.AtomType.MEMBER_LINK]
        self.assertGreater(len(member_links), 0)
        conf = _confidence(member_links[0].get_truth_value())
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)


# ---------------------------------------------------------------------------
# InferencePipeline PLN fluent API
# ---------------------------------------------------------------------------

class TestInferencePipelinePLN(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def test_pln_deduction_method(self):
        """InferencePipeline.pln_deduction() returns self for chaining"""
        p = at.InferencePipeline(self.space)
        ret = p.pln_deduction()
        self.assertIsNotNone(ret)

    def test_pln_revision_method(self):
        """InferencePipeline.pln_revision() appends a step"""
        p = at.InferencePipeline(self.space)
        p.pln_revision()
        self.assertEqual(p.size(), 1)

    def test_pln_abduction_method(self):
        """InferencePipeline.pln_abduction() appends a step"""
        p = at.InferencePipeline(self.space)
        p.pln_abduction()
        self.assertEqual(p.size(), 1)

    def test_pln_induction_method(self):
        """InferencePipeline.pln_induction() appends a step"""
        p = at.InferencePipeline(self.space)
        p.pln_induction()
        self.assertEqual(p.size(), 1)

    def test_combined_pln_pipeline(self):
        """PLN deduction + revision pipeline runs without error"""
        A = at.create_concept_node(self.space, "A")
        B = at.create_concept_node(self.space, "B")
        C = at.create_concept_node(self.space, "C")
        ab = self.space.add_link(at.AtomType.INHERITANCE_LINK, [A, B])
        bc = self.space.add_link(at.AtomType.INHERITANCE_LINK, [B, C])
        ab.set_truth_value(_tv(0.9, 0.8))
        bc.set_truth_value(_tv(0.8, 0.7))

        p = at.InferencePipeline(self.space)
        p.pln_deduction().pln_revision()
        result = p.run([ab, bc])
        self.assertGreaterEqual(len(result.atoms), 2)

    def test_make_pln_reasoning_pipeline(self):
        """make_pln_reasoning_pipeline creates a 3-step pipeline"""
        p = at.make_pln_reasoning_pipeline(self.space)
        self.assertEqual(p.size(), 3)
        names = p.step_names()
        self.assertEqual(names[0], "PLNDeduction")
        self.assertEqual(names[1], "PLNRevision")
        self.assertTrue(names[2].startswith("TVThreshold"))


# ---------------------------------------------------------------------------
# PatternMatcher static helpers
# ---------------------------------------------------------------------------

class TestPatternMatcher(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def test_match_ground_atoms(self):
        """PatternMatcher.match: ground atoms match when structurally equal"""
        cat    = at.create_concept_node(self.space, "Cat")
        cat2   = at.create_concept_node(self.space, "Cat")
        animal = at.create_concept_node(self.space, "Animal")
        self.assertTrue(at.PatternMatcher.match(cat, cat2))
        self.assertFalse(at.PatternMatcher.match(cat, animal))

    def test_match_with_variable(self):
        """PatternMatcher.match: variable matches any atom"""
        var_x  = at.create_variable_node(self.space, "?X")
        dog    = at.create_concept_node(self.space, "Dog")
        self.assertTrue(at.PatternMatcher.match(var_x, dog))

    def test_match_with_bindings_returns_dict(self):
        """PatternMatcher.match_with_bindings returns (bool, dict)"""
        var_x  = at.create_variable_node(self.space, "?X")
        cat    = at.create_concept_node(self.space, "Cat")
        ok, bindings = at.PatternMatcher.match_with_bindings(var_x, cat)
        self.assertTrue(ok)
        self.assertIsInstance(bindings, dict)

    def test_find_matches(self):
        """PatternMatcher.find_matches returns matching atoms"""
        dog    = at.create_concept_node(self.space, "Dog")
        animal = at.create_concept_node(self.space, "Animal")
        cat    = at.create_concept_node(self.space, "Cat")
        self.space.add_link(at.AtomType.INHERITANCE_LINK, [dog, animal])
        self.space.add_link(at.AtomType.INHERITANCE_LINK, [cat, animal])

        var_x   = at.create_variable_node(self.space, "?X")
        pattern = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [var_x, animal])

        results = at.PatternMatcher.find_matches(self.space, pattern)
        self.assertGreaterEqual(len(results), 2)

    def test_substitute(self):
        """PatternMatcher.substitute replaces variables with bindings"""
        var_x  = at.create_variable_node(self.space, "?X")
        cat    = at.create_concept_node(self.space, "Cat")
        animal = at.create_concept_node(self.space, "Animal")
        pattern = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [var_x, animal])

        bindings = {var_x: cat}
        concrete = at.PatternMatcher.substitute(pattern, bindings, self.space)
        self.assertTrue(concrete.is_link())
        out = concrete.get_outgoing_set()
        self.assertEqual(out[0].get_name(), "Cat")
        self.assertEqual(out[1].get_name(), "Animal")

    def test_unify(self):
        """PatternMatcher.unify: variable unifies with concrete atom"""
        var_x = at.create_variable_node(self.space, "?X")
        cat   = at.create_concept_node(self.space, "Cat")
        ok, bindings = at.PatternMatcher.unify(var_x, cat)
        self.assertTrue(ok)
        self.assertIsInstance(bindings, dict)
        self.assertGreater(len(bindings), 0)

    def test_is_variable(self):
        """PatternMatcher.is_variable: True only for VariableNodes"""
        var_x = at.create_variable_node(self.space, "?X")
        cat   = at.create_concept_node(self.space, "Cat")
        self.assertTrue(at.PatternMatcher.is_variable(var_x))
        self.assertFalse(at.PatternMatcher.is_variable(cat))

    def test_is_typed_variable(self):
        """PatternMatcher.is_typed_variable: True only for TypedVariableNode"""
        tvar  = at.create_typed_variable_node(self.space, "?X", "ConceptNode")
        var_x = at.create_variable_node(self.space, "?Y")
        self.assertTrue(at.PatternMatcher.is_typed_variable(tvar))
        self.assertFalse(at.PatternMatcher.is_typed_variable(var_x))

    def test_is_glob(self):
        """PatternMatcher.is_glob: True only for GlobNode"""
        glob = at.create_glob_node(self.space, "@rest")
        cat  = at.create_concept_node(self.space, "Cat")
        self.assertTrue(at.PatternMatcher.is_glob(glob))
        self.assertFalse(at.PatternMatcher.is_glob(cat))

    def test_get_type_constraint(self):
        """PatternMatcher.get_type_constraint extracts constraint string"""
        tvar = at.create_typed_variable_node(self.space, "?X", "ConceptNode")
        constraint = at.PatternMatcher.get_type_constraint(tvar)
        self.assertEqual(constraint, "ConceptNode")


# ---------------------------------------------------------------------------
# Pattern class
# ---------------------------------------------------------------------------

class TestPattern(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def test_has_variables_true(self):
        """Pattern.has_variables: True when pattern contains VariableNode"""
        var_x  = at.create_variable_node(self.space, "?X")
        animal = at.create_concept_node(self.space, "Animal")
        pattern = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [var_x, animal])
        self.assertTrue(at.Pattern.has_variables(pattern))

    def test_has_variables_false(self):
        """Pattern.has_variables: False for ground atom"""
        cat    = at.create_concept_node(self.space, "Cat")
        animal = at.create_concept_node(self.space, "Animal")
        link   = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [cat, animal])
        self.assertFalse(at.Pattern.has_variables(link))

    def test_get_variables(self):
        """Pattern.get_variables returns all variables in nested pattern"""
        var_x  = at.create_variable_node(self.space, "?X")
        var_y  = at.create_variable_node(self.space, "?Y")
        animal = at.create_concept_node(self.space, "Animal")
        inner  = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [var_x, var_y])
        outer  = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [inner, animal])
        variables = at.Pattern.get_variables(outer)
        self.assertEqual(len(variables), 2)

    def test_get_variables_empty_for_ground(self):
        """Pattern.get_variables: empty list for ground atom"""
        cat    = at.create_concept_node(self.space, "Cat2")
        animal = at.create_concept_node(self.space, "Animal2")
        link   = self.space.add_link(
            at.AtomType.INHERITANCE_LINK, [cat, animal])
        self.assertEqual(len(at.Pattern.get_variables(link)), 0)


# ---------------------------------------------------------------------------
# Module version
# ---------------------------------------------------------------------------

class TestModuleVersion(unittest.TestCase):
    def test_version(self):
        """Module version is 0.11.0"""
        self.assertEqual(at.__version__, "0.11.0")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
