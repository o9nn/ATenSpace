"""
test_phase13.py - Phase 13 Python tests

Covers:
  - PLNImplicationStep (evaluate existing IMPLICATION_LINKs, create new ones)
  - PLNImplicationChainStep (multi-hop transitive closure)
  - InferencePipeline Phase 13 fluent methods
  - make_pln_complete_pipeline factory helper
  - Module version 0.13.0
"""

import unittest
import sys
import os
import math

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
# PLNImplicationStep
# ---------------------------------------------------------------------------

class TestPLNImplicationStep(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def _node(self, name, s, c):
        n = at.create_concept_node(self.space, name)
        n.set_truth_value(at.TruthValue.create(s, c))
        return n

    def test_creates_implication_link(self):
        """PLNImplicationStep creates IMPLICATION_LINK for qualifying atoms"""
        A = self._node("ImpPyA1", 0.8, 0.9)
        B = self._node("ImpPyB1", 0.7, 0.8)
        step = at.PLNImplicationStep(
            min_antecedent_strength=0.5,
            min_implication_strength=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B])
        types = [a.get_type() for a in result.atoms]
        self.assertIn(at.AtomType.IMPLICATION_LINK, types)

    def test_implication_tv_formula(self):
        """PLNImplicationStep IMPLICATION_LINK TV: s = 1 - sA + sA*sB, c = min(cA,cB)"""
        sA, cA = 0.7, 0.9
        sB, cB = 0.6, 0.8
        A = self._node("ImpPyA2", sA, cA)
        B = self._node("ImpPyB2", sB, cB)
        step = at.PLNImplicationStep(
            min_antecedent_strength=0.5,
            min_implication_strength=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B])

        expected_s = 1.0 - sA + sA * sB
        expected_c = min(cA, cB)

        found = False
        for atom in result.atoms:
            if atom.get_type() != at.AtomType.IMPLICATION_LINK:
                continue
            outgoing = atom.get_outgoing()
            if outgoing[0].get_name() != "ImpPyA2":
                continue
            s = at.TruthValue.get_strength(atom.get_truth_value())
            c = at.TruthValue.get_confidence(atom.get_truth_value())
            self.assertAlmostEqual(s, expected_s, places=3)
            self.assertAlmostEqual(c, expected_c, places=3)
            found = True
        self.assertTrue(found, "A→B implication link not found")

    def test_no_link_when_antecedent_too_weak(self):
        """PLNImplicationStep skips atoms with strength below minAntecedentStrength"""
        A = self._node("ImpPyA3", 0.2, 0.9)   # strength < 0.5
        B = self._node("ImpPyB3", 0.8, 0.9)
        step = at.PLNImplicationStep(
            min_antecedent_strength=0.5,
            min_implication_strength=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B])

        # A→B must not be created (A too weak as antecedent)
        for atom in result.atoms:
            if atom.get_type() != at.AtomType.IMPLICATION_LINK:
                continue
            outgoing = atom.get_outgoing()
            self.assertNotEqual(
                outgoing[0].get_name(), "ImpPyA3",
                "A→B link should not be created when A is too weak")

    def test_no_link_below_implication_strength_threshold(self):
        """PLNImplicationStep skips when implication strength < minImplicationStrength"""
        # s(A→B) = 1 - 0.8 + 0.8*0.1 = 0.28 → below threshold 0.5
        sA, sB = 0.8, 0.1
        A = self._node("ImpPyA4", sA, 0.9)
        B = self._node("ImpPyB4", sB, 0.9)
        step = at.PLNImplicationStep(
            min_antecedent_strength=0.5,
            min_implication_strength=0.5)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B])

        for atom in result.atoms:
            if atom.get_type() != at.AtomType.IMPLICATION_LINK:
                continue
            outgoing = atom.get_outgoing()
            if (outgoing[0].get_name() == "ImpPyA4" and
                    outgoing[1].get_name() == "ImpPyB4"):
                self.fail("A→B link should not have been created")

    def test_evaluates_existing_implication_link(self):
        """PLNImplicationStep fills TV of existing IMPLICATION_LINK with confidence=0"""
        sA, cA = 0.6, 0.8
        sB, cB = 0.8, 0.9
        A = self._node("ImpPyA5", sA, cA)
        B = self._node("ImpPyB5", sB, cB)
        imp_link = self.space.add_link(at.AtomType.IMPLICATION_LINK, [A, B])
        # Explicitly mark as "unset" by setting confidence=0
        imp_link.set_truth_value(at.TruthValue.create(0.5, 0.0))
        self.assertAlmostEqual(
            at.TruthValue.get_confidence(imp_link.get_truth_value()), 0.0, places=5)

        step = at.PLNImplicationStep()
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        p.run([A, B, imp_link])

        s = at.TruthValue.get_strength(imp_link.get_truth_value())
        expected = 1.0 - sA + sA * sB
        self.assertAlmostEqual(s, expected, places=3)

    def test_no_duplicate_links(self):
        """PLNImplicationStep does not create a second A→B link when one exists"""
        A = self._node("ImpPyA6", 0.8, 0.9)
        B = self._node("ImpPyB6", 0.7, 0.9)
        # Pre-create A→B with a known TV
        existing = self.space.add_link(at.AtomType.IMPLICATION_LINK, [A, B])
        existing.set_truth_value(at.TruthValue.create(0.55, 0.7))

        step = at.PLNImplicationStep(
            min_antecedent_strength=0.5,
            min_implication_strength=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B, existing])

        count = sum(
            1 for a in result.atoms
            if a.get_type() == at.AtomType.IMPLICATION_LINK
            and len(a.get_outgoing()) == 2
            and a.get_outgoing()[0].get_name() == "ImpPyA6"
            and a.get_outgoing()[1].get_name() == "ImpPyB6"
        )
        self.assertEqual(count, 1)

    def test_directionality(self):
        """PLNImplicationStep creates A→B and B→A as distinct links with different TVs"""
        sA, sB = 0.8, 0.5
        A = self._node("DirPyA", sA, 0.9)
        B = self._node("DirPyB", sB, 0.9)

        step = at.PLNImplicationStep(
            min_antecedent_strength=0.3,
            min_implication_strength=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B])

        s_AB, s_BA = None, None
        for atom in result.atoms:
            if atom.get_type() != at.AtomType.IMPLICATION_LINK:
                continue
            out = atom.get_outgoing()
            if len(out) != 2:
                continue
            if out[0].get_name() == "DirPyA":
                s_AB = at.TruthValue.get_strength(atom.get_truth_value())
            elif out[0].get_name() == "DirPyB":
                s_BA = at.TruthValue.get_strength(atom.get_truth_value())

        self.assertIsNotNone(s_AB, "A→B link not found")
        self.assertIsNotNone(s_BA, "B→A link not found")
        self.assertAlmostEqual(s_AB, 1.0 - sA + sA * sB, places=3)
        self.assertAlmostEqual(s_BA, 1.0 - sB + sB * sA, places=3)
        self.assertNotAlmostEqual(s_AB, s_BA, places=3)

    def test_fluent_api(self):
        """InferencePipeline.pln_implication appends PLNImplicationStep"""
        p = at.InferencePipeline(self.space)
        p.pln_implication(min_antecedent_strength=0.5,
                          min_implication_strength=0.4)
        self.assertEqual(p.size(), 1)
        self.assertIn("PLNImplication", p.step_names())


# ---------------------------------------------------------------------------
# PLNImplicationChainStep
# ---------------------------------------------------------------------------

class TestPLNImplicationChainStep(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def _node(self, name):
        return at.create_concept_node(self.space, name)

    def _imp(self, src, dst, s, c):
        lnk = self.space.add_link(at.AtomType.IMPLICATION_LINK, [src, dst])
        lnk.set_truth_value(at.TruthValue.create(s, c))
        return lnk

    def test_follows_two_hops(self):
        """PLNImplicationChainStep derives A→C from A→B, B→C"""
        A, B, C = self._node("ChPyA"), self._node("ChPyB"), self._node("ChPyC")
        ab = self._imp(A, B, 0.8, 0.7)
        bc = self._imp(B, C, 0.7, 0.6)

        step = at.PLNImplicationChainStep(max_depth=3, min_chain_confidence=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B, C, ab, bc])

        found = False
        for atom in result.atoms:
            if atom.get_type() != at.AtomType.IMPLICATION_LINK:
                continue
            out = atom.get_outgoing()
            if (len(out) == 2 and out[0].get_name() == "ChPyA"
                    and out[1].get_name() == "ChPyC"):
                found = True
        self.assertTrue(found, "A→C not derived")

    def test_follows_three_hops(self):
        """PLNImplicationChainStep derives A→D from A→B, B→C, C→D"""
        A = self._node("TriPyA")
        B = self._node("TriPyB")
        C = self._node("TriPyC")
        D = self._node("TriPyD")
        ab = self._imp(A, B, 0.9, 0.8)
        bc = self._imp(B, C, 0.8, 0.7)
        cd = self._imp(C, D, 0.7, 0.6)

        step = at.PLNImplicationChainStep(max_depth=3, min_chain_confidence=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B, C, D, ab, bc, cd])

        found = False
        for atom in result.atoms:
            if atom.get_type() != at.AtomType.IMPLICATION_LINK:
                continue
            out = atom.get_outgoing()
            if (len(out) == 2 and out[0].get_name() == "TriPyA"
                    and out[1].get_name() == "TriPyD"):
                found = True
        self.assertTrue(found, "A→D not derived with 3-hop chain")

    def test_respects_max_depth_1(self):
        """PLNImplicationChainStep with max_depth=1 does not derive A→C"""
        A, B, C = self._node("D1A"), self._node("D1B"), self._node("D1C")
        ab = self._imp(A, B, 0.8, 0.7)
        bc = self._imp(B, C, 0.7, 0.6)

        step = at.PLNImplicationChainStep(max_depth=1, min_chain_confidence=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B, C, ab, bc])

        for atom in result.atoms:
            if atom.get_type() != at.AtomType.IMPLICATION_LINK:
                continue
            out = atom.get_outgoing()
            if (len(out) == 2 and out[0].get_name() == "D1A"
                    and out[1].get_name() == "D1C"):
                self.fail("A→C should not be derived with max_depth=1")

    def test_no_duplicate_derived_link(self):
        """PLNImplicationChainStep does not create a second A→C when one already exists"""
        A, B, C = self._node("DupPyA"), self._node("DupPyB"), self._node("DupPyC")
        ab = self._imp(A, B, 0.8, 0.7)
        bc = self._imp(B, C, 0.7, 0.6)
        ac = self._imp(A, C, 0.5, 0.5)  # existing A→C

        step = at.PLNImplicationChainStep(max_depth=3, min_chain_confidence=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B, C, ab, bc, ac])

        count = sum(
            1 for a in result.atoms
            if a.get_type() == at.AtomType.IMPLICATION_LINK
            and len(a.get_outgoing()) == 2
            and a.get_outgoing()[0].get_name() == "DupPyA"
            and a.get_outgoing()[1].get_name() == "DupPyC"
        )
        self.assertEqual(count, 1)

    def test_chained_tv_uses_deduction_formula(self):
        """PLNImplicationChainStep chained TV: strength = sAB * sBC (deduction)"""
        sAB, sBC = 0.8, 0.7
        A, B, C = self._node("TVPyA"), self._node("TVPyB"), self._node("TVPyC")
        ab = self._imp(A, B, sAB, 0.9)
        bc = self._imp(B, C, sBC, 0.6)

        step = at.PLNImplicationChainStep(max_depth=3, min_chain_confidence=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B, C, ab, bc])

        expected_s = sAB * sBC
        for atom in result.atoms:
            if atom.get_type() != at.AtomType.IMPLICATION_LINK:
                continue
            out = atom.get_outgoing()
            if (len(out) == 2 and out[0].get_name() == "TVPyA"
                    and out[1].get_name() == "TVPyC"):
                s = at.TruthValue.get_strength(atom.get_truth_value())
                self.assertAlmostEqual(s, expected_s, places=3)
                return
        self.fail("A→C derived link not found")

    def test_ignores_zero_confidence_links(self):
        """PLNImplicationChainStep does not follow links with confidence=0"""
        A, B, C = self._node("LC_PyA"), self._node("LC_PyB"), self._node("LC_PyC")
        # Explicitly set A→B with confidence=0 (unset / not yet derived)
        ab = self.space.add_link(at.AtomType.IMPLICATION_LINK, [A, B])
        ab.set_truth_value(at.TruthValue.create(0.8, 0.0))  # confidence = 0
        bc = self._imp(B, C, 0.8, 0.7)

        step = at.PLNImplicationChainStep(max_depth=3, min_chain_confidence=0.0)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B, C, ab, bc])

        for atom in result.atoms:
            if atom.get_type() != at.AtomType.IMPLICATION_LINK:
                continue
            out = atom.get_outgoing()
            if (len(out) == 2 and out[0].get_name() == "LC_PyA"
                    and out[1].get_name() == "LC_PyC"):
                self.fail("A→C must not be derived when A→B has c=0")

    def test_noop_without_implication_links(self):
        """PLNImplicationChainStep is a no-op when the working set has no implication links"""
        A = at.create_concept_node(self.space, "NL_PyA")
        B = at.create_concept_node(self.space, "NL_PyB")
        A.set_truth_value(at.TruthValue.create(0.8, 0.9))
        B.set_truth_value(at.TruthValue.create(0.7, 0.8))

        step = at.PLNImplicationChainStep(max_depth=3)
        p = at.InferencePipeline(self.space)
        p.add_step(step)
        result = p.run([A, B])
        # No new atoms — only A and B remain
        self.assertEqual(len(result.atoms), 2)

    def test_fluent_api(self):
        """InferencePipeline.pln_implication_chain appends PLNImplicationChainStep"""
        p = at.InferencePipeline(self.space)
        p.pln_implication_chain(max_depth=4, min_chain_confidence=0.1)
        self.assertEqual(p.size(), 1)
        self.assertIn("PLNImplicationChain", p.step_names())


# ---------------------------------------------------------------------------
# InferencePipeline - Phase 13 methods and factories
# ---------------------------------------------------------------------------

class TestInferencePipelinePhase13(unittest.TestCase):
    def setUp(self):
        self.space = at.AtomSpace()

    def test_pln_implication_step_name(self):
        """pln_implication() shortcut registers correct step name"""
        p = at.InferencePipeline(self.space)
        p.pln_implication()
        self.assertEqual(p.step_names()[0], "PLNImplication")

    def test_pln_implication_chain_step_name(self):
        """pln_implication_chain() shortcut registers correct step name"""
        p = at.InferencePipeline(self.space)
        p.pln_implication_chain()
        self.assertEqual(p.step_names()[0], "PLNImplicationChain")

    def test_fluent_chaining(self):
        """Fluent chaining of Phase 13 methods builds a 2-step pipeline"""
        p = at.InferencePipeline(self.space)
        p.pln_implication().pln_implication_chain()
        self.assertEqual(p.size(), 2)
        names = p.step_names()
        self.assertEqual(names[0], "PLNImplication")
        self.assertEqual(names[1], "PLNImplicationChain")

    def test_make_pln_complete_pipeline_step_count(self):
        """make_pln_complete_pipeline creates an 8-step pipeline"""
        p = at.make_pln_complete_pipeline(self.space)
        self.assertEqual(p.size(), 8)

    def test_make_pln_complete_pipeline_step_names(self):
        """make_pln_complete_pipeline step names are in expected order"""
        p = at.make_pln_complete_pipeline(self.space)
        names = p.step_names()
        self.assertEqual(names[0], "PLNDeduction")
        self.assertEqual(names[1], "PLNConjunction")
        self.assertEqual(names[2], "PLNDisjunction")
        self.assertEqual(names[3], "PLNSimilarity")
        self.assertEqual(names[4], "PLNImplication")
        self.assertEqual(names[5], "PLNImplicationChain")
        self.assertEqual(names[6], "PLNRevision")
        self.assertTrue(names[7].startswith("TVThreshold"))

    def test_make_pln_full_pipeline_unchanged(self):
        """make_pln_full_pipeline still creates 6 steps (backward compat)"""
        p = at.make_pln_full_pipeline(self.space)
        self.assertEqual(p.size(), 6)

    def test_complete_pipeline_end_to_end(self):
        """make_pln_complete_pipeline derives transitive implication link end-to-end"""
        A = at.create_concept_node(self.space, "E2E_PyA")
        B = at.create_concept_node(self.space, "E2E_PyB")
        C = at.create_concept_node(self.space, "E2E_PyC")
        A.set_truth_value(at.TruthValue.create(0.9, 0.85))
        B.set_truth_value(at.TruthValue.create(0.8, 0.75))
        C.set_truth_value(at.TruthValue.create(0.7, 0.65))

        ab = self.space.add_link(at.AtomType.IMPLICATION_LINK, [A, B])
        bc = self.space.add_link(at.AtomType.IMPLICATION_LINK, [B, C])
        ab.set_truth_value(at.TruthValue.create(0.85, 0.80))
        bc.set_truth_value(at.TruthValue.create(0.75, 0.70))

        p = at.make_pln_complete_pipeline(
            self.space,
            tv_threshold=0.0,
            min_confidence=0.0,
            min_strength=0.4,
            min_similarity=0.4,
            min_antecedent_strength=0.4,
            min_implication_strength=0.0,
            chain_max_depth=3,
        )
        result = p.run([A, B, C, ab, bc])

        # Result must contain more atoms than the seeds (new links derived)
        self.assertGreater(len(result.atoms), 5)
        self.assertEqual(result.iterations_run, 1)

        # A→C must have been derived by the chain step
        found_ac = False
        for atom in result.atoms:
            if atom.get_type() != at.AtomType.IMPLICATION_LINK:
                continue
            out = atom.get_outgoing()
            if (len(out) == 2 and out[0].get_name() == "E2E_PyA"
                    and out[1].get_name() == "E2E_PyC"):
                found_ac = True
        self.assertTrue(found_ac, "A→C not derived in end-to-end test")


# ---------------------------------------------------------------------------
# Module version
# ---------------------------------------------------------------------------

class TestModuleVersion(unittest.TestCase):
    def test_version(self):
        """Module version is 0.13.0"""
        self.assertEqual(at.__version__, "0.13.0")


if __name__ == "__main__":
    unittest.main()
