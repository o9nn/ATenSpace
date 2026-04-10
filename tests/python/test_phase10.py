#!/usr/bin/env python3
"""
ATenSpace Python Bindings - Phase 10 Tests (covering Phase 9 bindings)

Tests for:
  - QueryEngine and QueryBuilder
  - BinarySerializer (save/load/serialize/deserialize)
  - InferencePipeline and pipeline steps
  - HebbianLearner
  - TypedVariableNode / GlobNode (Phase 10)
  - New AtomType enum values
"""

import unittest
import torch
import tempfile
import os
import sys

try:
    import atenspace as at
except ImportError:
    print("ERROR: atenspace module not found. Please build and install first:")
    print("  pip install -e .")
    sys.exit(1)


class TestQueryEngine(unittest.TestCase):
    """Tests for QueryEngine and QueryBuilder"""

    def setUp(self):
        self.space = at.AtomSpace()

    def test_find_by_type(self):
        """QueryEngine.find_by_type returns atoms of correct type"""
        at.create_concept_node(self.space, "dog")
        at.create_concept_node(self.space, "cat")
        at.create_predicate_node(self.space, "is-animal")

        qe = at.QueryEngine(self.space)
        concepts = qe.find_by_type(at.AtomType.CONCEPT_NODE)
        self.assertEqual(len(concepts), 2)

        preds = qe.find_by_type(at.AtomType.PREDICATE_NODE)
        self.assertEqual(len(preds), 1)

    def test_find_matches_single_pattern(self):
        """QueryEngine.find_matches returns correct bindings"""
        mammal = at.create_concept_node(self.space, "qe-mammal")
        dog    = at.create_concept_node(self.space, "qe-dog")
        at.create_inheritance_link(self.space, dog, mammal)

        var_x  = at.create_variable_node(self.space, "?X")
        pattern = self.space.add_link(at.AtomType.INHERITANCE_LINK, [var_x, mammal])

        qe = at.QueryEngine(self.space)
        results = qe.find_matches(pattern)
        self.assertGreaterEqual(len(results), 1)

    def test_neighbourhood(self):
        """QueryEngine.neighbourhood expands correctly"""
        a = at.create_concept_node(self.space, "nb-a")
        b = at.create_concept_node(self.space, "nb-b")
        at.create_inheritance_link(self.space, a, b)

        qe = at.QueryEngine(self.space)
        nb = qe.neighbourhood(a, 1)
        self.assertIsInstance(nb, list)

    def test_query_builder_match(self):
        """QueryBuilder.match returns results"""
        mammal = at.create_concept_node(self.space, "qb-mammal")
        dog    = at.create_concept_node(self.space, "qb-dog")
        wolf   = at.create_concept_node(self.space, "qb-wolf")
        at.create_inheritance_link(self.space, dog,  mammal)
        at.create_inheritance_link(self.space, wolf, mammal)

        var_x  = at.create_variable_node(self.space, "?QBX")
        pattern = self.space.add_link(at.AtomType.INHERITANCE_LINK, [var_x, mammal])

        results = at.QueryBuilder(self.space).match(pattern).execute()
        self.assertEqual(len(results), 2)

    def test_query_builder_limit(self):
        """QueryBuilder.limit restricts result count"""
        base = at.create_concept_node(self.space, "lim-base")
        for i in range(10):
            entity = at.create_concept_node(self.space, f"lim-e-{i}")
            at.create_inheritance_link(self.space, entity, base)

        var_x   = at.create_variable_node(self.space, "?LX")
        pattern = self.space.add_link(at.AtomType.INHERITANCE_LINK, [var_x, base])

        results = at.QueryBuilder(self.space).match(pattern).limit(3).execute()
        self.assertLessEqual(len(results), 3)

    def test_query_builder_count(self):
        """QueryBuilder.count returns correct integer"""
        base = at.create_concept_node(self.space, "cnt-base")
        for i in range(5):
            e = at.create_concept_node(self.space, f"cnt-e-{i}")
            at.create_inheritance_link(self.space, e, base)

        var_x   = at.create_variable_node(self.space, "?CX")
        pattern = self.space.add_link(at.AtomType.INHERITANCE_LINK, [var_x, base])

        n = at.QueryBuilder(self.space).match(pattern).count()
        self.assertEqual(n, 5)

    def test_find_by_truth_strength(self):
        """QueryEngine.find_by_truth_strength filters by TV"""
        a = at.create_concept_node(self.space, "tvs-a")
        b = at.create_concept_node(self.space, "tvs-b")
        lnk = at.create_inheritance_link(self.space, a, b)
        lnk.set_truth_value(torch.tensor([0.9, 0.9]))

        qe = at.QueryEngine(self.space)
        strong = qe.find_by_truth_strength(0.8)
        self.assertGreaterEqual(len(strong), 1)


class TestBinarySerializer(unittest.TestCase):
    """Tests for module-level BinarySerializer functions"""

    def setUp(self):
        self.space = at.AtomSpace()
        for i in range(5):
            a = at.create_concept_node(self.space, f"ser-a-{i}")
            b = at.create_concept_node(self.space, f"ser-b-{i}")
            at.create_inheritance_link(self.space, a, b)

    def test_save_and_load(self):
        """save_atomspace / load_atomspace round-trip"""
        with tempfile.NamedTemporaryFile(suffix=".atsp", delete=False) as f:
            path = f.name
        try:
            ok = at.save_atomspace(self.space, path)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(path))

            dst = at.AtomSpace()
            ok2 = at.load_atomspace(dst, path)
            self.assertTrue(ok2)
            self.assertEqual(dst.get_size(), self.space.get_size())
        finally:
            os.unlink(path)

    def test_serialize_deserialize_bytes(self):
        """serialize_atomspace / deserialize_atomspace via bytes"""
        data = at.serialize_atomspace(self.space)
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

        dst = at.AtomSpace()
        at.deserialize_atomspace(dst, data)
        self.assertEqual(dst.get_size(), self.space.get_size())

    def test_serialize_empty_space(self):
        """Serializing an empty AtomSpace produces valid bytes"""
        empty = at.AtomSpace()
        data = at.serialize_atomspace(empty)
        self.assertIsInstance(data, bytes)

        dst = at.AtomSpace()
        at.deserialize_atomspace(dst, data)
        self.assertEqual(dst.get_size(), 0)


class TestInferencePipeline(unittest.TestCase):
    """Tests for InferencePipeline and step classes"""

    def setUp(self):
        self.space = at.AtomSpace()

    def test_forward_chain_step(self):
        """ForwardChainingStep executes without error"""
        a = at.create_concept_node(self.space, "fc-a")
        b = at.create_concept_node(self.space, "fc-b")
        lnk = at.create_inheritance_link(self.space, a, b)
        lnk.set_truth_value(torch.tensor([0.9, 0.9]))

        pipeline = at.InferencePipeline(self.space)
        pipeline.forward_chain(2)
        result = pipeline.run([a])
        self.assertIsInstance(result, at.PipelineResult)

    def test_tv_threshold_step(self):
        """TruthValueThresholdStep filters atoms correctly"""
        a = at.create_concept_node(self.space, "tvt-a")
        b = at.create_concept_node(self.space, "tvt-b")
        lhigh = at.create_inheritance_link(self.space, a, b)
        llow  = at.create_inheritance_link(self.space, b, a)
        lhigh.set_truth_value(torch.tensor([0.9, 0.9]))
        llow.set_truth_value(torch.tensor([0.1, 0.1]))

        pipeline = at.InferencePipeline(self.space)
        pipeline.filter_by_tv(0.5, 0.5)
        result = pipeline.run([lhigh, llow])

        self.assertIsInstance(result, at.PipelineResult)
        self.assertIsInstance(result.atoms, list)

    def test_pipeline_result_fields(self):
        """PipelineResult exposes correct fields"""
        pipeline = at.InferencePipeline(self.space)
        result = pipeline.run()
        self.assertIsInstance(result.atoms, list)
        self.assertIsInstance(result.stats, list)
        self.assertIsInstance(result.converged, bool)
        self.assertIsInstance(result.iterations_run, int)
        self.assertIsInstance(result.total_ms(), float)

    def test_step_stats_fields(self):
        """StepStats exposes correct fields after run"""
        pipeline = at.InferencePipeline(self.space)
        pipeline.forward_chain(1)
        result = pipeline.run()
        if result.stats:
            stat = result.stats[0]
            self.assertIsInstance(stat.step_name, str)
            self.assertIsInstance(stat.produced, bool)
            self.assertIsInstance(stat.working_set_size, int)
            self.assertIsInstance(stat.elapsed_ms, float)

    def test_pattern_match_step(self):
        """PatternMatchStep adds matching atoms to working set"""
        cat  = at.create_concept_node(self.space, "pm-cat")
        anim = at.create_concept_node(self.space, "pm-animal")
        at.create_inheritance_link(self.space, cat, anim)

        var_x   = at.create_variable_node(self.space, "?PMX")
        pattern = self.space.add_link(at.AtomType.INHERITANCE_LINK, [var_x, anim])

        pipeline = at.InferencePipeline(self.space)
        pipeline.match_pattern(pattern)
        result = pipeline.run()
        self.assertIsInstance(result.atoms, list)

    def test_filter_step_python_predicate(self):
        """FilterStep with Python predicate filters correctly"""
        a = at.create_concept_node(self.space, "fs-a")
        b = at.create_concept_node(self.space, "fs-b")

        # Keep only atoms whose name starts with "fs-a"
        step = at.FilterStep("keep-a", lambda atom: atom.get_name().startswith("fs-a"))
        pipeline = at.InferencePipeline(self.space)
        pipeline.add_step(step)
        result = pipeline.run([a, b])
        self.assertEqual(len(result.atoms), 1)
        self.assertEqual(result.atoms[0].get_name(), "fs-a")

    def test_custom_step_python_fn(self):
        """CustomStep with Python callable executes correctly"""
        a = at.create_concept_node(self.space, "cs-a")
        called = [False]

        def my_step(working_set, space):
            called[0] = True
            return False  # no change

        step = at.CustomStep("my-step", my_step)
        pipeline = at.InferencePipeline(self.space)
        pipeline.add_step(step)
        pipeline.run([a])
        self.assertTrue(called[0])

    def test_make_forward_reasoning_pipeline(self):
        """make_forward_reasoning_pipeline factory works"""
        a = at.create_concept_node(self.space, "mfr-a")
        b = at.create_concept_node(self.space, "mfr-b")
        lnk = at.create_inheritance_link(self.space, a, b)
        lnk.set_truth_value(torch.tensor([0.9, 0.9]))

        var_x   = at.create_variable_node(self.space, "?MFRX")
        pattern = self.space.add_link(at.AtomType.INHERITANCE_LINK, [var_x, b])

        pipeline = at.make_forward_reasoning_pipeline(self.space, pattern, 0.5, 2)
        self.assertIsInstance(pipeline, at.InferencePipeline)
        result = pipeline.run([a])
        self.assertIsInstance(result, at.PipelineResult)

    def test_pipeline_step_names(self):
        """InferencePipeline.step_names returns list of strings"""
        pipeline = at.InferencePipeline(self.space)
        pipeline.forward_chain(1)
        pipeline.filter_by_tv(0.5)
        names = pipeline.step_names()
        self.assertIsInstance(names, list)
        self.assertEqual(len(names), 2)
        for n in names:
            self.assertIsInstance(n, str)

    def test_pipeline_size(self):
        """InferencePipeline.size returns step count"""
        pipeline = at.InferencePipeline(self.space)
        self.assertEqual(pipeline.size(), 0)
        pipeline.forward_chain(1)
        self.assertEqual(pipeline.size(), 1)


class TestHebbianLearner(unittest.TestCase):
    """Tests for HebbianLearner"""

    def setUp(self):
        self.space = at.AtomSpace()
        self.bank  = at.AttentionBank()

    def test_config_defaults(self):
        """HebbianLearnerConfig has correct defaults"""
        cfg = at.HebbianLearnerConfig()
        self.assertAlmostEqual(cfg.learning_rate, 0.1, places=4)
        self.assertAlmostEqual(cfg.decay_rate,    0.01, places=4)
        self.assertFalse(cfg.asymmetric)
        self.assertFalse(cfg.oja_rule)

    def test_record_co_activation_creates_link(self):
        """recordCoActivation creates a HebbianLink"""
        learner = at.HebbianLearner(self.space, self.bank)
        a = at.create_concept_node(self.space, "hl-py-a")
        b = at.create_concept_node(self.space, "hl-py-b")

        learner.record_co_activation(a, b)
        link = learner.get_link(a, b)
        self.assertIsNotNone(link)

    def test_strength_increases_with_activations(self):
        """Repeated co-activations increase Hebbian strength"""
        cfg = at.HebbianLearnerConfig()
        cfg.learning_rate = 0.2
        learner = at.HebbianLearner(self.space, self.bank, cfg)

        a = at.create_concept_node(self.space, "hl-py-str-a")
        b = at.create_concept_node(self.space, "hl-py-str-b")

        before = learner.get_strength(a, b)
        for _ in range(10):
            learner.record_co_activation(a, b)
        after = learner.get_strength(a, b)
        self.assertGreater(after, before)

    def test_decay_reduces_strength(self):
        """decay() reduces Hebbian link strength"""
        learner = at.HebbianLearner(self.space, self.bank)
        a = at.create_concept_node(self.space, "hl-py-dec-a")
        b = at.create_concept_node(self.space, "hl-py-dec-b")

        for _ in range(20):
            learner.record_co_activation(a, b)
        before = learner.get_strength(a, b)
        learner.decay()
        after = learner.get_strength(a, b)
        self.assertLessEqual(after, before)

    def test_get_associates(self):
        """get_associates returns sorted neighbour list"""
        learner = at.HebbianLearner(self.space, self.bank)
        a = at.create_concept_node(self.space, "hl-py-asc-a")
        b = at.create_concept_node(self.space, "hl-py-asc-b")
        c = at.create_concept_node(self.space, "hl-py-asc-c")

        for _ in range(5):
            learner.record_co_activation(a, b)
        for _ in range(2):
            learner.record_co_activation(a, c)

        associates = learner.get_associates(a, 0.0)
        self.assertIsInstance(associates, list)
        self.assertGreaterEqual(len(associates), 2)

    def test_total_co_activations(self):
        """total_co_activations returns correct count"""
        learner = at.HebbianLearner(self.space, self.bank)
        a = at.create_concept_node(self.space, "hl-py-tot-a")
        b = at.create_concept_node(self.space, "hl-py-tot-b")

        self.assertEqual(learner.total_co_activations(), 0)
        learner.record_co_activation(a, b)
        self.assertEqual(learner.total_co_activations(), 1)

    def test_run_cycles(self):
        """run_cycles completes without error"""
        learner = at.HebbianLearner(self.space, self.bank)
        a = at.create_concept_node(self.space, "hl-py-cyc-a")
        b = at.create_concept_node(self.space, "hl-py-cyc-b")
        learner.record_co_activation(a, b)
        learner.run_cycles(3)  # should not raise

    def test_reset(self):
        """reset() clears all Hebbian links and stats"""
        learner = at.HebbianLearner(self.space, self.bank)
        a = at.create_concept_node(self.space, "hl-py-rst-a")
        b = at.create_concept_node(self.space, "hl-py-rst-b")
        learner.record_co_activation(a, b)
        self.assertGreater(learner.total_co_activations(), 0)
        learner.reset()
        self.assertEqual(learner.total_co_activations(), 0)


class TestPhase10NewAtomTypes(unittest.TestCase):
    """Tests for Phase 10 new atom types in Python"""

    def setUp(self):
        self.space = at.AtomSpace()

    def test_typed_variable_node_enum_value(self):
        """AtomType.TYPED_VARIABLE_NODE is accessible"""
        self.assertIsNotNone(at.AtomType.TYPED_VARIABLE_NODE)
        self.assertNotEqual(at.AtomType.TYPED_VARIABLE_NODE,
                            at.AtomType.VARIABLE_NODE)

    def test_glob_node_enum_value(self):
        """AtomType.GLOB_NODE is accessible"""
        self.assertIsNotNone(at.AtomType.GLOB_NODE)
        self.assertNotEqual(at.AtomType.GLOB_NODE, at.AtomType.VARIABLE_NODE)

    def test_create_typed_variable_node(self):
        """create_typed_variable_node factory produces TypedVariableNode"""
        tvar = at.create_typed_variable_node(self.space, "?X", "ConceptNode")
        self.assertIsNotNone(tvar)
        self.assertEqual(tvar.get_type(), at.AtomType.TYPED_VARIABLE_NODE)

    def test_create_glob_node(self):
        """create_glob_node factory produces GlobNode"""
        glob = at.create_glob_node(self.space)
        self.assertIsNotNone(glob)
        self.assertEqual(glob.get_type(), at.AtomType.GLOB_NODE)

    def test_hebbian_link_enum_values(self):
        """Hebbian link AtomType values are accessible"""
        self.assertIsNotNone(at.AtomType.HEBBIAN_LINK)
        self.assertIsNotNone(at.AtomType.SYMMETRIC_HEBBIAN_LINK)
        self.assertIsNotNone(at.AtomType.ASYMMETRIC_HEBBIAN_LINK)
        self.assertIsNotNone(at.AtomType.INVERSE_HEBBIAN_LINK)


class TestModuleVersion(unittest.TestCase):
    """Test module version reflects Phase 10"""

    def test_version_string(self):
        """Module version is 0.10.0"""
        self.assertEqual(at.__version__, "0.10.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
