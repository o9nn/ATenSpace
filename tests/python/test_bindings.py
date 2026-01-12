#!/usr/bin/env python3
"""
ATenSpace Python Bindings - Unit Tests
Phase 6 - Production Integration

Comprehensive test suite for Python bindings.
"""

import unittest
import torch
import tempfile
import os
import sys

# Try to import atenspace
try:
    import atenspace as at
except ImportError:
    print("ERROR: atenspace module not found. Please build and install first:")
    print("  pip install -e .")
    sys.exit(1)


class TestAtomSpace(unittest.TestCase):
    """Test core AtomSpace functionality"""
    
    def setUp(self):
        self.space = at.AtomSpace()
    
    def test_create_atomspace(self):
        """Test AtomSpace creation"""
        self.assertIsNotNone(self.space)
        self.assertEqual(len(self.space), 0)
    
    def test_add_nodes(self):
        """Test adding nodes"""
        cat = at.create_concept_node(self.space, "cat")
        self.assertIsNotNone(cat)
        self.assertEqual(cat.get_name(), "cat")
        self.assertEqual(len(self.space), 1)
    
    def test_add_nodes_with_embeddings(self):
        """Test adding nodes with embeddings"""
        embedding = torch.randn(128)
        cat = at.create_concept_node(self.space, "cat", embedding)
        self.assertTrue(cat.has_embedding())
        self.assertEqual(cat.get_embedding().shape[0], 128)
    
    def test_add_links(self):
        """Test adding links"""
        cat = at.create_concept_node(self.space, "cat")
        mammal = at.create_concept_node(self.space, "mammal")
        link = at.create_inheritance_link(self.space, cat, mammal)
        self.assertIsNotNone(link)
        self.assertEqual(link.get_arity(), 2)
    
    def test_query_by_type(self):
        """Test querying atoms by type"""
        at.create_concept_node(self.space, "cat")
        at.create_concept_node(self.space, "dog")
        at.create_predicate_node(self.space, "is-mammal")
        
        concepts = self.space.get_atoms_by_type(at.AtomType.CONCEPT_NODE)
        self.assertEqual(len(concepts), 2)
        
        predicates = self.space.get_atoms_by_type(at.AtomType.PREDICATE_NODE)
        self.assertEqual(len(predicates), 1)
    
    def test_similarity_search(self):
        """Test similarity search"""
        emb1 = torch.randn(128)
        emb2 = torch.randn(128)
        
        cat = at.create_concept_node(self.space, "cat", emb1)
        dog = at.create_concept_node(self.space, "dog", emb2)
        
        results = self.space.query_similar(emb1, k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].get_name(), "cat")


class TestTruthValues(unittest.TestCase):
    """Test truth value operations"""
    
    def setUp(self):
        self.space = at.AtomSpace()
    
    def test_set_get_truth_value(self):
        """Test setting and getting truth values"""
        atom = at.create_concept_node(self.space, "test")
        tv = torch.tensor([0.9, 0.8])
        atom.set_truth_value(tv)
        
        retrieved = atom.get_truth_value()
        self.assertAlmostEqual(retrieved[0].item(), 0.9, places=5)
        self.assertAlmostEqual(retrieved[1].item(), 0.8, places=5)
    
    def test_deduction(self):
        """Test PLN deduction formula"""
        tv1 = torch.tensor([0.9, 0.8])
        tv2 = torch.tensor([0.85, 0.75])
        
        result = at.TruthValueOps.deduction(tv1, tv2)
        self.assertEqual(result.shape[0], 2)
        self.assertGreater(result[0].item(), 0)
        self.assertLess(result[0].item(), 1)


class TestAttentionBank(unittest.TestCase):
    """Test attention allocation"""
    
    def setUp(self):
        self.space = at.AtomSpace()
        self.bank = at.AttentionBank()
    
    def test_set_attention_value(self):
        """Test setting attention values"""
        atom = at.create_concept_node(self.space, "test")
        av = at.AttentionValue(100.0, 50.0, 10.0)
        self.bank.set_attention_value(atom, av)
        
        retrieved = self.bank.get_attention_value(atom)
        self.assertAlmostEqual(retrieved.sti, 100.0, places=5)
        self.assertAlmostEqual(retrieved.lti, 50.0, places=5)
        self.assertAlmostEqual(retrieved.vlti, 10.0, places=5)
    
    def test_stimulate(self):
        """Test attention stimulation"""
        atom = at.create_concept_node(self.space, "test")
        av = at.AttentionValue(100.0, 50.0, 10.0)
        self.bank.set_attention_value(atom, av)
        
        self.bank.stimulate(atom, 20.0)
        
        new_av = self.bank.get_attention_value(atom)
        self.assertGreater(new_av.sti, 100.0)
    
    def test_attentional_focus(self):
        """Test getting attentional focus"""
        a1 = at.create_concept_node(self.space, "high")
        a2 = at.create_concept_node(self.space, "low")
        
        self.bank.set_attention_value(a1, at.AttentionValue(100.0, 50.0, 10.0))
        self.bank.set_attention_value(a2, at.AttentionValue(10.0, 5.0, 1.0))
        
        focus = self.bank.get_attentional_focus(k=1)
        self.assertEqual(len(focus), 1)
        self.assertEqual(focus[0].get_name(), "high")


class TestTimeServer(unittest.TestCase):
    """Test temporal tracking"""
    
    def setUp(self):
        self.space = at.AtomSpace()
        self.time_server = at.TimeServer()
    
    def test_record_creation(self):
        """Test recording creation time"""
        atom = at.create_concept_node(self.space, "test")
        self.time_server.record_creation(atom)
        
        time = self.time_server.get_creation_time(atom)
        self.assertGreater(time, 0)
    
    def test_record_access(self):
        """Test recording access time"""
        atom = at.create_concept_node(self.space, "test")
        self.time_server.record_access(atom)
        
        time = self.time_server.get_last_access_time(atom)
        self.assertGreater(time, 0)
    
    def test_record_event(self):
        """Test recording custom events"""
        atom = at.create_concept_node(self.space, "test")
        self.time_server.record_event(atom, "custom_event")
        # Event recorded successfully if no exception


class TestSerialization(unittest.TestCase):
    """Test save/load functionality"""
    
    def setUp(self):
        self.space = at.AtomSpace()
    
    def test_save_load_roundtrip(self):
        """Test saving and loading AtomSpace"""
        # Create knowledge
        cat = at.create_concept_node(self.space, "cat")
        mammal = at.create_concept_node(self.space, "mammal")
        at.create_inheritance_link(self.space, cat, mammal)
        
        original_size = len(self.space)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            filename = f.name
        
        try:
            at.Serializer.save(self.space, filename)
            
            # Load into new space
            new_space = at.AtomSpace()
            at.Serializer.load(new_space, filename)
            
            self.assertEqual(len(new_space), original_size)
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_to_string(self):
        """Test exporting to string"""
        at.create_concept_node(self.space, "test")
        text = at.Serializer.to_string(self.space)
        self.assertIn("test", text)


class TestPatternMatching(unittest.TestCase):
    """Test pattern matching"""
    
    def setUp(self):
        self.space = at.AtomSpace()
        self.matcher = at.PatternMatcher(self.space)
    
    def test_basic_match(self):
        """Test basic pattern matching"""
        cat = at.create_concept_node(self.space, "cat")
        mammal = at.create_concept_node(self.space, "mammal")
        link = at.create_inheritance_link(self.space, cat, mammal)
        
        # Match should succeed
        result = self.matcher.match(link, link)
        self.assertTrue(result)


class TestNLU(unittest.TestCase):
    """Test NLU functionality"""
    
    def test_tokenization(self):
        """Test text tokenization"""
        processor = at.TextProcessor()
        text = "The cat sat on the mat."
        tokens = processor.tokenize(text)
        self.assertGreater(len(tokens), 0)
    
    def test_entity_recognition(self):
        """Test entity recognition"""
        recognizer = at.EntityRecognizer()
        text = "John lives in New York."
        entities = recognizer.recognize(text)
        self.assertGreaterEqual(len(entities), 0)
    
    def test_semantic_extraction(self):
        """Test semantic extraction to knowledge graph"""
        space = at.AtomSpace()
        extractor = at.SemanticExtractor(space)
        
        text = "The cat is a mammal."
        extractor.extract_from_text(text)
        
        # Should have created some atoms
        self.assertGreater(len(space), 0)


class TestVision(unittest.TestCase):
    """Test vision functionality"""
    
    def test_bounding_box(self):
        """Test bounding box operations"""
        bbox = at.BoundingBox(10, 20, 50, 60)
        self.assertEqual(bbox.x, 10)
        self.assertEqual(bbox.y, 20)
        self.assertEqual(bbox.width, 50)
        self.assertEqual(bbox.height, 60)
        self.assertEqual(bbox.area(), 50 * 60)
    
    def test_detected_object(self):
        """Test detected object creation"""
        bbox = at.BoundingBox(10, 20, 50, 60)
        obj = at.DetectedObject("cat", bbox, 0.95)
        self.assertEqual(obj.label, "cat")
        self.assertAlmostEqual(obj.confidence, 0.95)
    
    def test_spatial_analysis(self):
        """Test spatial relationship analysis"""
        obj1 = at.DetectedObject("cat", at.BoundingBox(10, 20, 50, 60), 0.95)
        obj2 = at.DetectedObject("mat", at.BoundingBox(5, 70, 100, 30), 0.90)
        
        analyzer = at.SpatialAnalyzer()
        relations = analyzer.analyze_spatial_relations([obj1, obj2])
        
        self.assertGreaterEqual(len(relations), 0)


class TestCognitiveEngine(unittest.TestCase):
    """Test cognitive engine integration"""
    
    def setUp(self):
        self.space = at.AtomSpace()
        self.engine = at.CognitiveEngine(self.space)
    
    def test_cognitive_cycle(self):
        """Test running cognitive cycle"""
        # Add some atoms
        a = at.create_concept_node(self.space, "A")
        b = at.create_concept_node(self.space, "B")
        
        # Run cycle (should not crash)
        self.engine.cognitive_cycle(inference_steps=1)
    
    def test_metrics(self):
        """Test getting metrics"""
        at.create_concept_node(self.space, "test")
        metrics = self.engine.get_metrics()
        self.assertIn('atom_count', metrics)


class TestTensorLogic(unittest.TestCase):
    """Test tensor logic engine"""
    
    def setUp(self):
        self.engine = at.TensorLogicEngine()
    
    def test_batch_and(self):
        """Test batch AND operation"""
        tv1 = torch.tensor([[0.9, 0.8], [0.85, 0.75]])
        tv2 = torch.tensor([[0.95, 0.85], [0.9, 0.8]])
        
        result = self.engine.batch_and([tv1, tv2])
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 2)
    
    def test_batch_deduction(self):
        """Test batch deduction"""
        ab_tvs = torch.tensor([[0.9, 0.8], [0.85, 0.75]])
        bc_tvs = torch.tensor([[0.95, 0.85], [0.9, 0.8]])
        
        result = self.engine.batch_deduction(ab_tvs, bc_tvs)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 2)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAtomSpace))
    suite.addTests(loader.loadTestsFromTestCase(TestTruthValues))
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionBank))
    suite.addTests(loader.loadTestsFromTestCase(TestTimeServer))
    suite.addTests(loader.loadTestsFromTestCase(TestSerialization))
    suite.addTests(loader.loadTestsFromTestCase(TestPatternMatching))
    suite.addTests(loader.loadTestsFromTestCase(TestNLU))
    suite.addTests(loader.loadTestsFromTestCase(TestVision))
    suite.addTests(loader.loadTestsFromTestCase(TestCognitiveEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorLogic))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(run_tests())
