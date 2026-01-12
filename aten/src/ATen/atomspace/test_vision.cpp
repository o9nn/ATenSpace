#include <ATen/atomspace/ATenSpace.h>
#include <ATen/atomspace/Vision.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace at::atomspace;

void test_boundingBox() {
    std::cout << "Test: Bounding Box Operations... ";
    
    BoundingBox box1(0.2f, 0.3f, 0.4f, 0.5f);
    
    assert(std::abs(box1.centerX() - 0.4f) < 0.01f);
    assert(std::abs(box1.centerY() - 0.55f) < 0.01f);
    assert(std::abs(box1.area() - 0.2f) < 0.01f);
    
    std::cout << "PASSED\n";
}

void test_boundingBoxIOU() {
    std::cout << "Test: Bounding Box IoU... ";
    
    BoundingBox box1(0.0f, 0.0f, 0.5f, 0.5f);
    BoundingBox box2(0.25f, 0.25f, 0.5f, 0.5f);
    
    float iou = box1.iou(box2);
    
    // Boxes overlap, so IoU should be > 0 and < 1
    assert(iou > 0.0f);
    assert(iou < 1.0f);
    
    // Test non-overlapping boxes
    BoundingBox box3(0.8f, 0.8f, 0.1f, 0.1f);
    float iou2 = box1.iou(box3);
    assert(iou2 == 0.0f);
    
    std::cout << "PASSED\n";
}

void test_detectedObject() {
    std::cout << "Test: Detected Object Creation... ";
    
    BoundingBox bbox(0.2f, 0.3f, 0.4f, 0.5f);
    Tensor features = torch::randn({128});
    DetectedObject obj("dog", bbox, 0.95f, features);
    
    assert(obj.label == "dog");
    assert(obj.confidence == 0.95f);
    assert(obj.bbox.x == 0.2f);
    assert(obj.features.numel() == 128);
    
    std::cout << "PASSED\n";
}

void test_spatialRelation() {
    std::cout << "Test: Spatial Relation... ";
    
    SpatialRelation rel("above", "cat", "mat", 0.8f);
    
    assert(rel.type == "above");
    assert(rel.object1 == "cat");
    assert(rel.object2 == "mat");
    assert(rel.confidence == 0.8f);
    
    std::cout << "PASSED\n";
}

void test_spatialAnalysis() {
    std::cout << "Test: Spatial Relationship Analysis... ";
    
    std::vector<DetectedObject> objects;
    objects.emplace_back("top_obj", BoundingBox(0.4f, 0.2f, 0.2f, 0.2f), 0.9f);
    objects.emplace_back("bottom_obj", BoundingBox(0.4f, 0.6f, 0.2f, 0.2f), 0.9f);
    
    auto relations = SpatialAnalyzer::analyzeSpatialRelations(objects);
    
    // Should detect vertical relationship
    assert(relations.size() > 0);
    
    // The relationship should be "above" or "below"
    bool foundVerticalRelation = false;
    for (const auto& rel : relations) {
        if (rel.type == "above" || rel.type == "below") {
            foundVerticalRelation = true;
        }
    }
    assert(foundVerticalRelation);
    
    std::cout << "PASSED\n";
}

void test_sceneGraph() {
    std::cout << "Test: Scene Graph Building... ";
    
    AtomSpace space;
    
    std::vector<DetectedObject> objects;
    objects.emplace_back("dog", BoundingBox(0.2f, 0.3f, 0.2f, 0.3f), 0.95f, torch::randn({128}));
    objects.emplace_back("person", BoundingBox(0.6f, 0.3f, 0.2f, 0.4f), 0.98f, torch::randn({128}));
    
    auto relations = SpatialAnalyzer::analyzeSpatialRelations(objects);
    
    SceneUnderstanding::buildSceneGraph(space, objects, relations);
    
    // Should have created atoms
    assert(space.getSize() > 0);
    
    auto nodes = space.getAtomsByType(Atom::Type::CONCEPT_NODE);
    assert(nodes.size() >= 2);  // At least dog and person
    
    std::cout << "PASSED\n";
}

void test_sceneDescription() {
    std::cout << "Test: Scene Description... ";
    
    std::vector<DetectedObject> objects;
    objects.emplace_back("cat", BoundingBox(0.3f, 0.5f, 0.2f, 0.3f), 0.95f);
    objects.emplace_back("dog", BoundingBox(0.6f, 0.5f, 0.2f, 0.3f), 0.92f);
    
    auto relations = SpatialAnalyzer::analyzeSpatialRelations(objects);
    
    std::string description = SceneUnderstanding::describeScene(objects, relations);
    
    assert(!description.empty());
    assert(description.find("cat") != std::string::npos);
    assert(description.find("dog") != std::string::npos);
    
    std::cout << "PASSED\n";
}

void test_emptyScene() {
    std::cout << "Test: Empty Scene Handling... ";
    
    std::vector<DetectedObject> objects;
    auto relations = SpatialAnalyzer::analyzeSpatialRelations(objects);
    
    assert(relations.empty());
    
    std::string description = SceneUnderstanding::describeScene(objects, relations);
    assert(!description.empty());  // Should return a message about empty scene
    
    std::cout << "PASSED\n";
}

void test_visualGrounding() {
    std::cout << "Test: Visual Concept Grounding... ";
    
    AtomSpace space;
    
    std::vector<DetectedObject> examples;
    examples.emplace_back("dog", BoundingBox(0.2f, 0.3f, 0.2f, 0.3f), 0.95f, torch::randn({128}));
    examples.emplace_back("cat", BoundingBox(0.5f, 0.4f, 0.15f, 0.25f), 0.92f, torch::randn({128}));
    
    VisualReasoning::groundConcept(space, "pet", examples);
    
    // Should create concept node and inheritance links
    auto petNode = space.getNode(Atom::Type::CONCEPT_NODE, "pet");
    assert(petNode != nullptr);
    
    auto inheritanceLinks = space.getAtomsByType(Atom::Type::INHERITANCE_LINK);
    assert(inheritanceLinks.size() > 0);
    
    std::cout << "PASSED\n";
}

void test_objectDetectorInterface() {
    std::cout << "Test: Object Detector Interface... ";
    
    // Create a dummy image tensor
    Tensor image = torch::randn({3, 224, 224});
    
    // Test with default detector (should return empty for now)
    auto objects = ObjectDetector::detect(image);
    assert(objects.size() == 0);  // Default implementation returns empty
    
    // Test with custom detector function
    auto customDetector = [](const Tensor& img) -> std::vector<DetectedObject> {
        std::vector<DetectedObject> results;
        results.emplace_back("test_object", BoundingBox(0.5f, 0.5f, 0.2f, 0.2f), 0.9f);
        return results;
    };
    
    auto customObjects = ObjectDetector::detect(image, customDetector);
    assert(customObjects.size() == 1);
    assert(customObjects[0].label == "test_object");
    
    std::cout << "PASSED\n";
}

void test_visionProcessor() {
    std::cout << "Test: Vision Processor... ";
    
    AtomSpace space;
    
    // Create dummy image
    Tensor image = torch::randn({3, 224, 224});
    
    // Custom detector that returns some objects
    auto detector = [](const Tensor& img) -> std::vector<DetectedObject> {
        std::vector<DetectedObject> results;
        results.emplace_back("dog", BoundingBox(0.2f, 0.3f, 0.2f, 0.3f), 0.95f, torch::randn({128}));
        results.emplace_back("person", BoundingBox(0.6f, 0.2f, 0.25f, 0.5f), 0.98f, torch::randn({128}));
        return results;
    };
    
    VisionProcessor::processImage(space, image, detector);
    
    // Should have created knowledge graph
    assert(space.getSize() > 0);
    
    auto nodes = space.getAtomsByType(Atom::Type::CONCEPT_NODE);
    assert(nodes.size() >= 2);
    
    std::cout << "PASSED\n";
}

void test_multimodalCaption() {
    std::cout << "Test: Multimodal Image Captioning... ";
    
    std::vector<DetectedObject> objects;
    objects.emplace_back("cat", BoundingBox(0.3f, 0.6f, 0.2f, 0.3f), 0.95f);
    objects.emplace_back("mat", BoundingBox(0.2f, 0.7f, 0.6f, 0.2f), 0.90f);
    
    auto relations = SpatialAnalyzer::analyzeSpatialRelations(objects);
    
    std::string caption = MultimodalIntegration::caption(objects, relations);
    
    assert(!caption.empty());
    assert(caption.find("cat") != std::string::npos);
    assert(caption.find("mat") != std::string::npos);
    
    std::cout << "PASSED\n";
}

void test_temporalVision() {
    std::cout << "Test: Temporal Vision Processing... ";
    
    AtomSpace space;
    TimeServer timeServer;
    
    // Create dummy video frames
    std::vector<Tensor> frames;
    for (int i = 0; i < 3; ++i) {
        frames.push_back(torch::randn({3, 224, 224}));
    }
    
    // Note: processVideo with default detector won't add objects
    // but should not crash
    VisionProcessor::processVideo(space, frames, &timeServer);
    
    // Should complete without errors
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║            ATenSpace Vision Unit Tests                     ║\n";
    std::cout << "║                    Phase 5 Testing                         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    try {
        test_boundingBox();
        test_boundingBoxIOU();
        test_detectedObject();
        test_spatialRelation();
        test_spatialAnalysis();
        test_sceneGraph();
        test_sceneDescription();
        test_emptyScene();
        test_visualGrounding();
        test_objectDetectorInterface();
        test_visionProcessor();
        test_multimodalCaption();
        test_temporalVision();
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║             All Vision Tests Passed! ✓                     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
