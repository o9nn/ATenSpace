#include <ATen/atomspace/ATenSpace.h>
#include <ATen/atomspace/Vision.h>
#include <iostream>
#include <iomanip>

using namespace at::atomspace;

void printSeparator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

/**
 * Example 1: Object Detection Simulation
 */
void example1_objectDetection() {
    printSeparator("Example 1: Object Detection Simulation");
    
    std::cout << "Simulating object detection in an image...\n\n";
    
    // Simulate detected objects (in real scenario, from CV model)
    std::vector<DetectedObject> objects;
    
    // Create random embeddings for objects
    auto dogFeatures = torch::randn({128});
    auto catFeatures = torch::randn({128});
    auto personFeatures = torch::randn({128});
    
    objects.emplace_back("dog", BoundingBox(0.2f, 0.3f, 0.2f, 0.4f), 0.95f, dogFeatures);
    objects.emplace_back("cat", BoundingBox(0.6f, 0.5f, 0.15f, 0.3f), 0.92f, catFeatures);
    objects.emplace_back("person", BoundingBox(0.4f, 0.1f, 0.25f, 0.6f), 0.98f, personFeatures);
    
    std::cout << "Detected Objects:\n";
    for (const auto& obj : objects) {
        std::cout << "  " << std::setw(10) << std::left << obj.label
                  << " | Confidence: " << std::fixed << std::setprecision(2) << obj.confidence
                  << " | BBox: [" << obj.bbox.x << ", " << obj.bbox.y << ", "
                  << obj.bbox.width << ", " << obj.bbox.height << "]\n";
    }
}

/**
 * Example 2: Spatial Relationship Analysis
 */
void example2_spatialRelations() {
    printSeparator("Example 2: Spatial Relationship Analysis");
    
    // Create detected objects
    std::vector<DetectedObject> objects;
    objects.emplace_back("table", BoundingBox(0.1f, 0.6f, 0.8f, 0.3f), 0.95f);
    objects.emplace_back("cup", BoundingBox(0.3f, 0.5f, 0.1f, 0.15f), 0.90f);
    objects.emplace_back("book", BoundingBox(0.6f, 0.55f, 0.15f, 0.1f), 0.88f);
    objects.emplace_back("lamp", BoundingBox(0.8f, 0.2f, 0.15f, 0.3f), 0.93f);
    
    std::cout << "Detected Objects:\n";
    for (const auto& obj : objects) {
        std::cout << "  " << obj.label << " at (" << obj.bbox.centerX() 
                  << ", " << obj.bbox.centerY() << ")\n";
    }
    
    // Analyze spatial relationships
    auto relations = SpatialAnalyzer::analyzeSpatialRelations(objects);
    
    std::cout << "\nSpatial Relationships:\n";
    for (const auto& rel : relations) {
        std::cout << "  " << rel.object1 << " is " << rel.type << " " 
                  << rel.object2 << " (confidence: " << rel.confidence << ")\n";
    }
}

/**
 * Example 3: Scene Understanding and Knowledge Graph
 */
void example3_sceneUnderstanding() {
    printSeparator("Example 3: Scene Understanding and Knowledge Graph");
    
    AtomSpace space;
    
    // Simulate a scene
    std::vector<DetectedObject> objects;
    objects.emplace_back("dog", BoundingBox(0.2f, 0.5f, 0.2f, 0.3f), 0.95f, torch::randn({128}));
    objects.emplace_back("person", BoundingBox(0.6f, 0.3f, 0.25f, 0.6f), 0.98f, torch::randn({128}));
    objects.emplace_back("frisbee", BoundingBox(0.45f, 0.25f, 0.08f, 0.08f), 0.87f, torch::randn({128}));
    
    auto relations = SpatialAnalyzer::analyzeSpatialRelations(objects);
    
    std::cout << "Building knowledge graph from visual scene...\n\n";
    SceneUnderstanding::buildSceneGraph(space, objects, relations);
    
    std::cout << "Knowledge Graph Statistics:\n";
    std::cout << "  Total atoms: " << space.getSize() << "\n";
    std::cout << "  Concept nodes (objects): " 
              << space.getAtomsByType(Atom::Type::CONCEPT_NODE).size() << "\n";
    std::cout << "  Predicate nodes (spatial relations): " 
              << space.getAtomsByType(Atom::Type::PREDICATE_NODE).size() << "\n";
    std::cout << "  Evaluation links (relations): " 
              << space.getAtomsByType(Atom::Type::EVALUATION_LINK).size() << "\n";
    
    // Describe scene
    std::cout << "\nScene Description:\n  ";
    std::cout << SceneUnderstanding::describeScene(objects, relations) << "\n";
}

/**
 * Example 4: Visual Grounding
 */
void example4_visualGrounding() {
    printSeparator("Example 4: Visual Grounding - Concept Learning");
    
    AtomSpace space;
    
    std::cout << "Learning concept 'pet' from visual examples...\n\n";
    
    // Create visual examples of pets
    std::vector<DetectedObject> petExamples;
    petExamples.emplace_back("dog", BoundingBox(0.2f, 0.3f, 0.2f, 0.4f), 0.95f, torch::randn({128}));
    petExamples.emplace_back("cat", BoundingBox(0.6f, 0.5f, 0.15f, 0.3f), 0.92f, torch::randn({128}));
    petExamples.emplace_back("rabbit", BoundingBox(0.4f, 0.6f, 0.12f, 0.2f), 0.88f, torch::randn({128}));
    
    // Ground the concept in visual perception
    VisualReasoning::groundConcept(space, "pet", petExamples);
    
    std::cout << "Concept 'pet' grounded with " << petExamples.size() << " visual examples\n\n";
    
    // Show the knowledge structure
    auto petNode = space.getNode(Atom::Type::CONCEPT_NODE, "pet");
    if (petNode) {
        std::cout << "Created concept node: " << petNode->getName() << "\n";
        std::cout << "Incoming links (what is-a pet):\n";
        
        auto inheritanceLinks = space.getAtomsByType(Atom::Type::INHERITANCE_LINK);
        for (const auto& link : inheritanceLinks) {
            const auto& outgoing = link->getOutgoing();
            if (outgoing.size() == 2 && outgoing[1] == petNode) {
                std::cout << "  - " << outgoing[0]->getName() << " is a pet\n";
            }
        }
    }
}

/**
 * Example 5: Multimodal Integration
 */
void example5_multimodal() {
    printSeparator("Example 5: Multimodal Integration (Vision + Language)");
    
    // Simulate visual detection
    std::vector<DetectedObject> objects;
    objects.emplace_back("cat", BoundingBox(0.3f, 0.6f, 0.2f, 0.3f), 0.95f);
    objects.emplace_back("mat", BoundingBox(0.2f, 0.7f, 0.6f, 0.25f), 0.90f);
    
    auto relations = SpatialAnalyzer::analyzeSpatialRelations(objects);
    
    // Generate caption
    std::string caption = MultimodalIntegration::caption(objects, relations);
    
    std::cout << "Visual Scene:\n";
    for (const auto& obj : objects) {
        std::cout << "  - " << obj.label << "\n";
    }
    
    std::cout << "\nSpatial Relations:\n";
    for (const auto& rel : relations) {
        std::cout << "  - " << rel.object1 << " " << rel.type << " " << rel.object2 << "\n";
    }
    
    std::cout << "\nGenerated Caption:\n";
    std::cout << "  \"" << caption << "\"\n";
}

/**
 * Example 6: Temporal Visual Processing
 */
void example6_temporalVision() {
    printSeparator("Example 6: Temporal Visual Processing (Video)");
    
    AtomSpace space;
    TimeServer timeServer;
    
    std::cout << "Processing video frames (simulated)...\n\n";
    
    // Simulate 3 video frames
    std::vector<Tensor> frames;
    for (int i = 0; i < 3; ++i) {
        frames.push_back(torch::randn({3, 224, 224}));  // Random image tensors
    }
    
    std::cout << "Frame 0: Dog appears\n";
    std::cout << "Frame 1: Dog moves right\n";
    std::cout << "Frame 2: Dog runs\n\n";
    
    // In real scenario, would process actual video
    // For simulation, just record some temporal events
    auto dog = space.addNode(Atom::Type::CONCEPT_NODE, "dog");
    
    timeServer.recordEvent(dog, "detected_frame_0");
    timeServer.recordEvent(dog, "moved_right_frame_1");
    timeServer.recordEvent(dog, "running_frame_2");
    
    std::cout << "Temporal events recorded for object 'dog':\n";
    // Note: In real implementation, would query temporal information
    std::cout << "  - detected_frame_0\n";
    std::cout << "  - moved_right_frame_1\n";
    std::cout << "  - running_frame_2\n";
}

/**
 * Example 7: Bounding Box Operations
 */
void example7_boundingBoxOps() {
    printSeparator("Example 7: Bounding Box Operations");
    
    BoundingBox box1(0.2f, 0.3f, 0.3f, 0.4f);
    BoundingBox box2(0.3f, 0.4f, 0.3f, 0.4f);
    
    std::cout << "Box 1: [" << box1.x << ", " << box1.y << ", " 
              << box1.width << ", " << box1.height << "]\n";
    std::cout << "  Center: (" << box1.centerX() << ", " << box1.centerY() << ")\n";
    std::cout << "  Area: " << box1.area() << "\n\n";
    
    std::cout << "Box 2: [" << box2.x << ", " << box2.y << ", " 
              << box2.width << ", " << box2.height << "]\n";
    std::cout << "  Center: (" << box2.centerX() << ", " << box2.centerY() << ")\n";
    std::cout << "  Area: " << box2.area() << "\n\n";
    
    float iou = box1.iou(box2);
    std::cout << "Intersection over Union (IoU): " << iou << "\n";
    
    if (iou > 0.5f) {
        std::cout << "  -> Boxes significantly overlap\n";
    } else if (iou > 0.0f) {
        std::cout << "  -> Boxes partially overlap\n";
    } else {
        std::cout << "  -> Boxes don't overlap\n";
    }
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║             ATenSpace Visual Perception                    ║\n";
    std::cout << "║                      Phase 5 Examples                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    
    try {
        example1_objectDetection();
        example2_spatialRelations();
        example3_sceneUnderstanding();
        example4_visualGrounding();
        example5_multimodal();
        example6_temporalVision();
        example7_boundingBoxOps();
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          All Vision Examples Completed Successfully!       ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
