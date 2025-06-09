import psutil
import time

from restaurant_model_training.modeling import predict

def measure_performance(action, memory_threshold=200 * 1024 * 1024, latency_threshold=5):
    """Helper function for memory and latency tests"""
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    start_time = time.time()
    action()
    end_time = time.time()

    # memory and latency measurements
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    latency = end_time - start_time

    assert memory_used < memory_threshold, f"Memory usage exceeded {memory_threshold / (1024 * 1024):.2f} MB! Used: {memory_used / (1024 * 1024):.2f} MB"
    assert latency < latency_threshold, f"Action took too long! Latency: {latency:.2f} seconds"

def test_feature_generation_cost(model_setup):
    """Test cost of feature generation"""
    # get data + temp paths
    _, _, _, model_p, bow_p = model_setup
    vectorizer, _ = predict.load_models(bow_p, model_p)

    def feature_generation_action():
        _ = vectorizer.transform(["example review text"])

    # measure performance
    try:
        measure_performance(feature_generation_action)
    except AssertionError as e:
        print(f"Feature generation cost test failed: {e}")
        raise

def test_model_prediction_cost(model_setup):
    """Test cost of model prediction"""
    # get data + temp paths
    _, _, _, model_p, bow_p = model_setup
    vectorizer, classifier = predict.load_models(bow_p, model_p)

    test_reviews = ["Wow... Loved this place.", "Crust is not good.",
                    "Not tasty and the texture was wrong."]
    
    def model_prediction_action():
        _ = predict.predict(test_reviews, vectorizer, classifier)

    try:
        measure_performance(model_prediction_action)
    except AssertionError as e:
        print(f"Model prediction cost test failed: {e}")
        raise