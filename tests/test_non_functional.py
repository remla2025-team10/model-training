import psutil
import time

from restaurant_model_training.modeling import predict

# Test cost of feature generation (latency and memory usage)
def test_feature_generation_cost(model_setup):
    """ test that feature generation does not exceed set memory and latency constraints"""
    # get data + temp paths
    features, labels, model, model_p, bow_p = model_setup
    vectorizer, classifier = predict.load_models(bow_p, model_p)

    # measure memory usage and latency
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # start feature generation and timer
    start_time = time.time()
    _ = vectorizer.transform(["example review text"])
    end_time = time.time()

    # calculate memory usage and latency
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    latency = end_time - start_time

    # check if memory usage is below threshold
    assert memory_used < 200 * 1024 * 1024, f"Memory usage exceeded 200MB! Used: {memory_used / (1024 * 1024):.2f} MB"

    # check if latency is below threshold
    assert latency < 5, f"Feature generation took too long! Latency: {latency:.2f} seconds"