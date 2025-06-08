import psutil
import time

from restaurant_model_training.modeling import predict

# Test cost of feature generation (latency and memory usage)
def test_feature_generation_cost(model_setup):
    """ test that feature generation does not exceed set memory and latency constraints"""
    # get data + temp paths
    vectorizer, _ = model_setup

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

# DATA SLICE TESTS (from lectures)

# define sentiment data slices
positive_words = ["excellent", "amazing", "great", "delicious", "fantastic", "perfect", "awesome"]
negative_words = ["awful", "terrible", "bad", "disgusting", "worst", "horrible", "poor"]

def test_positive_sentiment_slice(model_setup):
    """Test that positive sentiment words are generally classified as positive (1)"""
    vectorizer, classifier = model_setup
    preds = predict.predict(positive_words, vectorizer, classifier)

    # expect these words to be classified as positive (class 1)
    positive_count = sum(preds)
    assert positive_count >= int(len(positive_words) * 0.8), \
        f"Expected >= 80% positive classifications, got {positive_count}/{len(positive_words)}"

def test_negative_sentiment_slice(model_setup):
    """Test that negative sentiment words are generally classified as negative (0)"""
    vectorizer, classifier = model_setup
    preds = predict.predict(negative_words, vectorizer, classifier)

    # expect these words to be classified as negative (class 0)
    negative_count = sum([1 for p in preds if p == 0])
    assert negative_count >= int(len(negative_words) * 0.8), \
        f"Expected >= 80% negative classifications, got {negative_count}/{len(negative_words)}"