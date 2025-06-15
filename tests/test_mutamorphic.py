"""
Contains mutamorphic tests for the model.
"""
from restaurant_model_training.modeling import predict

# mutamorphic pairs: semantically similar but not syntactically equivalent
mutamorphic_pairs = [
    ("The food was okay.", "The food was fine."),
    ("Great service!", "Excellent service!"),
    ("The ambiance was nice.", "The atmosphere was pleasant."),
    ("The food was pretty good.", "It was pretty decent."),
    ("The experience was poor.", "It was a bad experience."),
]

def test_mutamorphic_equivalence(model_setup):
    """Test that mutamorphic pairs produce the same predictions"""
    _, _, _, model_p, bow_p = model_setup
    vectorizer, classifier = predict.load_models(bow_p, model_p)

    # for each pair, check if predictions are the same
    for original, variant in mutamorphic_pairs:
        preds = predict.predict([original, variant], vectorizer, classifier)
        assert preds[0] == preds[1], (
            f"Mutamorphic pair diverged: '{original}' => {preds[0]}, '{variant}' => {preds[1]}"
        )
