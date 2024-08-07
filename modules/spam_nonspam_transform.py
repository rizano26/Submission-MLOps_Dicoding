"""
This module defines the preprocessing function for transforming raw input features
into features suitable for machine learning model training.
"""

import tensorflow as tf

LABEL_KEY = "Category"
FEATURE_KEY = "Message"

def transformed_name(key):
    """Renaming transformed features."""
    return key + "_xf"

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features.

    Args:
        inputs: map from feature keys to raw features.

    Returns:
        outputs: map from feature keys to transformed features.
    """
    outputs = {}

    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
