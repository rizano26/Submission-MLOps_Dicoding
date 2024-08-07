"""
This module defines the tuner function for hyperparameter tuning using Keras Tuner
and TensorFlow Extended (TFX). It builds and tunes a text classification model.
"""

import kerastuner as kt
import tensorflow as tf
from tfx.v1.components import TunerFnResult
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow_transform as tft

LABEL_KEY = "Category"
FEATURE_KEY = "Message"

def transformed_name(key):
    """Renaming transformed features."""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64) -> tf.data.Dataset:
    """Get post_transform feature & create batches of data."""
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))
    return dataset

def model_builder(hp, vectorize_layer):
    """Build machine learning model for hyperparameter tuning."""
    vocab_size = 10000
    embedding_dim = 16

    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Hyperparameter tuning for learning rate
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(hp_learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Tune the hyperparameters for the text classification model."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    # Create and adapt the TextVectorization layer
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize="lower_and_strip_punctuation",
        max_tokens=10000,
        output_mode='int',
        output_sequence_length=100
    )
    # Adapt the layer to the training dataset
    text_ds = train_set.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
    vectorize_layer.adapt(text_ds)

    tuner = kt.Hyperband(
        lambda hp: model_builder(hp, vectorize_layer),
        objective='val_binary_accuracy',
        max_epochs=10,
        hyperband_iterations=2
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": val_set,
            "epochs": 10
        }
    )
