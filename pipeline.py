"""
This module defines a TFX pipeline for spam and non-spam email classification.
It uses BeamDagRunner for orchestration and various TFX components for data processing
and model training.

The pipeline is configured with paths for data, transformation, tuning, training,
and model serving. It utilizes a SQLite metadata store. test
"""

import os
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration.pipeline import Pipeline
from tfx.orchestration import metadata
from modules.components import init_components

# Directory kerja saat ini
CWD = os.getcwd()


def create_pipeline(
        name: str,
        config: dict) -> Pipeline:
    """Create TFX pipeline."""
    components = init_components(
        data_dir=config['data_dir'],
        transform_module=config['transform_mod'],
        tuner_module=config['tuner_mod'],
        training_module=config['training_mod']
    )

    return Pipeline(
        pipeline_name=name,
        pipeline_root=config['root'],
        components=components,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            config['metadata_db_path'])
    )


if __name__ == '__main__':
    # Define paths
    PIPELINE_NAME = "wnefst26_pipeline"
    PIPELINE_ROOT = os.path.join(CWD, PIPELINE_NAME)
    DATA_ROOT = os.path.join(CWD, 'dataset')
    TRANSFORM_MODULE = os.path.join(
        CWD, 'modules', 'spam_nonspam_transform.py')
    TUNER_MODULE = os.path.join(CWD, 'modules', 'spam_nonspam_tuner.py')
    TRAINING_MODULE = os.path.join(
        CWD, 'modules', 'spam_nonspam_trainer_withtuner.py')
    METADATA_PATH = os.path.join(CWD, PIPELINE_NAME, 'metadata.db')

    # Create configuration dictionary
    pipeline_config = {
        'root': PIPELINE_ROOT,
        'data_dir': DATA_ROOT,
        'transform_mod': TRANSFORM_MODULE,
        'tuner_mod': TUNER_MODULE,
        'training_mod': TRAINING_MODULE,
        'metadata_db_path': METADATA_PATH
    }

    # Create and run the pipeline
    BeamDagRunner().run(
        create_pipeline(
            name=PIPELINE_NAME,
            config=pipeline_config))
