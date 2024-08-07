"""This module initializes TFX pipeline components."""

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)

def create_example_gen(data_dir):
    """Create ExampleGen component."""
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2)
        ])
    )
    return CsvExampleGen(input_base=data_dir, output_config=output)

def create_statistics_gen(example_gen):
    """Create StatisticsGen component."""
    return StatisticsGen(examples=example_gen.outputs["examples"])

def create_schema_gen(statistics_gen):
    """Create SchemaGen component."""
    return SchemaGen(statistics=statistics_gen.outputs["statistics"])

def create_example_validator(statistics_gen, schema_gen):
    """Create ExampleValidator component."""
    return ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

def create_transform(example_gen, schema_gen, transform_module):
    """Create Transform component."""
    return Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module
    )

def create_tuner(transform, tuner_module):
    """Create Tuner component."""
    return Tuner(
        module_file=tuner_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'])
    )

def create_trainer(transform, schema_gen, tuner, training_module):
    """Create Trainer component."""
    return Trainer(
        module_file=training_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(splits=['train']),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'])
    )

def create_model_resolver():
    """Create model resolver component."""
    return Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('Latest_blessed_model_resolver')

def create_eval_config():
    """Create evaluation configuration."""
    return tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='Category')],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='F1Score'),
                tfma.MetricConfig(class_name='FalsePositives'),
                tfma.MetricConfig(class_name='TruePositives'),
                tfma.MetricConfig(class_name='FalseNegatives'),
                tfma.MetricConfig(class_name='TrueNegatives'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.5}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': 0.0001}
                        )
                    )
                )
            ])
        ]
    )

def create_evaluator(example_gen, trainer, model_resolver):
    """Create Evaluator component."""
    eval_config = create_eval_config()
    return Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config
    )

def create_pusher(trainer, evaluator):
    """Create Pusher component."""
    return Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory='serving_model_dir/spam-detection-model'
            )
        )
    )

def init_components(data_dir, transform_module, tuner_module, training_module):
    """Initiate TFX pipeline components.

    Args:
        data_dir (str): a path to the data
        transform_module (str): a path to the transform_module
        training_module (str): a path to the training module

    Returns:
        TFX components
    """
    example_gen = create_example_gen(data_dir)
    statistics_gen = create_statistics_gen(example_gen)
    schema_gen = create_schema_gen(statistics_gen)
    example_validator = create_example_validator(statistics_gen, schema_gen)
    transform = create_transform(example_gen, schema_gen, transform_module)
    tuner = create_tuner(transform, tuner_module)
    trainer = create_trainer(transform, schema_gen, tuner, training_module)
    evaluator = create_evaluator(example_gen, trainer, create_model_resolver())
    pusher = create_pusher(trainer, evaluator)

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        create_model_resolver(),
        evaluator,
        pusher
    ]

    return components
