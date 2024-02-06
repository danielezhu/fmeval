from test.integration.models.model_runners import sm_model_runner
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.model_runners.composers.composers import PromptComposer
from typing import Dict, Any, List, Tuple, Callable, Union, Concatenate
from abc import ABC, abstractmethod
import ray
import ray.data
import inspect

Record = Dict[str, Any]


# RecordTransform = Callable[[Record, str, List[str]], Record]

class Transform:
    @abstractmethod
    def __init__(self, **kwargs):
        """"""""

    @abstractmethod
    def __call__(self, record: Record) -> Record:
        """"""


class GetModelResponse(Transform):
    def __init__(self, model_runner: ModelRunner, input_col: str, output_cols: Tuple[str]):
        self.model_runner = model_runner
        self.input_col = input_col
        self.output_cols = output_cols

    def __call__(self, record: Record) -> Record:
        model_response = self.model_runner.predict(record[self.input_col])
        # We should add validation that the number of output cols matches the number of fields
        # in the ModelResponse object that is the output of self.model_runner.predict()
        for model_response_item, output_col_name in zip(model_response, self.output_cols):
            record[output_col_name] = model_response_item
        return record


class GeneratePrompt(Transform):
    def __init__(self, prompt_composer: PromptComposer, model_input_key: str):
        self.prompt_composer = prompt_composer
        self.model_input_key = model_input_key

    def __call__(self, record: Record) -> Record:
        record["prompt"] = self.prompt_composer.compose(record[self.model_input_key])
        return record


class TransformPipeline:
    def __init__(self, pipeline: List[Transform]):
        self.pipeline = pipeline

    def execute(self, dataset: ray.data.Dataset):
        for transform in self.pipeline:
            init_arg_names = inspect.signature(transform.__init__).parameters.keys()
            dataset = dataset.map(
                transform.__class__,
                fn_constructor_kwargs={k: v for k, v in transform.__dict__.items() if k in init_arg_names},
                compute=ray.data.ActorPoolStrategy(size=2)
            )
        return dataset


generate_prompt = GeneratePrompt(prompt_composer=PromptComposer("Answer the following question: $feature"), model_input_key="model_input")
get_model_response = GetModelResponse(model_runner=sm_model_runner, input_col="prompt", output_cols=["model_output, log_prob"])
pipeline = TransformPipeline([generate_prompt, get_model_response])

ds = ray.data.from_items([
    {"model_input": "The capital of England is", "untouched": "blah"},
    {"model_input": "The capital of the USA is", "untouched": "blah blah"},
    {"model_input": "Red and blue combined make", "untouched": "blah blah blah"},
])

ds = pipeline.execute(ds)

ds.show()
