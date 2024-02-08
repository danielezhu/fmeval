from typing import Tuple, List

from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform, Record


class GetModelResponse(Transform):
    def __init__(self, model_runner: ModelRunner, model_input_keys: List[str], model_response_keys: Tuple[str]):
        self.model_runner = model_runner
        self.model_input_keys = model_input_keys
        self.model_response_keys = model_response_keys
        self.model_output_keys = {
            model_input_key: [
                f"[{model_input_key}]_[{model_response_key}]"
                for model_response_key in self.model_response_keys
            ]
            for model_input_key in self.model_input_keys
        }

    def __call__(self, record: Record) -> Record:
        for model_input_key in self.model_input_keys:
            model_response = self.model_runner.predict(record[model_input_key])
            # We should add validation that the number of output cols matches the number of fields
            # in the ModelResponse object that is the output of self.model_runner.predict()
            for model_response_item, model_output_key in zip(model_response, self.model_output_keys[model_input_key]):
                record[model_output_key] = model_response_item
        return record


class GeneratePrompt(Transform):
    def __init__(self, prompt_composer: PromptComposer, input_keys: List[str]):
        self.prompt_composer = prompt_composer
        self.input_keys = input_keys
        self.prompt_keys = [f"{input_key}_prompt" for input_key in self.input_keys]

    def __call__(self, record: Record) -> Record:
        for input_key, prompt_key in zip(self.input_keys, self.prompt_keys):
            assert prompt_key not in record, f"Prompt key {prompt_key} already exists in record {record}"
            record[prompt_key] = self.prompt_composer.compose(record[input_key])
        return record
