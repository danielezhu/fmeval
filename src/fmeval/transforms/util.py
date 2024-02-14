from typing import Tuple, List

from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform, Record


class GetModelResponse(Transform):
    _transform_name = "GetModelResponse"

    def __init__(self, model_runner: ModelRunner, model_input_keys: List[str], model_response_keys: Tuple[str]):
        self.model_runner = model_runner
        self.model_input_keys = model_input_keys
        self.model_response_keys = model_response_keys
        self.model_output_keys = {
            model_input_key: [
                self._model_output_key(model_input_key, model_response_key)
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

    def _model_output_key(self, model_input_key: str, model_response_key: str) -> str:
        return f"{self._transform_name}({model_input_key}, {model_response_key})"

    @property
    def output_keys(self):
        return [
            output_key
            for output_key_list in self.model_output_keys.values()
            for output_key in output_key_list
        ]


class GeneratePrompt(Transform):
    _transform_name = "GeneratePrompt"

    def __init__(self, prompt_composer: PromptComposer, input_keys: List[str]):
        self.prompt_composer = prompt_composer
        self.input_keys = input_keys
        self.prompt_keys = [self._prompt_key(input_key) for input_key in self.input_keys]

    def __call__(self, record: Record) -> Record:
        for input_key, prompt_key in zip(self.input_keys, self.prompt_keys):
            assert prompt_key not in record, f"Prompt key {prompt_key} already exists in record {record}"
            record[prompt_key] = self.prompt_composer.compose(record[input_key])
        return record

    def _prompt_key(self, input_key: str) -> str:
        return f"{self._transform_name}({input_key})"

    @property
    def output_keys(self):
        return self.prompt_keys


class GenerateDeltaScores(Transform):
    _transform_name = "GenerateDeltaScores"

    def __init__(self, original_score_key: str, perturbed_score_keys: List[str]):
        self.original_score_key = original_score_key
        self.perturbed_score_keys = perturbed_score_keys

    def __call__(self, record: Record) -> Record:
        for perturbed_score_key in self.perturbed_score_keys:
            record[self._delta_score_key(perturbed_score_key)] = \
                abs(record[self.original_score_key] - record[perturbed_score_key])
        return record

    def _delta_score_key(self, perturbed_score_key: str) -> str:
        return f"{self._transform_name}({self.original_score_key}, {perturbed_score_key})"

    @property
    def output_keys(self):
        return [
            self._delta_score_key(perturbed_score_key)
            for perturbed_score_key in self.perturbed_score_keys
        ]


class Mean(Transform):
    _transform_name = "Mean"

    def __init__(self, input_keys: List[str]):
        self.input_keys = input_keys
        self.output_key = f"{self._transform_name}({self.input_keys})"

    def __call__(self, record: Record) -> Record:
        avg = sum(record[input_key] for input_key in self.input_keys) / len(self.input_keys)
        record[self.output_key] = avg
        return record

    @property
    def output_keys(self):
        return [self.output_key]
