import ray
from typing import Tuple, List, Dict, Any
from ray import ObjectRef

from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.model_runners.model_runner import ModelRunner
from fmeval.transforms.transform import Transform, Record


def shared_resource(resource: object, num_cpus: int = 1) -> ObjectRef:
    """
    Creates a Ray Actor out of `resource`.

    :param resource: The object to be converted into a Ray Actor.
        This should be some shared resource like a BertscoreHelperModel instance.
    :param num_cpus: The num_cpus parameter to pass to ray.remote()
    :returns: A Ray Actor handle.
    """
    resource_cls, serialized_data = resource.__reduce__()
    wrapped_resource_cls = ray.remote(num_cpus=num_cpus)(resource_cls)
    return wrapped_resource_cls.remote(*serialized_data)


def create_output_key(transform_name: str, *args, **kwargs) -> str:
    def args_to_str(positional_args: Tuple[str]) -> str:
        return ", ".join(str(arg) for arg in positional_args)

    def kwargs_to_str(keyword_args: Dict[str, Any]) -> str:
        return ", ".join(f"{k}={str(v)}" for k, v in keyword_args.items())

    args_string = args_to_str(args)
    kwargs_string = kwargs_to_str(kwargs)
    output_key = f"{transform_name}" \
                 f"({args_string if args_string else ''}" \
                 f"{', ' if args_string and kwargs_string else ''}" \
                 f"{kwargs_string if kwargs_string else ''})"
    return output_key


class GetModelResponse(Transform):

    def __init__(
            self,
            input_keys: List[str],
            output_keys: List[str],
            model_runner: ModelRunner,
    ):
        assert len(input_keys) == 1, "GetModelResponse transform takes a single input key."
        super().__init__(input_keys, output_keys, model_runner)
        self.model_runner = model_runner

    def __call__(self, record: Record) -> Record:
        model_input_key = self.input_keys[0]
        model_response = self.model_runner.predict(record[model_input_key])
        # Before constructing a GetModelResponse instance, we should validate that
        # the number of output keys matches the size of the model response payload.
        # Ex: if the payload consists of a model_output and log_prob, there should
        # be two output keys. Note that the order of the values in self.output_keys
        # matters here.
        for model_response_item, model_output_key in zip(model_response, self.output_keys):
            record[model_output_key] = model_response_item
        return record


class GeneratePrompt(Transform):

    def __init__(self, input_keys: List[str], output_keys: List[str], prompt_template: str):
        super().__init__(input_keys, output_keys, prompt_template)
        self.prompt_composer = PromptComposer(prompt_template)

    def __call__(self, record: Record) -> Record:
        for input_key, prompt_key in zip(self.input_keys, self.output_keys):
            record[prompt_key] = self.prompt_composer.compose(record[input_key])
        return record


class GenerateDeltaScores(Transform):

    def __init__(self, input_keys: List[str], output_keys: List[str]):
        assert len(input_keys) == 1, "GenerateDeltaScores transform takes a single input key."
        super().__init__(input_keys, output_keys)

    def __call__(self, record: Record) -> Record:
        original_score_key = self.input_keys[0]
        for perturbed_score_key in self.output_keys:
            record[perturbed_score_key] = abs(record[original_score_key] - record[perturbed_score_key])
        return record


class Mean(Transform):

    def __init__(self, input_keys: List[str], output_keys: List[str]):
        super().__init__(input_keys, output_keys)

    def __call__(self, record: Record) -> Record:
        avg = sum(record[input_key] for input_key in self.input_keys) / len(self.input_keys)
        output_key = self.output_keys[0]
        record[output_key] = avg
        return record
