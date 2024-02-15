from typing import List
from fmeval.model_runners.composers.composers import PromptComposer
from fmeval.transforms.util import generate_prompt, get_model_response
from test.integration.models.model_runners import sm_model_runner
from fmeval.transforms.util import TransformFunction
import ray.data


ds = ray.data.from_items(
    [
        {"model_input": "What year did Obama first become president?"},
        {"model_input": "What color do you get when combining red and blue?"},
        {"model_input": "How many meters are in a mile?"}
    ]
)

prompt_composer = PromptComposer("Answer the following question: $feature")

pipeline = [
    generate_prompt(prompt_composer, ["model_input"]),
    get_model_response(sm_model_runner, ["generate_prompt(model_input)"], ["model_output"])
]

# Without the register_transform decorator (possibly less confusing to the user):
# pipeline = [
#     TransformFunction(generate_prompt, prompt_composer, ["model_input"]),
#     TransformFunction(get_model_response, sm_model_runner, ["generate_prompt(model_input)"], ["model_output"])
# ]


def execute(dataset: ray.data.Dataset, transform_pipeline: List[TransformFunction]) -> ray.data.Dataset:
    for transform in transform_pipeline:
        dataset = dataset.map(transform.f, fn_args=transform.args, fn_kwargs=transform.kwargs)
    return dataset


ds = execute(ds, pipeline)
ds.show()


# ds = ds.map(
#     generate_prompt,
#     fn_args=[prompt_composer, ["model_input"]],
# )
# ds = ds.map(
#     get_model_response,
#     fn_args=[sm_model_runner, ["generate_prompt(model_input)"], ["model_output"]]
# )
