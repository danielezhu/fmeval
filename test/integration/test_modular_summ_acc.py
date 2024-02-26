from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.transforms.summarization_accuracy import SummarizationAccuracy, DEFAULT_MODEL_TYPE
from fmeval.transforms.util import GeneratePrompt, GetModelResponse, shared_resource, create_output_key
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.eval_algorithms import DATASET_CONFIGS, XSUM
from test.integration.models.model_runners import sm_model_runner


data_config = DATASET_CONFIGS[XSUM]
ds = get_dataset(data_config, 20)

gen_prompt = GeneratePrompt(
    input_keys=["model_input"],
    output_keys=["prompt"],
    prompt_template="Summarize the following text in one sentence: $feature",
)

get_model_response = GetModelResponse(
    input_keys=gen_prompt.output_keys,
    output_keys=["model_output"],
    model_runner=sm_model_runner,
)

bertscore_model = shared_resource(BertscoreHelperModel(DEFAULT_MODEL_TYPE))
summ_acc = SummarizationAccuracy(
    target_output_key="target_output",
    model_output_key=get_model_response.output_keys[0],
    bertscore_model=bertscore_model
)

pipeline = TransformPipeline([gen_prompt, get_model_response, summ_acc.pipeline])
ds = pipeline.execute(ds)
ds.show()

sample = ds.take(1)[0]
for key in sample:
    print(f"{key}\n")
