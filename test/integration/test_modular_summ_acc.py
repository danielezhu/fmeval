from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.transforms.summarization_accuracy import SummarizationAccuracy, DEFAULT_MODEL_TYPE
from fmeval.transforms.util import GeneratePrompt, GetModelResponse, PromptComposer, shared_resource
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.eval_algorithms import DATASET_CONFIGS, XSUM
from test.integration.models.model_runners import sm_model_runner


data_config = DATASET_CONFIGS[XSUM]
ds = get_dataset(data_config, 20)

gen_prompt = GeneratePrompt(
    prompt_composer=PromptComposer("Summarize the following text in one sentence: $feature"),
    input_keys=["model_input"],
)

get_model_response = GetModelResponse(
    model_runner=sm_model_runner,
    model_input_keys=gen_prompt.output_keys,
    model_response_keys=["model_output"],
)

bertscore_model = shared_resource(BertscoreHelperModel(DEFAULT_MODEL_TYPE))
summ_acc = SummarizationAccuracy(get_model_response.output_keys[0], bertscore_model)

pipeline = TransformPipeline([gen_prompt, get_model_response, summ_acc.pipeline])
ds = pipeline.execute(ds)
ds.show()
