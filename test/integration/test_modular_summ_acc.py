from fmeval.data_loaders.data_config import DataConfig
from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.transforms.summarization_accuracy import SummarizationAccuracy, SummarizationAccuracyConfig
from fmeval.transforms.util import GeneratePrompt, GetModelResponse, PromptComposer
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
    model_input_keys=gen_prompt.prompt_keys,
    model_response_keys=["model_output"],
)

config = SummarizationAccuracyConfig(
    model_output_key=get_model_response.model_output_keys[get_model_response.model_input_keys[0]][0]
)
bertscore_model = BertscoreHelperModel.remote(
    model_type=config.bertscore_model_type
)
summ_acc = SummarizationAccuracy(config, bertscore_model)

pipeline = TransformPipeline([gen_prompt, get_model_response, summ_acc])
ds = pipeline.execute(ds)
ds.show()
