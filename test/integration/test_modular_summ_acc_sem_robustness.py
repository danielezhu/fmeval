from fmeval.data_loaders.data_config import DataConfig
from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.transforms.perturbations import ButterFinger
from fmeval.transforms.summarization_accuracy import SummarizationAccuracy, SummarizationAccuracyConfig
from fmeval.transforms.util import GeneratePrompt, GetModelResponse, PromptComposer
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.eval_algorithms import DATASET_CONFIGS, XSUM
from test.integration.models.model_runners import sm_model_runner


data_config = DATASET_CONFIGS[XSUM]
ds = get_dataset(data_config, 100)
prompt_composer = PromptComposer("Summarize the following text in one sentence: $feature")

gen_og_prompt = GeneratePrompt(
    prompt_composer=prompt_composer,
    input_keys=["model_input"],
)

get_og_response = GetModelResponse(
    model_runner=sm_model_runner,
    model_input_keys=gen_og_prompt.prompt_keys,
    model_response_keys=["model_output"],
)

get_perturbed_inputs = ButterFinger(
    input_text_key="model_input",
    perturbation_prob=0.1,
    num_perturbations=3,
)

gen_perturbed_prompt = GeneratePrompt(
    prompt_composer=prompt_composer,
    input_keys=get_perturbed_inputs.perturbed_text_keys
)

get_perturbed_responses = GetModelResponse(
    model_runner=sm_model_runner,
    model_input_keys=gen_perturbed_prompt.prompt_keys,
    model_response_keys=["model_output"]
)

summ_acc_config = SummarizationAccuracyConfig(
    target_output_key="target_output",
    model_output_key=get_og_response.model_output_keys[get_og_response.model_input_keys[0]][0]
)
bertscore_model = BertscoreHelperModel.remote(
    model_type=summ_acc_config.bertscore_model_type
)
get_og_summ_acc_scores = SummarizationAccuracy(summ_acc_config, bertscore_model)

pipeline = TransformPipeline(
    [
        gen_og_prompt,
        get_og_response,
        get_perturbed_inputs,
        gen_perturbed_prompt,
        get_perturbed_responses,
        get_og_summ_acc_scores,
    ]
)
ds = pipeline.execute(ds)
ds.show()
