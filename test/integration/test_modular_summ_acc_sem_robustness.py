from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.eval_algorithms.summarization_accuracy import DEFAULT_MODEL_TYPE
from fmeval.transforms.perturbations import ButterFinger
from fmeval.transforms.summarization_accuracy import SummarizationAccuracy, METEOR_SCORE, ROUGE_SCORE, BERT_SCORE
from fmeval.transforms.util import GeneratePrompt, GetModelResponse, PromptComposer, GenerateDeltaScores, Mean, \
    shared_resource
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.eval_algorithms import DATASET_CONFIGS, XSUM
from test.integration.models.model_runners import sm_model_runner


data_config = DATASET_CONFIGS[XSUM]
ds = get_dataset(data_config, 20)
prompt_composer = PromptComposer("Summarize the following text in one sentence: $feature")

gen_og_prompt = GeneratePrompt(
    prompt_composer=prompt_composer,
    input_keys=["model_input"],
)

get_og_response = GetModelResponse(
    model_runner=sm_model_runner,
    model_input_keys=gen_og_prompt.output_keys,
    model_response_keys=["model_output"],
)

get_perturbed_inputs = ButterFinger(
    input_text_key="model_input",
    perturbation_prob=0.1,
    num_perturbations=3,
)

gen_perturbed_prompt = GeneratePrompt(
    prompt_composer=prompt_composer,
    input_keys=get_perturbed_inputs.output_keys,
)

get_perturbed_responses = GetModelResponse(
    model_runner=sm_model_runner,
    model_input_keys=gen_perturbed_prompt.output_keys,
    model_response_keys=["model_output"]
)

bertscore_model = shared_resource(BertscoreHelperModel(DEFAULT_MODEL_TYPE))
get_og_summ_acc_scores = SummarizationAccuracy(get_og_response.output_keys[0], bertscore_model)

get_perturbed_summ_acc_scores = [
    SummarizationAccuracy(
        model_output_key=perturbed_model_response_key,
        bertscore_model=bertscore_model,
        meteor_score_output_key=f"{METEOR_SCORE}({perturbed_model_response_key})",
        rouge_score_output_key=f"{ROUGE_SCORE}({perturbed_model_response_key})",
        bertscore_output_key=f"{BERT_SCORE}({perturbed_model_response_key})"
    )
    for perturbed_model_response_key in get_perturbed_responses.output_keys
]

original_score_keys = [
    transform.output_keys[0]
    for transform in get_og_summ_acc_scores.transforms.values()
]

perturbed_score_keys = {
    original_score_key: [
        summ_acc.transforms[original_score_key].output_keys[0]
        for summ_acc in get_perturbed_summ_acc_scores
    ]
    for original_score_key in original_score_keys
}

get_delta_scores = [
    GenerateDeltaScores(original_score_key, perturbed_score_keys[original_score_key])
    for original_score_key in original_score_keys
]

get_mean_delta_scores = [
    Mean(delta_score.output_keys)
    for delta_score in get_delta_scores
]

pipeline = TransformPipeline(
    [
        gen_og_prompt,
        get_og_response,
        get_perturbed_inputs,
        gen_perturbed_prompt,
        get_perturbed_responses,
        get_og_summ_acc_scores.pipeline,
        [summ_acc.pipeline for summ_acc in get_perturbed_summ_acc_scores],
        get_delta_scores,
        get_mean_delta_scores
    ]
)

ds = pipeline.execute(ds)
ds.show()

sample = ds.take(1)[0]
for key in sample:
    print(f"{key}\n")
