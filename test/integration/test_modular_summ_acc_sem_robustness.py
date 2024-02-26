from fmeval.data_loaders.util import get_dataset
from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.eval_algorithms.summarization_accuracy import DEFAULT_MODEL_TYPE
from fmeval.transforms.perturbations import ButterFinger
from fmeval.transforms.summarization_accuracy import SummarizationAccuracy, METEOR_SCORE, ROUGE_SCORE, BERT_SCORE, \
    MeteorScore, RougeScore, BertScore
from fmeval.transforms.util import GeneratePrompt, GetModelResponse, GenerateDeltaScores, Mean, \
    shared_resource, create_output_key
from fmeval.transforms.transform_pipeline import TransformPipeline
from fmeval.eval_algorithms import DATASET_CONFIGS, XSUM
from test.integration.models.model_runners import sm_model_runner


data_config = DATASET_CONFIGS[XSUM]
ds = get_dataset(data_config, 20)
prompt_template = "Summarize the following text in one sentence: $feature"

gen_og_prompt = GeneratePrompt(
    input_keys=["model_input"],
    output_keys=["prompt"],
    prompt_template="Summarize the following text in one sentence: $feature",
)

get_og_response = GetModelResponse(
    input_keys=gen_og_prompt.output_keys,
    output_keys=["model_output"],
    model_runner=sm_model_runner,
)

perturbation_prob = 0.1
num_perturbations = 3
get_perturbed_inputs = ButterFinger(
    input_keys=["model_input"],
    output_keys=[
        create_output_key(ButterFinger.__name__, "model_input", i)
        for i in range(num_perturbations)
    ],
    perturbation_prob=perturbation_prob,
    num_perturbations=num_perturbations,
)

gen_perturbed_prompt = GeneratePrompt(
    input_keys=get_perturbed_inputs.output_keys,
    output_keys=[
        create_output_key(GeneratePrompt.__name__, perturbed_input_key)
        for perturbed_input_key in get_perturbed_inputs.output_keys
    ],
    prompt_template=prompt_template,
)

get_perturbed_responses = [
    GetModelResponse(
        input_keys=[perturbed_prompt_key],
        output_keys=[create_output_key(GetModelResponse.__name__, perturbed_prompt_key)],
        model_runner=sm_model_runner,
    )
    for perturbed_prompt_key in gen_perturbed_prompt.output_keys
]


bertscore_model = shared_resource(BertscoreHelperModel(DEFAULT_MODEL_TYPE))
og_summ_acc = SummarizationAccuracy(
    target_output_key="target_output",
    model_output_key=get_og_response.output_keys[0],
    bertscore_model=bertscore_model,
)

perturbed_summ_accs = [
    SummarizationAccuracy(
        target_output_key="target_output",
        model_output_key=get_perturbed_response.output_keys[0],
        bertscore_model=bertscore_model,
        meteor_score_output_key=create_output_key(
            MeteorScore.__name__,
            get_perturbed_response.output_keys[0]
        ),
        rouge_score_output_key=create_output_key(
            RougeScore.__name__,
            get_perturbed_response.output_keys[0]
        ),
        bert_score_output_key=create_output_key(
            BertScore.__name__,
            get_perturbed_response.output_keys[0]
        ),
    )
    for get_perturbed_response in get_perturbed_responses
]

SUMM_ACC_SCORE_NAMES = {METEOR_SCORE, ROUGE_SCORE, BERT_SCORE}

perturbed_score_keys = {
    score_name: [
        summ_acc.transforms[score_name].output_keys[0]
        for summ_acc in perturbed_summ_accs
    ]
    for score_name in SUMM_ACC_SCORE_NAMES
}

get_delta_scores = [
    GenerateDeltaScores(
        input_keys=[original_score_key],
        output_keys=perturbed_score_keys[original_score_key]
    )
    for original_score_key in SUMM_ACC_SCORE_NAMES
]

get_mean_delta_scores = [
    Mean(
        input_keys=delta_score.output_keys,
        output_keys=[create_output_key(Mean.__name__, delta_score.input_keys[0])],
    )
    for delta_score in get_delta_scores
]

pipeline = TransformPipeline(
    [
        gen_og_prompt,
        get_og_response,
        get_perturbed_inputs,
        gen_perturbed_prompt,
        get_perturbed_responses,
        og_summ_acc.pipeline,
        [summ_acc.pipeline for summ_acc in perturbed_summ_accs],
        get_delta_scores,
        get_mean_delta_scores
    ]
)

ds = pipeline.execute(ds)
ds.show()

sample = ds.take(1)[0]
for key in sample:
    print(f"{key}\n")
