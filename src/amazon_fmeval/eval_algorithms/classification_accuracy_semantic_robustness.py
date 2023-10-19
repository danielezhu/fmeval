import logging
import warnings
from collections import defaultdict


from typing import Any, Callable, List, Optional, Dict

from dataclasses import dataclass


import amazon_fmeval.util as util
from amazon_fmeval.constants import (
    MODEL_INPUT_COLUMN_NAME,
    TARGET_OUTPUT_COLUMN_NAME,
    MEAN,
    BUTTER_FINGER,
    RANDOM_UPPER_CASE,
    WHITESPACE_ADD_REMOVE,
    PREFIX_FOR_DELTA_SCORES,
)
from amazon_fmeval.data_loaders.util import get_dataset
from amazon_fmeval.data_loaders.data_config import DataConfig
from amazon_fmeval.eval_algorithms.semantic_perturbation_utils import (
    ButterFingerConfig,
    RandomUpperCaseConfig,
    WhitespaceAddRemoveConfig,
)
from amazon_fmeval.eval_algorithms.util import (
    generate_prompt_column_for_dataset,
    aggregate_evaluation_scores,
    validate_dataset,
    save_dataset,
    generate_output_dataset_path,
    generate_mean_delta_score,
)
from amazon_fmeval.eval_algorithms.eval_algorithm import (
    EvalAlgorithmInterface,
    EvalAlgorithmConfig,
)
from amazon_fmeval.eval_algorithms import (
    EvalAlgorithm,
    EvalOutput,
    EvalScore,
    EVAL_DATASETS,
    EVAL_PROMPT_TEMPLATES,
    DATASET_CONFIGS,
)
from amazon_fmeval.exceptions import EvalAlgorithmClientError
from amazon_fmeval.model_runners.composers.composers import PromptComposer
from amazon_fmeval.model_runners.model_runner import ModelRunner
from amazon_fmeval.perf_util import timed_block
from amazon_fmeval.eval_algorithms.classification_accuracy import (
    convert_model_output_to_label,
    ClassificationAccuracy,
    ClassificationAccuracyConfig,
    CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME,
    CLASSIFICATION_ACCURACY_SCORE,
)
from amazon_fmeval.eval_algorithms.general_semantic_robustness import ButterFinger, RandomUpperCase, WhitespaceAddRemove

# All the perturbation types supported by this eval algo
PERTURBATION_TYPE_TO_HELPER_CLASS = {
    BUTTER_FINGER: ButterFinger,
    RANDOM_UPPER_CASE: RandomUpperCase,
    WHITESPACE_ADD_REMOVE: WhitespaceAddRemove,
}

PREFIX_FOR_DELTA_SCORES = "delta_"
DELTA_CLASSIFICATION_ACCURACY_SCORE = PREFIX_FOR_DELTA_SCORES + CLASSIFICATION_ACCURACY_SCORE


PROMPT_COLUMN_NAME = "prompt"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationAccuracySemanticRobustnessConfig(EvalAlgorithmConfig):
    """
    Configuration for the Classification Accuracy Semantic Robustness Evaluation

    :param valid_labels: List of valid string label
    :param converter_fn: Function to process model output to labels, defaults to simple integer conversion
    :param perturbation_type: perturbation type for generating perturbed inputs
    :param num_perturbations: Number of perturbed inputs to be generated for robustness evaluation
    :param seed: Seed to be configured for generating perturbations
    :param butter_finger_perturbation_prob: The probability that a given character will be perturbed. Used for
        butter_finger perturbation_type
    :param random_uppercase_corrupt_proportion: Fraction of characters to be changed to uppercase. Used for
        random_upper_case perturbation_type
    :param whitespace_remove_prob: Given a whitespace, remove it with this much probability. Used for
        whitespace_add_remove perturbation_type
    :param whitespace_add_prob: Given a non-whitespace, add a whitespace before it with this probability. Used for
        whitespace_add_remove perturbation_type
    """

    valid_labels: List[str]
    converter_fn: Callable[[str, List[str]], str] = convert_model_output_to_label
    perturbation_type: str = BUTTER_FINGER
    num_perturbations: int = 5
    seed: int = 5
    butter_finger_perturbation_prob: float = 0.1
    random_uppercase_corrupt_proportion: float = 0.1
    whitespace_remove_prob: float = 0.1
    whitespace_add_prob: float = 0.05

    def __post_init__(self):
        if self.perturbation_type not in PERTURBATION_TYPE_TO_HELPER_CLASS.keys():
            raise EvalAlgorithmClientError(
                f"Invalid perturbation type '{self.perturbation_type} requested, please "
                f"choose from acceptable values: {PERTURBATION_TYPE_TO_HELPER_CLASS.keys()}"
            )
        for i, label in enumerate(self.valid_labels):
            if not isinstance(label, str):
                warnings.warn("Valid labels should be strings, casting.")
                self.valid_labels[i] = str(label)


CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS = EvalAlgorithm.CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS.value


class ClassificationAccuracySemanticRobustness(EvalAlgorithmInterface):
    """
    Classification Accuracy Eval algorithm
    """

    eval_name = EvalAlgorithm.CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS.value

    def __init__(self, eval_algorithm_config: ClassificationAccuracySemanticRobustnessConfig):
        """Default constructor

        :param eval_algorithm_config: Classification Accuracy Semantic Robustness eval algorithm config.
        """
        super().__init__(eval_algorithm_config)
        self.eval_name = CLASSIFICATION_ACCURACY_SEMANTIC_ROBUSTNESS
        self._eval_algorithm_config = eval_algorithm_config

        if self._eval_algorithm_config.perturbation_type == BUTTER_FINGER:
            self._perturbation_config = ButterFingerConfig(self._eval_algorithm_config.butter_finger_perturbation_prob)
        elif self._eval_algorithm_config.perturbation_type == RANDOM_UPPER_CASE:
            self._perturbation_config = RandomUpperCaseConfig(
                self._eval_algorithm_config.random_uppercase_corrupt_proportion
            )
        else:
            self._perturbation_config = WhitespaceAddRemoveConfig(
                self._eval_algorithm_config.whitespace_remove_prob, self._eval_algorithm_config.whitespace_add_prob
            )

        self._classification_accuracy_eval_algo = ClassificationAccuracy(
            eval_algorithm_config=ClassificationAccuracyConfig(
                valid_labels=eval_algorithm_config.valid_labels, converter_fn=eval_algorithm_config.converter_fn
            )
        )

    def __reduce__(self):  # pragma: no cover
        """
        Custom serializer method used by Ray when it serializes instances of this
        class during dataset.map() operations.
        """
        serialized_data = (self._eval_algorithm_config,)
        return ClassificationAccuracySemanticRobustness, serialized_data

    def evaluate(
        self,
        model: Optional[ModelRunner],
        dataset_config: Optional[DataConfig] = None,
        prompt_template: Optional[str] = None,
        save: bool = False,
    ) -> List[EvalOutput]:
        """
        Classification Accuracy Semantic Robustness evaluate.

        :param model: An instance of ModelRunner which is the model under evaluation
        :param dataset_config: Configures the single dataset used for evaluation. If not provided,
            evaluation will use all of it's supported built-in datasets
        :param prompt_template: A template which can be used to compose prompt using model_input
        :param save: If set to true, prompt responses and scores will be saved to file. The output is written to
                     EvalAlgorithmInterface.EVAL_RESULTS_PATH
        :returns: A List of EvalOutput objects.
        """
        util.require(
            model,
            "Missing required input: model i.e. ModelRunner, for ClassificationAccuracySemanticRobustness evaluate",
        )
        is_custom_dataset_evaluation = False
        if dataset_config:
            is_custom_dataset_evaluation = True
            dataset_configs = [dataset_config]
        else:
            dataset_configs = [DATASET_CONFIGS[dataset_name] for dataset_name in EVAL_DATASETS[self.eval_name]]

        eval_outputs: List[EvalOutput] = []
        for dataset_config in dataset_configs:
            dataset = get_dataset(dataset_config)
            validate_dataset(dataset, [TARGET_OUTPUT_COLUMN_NAME, MODEL_INPUT_COLUMN_NAME])
            if is_custom_dataset_evaluation:
                # TODO when user provide built-in DataConfig, we should provide default prompt_template
                util.require(
                    prompt_template is not None,
                    f"Missing required input: prompt_template for evaluating custom dataset : {dataset_config}",
                )
            else:
                prompt_template = EVAL_PROMPT_TEMPLATES[self.eval_name, dataset_config.dataset_name]
                util.assert_condition(
                    prompt_template is not None,
                    f"No Prompt Template configured for ({self.eval_name}, {dataset_config.dataset_name})",
                )
            assert prompt_template  # to satisfy mypy
            dataset = generate_prompt_column_for_dataset(
                prompt_template=prompt_template,
                data=dataset,
                model_input_column_name=MODEL_INPUT_COLUMN_NAME,
                prompt_column_name=PROMPT_COLUMN_NAME,
            )
            with timed_block(f"Computing score and aggregation on dataset {dataset_config.dataset_name}", logger):

                def _generate_score_columns(row: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
                    scores = self.evaluate_sample(
                        row[MODEL_INPUT_COLUMN_NAME], model, row[TARGET_OUTPUT_COLUMN_NAME], prompt_template
                    )
                    for score in scores:
                        row[score.name] = score.value
                    return row

                dataset = dataset.map(_generate_score_columns).materialize()

                dataset_scores, category_scores = aggregate_evaluation_scores(
                    dataset, [DELTA_CLASSIFICATION_ACCURACY_SCORE], agg_method=MEAN
                )

                eval_outputs.append(
                    EvalOutput(
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                        prompt_template=prompt_template,
                        dataset_scores=dataset_scores,
                        category_scores=category_scores,
                        output_path=self._eval_results_path,
                    )
                )
            if save:
                save_dataset(
                    dataset=dataset,
                    score_names=[CLASSIFIED_MODEL_OUTPUT_COLUMN_NAME],
                    path=generate_output_dataset_path(
                        path_to_parent_dir=self._eval_results_path,
                        eval_name=self.eval_name,
                        dataset_name=dataset_config.dataset_name,
                    ),
                )

        return eval_outputs

    def evaluate_sample(self, model_input: str, model: ModelRunner, target_output: str, prompt_template: str = "$feature") -> List[EvalScore]:  # type: ignore[override]
        """
        Evaluate a single record for Classification Accuracy Semantic Robustness.

        :param model_input: text input for model
        :param model: An instance of ModelRunner which is the model under evaluation
        :param target_output: The expected responses from the model
        :param prompt_template: A template which can be used to compose prompt using model_input
        :returns: A List of EvalScores computed for prompts and responses.
        """
        util.require(
            model_input,
            "Missing required input: model_input, for ClassificationAccuracySemanticRobustness evaluate_sample",
        )
        util.require(
            model,
            "Missing required input: model i.e. ModelRunner, for ClassificationAccuracySemanticRobustness evaluate_sample",
        )
        util.require(
            target_output,
            "Missing required input: target_output, for " "ClassificationAccuracySemanticRobustness evaluate_sample",
        )

        prompt_composer = PromptComposer(prompt_template)
        original_prompt = prompt_composer.compose(model_input)
        original_model_output = model.predict(original_prompt)[0]

        # Check if predictor is deterministic
        if model.predict(original_prompt)[0] != original_model_output:
            raise EvalAlgorithmClientError("For evaluating semantic robustness, the model should be deterministic.")

        perturbation = PERTURBATION_TYPE_TO_HELPER_CLASS[self._eval_algorithm_config.perturbation_type](
            seed=self._eval_algorithm_config.seed
        )

        perturbed_inputs = perturbation.perturb(
            text=model_input,
            config=self._perturbation_config,
            num_perturbations=self._eval_algorithm_config.num_perturbations,
        )

        perturbed_input_prompts = [prompt_composer.compose(perturbed_input) for perturbed_input in perturbed_inputs]
        perturbed_input_outputs = [model.predict(prompt)[0] for prompt in perturbed_input_prompts]

        original_classification_accuracy_scores = self._classification_accuracy_eval_algo.evaluate_sample(
            target_output=target_output, model_output=original_model_output
        )

        perturbed_outputs_classification_accuracy_scores = defaultdict(lambda: [])
        for perturbed_input_output in perturbed_input_outputs:
            accuracy_scores = self._classification_accuracy_eval_algo.evaluate_sample(
                target_output=target_output, model_output=perturbed_input_output
            )
            for accuracy_score in accuracy_scores:
                perturbed_outputs_classification_accuracy_scores[accuracy_score.name].append(accuracy_score)

        return [
            EvalScore(
                name=PREFIX_FOR_DELTA_SCORES + original_score.name,
                value=generate_mean_delta_score(
                    original_score, perturbed_outputs_classification_accuracy_scores[original_score.name]
                ),
            )
            for original_score in original_classification_accuracy_scores
        ]