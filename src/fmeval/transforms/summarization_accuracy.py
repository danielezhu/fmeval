from dataclasses import dataclass
from typing import Optional

from ray import ObjectRef

from fmeval.eval_algorithms.helper_models.helper_model import BertscoreHelperModel
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.transforms.transform import Transform, Record

import ray
import nltk
import evaluate as hf_evaluate

from nltk import word_tokenize
from nltk.translate import meteor_score

METEOR_SCORE = "meteor"
ROUGE_SCORE = "rouge"
BERT_SCORE = "bertscore"

# rouge constants
ROUGE_1 = "rouge1"
ROUGE_2 = "rouge2"
ROUGE_L = "rougeL"

ROUGE_TYPES = [ROUGE_1, ROUGE_2, ROUGE_L]

# bertscore constants
MICROSOFT_DEBERTA_MODEL = "microsoft/deberta-xlarge-mnli"
ROBERTA_MODEL = "roberta-large-mnli"
DEFAULT_MODEL_TYPE = MICROSOFT_DEBERTA_MODEL
SUPPORTED_MODEL_TYPES = [MICROSOFT_DEBERTA_MODEL, ROBERTA_MODEL]


def _load_eval_helpers():
    """
    Method to download required helpers for eval_algo in constructor call
    """
    # load helper modules for meteor
    nltk.download("wordnet")
    nltk.download("punkt")
    nltk.download("omw-1.4")


@dataclass(frozen=False)
class SummarizationAccuracyConfig:
    """
    Configuration for the summarization accuracy eval algorithm

    :param rouge_type: Type of rouge metric in eval results
    :param use_stemmer_for_rouge: bool value to set using stemmer for rouge metric
    :param bertscore_model_type: model type to use for BERT score
    :param target_output_key: the Record key for the target output
    :param model_output_key: the Record key for the model output
    """
    rouge_type: str = ROUGE_2
    use_stemmer_for_rouge: bool = True
    bertscore_model_type: str = DEFAULT_MODEL_TYPE
    target_output_key: str = "target_output"
    model_output_key: str = "model_output"

    def __post_init__(self):
        if self.rouge_type not in ROUGE_TYPES:
            raise EvalAlgorithmClientError(
                f"Invalid rouge_type: {self.rouge_type} requested in SummarizationAccuracyConfig, "
                f"please choose from acceptable values: {ROUGE_TYPES}"
            )
        if self.bertscore_model_type not in SUPPORTED_MODEL_TYPES:
            raise EvalAlgorithmClientError(
                f"Invalid model_type_for_bertscore: {self.bertscore_model_type} requested in "
                f"SummarizationAccuracyConfig, please choose from acceptable values: {SUPPORTED_MODEL_TYPES}"
            )


class SummarizationAccuracy(Transform):
    def __init__(
            self,
            config: SummarizationAccuracyConfig,
            bertscore_model: ObjectRef,
    ):
        self.config = config
        self.score_functions = {
            METEOR_SCORE: self.get_meteor_score,
            ROUGE_SCORE: self.get_rouge_score,
            BERT_SCORE: self.get_bert_score,
        }
        self.bertscore_model = bertscore_model
        _load_eval_helpers()

    def __call__(self, record: Record):
        for score_name, score_fn in self.score_functions.items():
            record[score_name] = score_fn(record[self.config.target_output_key], record[self.config.model_output_key])
        return record

    def get_meteor_score(self, target_output: str, model_output: str) -> float:
        """
        METEOR is a metric for text similarity between the machine-produced summary and human-produced reference summaries.
        Unigrams can be matched based on their surface forms, stemmed forms,
        and meanings; furthermore, METEOR can be easily extended to include more
        advanced matching strategies. Once all generalized unigram matches
        between the two strings have been found, METEOR computes a score for
        this matching using a combination of unigram-precision, unigram-recall, and
        a measure of fragmentation that is designed to directly capture how
        well-ordered the matched words in the machine translation are in relation
        to the reference.

        :param target_output: The expected responses from the model
        :param model_output: The output of a model that we want to evaluate.
        :returns: meteor score
        """
        return meteor_score.single_meteor_score(
            reference=word_tokenize(target_output), hypothesis=word_tokenize(model_output)
        )

    def get_rouge_score(self, target_output: str, model_output: str) -> float:
        """
        The ROUGE-N, where N=[1,2,L], score is a standard metric for summarization quality.
        It computes the word overlap between the reference and model summary. Given that this metric is based on simple
        word overlap statistics, it works best for extractive summaries.
        Note that if we rephrase the summary without changing its meaning the ROUGE-N score will drop.

        Reference: https://huggingface.co/spaces/evaluate-metric/rouge

        :param target_output: The expected responses from the model
        :param model_output: The output of a model that we want to evaluate.
        :returns: rouge score
        """
        rouge = hf_evaluate.load("rouge")
        return rouge.compute(
            predictions=[model_output],
            references=[target_output],
            use_stemmer=self.config.use_stemmer_for_rouge,
            rouge_types=[self.config.rouge_type],
        )[self.config.rouge_type]

    def get_bert_score(self, target_output: str, model_output: str) -> float:
        """
        BERTscore is a similarity-based metric that compares the embedding of the prediction and target sentences
        under a learned model, typically, from the BERT family.
        This score may lead to increased flexibility compared to ROUGE and METEOR in terms of rephrasing since
        semantically similar sentences are (typically) embedded similarly.

        https://huggingface.co/spaces/evaluate-metric/bertscore

        :param target_output: The expected responses from the model
        :param model_output: The output of a model that we want to evaluate.
        :returns: bert score
        """
        return ray.get(self.bertscore_model.get_helper_scores.remote(target_output, model_output))
