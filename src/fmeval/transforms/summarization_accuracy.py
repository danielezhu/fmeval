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

from fmeval.transforms.transform_pipeline import TransformPipeline

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


class MeteorScore(Transform):
    def __init__(
            self,
            target_output_key: str,
            model_output_key: str,
            meteor_score_output_key: str,
            load_helpers: bool = False) -> Record:
        super().__init__(target_output_key, model_output_key, meteor_score_output_key, load_helpers)
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.meteor_score_output_key = meteor_score_output_key
        if load_helpers:
            _load_eval_helpers()
            self.load_helpers = False

    def __call__(self, record: Record):
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
        """
        record[self.meteor_score_output_key] = meteor_score.single_meteor_score(
            reference=word_tokenize(record[self.target_output_key]),
            hypothesis=word_tokenize(record[self.model_output_key])
        )
        return record

    @property
    def output_keys(self):
        return [self.meteor_score_output_key]


class RougeScore(Transform):
    def __init__(
            self,
            target_output_key: str,
            model_output_key: str,
            rouge_score_output_key: str,
            rouge_type: str = ROUGE_2,
            use_stemmer: bool = True
    ) -> Record:
        if rouge_type not in ROUGE_TYPES:
            raise EvalAlgorithmClientError(
                f"Invalid rouge_type: {rouge_type} requested in SummarizationAccuracyConfig, "
                f"please choose from acceptable values: {ROUGE_TYPES}"
            )
        super().__init__(target_output_key, model_output_key, rouge_score_output_key, rouge_type=rouge_type, use_stemmer=use_stemmer)
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.rouge_score_output_key = rouge_score_output_key
        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer

    def __call__(self, record: Record) -> Record:
        """
        The ROUGE-N, where N=[1,2,L], score is a standard metric for summarization quality.
        It computes the word overlap between the reference and model summary. Given that this metric is based on simple
        word overlap statistics, it works best for extractive summaries.
        Note that if we rephrase the summary without changing its meaning the ROUGE-N score will drop.

        Reference: https://huggingface.co/spaces/evaluate-metric/rouge
        """
        rouge = hf_evaluate.load("rouge")
        record[self.rouge_score_output_key] = rouge.compute(
            predictions=[record[self.model_output_key]],
            references=[record[self.target_output_key]],
            use_stemmer=self.use_stemmer,
            rouge_types=[self.rouge_type],
        )[self.rouge_type]
        return record

    @property
    def output_keys(self):
        return [self.rouge_score_output_key]


class BertScore(Transform):
    def __init__(
            self,
            target_output_key: str,
            model_output_key: str,
            bertscore_output_key: str,
            bertscore_model_ref: ObjectRef
    ):
        super().__init__(target_output_key, model_output_key, bertscore_output_key, bertscore_model_ref)
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.bertscore_output_key = bertscore_output_key
        self.bertscore_model_ref = bertscore_model_ref

    def __call__(self, record: Record) -> Record:
        """
        BERTscore is a similarity-based metric that compares the embedding of the prediction and target sentences
        under a learned model, typically, from the BERT family.
        This score may lead to increased flexibility compared to ROUGE and METEOR in terms of rephrasing since
        semantically similar sentences are (typically) embedded similarly.

        https://huggingface.co/spaces/evaluate-metric/bertscore
        """
        record[self.bertscore_output_key] = ray.get(
            self.bertscore_model_ref.get_helper_scores.remote(
                record[self.target_output_key],
                record[self.model_output_key],
            )
        )
        return record

    @property
    def output_keys(self):
        return [self.bertscore_output_key]


class SummarizationAccuracy:
    def __init__(
            self,
            model_output_key: str,
            rouge_type: str = ROUGE_2,
            use_stemmer_for_rouge: bool = True,
            bertscore_model_type: str = DEFAULT_MODEL_TYPE,
            bertscore_model_ref: Optional[ObjectRef] = None,
            meteor_score_output_key: str = METEOR_SCORE,
            rouge_score_output_key: str = ROUGE_SCORE,
            bertscore_output_key: str = BERT_SCORE,
    ):
        meteor_transform = MeteorScore(
            target_output_key="target_output",  # use a constant in final version
            model_output_key=model_output_key,
            meteor_score_output_key=meteor_score_output_key,
            load_helpers=True,
        )
        rouge_transform = RougeScore(
            target_output_key="target_output",
            model_output_key=model_output_key,
            rouge_score_output_key=rouge_score_output_key,
            rouge_type=rouge_type,
            use_stemmer=use_stemmer_for_rouge
        )
        bertscore_model_ref = (
            bertscore_model_ref
            if bertscore_model_ref
            else BertscoreHelperModel.remote(model_type=bertscore_model_type)
        )
        bertscore_transform = BertScore(
            target_output_key="target_output",
            model_output_key=model_output_key,
            bertscore_output_key=bertscore_output_key,
            bertscore_model_ref=bertscore_model_ref
        )
        self.pipeline = TransformPipeline([meteor_transform, rouge_transform, bertscore_transform])
        self.transforms = {
            METEOR_SCORE: meteor_transform,
            ROUGE_SCORE: rouge_transform,
            BERT_SCORE: bertscore_transform
        }
