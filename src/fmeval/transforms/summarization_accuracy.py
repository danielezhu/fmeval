from typing import List

import ray
import nltk
import evaluate as hf_evaluate

from ray import ObjectRef
from fmeval.exceptions import EvalAlgorithmClientError
from fmeval.transforms.transform import Transform, Record
from fmeval.transforms.transform_pipeline import TransformPipeline

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


class MeteorScore(Transform):
    def __init__(
            self,
            input_keys: List[str],
            output_keys: List[str],
            target_output_key: str,
            model_output_key: str,
            load_helpers: bool = False
    ) -> Record:
        assert set(input_keys) == {target_output_key, model_output_key}
        assert len(output_keys) == 1
        super().__init__(
            input_keys,
            output_keys,
            target_output_key,
            model_output_key,
            load_helpers=False,  # after calling _load_eval_helpers during initialization, we never need to do so again
        )
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        if load_helpers:
            _load_eval_helpers()

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
        output_key = self.output_keys[0]
        record[output_key] = meteor_score.single_meteor_score(
            reference=word_tokenize(record[self.target_output_key]),
            hypothesis=word_tokenize(record[self.model_output_key])
        )
        return record


class RougeScore(Transform):
    def __init__(
            self,
            input_keys: List[str],
            output_keys: List[str],
            target_output_key: str,
            model_output_key: str,
            rouge_type: str = ROUGE_2,
            use_stemmer: bool = True
    ) -> Record:
        assert set(input_keys) == {target_output_key, model_output_key}
        assert len(output_keys) == 1
        if rouge_type not in ROUGE_TYPES:
            raise EvalAlgorithmClientError(
                f"Invalid rouge_type: {rouge_type} requested in SummarizationAccuracyConfig, "
                f"please choose from acceptable values: {ROUGE_TYPES}"
            )
        super().__init__(
            input_keys,
            output_keys,
            target_output_key,
            model_output_key,
            rouge_type=rouge_type,
            use_stemmer=use_stemmer
        )
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
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
        record[self.output_keys[0]] = rouge.compute(
            predictions=[record[self.model_output_key]],
            references=[record[self.target_output_key]],
            use_stemmer=self.use_stemmer,
            rouge_types=[self.rouge_type],
        )[self.rouge_type]
        return record


class BertScore(Transform):
    def __init__(
            self,
            input_keys: List[str],
            output_keys: List[str],
            target_output_key: str,
            model_output_key: str,
            bertscore_model: ObjectRef,
    ):
        assert set(input_keys) == {target_output_key, model_output_key}
        assert len(output_keys) == 1
        super().__init__(input_keys, output_keys, target_output_key, model_output_key, bertscore_model)
        self.target_output_key = target_output_key
        self.model_output_key = model_output_key
        self.bertscore_model = bertscore_model

    def __call__(self, record: Record) -> Record:
        """
        BERTscore is a similarity-based metric that compares the embedding of the prediction and target sentences
        under a learned model, typically, from the BERT family.
        This score may lead to increased flexibility compared to ROUGE and METEOR in terms of rephrasing since
        semantically similar sentences are (typically) embedded similarly.

        https://huggingface.co/spaces/evaluate-metric/bertscore
        """
        record[self.output_keys[0]] = ray.get(
            self.bertscore_model.get_helper_scores.remote(
                record[self.target_output_key],
                record[self.model_output_key],
            )
        )
        return record


class SummarizationAccuracyTransforms:
    def __init__(
            self,
            target_output_key: str,
            model_output_key: str,
            bertscore_model: ObjectRef,
            rouge_type: str = ROUGE_2,
            use_stemmer_for_rouge: bool = True,
            meteor_score_output_key: str = METEOR_SCORE,
            rouge_score_output_key: str = ROUGE_SCORE,
            bert_score_output_key: str = BERT_SCORE,
    ):
        meteor_transform = MeteorScore(
            input_keys=[target_output_key, model_output_key],
            output_keys=[meteor_score_output_key],
            target_output_key=target_output_key,
            model_output_key=model_output_key,
            load_helpers=True,
        )
        rouge_transform = RougeScore(
            input_keys=[target_output_key, model_output_key],
            output_keys=[rouge_score_output_key],
            target_output_key=target_output_key,
            model_output_key=model_output_key,
            rouge_type=rouge_type,
            use_stemmer=use_stemmer_for_rouge
        )
        bertscore_transform = BertScore(
            input_keys=[target_output_key, model_output_key],
            output_keys=[bert_score_output_key],
            target_output_key=target_output_key,
            model_output_key=model_output_key,
            bertscore_model=bertscore_model
        )
        self.pipeline = TransformPipeline([meteor_transform, rouge_transform, bertscore_transform])
        self.transforms = {
            METEOR_SCORE: meteor_transform,
            ROUGE_SCORE: rouge_transform,
            BERT_SCORE: bertscore_transform
        }
