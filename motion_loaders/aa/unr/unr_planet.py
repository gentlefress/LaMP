# coding=utf-8

""" UNR metric. """

import evaluate
from typing import Callable
from nltk.util import ngrams
from nltk import word_tokenize

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration


_CITATION = """

"""

_DESCRIPTION = """
The UNR metric measures the summary of n-grams uniqueness:

    count(uniq_n_gram(y))/count(n_gram(y))

Where we take n in [1,3] and divide the average by variance.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: List of predictions to score. Each prediction should be a string.
    references: List of references for each prediction. Each reference should be a string.
Returns:
    unr: List of unr scores for each prediction. Each score should be a float.
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("unr"))
    >>> predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
    >>> references = ["Peace at home, peace in th world.", "The cat is playing on the mat."]
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    { "total_items": 2, "empty_items": 0, "unr": { "unr_1": 0.9, "unr_2": 0.9444, "unr_3": 1.0, "unr_avg": 0.9481333333333334 }}
"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class UNRPlanet(MetricForLanguageGeneration):
    def _info(self):
        """
        Returns metadata about the metric.

        Returns:
            MetricInfo: An object containing metadata about the metric.
        """
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=1,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=[""],
            reference_urls=[
                ""
            ],
        )

    def _compute_single_pred_single_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the unr score for a single predicted text and a single reference text.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference text.
            reduce_fn (Callable): A function to use for reducing the unr scores across multiple examples.
        """
        unr_1, unr_2, unr_3, unr_avg = self._compute_unr(predictions)
        return {"unr_1": unr_1, "unr_2": unr_2, "unr_3": unr_3, "unr_avg": unr_avg}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the unr score for a single predicted text and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the unr scores across multiple examples.
        """
        unr_1, unr_2, unr_3, unr_avg = self._compute_unr(predictions)
        return {"unr_1": unr_1, "unr_2": unr_2, "unr_3": unr_3, "unr_avg": unr_avg}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the unr score for multiple predicted texts and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted texts.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the unr scores across multiple examples.
        """
        predList = []
        for pred in predictions:
            predList += pred
        unr_1, unr_2, unr_3, unr_avg = self._compute_unr(predList)
        return {"unr_1": unr_1, "unr_2": unr_2, "unr_3": unr_3, "unr_avg": unr_avg}

    @staticmethod
    def _compute_unr(predictions):
        sum_unigram_ratio = 0
        sum_bigram_ratio = 0
        sum_trigram_ratio = 0
        all_unigram_ratio = []
        all_bigram_ratio = []
        all_trigram_ratio = []
        number_file = len(predictions)
        for p in predictions:
            all_txt = []
            all_txt.extend(word_tokenize(p.strip()))

            all_unigram = list(ngrams(all_txt, 1))
            uniq_unigram = set(all_unigram)
            if len(uniq_unigram) > 0:
                unigram_ratio = len(uniq_unigram) / len(all_unigram)
                sum_unigram_ratio += unigram_ratio
                all_unigram_ratio.append(unigram_ratio)

            all_bigram = list(ngrams(all_txt, 2))
            uniq_bigram = set(all_bigram)
            if len(uniq_bigram) > 0:
                bigram_ratio = len(uniq_bigram) / len(all_bigram)
                sum_bigram_ratio += bigram_ratio
                all_bigram_ratio.append(bigram_ratio)

            all_trigram = list(ngrams(all_txt, 3))
            uniq_trigram = set(all_trigram)
            if len(uniq_trigram) > 0:
                trigram_ratio = len(uniq_trigram) / len(all_trigram)
                sum_trigram_ratio += trigram_ratio
                all_trigram_ratio.append(trigram_ratio)
                
        uniq_unigram_ratio = round(sum_unigram_ratio / number_file, 4)
        uniq_bigram_ratio = round(sum_bigram_ratio / number_file, 4)
        uniq_trigram_ratio = round(sum_trigram_ratio / number_file, 4)
        unr_avg = (uniq_unigram_ratio + uniq_bigram_ratio + uniq_trigram_ratio) / 3
        return uniq_unigram_ratio, uniq_bigram_ratio, uniq_trigram_ratio, unr_avg
