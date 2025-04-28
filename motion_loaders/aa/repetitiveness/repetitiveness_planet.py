# coding=utf-8

""" Repetitiveness metric. """

import evaluate
from typing import Callable

from collections import Counter

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration

_CITATION = """\
@inproceedings{fu2020a,
  title={A Theoretical Analysis of the Repetition Problem in Text Generation.},
  author={Fu, Zihao and Lam, Wai and So, Anthony Man-Cho and Shi, Bei },
  booktitle={Thirty-Fifth AAAI Conference on Artificial Intelligence},
  year={2021}
}
"""

_DESCRIPTION = """\
The [repetition problem](https://github.com/fuzihaofzh/repetition-problem-nlg) has been observed in nearly all text generation models.
This problem is, unfortunately, caused by the traits of our language itself. There exists too many words predicting the same word 
as the subsequent word with high probability. Consequently, it is easy to go back to that word and form repetitions.
The Repetitiveness metric evaluates how many n-grams are repeated on average in the hypothesis sentences, the result is normalized by 
the length of the sentence.
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions (EvaluationInstance): List of generated text predictions.
    references (EvaluationInstance): List of reference texts for comparison.

Returns:
    'score': A dictionary containing the computed Repetitiveness metric score.
Examples:
    >>> predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
    >>> references = ["Peace at home, peace in the world.", "The cat is playing on the mat."]
    >>> scorer = NLGMetricverse(metrics=load_metric("repetitiveness"))
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    { "repetitiveness": { "score": 0.85 } }
"""

_LICENSE = """

"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class RepetitivenessPlanet(MetricForLanguageGeneration):
    def _info(self):
        """
        Returns information about the Repetitiveness metric, including its description, citation,
        input parameters description, default features, codebase URLs, and reference URLs.

        Returns:
            MetricInfo: An object containing information about the metric.
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

    def _compute_single_pred_single_ref(self, predictions: EvaluationInstance, references: EvaluationInstance,
                                        reduce_fn: Callable = None, **kwargs):
        """
        Computes the Repetitiveness metric score for a single prediction and a single reference.

        Args:
            predictions (EvaluationInstance): The generated text prediction.
            references (EvaluationInstance): List of reference texts for comparison.
            reduce_fn (Callable, optional): A reduction function (if applicable). Defaults to None.
            **kwargs: Additional keyword arguments for future expansion.

        Returns:
            dict: A dictionary containing the computed Repetitiveness metric score.
                  The score is stored under the key "score".
        """
        result = self.__compute_repetitiveness(predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(self, predictions: EvaluationInstance, references: EvaluationInstance,
                                       reduce_fn: Callable = None, **kwargs):
        """
        Computes the Repetitiveness metric score for a single prediction and multiple references.

        Args:
            predictions (EvaluationInstance): The generated text prediction.
            references (EvaluationInstance): List of reference texts for comparison.
            reduce_fn (Callable, optional): A reduction function (if applicable). Defaults to None.
            **kwargs: Additional keyword arguments for future expansion.

        Returns:
            dict: A dictionary containing the computed Repetitiveness metric score.
                  The score is stored under the key "score".
        """
        result = self.__compute_repetitiveness(predictions)
        return {"score": result}

    def _compute_multi_pred_multi_ref(self, predictions: EvaluationInstance, references: EvaluationInstance,
                                      reduce_fn: Callable = None, **kwargs):
        """
        Computes the Repetitiveness metric score for multiple predictions and multiple references.

        Args:
            predictions (EvaluationInstance): List of generated text predictions.
            references (EvaluationInstance): List of reference texts for comparison.
            reduce_fn (Callable, optional): A reduction function (if applicable). Defaults to None.
            **kwargs: Additional keyword arguments for future expansion.

        Returns:
            dict: A dictionary containing the computed Repetitiveness metric score.
                  The score is stored under the key "score".
        """
        inputList = []
        for prediction in predictions:
            inputList += prediction
        result = self.__compute_repetitiveness(inputList)
        return {"score": result}

    @staticmethod
    def __compute_repetitiveness(predictions):
        """
        Helper method to compute the Repetitiveness metric for a given list of texts.

        Args:
            predictions (list): List of texts for which repetitiveness is to be computed.

        Returns:
            float: The computed Repetitiveness metric score.
        """
        counter = 0
        for candidate in predictions:
            monograms = candidate.split(" ")
            n_words = len(monograms)
            m_counted = Counter(monograms)
            for ngram in m_counted.values():
                if ngram > 1:
                    counter = counter + 1  # if a word  that repeats itself is found
            counter = counter + n_words
        return (counter / len(predictions)) / 10
