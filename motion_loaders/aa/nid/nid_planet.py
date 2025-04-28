# coding=utf-8

""" NID metric. """

import evaluate
import numpy as np
from scipy.stats import entropy
from typing import Callable
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.corpus import stopwords

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration


_CITATION = """

"""

_DESCRIPTION = """
The NID metric reckons redundancy by inverting the entropy of summary unigrams and applying
lenght normalization:

        1 - entropy(y)/log(|y|)
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: List of predictions to score. Each prediction should be a string.
    references: List of references for each prediction. Each reference should be a string.
Returns:
    nid: The average nid of the predictions.
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("nid"))
    >>> predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
    >>> references = ["Peace at home, peace in th world.", "The cat is playing on the mat."]
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    { "total_items": 2, "empty_items": 0, "nid": { "score": 0.5101 }}
"""

_LICENSE = """

"""

CHECKPOINT_URLS = {

}


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NIDPlanet(MetricForLanguageGeneration):
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
        Computes the nid score for a single predicted text and a single reference text.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference text.
            reduce_fn (Callable): A function to use for reducing the nid scores across multiple examples.
        """
        result = self._compute_nid(predictions)
        return {"score": result}

    def _compute_single_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the nid score for a single predicted text and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted text.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the nid scores across multiple examples.
        """
        result = self._compute_nid(predictions)
        return {"score": result}

    def _compute_multi_pred_multi_ref(
            self, predictions: EvaluationInstance, references: EvaluationInstance, reduce_fn: Callable = None
    ):
        """
        Computes the nid score for multiple predicted texts and multiple reference texts.

        Args:
            predictions (EvaluationInstance): An object containing the predicted texts.
            references (EvaluationInstance): An object containing the reference texts.
            reduce_fn (Callable): A function to use for reducing the nid scores across multiple examples.
        """
        predList = []
        for pred in predictions:
            predList += pred
        result = self._compute_nid(predList)
        return {"score": result}

    @staticmethod
    def _compute_nid(predictions):
        sum_redundancy = 0
        stop_words = set(stopwords.words("english"))
        count = CountVectorizer()
        all_redundancy = []
        number_file = len(predictions)
        for p in predictions:
            all_txt = []
            all_txt.extend(word_tokenize(p.strip()))
            num_word = len(all_txt)
            new_all_txt = [w for w in all_txt if not w in stop_words]
            new_all_txt = [" ".join(new_all_txt)]
            try:
                x = count.fit_transform(new_all_txt)
                bow = x.toarray()[0]
                max_possible_entropy = np.log(num_word)
                e = entropy(bow)
                redundancy = (1 - e / max_possible_entropy)
                sum_redundancy += redundancy
                all_redundancy.append(redundancy)
            except ValueError:
                continue
        NID = round(sum_redundancy / number_file, 4)
        return NID
