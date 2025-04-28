# coding=utf-8
import evaluate
from nltk.translate import nist_score as nist
import numpy as np

from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core.utils import requirement_message

_CITATION = """\
@inproceedings{doddington02,
    author = {Doddington, George},
    title = {Automatic Evaluation of Machine Translation Quality Using N-Gram Co-Occurrence Statistics},
    year = {2002},
    publisher = {Morgan Kaufmann Publishers Inc.},
    address = {San Francisco, CA, USA},
    booktitle = {Proceedings of the Second International Conference on Human Language Technology Research},
    pages = {138–145},
    numpages = {8},
    location = {San Diego, California},
    series = {HLT '02}
}
"""

_DESCRIPTION = """
NIST is a method for evaluating the quality of text which has been translated using machine translation. Its name comes 
from the US National Institute of Standards and Technology. The NIST metric was designed to improve BLEU by rewarding
the translation of infrequently used words. It is based on the BLEU metric, but with some alterations. 
Where BLEU simply calculates n-gram precision adding equal weight to each one, NIST also calculates how informative a 
particular n-gram is. That is to say when a correct n-gram is found, the rarer that n-gram is, the more weight it will 
be given. For example, if the bigram "on the" is correctly matched, it will receive lower weight than the correct 
matching of bigram "interesting calculations", as this is less likely to occur. The final NIST score is calculated
using the arithmetic mean of the ngram matches between candidate and reference translations. In addition, a smaller
brevity penalty is used for smaller variations in phrase lengths. NIST also differs from BLEU in its calculation of the
brevity penalty insofar as small variations in translation length do not impact the overall score as much. The
reliability and quality of the NIST metric has been shown to be superior to the BLEU metric in many cases.
The metric can be thought of as a variant of BLEU which weighs each matched n-gram based on its information gain,
calculated as:
$Info(n-gram) = Info(w_1,\dots,w_n) = log_2 \frac{# of occurences of w_1,\dots,w_{n-1}}{# of occurences of w_1,\dots,w_n}$
To sum up, the idea is to give more credit if a matched n-gram is rare and less credit if a matched n-gram is common.
This also reduces the chance of gaming the metric by producing trivial n-grams.
"""

_KWARGS_DESCRIPTION = """
Computes NIST score.
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    n: length of n-grams. default: 5.
Returns:
    'score': nist score.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NistPlanet(MetricForLanguageGeneration):
    def _info(self):
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=1,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=["https://github.com/nltk/nltk/blob/develop/nltk/translate/nist_score.py"],
            reference_urls=[
                "https://github.com/nltk/nltk/blob/develop/nltk/translate/nist_score.py"
            ],
        )

    def _download_and_prepare(self, dl_manager):
        try:
            import nltk.translate.nist_score
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="nist", package_name="nltk"))
        else:
            super(NistPlanet, self)._download_and_prepare(dl_manager)

    def _compute_single_pred_single_ref(
            self,
            predictions,
            references,
            reduce_fn=None,
            n=5
    ):
        """
        Compute the nist score for a single prediction and a single reference.
        Args:
            predictions: A parameter containing a single text sample for prediction.
            references: A parameter containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            n (int): length of n-grams. default: 5.
        """
        newRefs = []
        scores = []
        for reference in references:
            newRef = reference.split()
            newRefs.append(newRef)
        for prediction in predictions:
            newPred = prediction.split()
            score = nist.sentence_nist(newRefs, newPred, n)
            scores.append(score)
        res = np.mean(scores)
        return {"score": res}

    def _compute_single_pred_multi_ref(
            self,
            predictions,
            references,
            reduce_fn=None,
            n=5
    ):
        """
        Compute the nist score for a single prediction and multiple reference.
        Args:
            predictions: A parameter containing a single text sample for prediction.
            references: A parameter containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            n (int): length of n-grams. default: 5.
        """
        scores = []
        predScores = []
        for refList in references:
            newRefs = []
            for reference in refList:
                newRef = reference.split()
                newRefs.append(newRef)
            for prediction in predictions:
                newPred = prediction.split()
                score = nist.sentence_nist(newRefs, newPred, n)
                predScores.append(score)
            scores.append(np.mean(predScores))
        res = np.mean(scores)
        return {"score": res}

    def _compute_multi_pred_multi_ref(
            self,
            predictions,
            references,
            reduce_fn=None,
            n=5
    ):
        """
        Compute the nist score for multiple prediction and multiple reference.
        Args:
            predictions: A parameter containing a multiple text sample for prediction.
            references: A paramter containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            n (int): length of n-grams. default: 5.
        """
        scores = []
        for prediction in predictions:
            score = self._compute_single_pred_multi_ref(predictions=prediction, references=references,
                                                        reduce_fn=reduce_fn, n=n)
            scores.append(score["score"])
        res = np.mean(scores)
        return {"score": res}
