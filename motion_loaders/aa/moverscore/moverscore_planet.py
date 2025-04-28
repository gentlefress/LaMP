# coding=utf-8
""" MOVERScore metric """
import os
from collections.abc import Callable

import evaluate
import numpy as np

from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.metrics._core.utils import PackagePlaceholder, requirement_message

# `import moverscore` placeholder
moverscore_v2 = PackagePlaceholder(version="2.0.0")

_CITATION = """\
@inproceedings{zhao2019moverscore,
  title = {MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance},
  month = {August},
  year = {2019},
  author = {Wei Zhao, Maxime Peyrard, Fei Liu, Yang Gao, Christian M. Meyer, Steffen Eger},
  address = {Hong Kong, China},
  publisher = {Association for Computational Linguistics},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
}
"""

_DESCRIPTION = """\
MoverScore (Zhao et.al, 2019) is an automated evaluation metric assigning a single holistic score to any
system-generated text (neural or non-neural) by comparing it against human references for semantic content
matching. It is a monolingual measure evluating meaning similarities between pairs of sentences written in
the same language. It combines contextualized representations coming from language models (trained to capture
distant semantic dependencies) with the Word Mover's distance (WMD). So, MoverScore generalizes WMD by working
on n-grams. Specifically, it computes the minimum cost of transforming (transportation distance) the generated
text to the reference text, taking into account Euclidean distance between vector representations of n-gram as
well as their document frequencies. According to the authors, MoverScore can be seen as a generalization of
BertScore. Both of them use contextualized representations, but they have a different focus. BertScore aligns
each hypothesis word with a single reference word (1:1), while MoverScore makes a soft alignment (1:N).
MoverScore demonstrates strong generalization capability across multiple tasks, achieving much higher correlation
with human judgments than BLEU on machine translation, summarization and image captioning.
We consider MOVERScore v2, which is faster than the original implementation (by disabling powermean) but a bit
worse in performance.

BOUNDS
Intuitively, the metric assigns a perfect score to the system text if it conveys the same meaning as the
reference text. Any deviation from the reference content can then lead to a reduced score, e.g., the
system text contains more (or less) content than the reference, or the system produces ill-formed text
that fails to deliver the intended meaning. In general, higher scores refer to better performance.

WEAKNESSES
The paradigm of reference-based measures is useful for targeted generation tasks such as translation and
summarization where matching a set of references is paramount. It is, however, unsuitable for open-ended
generation where there typically are several plausible continuations for each context and creative
generations are desirable.

PROPERTY
IDF-weighted n-gram soft-alignment via ELMo/BERT contextualized embeddings

CATEGORY
unsupervised; embedding-based

TASKS
MT, SUM, D2T, IC
"""

_KWARGS_DESCRIPTION = """
Computes MoverScore.
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    'score': mover score.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MoverscorePlanet(MetricForLanguageGeneration):
    def _info(self):
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=1,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=["https://github.com/AIPHES/emnlp19-moverscore"],
            reference_urls=[
                "https://github.com/AIPHES/emnlp19-moverscore"
            ],
        )

    def _download_and_prepare(self, dl_manager):
        global moverscore_v2
        try:
            import moverscore_v2
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="moverscore_v2", package_name="moverscore"))
        else:
            super(MoverscorePlanet, self)._download_and_prepare(dl_manager)

    def _compute_single_pred_single_ref(
            self,
            predictions,
            references,
            reduce_fn: Callable = None,
            idf_dict_ref=None,
            idf_dict_hyp=None,
            stop_words=None,
            n_gram=1,
            remove_subwords=True,
            model="distilbert-base-uncased"
    ):
        """
        Compute the moverscore score for a single prediction and a single reference.
        Args:
            predictions: A parameter containing a single text sample for prediction.
            references: A parameter containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            idf_dict_ref (Any, optional): idf dictionary extracted from the reference corpus.
            idf_dict_hyp (Any, optional): idf dictionary extracted from the system hypothesis corpus.
            stop_words (Any, optional): a set of functional words to ignore.
            n_gram (int, optional): unigram-based MoverScore (n-gram=1), bigram-based MoverScore (n-gram=2).
            remove_subwords (bool, optional): if `True`, remove the specified stopwords.
            model (str, optional): the pre-trained model to use. Default: `distilbert-base-uncased`.
        """
        os.environ['MOVERSCORE_MODEL'] = model
        if stop_words is None:
            stop_words = []
        if idf_dict_ref is None:
            idf_dict_ref = moverscore_v2.get_idf_dict(references)  # idf_dict_ref = defaultdict(lambda: 1.)
        if idf_dict_hyp is None:
            idf_dict_hyp = moverscore_v2.get_idf_dict(predictions)  # idf_dict_hyp = defaultdict(lambda: 1.)

        scores = moverscore_v2.word_mover_score(references, predictions, idf_dict_ref, idf_dict_hyp,
                                                stop_words=stop_words, n_gram=n_gram, remove_subwords=remove_subwords)
        return {"score": np.mean(scores)}

    def _compute_single_pred_multi_ref(
            self,
            predictions,
            references,
            reduce_fn: Callable = None,
            idf_dict_ref=None,
            idf_dict_hyp=None,
            stop_words=None,
            n_gram=1,
            remove_subwords=True,
            model="distilbert-base-uncased"
    ):
        """
        Compute the moverscore score for a single prediction and multiple reference.
        Args:
            predictions: A parameter containing a single text sample for prediction.
            references: A parameter containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            idf_dict_ref (Any, optional): idf dictionary extracted from the reference corpus.
            idf_dict_hyp (Any, optional): idf dictionary extracted from the system hypothesis corpus.
            stop_words (Any, optional): a set of functional words to ignore.
            n_gram (int, optional): unigram-based MoverScore (n-gram=1), bigram-based MoverScore (n-gram=2).
            remove_subwords (bool, optional): if `True`, remove the specified stopwords.
            model (str, optional): the pre-trained model to use. Default: `distilbert-base-uncased`.
        """
        os.environ['MOVERSCORE_MODEL'] = model
        if stop_words is None:
            stop_words = []
        if idf_dict_hyp is None:
            idf_dict_hyp = moverscore_v2.get_idf_dict(predictions)  # idf_dict_hyp = defaultdict(lambda: 1.)
        res = []
        for reference in references:
            if idf_dict_ref is None:
                idf_dict_ref = moverscore_v2.get_idf_dict(reference)  # idf_dict_ref = defaultdict(lambda: 1.)
            score = moverscore_v2.word_mover_score(references, predictions, idf_dict_ref, idf_dict_hyp,
                                                   stop_words=stop_words, n_gram=n_gram,
                                                   remove_subwords=remove_subwords)
            res.append(np.mean(score))
        scores = np.mean(res)
        return {"score": scores}

    def _compute_multi_pred_multi_ref(
            self,
            predictions,
            references,
            reduce_fn=None,
            idf_dict_ref=None,
            idf_dict_hyp=None,
            stop_words=None,
            n_gram=1,
            remove_subwords=True,
            model="distilbert-base-uncased"
    ):
        """
        Compute the moverscore score for multiple prediction and multiple reference.
        Args:
            predictions: A parameter containing a multiple text sample for prediction.
            references: A parameter containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            idf_dict_ref (Any, optional): idf dictionary extracted from the reference corpus.
            idf_dict_hyp (Any, optional): idf dictionary extracted from the system hypothesis corpus.
            stop_words (Any, optional): a set of functional words to ignore.
            n_gram (int, optional): unigram-based MoverScore (n-gram=1), bigram-based MoverScore (n-gram=2).
            remove_subwords (bool, optional): if `True`, remove the specified stopwords.
            model (str, optional): the pre-trained model to use. Default: `distilbert-base-uncased`.
        """
        os.environ['MOVERSCORE_MODEL'] = model
        if stop_words is None:
            stop_words = []
        res = []
        for prediction in predictions:
            score = self._compute_single_pred_multi_ref(predictions=prediction, references=references,
                                                        reduce_fn=reduce_fn, idf_dict_ref=idf_dict_ref,
                                                        idf_dict_hyp=idf_dict_hyp, stop_words=stop_words,
                                                        n_gram=n_gram, remove_subwords=remove_subwords)
            res.append(np.mean(score["score"]))
        scores = np.mean(res)
        return {"score": scores}
