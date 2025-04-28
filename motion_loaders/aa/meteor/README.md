# Metric Card for METEOR

## Metric Description
METEOR (Metric for Evaluation of Translation with Explicit ORdering) is an automatic metric originally designed to address some of the issues found in BLEU and has been widely used for evaluating machine translation models.
Compared to BLEU, which only measures precision, METEOR is based on the harmonic mean of the unigram precision and recall, in which recall is weighted higher than precision.
It is based on a generalized concept of unigram matching between the machine-produced translation and human-produced reference translations.
METEOR has several variants that extend exact word matching that most of the metrics in this category do not include, such as stemming and WordNet-based synonym matching (if English is the target).
These variants address the problem of reference translation variability, allowing for morphological variants and synonyms to be recognized as valid translations.
The metric has been found to produce good correlation with human judgments at the sentence or segment level (Agarwal & Lavie, 2008).
This differs from BLEU in that METEOR is explicitly designed to compare at the sentence level rather than the corpus level.
Once all generalized unigram matches between the two strings have been found, METEOR computes a score for this matching using a combination of unigram-precision, unigram-recall, and a measure of fragmentation that is designed to directly capture how well-ordered the matched words in the machine translation are in relation to the reference.
To take into account longer n-gram matches, a penalty factor is introduced: the longer the adjacent mappings between the candidate and the reference, the fewer chunks there are (a translation that is identical to the reference will give just one chunk).
The penalty has the effect of reducing the harmonic mean by up to 50% if there are no bigram or longer matches.
- precision: <img src="https://render.githubusercontent.com/render/math?math={P=\frac{m}{w_t}}##gh-light-mode-only">, where <img src="https://render.githubusercontent.com/render/math?math={m}##gh-light-mode-only"> is the number of unigrams in the hypothesis that are also found in the reference, and <img src="https://render.githubusercontent.com/render/math?math={w_t}##gh-light-mode-only"> is the number of unigrams in the hypothesis.
- recall: <img src="https://render.githubusercontent.com/render/math?math={R=\frac{m}{w_r}}##gh-light-mode-only">, where <img src="https://render.githubusercontent.com/render/math?math={w_r}##gh-light-mode-only"> is the number of unigrams in the reference.
- harmonic mean: <img src="https://render.githubusercontent.com/render/math?math={F_{mean}=\frac{10PR}{R+9P}}##gh-light-mode-only">, with recall weighted 9 times more than precision.
- penalty: <img src="https://render.githubusercontent.com/render/math?math={p=0.5(\frac{c}{u_m})^3}##gh-light-mode-only">, where <img src="https://render.githubusercontent.com/render/math?math={c}##gh-light-mode-only"> is the number of chunks, and <img src="https://render.githubusercontent.com/render/math?math={u_m}##gh-light-mode-only"> is the number of unigrams that have been mapped. <img src="https://render.githubusercontent.com/render/math?math={\frac{c}{m}}##gh-light-mode-only"> is also known as fragmentation fraction. The exponential value determines the functional relation between fragmentation and the penalty; it is also known as beta.
- final score: <img src="https://render.githubusercontent.com/render/math?math={M=F_{mean}(1-p)}##gh-light-mode-only">.
To calculate a score over a whole corpus, or collection of segments, the aggregate values for P, R and p are taken and then combined using the same formula.
The algorithm also works for comparing a candidate translation against more than one reference translations.
In this case the algorithm compares the candidate against each of the references and selects the highest score (f_reduce=max).

### Inputs
- **predictions** (`list`): prediction sentences. Each prediction should be a string with tokens separated by spaces.
- **references** (`list`): reference sentences. Each reference should be a string with tokens separated by spaces.
- **alpha** (`float`): optional, parameter for controlling relative weights of precision and recall. Default: 0.9.
- **beta** (`float`): optional, parameter for controlling shape of penalty as a function of fragmentation. Default: 3.
- **gamma** (`float`): optional, relative weight assigned to fragmentation penalty. Default: 0.5.

### Outputs
METEOR outputs a dictionary with the following values:
- **meteor** (`float`): average METEOR score.

### Results from popular papers
The [METEOR paper](https://aclanthology.org/W05-0909.pdf) does not report METEOR score values for different models, but it does report that METEOR gets an R correlation value of 0.347 with human evaluation on the Arabic data and 0.331 on the Chinese data. 

## Bounds
The range of METEOR score is  <img src="https://render.githubusercontent.com/render/math?math={[0, 1]}##gh-light-mode-only">.
The higher score means that the candidate is closer to reference translation (i.e., closer to human judgment).

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
scorer = NLGMetricverse(metrics=load_metric("meteor"))
predictions = ["the cat sat on the mat"]
references = ["on the mat sat the cat"]
scores = scorer(predictions=predictions, references=references)
print(scores)
{ 
  "meteor": { 
    'score': 0.5 
  } 
}
```

## Limitations and bias
While the correlation between METEOR and human judgments was measured for Chinese and Arabic and found to be significant, further experimentation is needed to check its correlation for other languages. 

Furthermore, while the alignment and matching done in METEOR is based on unigrams, using multiple word entities (e.g. bigrams) could contribute to improving its accuracy -- this has been proposed in [more recent publications](https://www.cs.cmu.edu/~alavie/METEOR/pdf/meteor-naacl-2010.pdf) on the subject.

## Citation
```bibtex
@inproceedings{banarjee2005,
  title     = {{METEOR}: An Automatic Metric for {MT} Evaluation with Improved Correlation with Human Judgments},
  author    = {Banerjee, Satanjeev  and Lavie, Alon},
  booktitle = {Proceedings of the {ACL} Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation
                and/or Summarization},
  month     = jun,
  year      = {2005},
  address   = {Ann Arbor, Michigan},
  publisher = {Association for Computational Linguistics},
  url       = {https://www.aclweb.org/anthology/W05-0909},
  pages     = {65--72},
}
```

## Further References
- [METEOR -- Wikipedia](https://en.wikipedia.org/wiki/METEOR)
- [METEOR score -- NLTK](https://www.nltk.org/_modules/nltk/translate/meteor_score.html)
- Refer to the [METEOR paper](https://aclanthology.org/W05-0909.pdf) for more information about parameter values and ranges.