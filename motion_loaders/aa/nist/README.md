# Metric Card for NIST

## Metric Description
NIST is a method for evaluating the quality of text which has been translated using machine translation.
Its name comes from the US National Institute of Standards and Technology.

The NIST metric was designed to improve BLEU by rewarding the translation of infrequently used words.
It is based on the BLEU metric, but with some alterations. 
Where BLEU simply calculates n-gram precision adding equal weight to each one, NIST also calculates how informative a particular n-gram is.
That is to say when a correct n-gram is found, the rarer that n-gram is, the more weight it will be given.
For example, if the bigram "on the" is correctly matched, it will receive lower weight than the correct matching of bigram "interesting calculations", as this is less likely to occur.
The final NIST score is calculated using the arithmetic mean of the ngram matches between candidate and reference translations.
In addition, a smaller brevity penalty is used for smaller variations in phrase lengths.
NIST also differs from BLEU in its calculation of the brevity penalty insofar as small variations in translation length do not impact the overall score as much.

The reliability and quality of the NIST metric has been shown to be superior to the BLEU metric in many cases.
The metric can be thought of as a variant of BLEU which weighs each matched n-gram based on its information gain, calculated as:
<img src="https://render.githubusercontent.com/render/math?math={Info(n-gram) = Info(w_1,\dots,w_n) = log_2 \frac{|\text{occurences of} w_1,\dots,w_{n-1}|}{|\text{occurences of} w_1,\dots,w_n|}}">.<br>
To sum up, the idea is to give more credit if a matched n-gram is rare and less credit if a matched n-gram is common.
This also reduces the chance of gaming the metric by producing trivial n-grams.

### Inputs
- **predictions** (`list`): prediction/candidate sentences. Each prediction should be a string with tokens separated by spaces.
- **references** (`list`): reference sentences. Each prediction should be a string with tokens separated by spaces.
- **n** (`str`): length of n-grams. Default: `5`.

### Outputs
NIST outputs a dictionary with the following values:
- **score** (`float`): the predicted NIST score.

### Results from popular papers

## Bounds
<img src="https://render.githubusercontent.com/render/math?math={[0, 1]}">, with 1 being the best.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
scorer = NLGMetricverse(metrics=load_metric("nist", n=5))
predictions = [
  ["the cat is on the mat", "There is cat playing on the mat"],
  ["Look! what a wonderful day, today.", "Today is a very wonderful day"]
]
references = [
  ["the cat is playing on the mat.", "The cat plays on the mat."], 
  ["Today is a wonderful day", "The weather outside is wonderful."]
]
scores = scorer(predictions=predictions, references=references)
print(scores)
{ 
  "nist": { 
    'score': 0.7240558145830365
  }
}
```

## Limitations and bias
It is sensitive to the n-gram frequency, but shares the same limits of BLEU.

## Citation
```bibtex
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
```

## Further References
- Refer to [Sai et al.](https://arxiv.org/pdf/2008.12009.pdf) for further details.