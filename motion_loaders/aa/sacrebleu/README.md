# Metric Card for SacreBLEU

## Metric Description
SacreBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores. Inspired by Rico Sennrich's `multi-bleu-detok.perl`, it produces the official Workshop on Machine Translation (WMT) scores but works with plain text. It also knows all the standard test sets and handles downloading, processing, and tokenization.

See the [README.md] file at https://github.com/mjpost/sacreBLEU for more information.

### Inputs
- **predictions** (`list` of `str`): list of translations to score. Each translation should be tokenized into a list of tokens.
- **references** (`list` of `list` of `str`): A list of lists of references. The contents of the first sub-list are the references for the first prediction, the contents of the second sub-list are for the second prediction, etc. Note that there must be the same number of references for each prediction (i.e. all sub-lists must be of the same length).
- **smooth_method** (`str`): The smoothing method to use, defaults to `'exp'`. Possible values are:
    - `'none'`: no smoothing
    - `'floor'`: increment zero counts
    - `'add-k'`: increment num/denom by k for n>1
    - `'exp'`: exponential decay
- **smooth_value** (`float`): The smoothing value. Only valid when `smooth_method='floor'` (in which case `smooth_value` defaults to `0.1`) or `smooth_method='add-k'` (in which case `smooth_value` defaults to `1`).
- **tokenize** (`str`): Tokenization method to use for BLEU. If not provided, defaults to `'zh'` for Chinese, `'ja-mecab'` for Japanese and `'13a'` (mteval) otherwise. Possible values are:
    - `'none'`: No tokenization.
    - `'zh'`: Chinese tokenization.
    - `'13a'`: mimics the `mteval-v13a` script from Moses.
    - `'intl'`: International tokenization, mimics the `mteval-v14` script from Moses
    - `'char'`: Language-agnostic character-level tokenization.
    - `'ja-mecab'`: Japanese tokenization. Uses the [MeCab tokenizer](https://pypi.org/project/mecab-python3).
- **lowercase** (`bool`): If `True`, lowercases the input, enabling case-insensitivity. Defaults to `False`.
- **force** (`bool`): If `True`, insists that your tokenized input is actually detokenized. Defaults to `False`.
- **use_effective_order** (`bool`): If `True`, stops including n-gram orders for which precision is 0. This should be `True`, if sentence-level BLEU will be computed. Defaults to `False`.

### Outputs
- **score**: BLEU score
- **counts**: Counts
- **totals**: Totals
- **precisions**: Precisions
- **bp**: Brevity penalty
- **sys_len**: predictions length
- **ref_len**: reference length

### Results from Popular Papers

## Bounds
The score can take any value between `0.0` and `100.0`, inclusive.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
scorer = NLGMetricverse(metrics=load_metric("sacrebleu"))
predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
references = [
    ["the cat is playing on the mat.", "The cat plays on the mat."], 
    ["Today is a wonderful day", "The weather outside is wonderful."]
]
scores = scorer(predictions=predictions, references=references)
print(scores)
{
    "sacrebleu": {
    "score": 0.32377227131456443,
    "counts": [
        11,
        6,
        3,
        0
    ],
    "totals": [
        13,
        11,
        9,
        7
    ],
    "precisions": [
        0.8461538461538461,
        0.5454545454545454,
        0.33333333333333337,
        0.07142857142857144
    ],
    "bp": 1.0,
    "sys_len": 11,
    "ref_len": 12,
    "adjusted_precisions": [
        0.8461538461538461,
        0.5454545454545454,
        0.33333333333333337,
        0.07142857142857144
    ]
    }
}

from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["hello there general kenobi", "foo bar foobar"]
references = [["hello there general kenobi", "hello there !"], ["foo bar foobar", "foo bar foobar"]]
scorer = NLGMetricverse(metrics=load_metric("sacrebleu"))
scores = scorer(predictions=predictions, references=references)
print(list(scores.keys()))
['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
print(round(scores["score"], 1))
100.0
```

## Limitations and Bias
Because what this metric calculates is BLEU scores, it has the same limitations as that metric, except that sacreBLEU is more easily reproducible.

## Citation
```bibtex
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
```

## Further References
- See the [sacreBLEU README.md file](https://github.com/mjpost/sacreBLEU) for more information.