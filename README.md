# BIO-R-BERT

**Relation Extraction** task on DDI 2013 Bio dataset

- Using [BioBERT](https://github.com/naver/biobert-pretrained) for pretrained model
- Using [R-BERT](https://arxiv.org/abs/1905.08284) for relation extraction modeling

## Model

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />  
</p>

## Dataset

- DDI (Drug-Drug Interaction) 2013 dataset ([link](http://labda.inf.uc3m.es/ddicorpus))
  - Relation Extraction task on Bioinformatics
  - 175 MEDLINE abstracts and 730 DrugBank documents
  - 5 DDI types (Negative, Mechanism, Effect, Advice, Int)
  - Use the preprocessed dataset from [this repo](https://github.com/arwhirang/DDI-recursive-NN)
  - Didn't replace the name of drug to `DRUG0`, `DRUG1`, or `DRUGN`, comparing to other researches

## How to use BioBERT for Transformers library

```python
# >>> from transformers import BertModel, BertTokenizer
# >>> model = BertModel.from_pretrained('monologg/biobert_v1.1_pubmed')
# >>> tokenizer = BertTokenizer.from_pretrained('monologg/biobert_v1.1_pubmed')
```

## Dependencies

- python>=3.5

- torch==1.1.0
- transformers>=2.2.2
- scikit-learn>=0.20.0

```bash
$ pip3 install -r requirements.txt
```

## How to run

You must give `--do_lower_case` option if pretrained model is uncased model.

```bash
$ python3 main.py --do_train --do_eval
```

## Results

`F1 micro score` on 4 Positive types (Mechanism, Effect, Advice, Int)

|                                                                                                                              | F1 micro (%) |
| ---------------------------------------------------------------------------------------------------------------------------- | ------------ |
| [CNN](https://www.researchgate.net/publication/292590022_Drug-Drug_Interaction_Extraction_via_Convolutional_Neural_Networks) | 69.75        |
| [AB-LSTM](https://arxiv.org/abs/1701.08303)                                                                                  | 69.39        |
| [MCCNN](https://www.hindawi.com/journals/bmri/2016/1850404/)                                                                 | 70.21        |
| [GCNN](https://arxiv.org/abs/1805.05593)                                                                                     | 72.55        |
| [Recursive NN](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190926)                                    | 73.50        |
| [RHCNN](https://www.mdpi.com/1099-4300/21/1/37)                                                                              | 75.48        |
| [SMGCN](https://www.aclweb.org/anthology/D19-6204.pdf)                                                                       | 76.64        |
| _BIO-R-BERT_                                                                                                                 | **82.66**    |

## Comparing the effect of pretrained language models

Using R-BERT architecture, with different pretrained weights

|                          | F1 Micro (%) |
| ------------------------ | ------------ |
| Random Init              | 47.04        |
| bert-base-cased          | 80.62        |
| scibert-scivocab-uncased | 81.30        |
| biobert_v1.0_pubmed_pmc  | 82.30        |
| biobert_v1.1_pubmed      | **82.66**    |

## References

- [BioBERT](https://github.com/naver/biobert-pretrained)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [R-BERT Repository](https://github.com/monologg/R-BERT)
- [DDI-recursive-NN Repository](https://github.com/arwhirang/DDI-recursive-NN)
- [AB-LSTM Repository](https://github.com/sunilitggu/DDI-extraction-through-LSTM)
- [MCCNN Repository](https://github.com/coddinglxf/DDI)
- [RHCNN Repository](https://github.com/DongKeee/DDIExtraction)
