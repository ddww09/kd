import os
import random
import logging

import sklearn.metrics as sm
import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as mp
from sklearn.metrics import precision_recall_fscore_support
# from transformers.tokenization_bert import BertTokenizer
from transformers import BertTokenizer

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    # Download vocabulary  and cache
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name, do_lower_case=args.do_lower_case)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def init_logger():
    # 字符串格式、日期格式、日志记录器级别
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def compute_metrics_test(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1_test(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


# def acc_and_f1(preds, labels, average='micro'):
# def acc_and_f1(preds, labels, average="macro"):
#     acc = simple_accuracy(preds, labels)
#     # f1 = f1_score(y_true=labels, y_pred=preds, labels=[1, 2, 3, 4], average=average)
#     f1 = f1_score(y_true=labels, y_pred=preds, average=average)
#     return {
#         "acc": acc,
#         "f1": f1,
#     }
def acc_and_f1(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)

    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    # f1 = f1_score(y_true=labels, y_pred=preds, labels=[1, 2, 3, 4], average=average)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    # p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
    #      y_true=labels, y_pred=preds, labels=[1, 2, 3, 4, 5], average=None)
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
        y_true=labels, y_pred=preds, labels=[0, 1, 2, 3, 4], average=None)
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f_class": f_class,
    }


def acc_and_f1_test(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)

    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    # f1 = f1_score(y_true=labels, y_pred=preds, labels=[1, 2, 3, 4], average=average)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    # p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
    #      y_true=labels, y_pred=preds, labels=[1, 2, 3, 4, 5], average=None)
    p_class, r_class, f_class, support_micro = precision_recall_fscore_support(
        y_true=labels, y_pred=preds, labels=[0, 1, 2, 3, 4], average=None)
    m = sm.confusion_matrix(y_true=labels, y_pred=preds)
    print('混淆矩阵为：', m, sep='\n')

    # 画出混淆矩阵
    mp.figure('Confusion Matrix')
    mp.xticks([])
    mp.yticks([])
    mp.imshow(m, cmap='gray')
    mp.show()
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f_class": f_class,
        "m": m,
    }

