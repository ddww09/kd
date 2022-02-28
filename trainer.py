import math
import os
import logging
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from model import RBERT
from utils import set_seed, compute_metrics, get_label, compute_metrics_test
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        # 加载模型配置参数隐层数、隐层维度、激活函数and so on
        self.bert_config = BertConfig.from_pretrained(args.pretrained_model_name, num_labels=self.num_labels,
                                                      finetuning_task=args.task)
        # 通过配置参数加载模型
        self.model = RBERT(self.bert_config, args)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        # 打乱数据集
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (
                    len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        # f=open('./loss.txt','a')
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)
        loss_holder = []  # 定义损失数组，用于可视化训练过程
        loss_value = np.inf  # 损失值设置为无限大， 每次迭代若损失值比loss_value小则保存模型，将最新的损失值赋值给loss_value

        f = open('./loss.txt', 'a')
        count_model = 0
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5]}
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                    print(2)

                loss.backward()
                loss_save = str(loss.item())
                f.write(loss_save)
                f.write('\n')
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    # 只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整
                    optimizer.step()  # 单次优化 所有的optimizer都实现了step()方法，这个方法会更新所有的参数
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()  # 梯度清零
                    global_step += 1  #

                    # 每10次保留一次loss
                    if global_step % 100 == 0:
                        loss_holder.append([global_step, loss])

                    # 每100次测试一下
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate('dev')

                    # 每50次保存比较一下模型 （比较loss）
                    #if self.args.save_steps > 0 and global_step % self.args.save_steps == 0 and loss < loss_value:
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        # loss_value = loss
                        # count_model = count_model + 1
                        # print("第%d次迭代", global_step)
                        # print("第%d次更新", count_model)
                        # print(loss_value, loss)  # 打印loss与loss_value 调试， 运行时删除即可
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
            # aa=str(loss_holder)
            # f.write(loss_holder)

        # 画图
        # fig = plt.figure(figsize=(20, 15))
        # fig.autofmt_xdate()
        # loss_df = pd.DataFrame(loss_holder, columns=["time", "loss"])
        # x_times = loss_df["time"].values
        # plt.ylabel("loss")
        # plt.xlabel("times")
        # plt.plot(loss_df["loss"].values)
        # plt.show()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        # eval_dataset = RandomSampler(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        preds = np.argmax(preds, axis=1)
        if mode=='dev':
            result = compute_metrics(preds, out_label_ids)
        else:
            result = compute_metrics_test(preds, out_label_ids)

        results.update(result)
        f = open('./eval.txt', 'a')
        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            f.writelines(key)
            f.write('\t')
            f.writelines(str(results[key]))
            f.write('\n')
        f.close()
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model path doesn't exists! ")
        output_dir = os.path.join(self.args.model_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = BertConfig.from_pretrained(self.args.model_dir)
            logger.info("***** Bert config loaded *****")
            self.model = RBERT.from_pretrained(self.args.model_dir, config=self.bert_config, args=self.args)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
