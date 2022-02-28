import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
import numpy as np
import pandas as pd


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(RBERT, self).__init__(bert_config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels

        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)  # fc全连接
        self.e1_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.e2_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 3, bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        self.out_layer = FCLayer(bert_config.hidden_size * 3, bert_config.hidden_size * 3, args.dropout_rate)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]   unsqueeze()--->扩充维度
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(
            1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    # 当调用trainer.py outputs = self.model(**inputs)执行
    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # 新加
        e1_arr = []
        e2_arr = []
        # 计算e1_id, e2_id
        e2_info = []
        e1_info = []
        e1info_attention_mask = []
        e2info_attention_mask = []
        batch = e1_h.shape[0]
        e1_id = (input_ids * e1_mask).cpu().numpy().tolist()
        e2_id = (input_ids * e2_mask).cpu().numpy().tolist()

        f = open('data/extra_dict.tsv', 'r')

        for i in range(batch):
            #  提取实体
            a = pd.DataFrame(e1_id[i]).replace(0, np.NAN)
            a.dropna(inplace=True)
            a = np.array(a).astype(np.int32).reshape(1, -1)[0]
            a = a.tolist()
            # 删除最后一个109
            a.pop()
            # 删除第一个109
            del (a[0])

            e1_arr.append(a)

            b = pd.DataFrame(e2_id[i]).replace(0, np.NAN)
            b.dropna(inplace=True)
            b = np.array(b).astype(np.int32).reshape(1, -1)[0]
            b = b.tolist()
            # 删除最后一个109
            b.pop()
            # 删除第一个109
            del (b[0])
            e2_arr.append(b)

        # 在extra文件中相同的实体，如果找到返回后边的实体解释

        for i in range(batch):
            lines = f.readlines()
            empty_list = [1] * 400
            ini_attention_mask = [0] * 400
            e1_info.append(empty_list)
            e2_info.append(empty_list)
            e1info_attention_mask.append(ini_attention_mask)
            e2info_attention_mask.append(ini_attention_mask)

            for line in lines:
                e1_ok = 0
                e2_ok = 0
                e1_id = str(e1_arr[i])
                w1, w2 = line.strip('\n').split('\t')
                if e1_id == w1 and e1_ok == 0:
                    w2 = w2.replace('[', '').replace(']', '')
                    w2 = w2.split(',')
                    c = list(map(int, w2))

                    aa = pd.DataFrame(c).replace(0, np.NAN)
                    aa.dropna(inplace=True)
                    aa = np.array(aa).astype(np.int32).reshape(1, -1)[0]
                    aa = aa.tolist()
                    e_mask_len = len(aa)
                    e1_info[i] = c
                    e1info_attention_mask[i] = [1] * e_mask_len + [0] * (400 - e_mask_len)
                    e1_ok = 1

                e2_id = str(e2_arr[i])
                if e2_id == w1 and e2_ok == 0:
                    w2 = w2.replace('[', '').replace(']', '')
                    w2 = w2.split(',')
                    d = list(map(int, w2))

                    aa = pd.DataFrame(d).replace(0, np.NAN)
                    aa.dropna(inplace=True)
                    aa = np.array(aa).astype(np.int32).reshape(1, -1)[0]
                    aa = aa.tolist()
                    e_mask_len = len(aa)
                    e2info_attention_mask[i] = [1] * e_mask_len + [0] * (400 - e_mask_len)
                    e2_info[i] = d
                    e2_ok = 1

                if e1_ok == 1 and e2_ok == 1:
                    break

        f.close()

        # 把得到的信息送入模型(之后把attention换成e1attention，e2attention)

        # 需要先变成tensor（）
        e1_info = torch.LongTensor(e1_info)
        e2_info = torch.LongTensor(e2_info)
        e1info_attention_mask = torch.LongTensor(e1info_attention_mask)
        e2info_attention_mask = torch.LongTensor(e2info_attention_mask)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        e1_info = e1_info.to(device)
        e2_info = e2_info.to(device)
        e1info_attention_mask = e1info_attention_mask.to(device)
        e2info_attention_mask = e2info_attention_mask.to(device)

        e1_outputs = self.bert(e1_info, e1info_attention_mask, token_type_ids=token_type_ids)
        e2_outputs = self.bert(e2_info, e2info_attention_mask, token_type_ids=token_type_ids)

        e1_sequence_output = e1_outputs[0]
        e2_sequence_output = e2_outputs[0]
        e1_pooled_output = e1_outputs[1]
        e2_pooled_output = e2_outputs[1]

        # 不参加实体平均

        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)

        # 与cls一样 information
        e1_pooled_output = self.cls_fc_layer(e1_pooled_output)
        e2_pooled_output = self.cls_fc_layer(e2_pooled_output)

        e1_h = self.e1_fc_layer(e1_h)
        e2_h = self.e2_fc_layer(e2_h)

        # Concat -> fc_layer
        lianhe1 = e1_h + e1_pooled_output * 0.4
        lianhe2 = e2_h + e2_pooled_output * 0.4

        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)

        concat_h2 = torch.cat([pooled_output, lianhe1, lianhe2], dim=-1)

        m1 = nn.LayerNorm(concat_h.size()[0:]).cuda()
        concat_h = m1(concat_h)
        # concat_h=self.out_layer(concat_h)  # dropout

        m2 = nn.LayerNorm(concat_h2.size()[0:]).cuda()
        concat_h2 = m2(concat_h2)
        # concat_h2 = self.out_layer(concat_h2)  # dropout

        logits = self.label_classifier(concat_h)
        logits2 = self.label_classifier(concat_h2)
        #lianhe = logits * 0.5 + logits2 * 0.5

        # 先尝试2
        #outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        outputs = (logits2,) + outputs[2:]  # add hidden states and attention if they are here
        #outputs3 = (lianhe,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits2.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()

                loss = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)
