import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # self.generator = nn.Linear(hidden_size, num_classes)
        self.char_head = nn.Linear(self.hidden_size, num_classes) if num_classes > 0 else nn.Identity()
        self.bpe_head = nn.Linear(self.hidden_size, 50257) if num_classes > 0 else nn.Identity()
        self.wp_head = nn.Linear(self.hidden_size, 30522) if num_classes > 0 else nn.Identity()

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1) #192,1
        batch_size = input_char.size(0) #192
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device) #192,38
        one_hot = one_hot.scatter_(1, input_char, 1) #192,38 得看一下scatter函数的作用才能理解该函数
        return one_hot

    def forward(self, batch_H, text, bpe_text, wp_text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0) #192
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.  26

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)  #192,26,256
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device)) #2个元素的元组 每个元素是192,256的张量

        self.context_history = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)  #192,26,256
        self.alpha_history = []
        if is_train:  #训练过程
            for i in range(num_steps): #26
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                # batch_H [batch_size,times, feaiture_dims)
                # alpha [batch_size,times,1]
                # cur_time: context [batch_size,feature_dims]
                hidden, alpha, context = self.attention_cell(hidden, batch_H, char_onehots)

                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
                self.alpha_history.append(alpha)
                self.context_history[:, i, :] =context

            char_probs = self.char_head(output_hiddens)
            bpe_probs = self.bpe_head(output_hiddens)
            wp_probs = self.wp_head(output_hiddens)


        else:  #预测过程
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token  长为192的全0张量
            char_probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device) #192,26,38的全0张量
            bpe_probs = torch.FloatTensor(batch_size, num_steps, 50257).fill_(0).to(device) #192,26,50257
            wp_probs = torch.FloatTensor(batch_size, num_steps, 30522).fill_(0).to(device) #192,26,30522

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)  #192,38 对应这一batch中所有图片第i个字符的one-hot嵌入 此时除了第一列为1，其他列全为0
                hidden, alpha, context = self.attention_cell(hidden, batch_H, char_onehots) #这里进入注意力机制的逻辑 还有GRU部分
                char_probs_step = self.char_head(hidden[0]) #192,38
                char_probs[:, i, :] = char_probs_step
                _, next_input = char_probs_step.max(1) #192的张量，每一元素是原先张量在第1维度的最大值和索引

                bpe_probs_step = self.bpe_head(hidden[0])
                bpe_probs[:, i, :] = bpe_probs_step

                wp_probs_step = self.wp_head(hidden[0])
                wp_probs[:, i, :] = wp_probs_step


                targets = next_input
                self.alpha_history.append(alpha)
                self.context_history[:, i, :] = context
        self.alpha_history = torch.cat(self.alpha_history, -1) #192,26,26
        self.alpha_history.permute(0,2,1)  # batch_size x num_steps x num_classes   192,26,26
        return char_probs, bpe_probs, wp_probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):  #prev_hidden:2个元素的元组，每个元素192,256的张量 batch_h：192,26,256 char_onehots:192,38 除了第一列为1其他列全为0
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]

        batch_H_proj = self.i2h(batch_H)  #192,26,256 这是f
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1) #192,1,256 这是hk-1
        # e= v^T tanh(Ws*s_{t-1} + Wh*H) : H batch_size,times,feature_dims,
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1  192,26,1
        alpha = F.softmax(e, dim=1) #在序列T维度上进行softmax 192,26,1
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel 192,256
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding) 192,256+38=294
        cur_hidden = self.rnn(concat_context, prev_hidden)  #GRU concat_context中，context代表ck,char_onehots是yk-1,prev_hidden代表hk-1，结果得到的cur_hidden是hk 也是2个元素的元组，每个元素是192,256的张量
        return cur_hidden, alpha,context
