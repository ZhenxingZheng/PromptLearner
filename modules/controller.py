import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
from modules.language_model import QuestionEmbedding, QuestionSelfAttention
import os


class Controller(nn.Module):

    def __init__(self, dim_word_output, T_ctrl):
        super(Controller, self).__init__()
        ctrl_dim = dim_word_output

        # define c_0 and reset_parameters
        self.c_init = Parameter(torch.FloatTensor(1, ctrl_dim))
        self.reset_parameters()

        # define fc operators
        self.encode_que_list = nn.ModuleList([nn.Sequential(nn.Linear(ctrl_dim, ctrl_dim),
                                                            nn.Tanh(),
                                                            nn.Linear(ctrl_dim, ctrl_dim))])
        for i in range(T_ctrl - 1):
            self.encode_que_list.append(nn.Sequential(nn.Linear(ctrl_dim, ctrl_dim),
                                                      nn.Tanh(),
                                                      nn.Linear(ctrl_dim, ctrl_dim)))
        self.fc1 = nn.Linear(2*ctrl_dim, ctrl_dim)
        self.fc2 = nn.Linear(ctrl_dim, 1)
        self.fc3 = nn.Linear(2*ctrl_dim, ctrl_dim)

        self.T_ctrl = T_ctrl

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.c_init.size(1))
        self.c_init.data.uniform_(-stdv, stdv)

    # self.controller(context, hidden, is_not_pad_sents)
    def forward(self, q_emb_seq, q_encoding, attn_mask):
        c_prev = self.c_init.expand(q_encoding.size(0), self.c_init.size(1)) # batch, hidden_size

        words_weight_list = []
        control_vector_list = []

        for t in range(self.T_ctrl):
            q_i = self.encode_que_list[t](q_encoding)# batch, hidden_size
            q_i_c = torch.cat([q_i, c_prev], dim=1)# batch, hidden_size*2

            cq_i = self.fc1(q_i_c)# batch, hidden_size
            cq_i_reshape = cq_i.unsqueeze(1).expand(-1, q_emb_seq.size(1), -1)# batch, max_len, hidden_size
            interactions = cq_i_reshape * q_emb_seq # batch, max_len, hidden_size

            interactions = torch.cat([interactions, q_emb_seq], dim=2)# batch, max_len, hidden_size*2
            interactions = torch.tanh(self.fc3(interactions))# batch, max_len, hidden_size
            logits = self.fc2(interactions).squeeze(2)# batch, max_len
            mask = (1.0 - attn_mask.float()) * (-1e30)# batch, max_len

            logits = logits + mask
            logits = F.softmax(logits, dim=1)# batch, max_len

            norm_cv_i = logits * attn_mask.float()# batch, max_len
            norm_cv_i_sum = torch.sum(norm_cv_i, dim=1).unsqueeze(1).expand(logits.size(0), logits.size(1))
            norm_cv_i[norm_cv_i_sum != 0] = norm_cv_i[norm_cv_i_sum != 0] / norm_cv_i_sum[norm_cv_i_sum != 0]


            words_weight_list.append(norm_cv_i)

            c_i = torch.sum(
                norm_cv_i.unsqueeze(2).expand(norm_cv_i.size(0), norm_cv_i.size(1), q_emb_seq.size(2)) * q_emb_seq, dim=1)
            c_prev = c_i
            control_vector_list.append(c_prev)

        return words_weight_list, control_vector_list


if __name__ == '__main__':
    word_dim = 768
    word_max_length = 24
    reason_step = 3

    controller = Controller(word_dim, reason_step).cuda()
    q_emb_seq = torch.randn(2, word_max_length, word_dim).cuda() # (batch, word_len, hidden)
    q_emb_self_att = torch.randn(2, word_dim).cuda()
    is_not_pad_sents = torch.ones(2, word_max_length).cuda()
    is_not_pad_sents[1, 12:word_max_length] = 0
    words_weight_list, control_vector_list = controller(q_emb_seq, q_emb_self_att, is_not_pad_sents) #output (batch, word_len), (batch, hidden)
    print(len(control_vector_list))
    print(control_vector_list[0].size())
    print(words_weight_list[0].size())
    print('haha')


