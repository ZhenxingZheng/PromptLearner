"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import torch
import torch.nn as nn
from modules.fusion import BAN, BUTD, MuTAN
from modules.language_model import WordEmbedding, QuestionEmbedding,\
                                 QuestionSelfAttention
from modules.relation_encoder import ImplicitRelationEncoder,\
                                   ExplicitRelationEncoder
from modules.classifier import SimpleClassifier
from modules.controller import Controller

class ReGAT(nn.Module):
    def __init__(self, q_att, v_relation, controller,
                 joint_embedding, classifier, fusion, reason_step):
        super(ReGAT, self).__init__()
        self.name = "ReGAT_%s_%s" % ('implicit', fusion)
        self.q_att = q_att
        self.v_relation = v_relation
        self.controller = controller
        self.joint_embedding = joint_embedding
        self.classifier = classifier
        self.fusion = fusion
        self.reason_step = reason_step

    def forward(self, v_emb, q_emb_seq, q_emb, mask, implicit_pos_emb=None):
        """Forward
        v_emb: [batch, num_objs, obj_dim]
        q_emb_seq: [batch_size, seq_length, text_dim]
        q_emb: [batch_size, text_dim]
        mask: [batch_size, seq_length]

        return: logits, not probs
        """
        q_emb_self_att = self.q_att(q_emb_seq)
        words_weight_list, control_vector_list = self.controller(q_emb_seq, q_emb_self_att, mask)

        # [batch_size, num_rois, out_dim]
        prediction = []
        for step in range(self.reason_step):
            v_emb = self.v_relation.forward(v_emb, implicit_pos_emb,
                                            control_vector_list[step])
            # v_emb = self.v_relation.forward(v_emb, implicit_pos_emb,
            #                                 control_vector_list[step])
            if self.fusion == "ban":
                joint_emb, att = self.joint_embedding(v_emb, q_emb_seq)
            elif self.fusion == "butd":
                joint_emb, att = self.joint_embedding(v_emb, q_emb)
            else:  # mutan
                joint_emb, att = self.joint_embedding(v_emb, q_emb_self_att)
            if self.classifier:
                logits = self.classifier(joint_emb)
            else:
                logits = joint_emb
            # prediction.append(logits)
        # prediction = torch.stack(prediction, dim=-1)
        # prediction = torch.mean(prediction, dim=-1)
        return joint_emb


def build_regat(question_dim=512, num_hid=512, reason_step=1, graph_step=1, v_dim=512, relation_dim=512, dir_num=2,
                imp_pos_emb_dim=-1, nongt_dim=20, num_heads=16, num_ans_candidates=3129,
                fusion='butd', relation_type='implicit'):
    print("Building ReGAT model with %s relation and %s fusion method" %
          (relation_type, fusion))

    q_att = QuestionSelfAttention(question_dim, .2)
    controller = Controller(question_dim, reason_step)

    v_relation = ImplicitRelationEncoder(
                    v_dim, question_dim, relation_dim,
                    dir_num, imp_pos_emb_dim, nongt_dim,
                    num_heads=num_heads, num_steps=graph_step,
                    residual_connection=True,
                    label_bias=False)
    classifier = SimpleClassifier(v_dim+relation_dim, 512,
                                  num_ans_candidates, 0.5)

    gamma = 0
    ban_gamma = 1
    mutan_gamma = 2
    if fusion == "ban":
        joint_embedding = BAN(v_dim, num_hid, ban_gamma)
        gamma = ban_gamma
    elif fusion == "butd":
        joint_embedding = BUTD(relation_dim, question_dim, num_hid)
    else:
        joint_embedding = MuTAN(v_dim, num_hid,
                                num_ans_candidates, mutan_gamma)
        gamma = mutan_gamma
        classifier = None
    return ReGAT(q_att, v_relation, controller, joint_embedding,
                 None, fusion, reason_step)

if __name__ == '__main__':
    print('haha')
    word_dim = 512
    visual_dim = 512
    word_max_length = 14
    reason_step = 1

    net = build_regat().cuda()
    v = torch.randn(2, 12, 512).cuda()
    q_emb_seq = torch.randn(2, word_max_length, word_dim).cuda()
    q_emb = torch.randn(2, word_dim).cuda()
    mask = torch.ones(2, word_max_length).cuda()
    mask[1, 12:word_max_length] = 0

    output = net(v, q_emb_seq, q_emb, mask)
    print(output.size())