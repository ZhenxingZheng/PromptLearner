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
    def __init__(self, dataset, w_emb, q_emb, q_att, v_relation, controller,
                 joint_embedding, classifier, glimpse, fusion, relation_type, reason_step):
        super(ReGAT, self).__init__()
        self.name = "ReGAT_%s_%s" % (relation_type, fusion)
        self.relation_type = relation_type
        self.fusion = fusion
        self.dataset = dataset
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.q_att = q_att
        self.controller = controller
        self.v_relation = v_relation
        self.joint_embedding = joint_embedding
        self.classifier = classifier
        self.reason_step = reason_step

    def forward(self, v, b, q, mask, implicit_pos_emb, sem_adj_matrix,
                spa_adj_matrix, labels):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        pos: [batch_size, num_objs, nongt_dim, emb_dim]
        sem_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]
        spa_adj_matrix: [batch_size, num_objs, num_objs, num_edge_labels]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb_seq = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_emb_self_att = self.q_att(q_emb_seq)
        words_weight_list, control_vector_list = self.controller(q_emb_seq, q_emb_self_att, mask)

        # [batch_size, num_rois, out_dim]
        prediction = []
        for step in range(self.reason_step):
            v_emb = self.v_relation.forward(v, implicit_pos_emb,
                                            control_vector_list[step])
            if self.fusion == "ban":
                joint_emb, att = self.joint_embedding(v_emb, q_emb_seq, b)
            elif self.fusion == "butd":
                q_emb = self.q_emb(w_emb)  # [batch, q_dim]
                joint_emb, att = self.joint_embedding(v_emb, q_emb)
            else:  # mutan
                joint_emb, att = self.joint_embedding(v_emb, q_emb_self_att)
            if self.classifier:
                logits = self.classifier(joint_emb)
            else:
                logits = joint_emb
            prediction.append(logits)
        prediction = torch.stack(prediction, dim=-1)
        prediction = torch.mean(prediction, dim=-1)
        return prediction, att


def build_regat(dataset, args):
    print("Building ReGAT model with %s relation and %s fusion method" %
          (args.relation_type, args.fusion))
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600,
                              args.num_hid, 1, False, .0)
    q_att = QuestionSelfAttention(args.num_hid, .2)
    controller = Controller(args.num_hid, args.reason_step)

    v_relation = ImplicitRelationEncoder(
                    dataset.v_dim, args.num_hid, args.relation_dim,
                    args.dir_num, args.imp_pos_emb_dim, args.nongt_dim,
                    num_heads=args.num_heads, num_steps=args.num_steps,
                    residual_connection=args.residual_connection,
                    label_bias=args.label_bias)
    classifier = SimpleClassifier(args.num_hid, args.num_hid * 2,
                                  dataset.num_ans_candidates, 0.5)
    # relation_list = nn.ModuleList([v_relation])
    # classifier_list = nn.ModuleList([classifier])
    #
    # for i in range(args.reason_step):
    #     relation_list.append(v_relation)
    #     classifier_list.append(classifier)

    gamma = 0
    if args.fusion == "ban":
        joint_embedding = BAN(args.relation_dim, args.num_hid, args.ban_gamma)
        gamma = args.ban_gamma
    elif args.fusion == "butd":
        joint_embedding = BUTD(args.relation_dim, args.num_hid, args.num_hid)
    else:
        joint_embedding = MuTAN(args.relation_dim, args.num_hid,
                                dataset.num_ans_candidates, args.mutan_gamma)
        gamma = args.mutan_gamma
        classifier = None
    return ReGAT(dataset, w_emb, q_emb, q_att, v_relation, controller, joint_embedding,
                 classifier, gamma, args.fusion, args.relation_type, args.reason_step)
