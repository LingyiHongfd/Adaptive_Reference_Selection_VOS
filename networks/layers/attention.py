import torch
import torch.nn.functional as F
import math
from torch import nn


class IA_gate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IA_gate, self).__init__()
        self.IA = nn.Linear(in_dim, out_dim)

    def forward(self, x, IA_head):
        a = self.IA(IA_head)
        a = 1. + torch.tanh(a)
        a = a.unsqueeze(-1).unsqueeze(-1)
        x = a * x
        return x


def calculate_attention_head_channel(ref_embedding, ref_label, prev_embedding, prev_label, curr_embedding,
                                     epsilon=1e-5):
    no, c, h, w = curr_embedding.size()

    '''
    print ('cur',torch.min(torch.norm(curr_embedding, dim=(2, 3), keepdim=True)))
    curr_embedding_n = curr_embedding / (torch.norm(curr_embedding, dim=(2, 3), keepdim=True)+1e-12)
    # print('curr_embedding_n 1', curr_embedding_n.size())
    curr_embedding_n = curr_embedding_n.reshape(no, c, -1)
    # print("curr_embedding_n 2", curr_embedding_n.size())
    print ('curr embed',torch.min(curr_embedding_n))


    # ref_head = ref_embedding * ref_label
    # ref_head_pos = torch.sum(ref_head, dim=(2, 3))
    # ref_head_neg = torch.sum(ref_embedding, dim=(2, 3)) - ref_head_pos

    ref_head_pos = ref_embedding * ref_label
    ref_head_neg = ref_embedding - ref_head_pos
    ref_head_pos_norm = torch.norm(ref_head_pos, dim=(2, 3), keepdim=True)+1e-12
    ref_head_neg_norm = torch.norm(ref_head_neg, dim=(2, 3), keepdim=True)+1e-12
    ref_head_pos_n = ref_head_pos / ref_head_pos_norm
    ref_head_neg_n = ref_head_neg / ref_head_neg_norm
    # print('ref_head_neg_n 1', ref_head_neg_n.size())
    ref_head_pos_n = ref_head_pos_n.permute(0, 2, 3, 1).reshape(no, -1, c)
    ref_head_neg_n = ref_head_neg_n.permute(0, 2, 3, 1).reshape(no, -1, c)
    # print('ref_head_neg_n 2', ref_head_neg_n.size())
    ref_head_p = torch.bmm(curr_embedding_n, ref_head_pos_n).reshape(no, -1)
    ref_head_n = torch.bmm(curr_embedding_n, ref_head_neg_n).reshape(no, -1)
    # print('ref_head_n 1', ref_head_n.size())
    ref_head_p = ref_head_p[:, 0:c * c:(c + 1)]
    ref_head_n = ref_head_n[:, 0:c * c:(c + 1)]
    # print('ref_head_n 2', ref_head_n.size())

    prev_head_pos = []
    prev_head_neg = []
    for i in range(prev_label.size(1)):
        prev_label_b = prev_label[:, i, ].unsqueeze(1)
        prev_embedding_b = prev_embedding[:, i * 100:(i + 1) * 100]
        # print('prev_embedding_b', prev_embedding_b.size())
        # print('prev_label_b', torch.unique(prev_label_b))
        prev_head_pos_b = prev_embedding_b * prev_label_b
        prev_head_neg_b = prev_embedding_b - prev_head_pos_b
        prev_head_pos_norm = torch.norm(prev_head_pos_b, dim=(2, 3), keepdim=True)+1e-12
        prev_head_neg_norm = torch.norm(prev_head_neg_b, dim=(2, 3), keepdim=True)+1e-12
        prev_head_pos_n = prev_head_pos_b / prev_head_pos_norm
        prev_head_neg_n = prev_head_neg_b / prev_head_neg_norm
        prev_head_pos_n = prev_head_pos_n.permute(0, 2, 3, 1).reshape(no, -1, c)
        prev_head_neg_n = prev_head_neg_n.permute(0, 2, 3, 1).reshape(no, -1, c)
        prev_head_p = torch.bmm(curr_embedding_n, prev_head_pos_n).reshape(no, -1)
        prev_head_n = torch.bmm(curr_embedding_n, prev_head_neg_n).reshape(no, -1)
        prev_head_p = prev_head_p[:, 0:c * c:(c + 1)]
        prev_head_n = prev_head_n[:, 0:c * c:(c + 1)]
        # print('prev_head_n', prev_head_n.size())
        prev_head_pos.append(prev_head_p)
        prev_head_neg.append(prev_head_n)
    prev_head_pos = torch.cat(prev_head_pos, 1)
    prev_head_neg = torch.cat(prev_head_neg, 1)
    # print('prev_head_pos', prev_head_pos.size())
    '''

    ref_head_pos = ref_embedding * ref_label
    ref_head_neg = ref_embedding - ref_head_pos
    ref_head_pos = ref_head_pos - curr_embedding
    ref_head_neg = ref_head_neg - curr_embedding
    ref_head_pos = 1 / (1 + torch.sqrt(torch.sum(torch.square(ref_head_pos), dim=(2, 3)) + 1e-5))
    ref_head_neg = 1 / (1 + torch.sqrt(torch.sum(torch.square(ref_head_neg), dim=(2, 3)) + 1e-5))
    prev_head_pos = []
    prev_head_neg = []
    for i in range(prev_label.size(1)):
        prev_label_b = prev_label[:, i, ].unsqueeze(1)
        prev_embedding_b = prev_embedding[:, i * 100:(i + 1) * 100]
        prev_head_pos_b = prev_embedding_b * prev_label_b
        prev_head_neg_b = prev_embedding_b - prev_head_pos_b
        prev_head_pos_b = prev_head_pos_b - curr_embedding
        prev_head_neg_b = prev_head_neg_b - curr_embedding
        prev_head_pos_b = 1 / (1 + torch.sqrt(torch.sum(torch.square(prev_head_pos_b), dim=(2, 3)) + 1e-5))
        prev_head_neg_b = 1 / (1 + torch.sqrt(torch.sum(torch.square(prev_head_neg_b), dim=(2, 3)) + 1e-5))
        prev_head_pos.append(prev_head_pos_b)
        prev_head_neg.append(prev_head_neg_b)
    prev_head_pos = torch.cat(prev_head_pos, 1)
    prev_head_neg = torch.cat(prev_head_neg, 1)

    # print ('ref_head_pos',torch.min(ref_head_pos),torch.max(ref_head_pos))

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], 1)
    # print('total head', torch.max(total_head), torch.min(total_head))
    # print ('init_head',total_head.size())
    # total_head = torch.zeros((no, 800)).cuda(0)
    # total_head = torch.cat([ref_head_p, ref_head_n, total_head], 1)

    return total_head


def calculate_attention_head(ref_embedding, ref_label, prev_embedding, prev_label, epsilon=1e-5):
    ref_head = ref_embedding * ref_label
    ref_head_pos = torch.sum(ref_head, dim=(2, 3))
    ref_head_neg = torch.sum(ref_embedding, dim=(2, 3)) - ref_head_pos
    ref_pos_num = torch.sum(ref_label, dim=(2, 3))
    ref_neg_num = torch.sum(1. - ref_label, dim=(2, 3))
    ref_head_pos = ref_head_pos / (ref_pos_num + epsilon)
    ref_head_neg = ref_head_neg / (ref_neg_num + epsilon)

    prev_head_pos = []
    prev_head_neg = []
    for i in range(prev_label.size(1)):
        prev_label_b = prev_label[:, i, ].unsqueeze(1)
        prev_embedding_b = prev_embedding[:, 100 * i:100 * (i + 1), ]
        prev_head = prev_embedding_b * prev_label_b
        prev_head_pos_b = torch.sum(prev_head, dim=(2, 3))
        prev_head_neg_b = torch.sum(prev_embedding_b, dim=(2, 3)) - prev_head_pos_b
        prev_pos_num_b = torch.sum(prev_label_b, dim=(2, 3))
        prev_neg_num_b = torch.sum(1. - prev_label_b, dim=(2, 3))
        prev_head_pos_b = prev_head_pos_b / (prev_pos_num_b + epsilon)
        prev_head_neg_b = prev_head_neg_b / (prev_neg_num_b + epsilon)
        prev_head_pos.append(prev_head_pos_b)
        prev_head_neg.append(prev_head_neg_b)
    prev_head_pos = torch.cat(prev_head_pos, 1)
    prev_head_neg = torch.cat(prev_head_neg, 1)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)

    return total_head


def calculate_attention_head_for_eval(ref_embeddings, ref_labels, prev_embedding, prev_label, epsilon=1e-5):
    total_ref_head_pos = 0.
    total_ref_head_neg = 0.
    total_ref_pos_num = 0.
    total_ref_neg_num = 0.

    for idx in range(len(ref_embeddings)):
        ref_embedding = ref_embeddings[idx]
        ref_label = ref_labels[idx]
        ref_head = ref_embedding * ref_label
        ref_head_pos = torch.sum(ref_head, dim=(2, 3))
        ref_head_neg = torch.sum(ref_embedding, dim=(2, 3)) - ref_head_pos
        ref_pos_num = torch.sum(ref_label, dim=(2, 3))
        ref_neg_num = torch.sum(1. - ref_label, dim=(2, 3))
        total_ref_head_pos = total_ref_head_pos + ref_head_pos
        total_ref_head_neg = total_ref_head_neg + ref_head_neg
        total_ref_pos_num = total_ref_pos_num + ref_pos_num
        total_ref_neg_num = total_ref_neg_num + ref_neg_num
    ref_head_pos = total_ref_head_pos / (total_ref_pos_num + epsilon)
    ref_head_neg = total_ref_head_neg / (total_ref_neg_num + epsilon)
    '''
    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)
    '''

    prev_head_pos = []
    prev_head_neg = []
    for i in range(prev_label.size(1)):
        prev_label_b = prev_label[:, i, ].unsqueeze(1)
        prev_embedding_b = prev_embedding[:, 100 * i:100 * (i + 1), ]
        prev_head = prev_embedding_b * prev_label_b
        prev_head_pos_b = torch.sum(prev_head, dim=(2, 3))
        prev_head_neg_b = torch.sum(prev_embedding_b, dim=(2, 3)) - prev_head_pos_b
        prev_pos_num_b = torch.sum(prev_label_b, dim=(2, 3))
        prev_neg_num_b = torch.sum(1. - prev_label_b, dim=(2, 3))
        prev_head_pos_b = prev_head_pos_b / (prev_pos_num_b + epsilon)
        prev_head_neg_b = prev_head_neg_b / (prev_neg_num_b + epsilon)
        prev_head_pos.append(prev_head_pos_b)
        prev_head_neg.append(prev_head_neg_b)
    prev_head_pos = torch.cat(prev_head_pos, 1)
    prev_head_neg = torch.cat(prev_head_neg, 1)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    return total_head
