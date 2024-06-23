import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from layer import HGATLayer
from torch_geometric.nn import GCNConv
import torch.nn.init as init
import Constants
from TransformerBlock import TransformerBlock
from torch.autograd import Variable


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):  #
        super(HGNN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.weight1 = nn.Parameter(torch.Tensor(in_ft, out_ft))
        init.xavier_uniform_(self.weight)  
        init.xavier_uniform_(self.weight1)
        # self.edge = nn.Embedding(984, out_ft)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)


        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, x, G):  # x: torch.Tensor, G: torch.Tensor
        edge_emb = nn.Embedding(984, 64)
        x = G.matmul(edge_emb.weight.cuda())
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        edge = G.t().matmul(x)
        edge = edge.matmul(self.weight1)
        x = G.matmul(edge)

        return x

class HGNN2(nn.Module):
    def __init__(self, emb_dim, dropout=0.15):
        super(HGNN2, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(emb_dim, emb_dim)
        self.hgc2 = HGNN_conv(emb_dim, emb_dim)
        # self.bn1 = nn.BatchNorm1d(emb_dim)
        # self.bn2 = nn.BatchNorm1d(emb_dim)
        # self.feat = nn.Embedding(n_node, emb_dim)
        # self.feat_idx = torch.arange(n_node).cuda()
        # nn.init.xavier_uniform_(self.feat.weight)
        self.fc1 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.fc2 = nn.Linear(emb_dim, emb_dim, bias=False)
        self.weight = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

    def forward(self, x, G):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = F.relu(x,inplace = False)
        x = self.hgc1(x, G)        
        x = self.hgc2(x, G)
        # x = F.dropout(x, self.dropout)
        x = F.softmax(x,dim = 1)
        return x


def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
    masked_seq = Variable(masked_seq, requires_grad=False)
    # print("masked_seq ",masked_seq.size())
    return masked_seq.cuda()


# Fusion gate
class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.1):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out


'''Learn friendship network'''


class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.1, is_norm=True):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)
        # in:inp,out:nip*2
        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)
        self.is_norm = is_norm

        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(ninp)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)
        # print(graph_output.shape)
        return graph_output.cuda()


'''Learn diffusion network'''

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.1):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out))
        # out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
        return hidden


class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.1, is_norm=True):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.is_norm = is_norm
        if self.is_norm:
            self.batch_norm1 = torch.nn.BatchNorm1d(output_size)
        self.gat1 = HGATLayer(input_size, output_size, dropout=self.dropout, transfer=False, concat=True, edge=True)
        self.fus1 = Fusion(output_size)
        # self.hgnn = DJconv(64, 64, 1)
        # self.hgnn = HGNN_conv(input_size, output_size, True)
        self.hgnn = HGNN2(input_size, 0.3)

    def forward(self, x, hypergraph_list):
        root_emb = F.embedding(hypergraph_list[1].cuda(), x)

        hypergraph_list = hypergraph_list[0]
        embedding_list = {}
        for sub_key in hypergraph_list.keys():
            sub_graph = hypergraph_list[sub_key]
            sub_node_embed, sub_edge_embed = self.gat1(x, sub_graph.cuda(), root_emb)
            sub_node_embed = self.hgnn(x, sub_graph.cuda())
            sub_node_embed = F.dropout(sub_node_embed, self.dropout, training=self.training)

            if self.is_norm:
                sub_node_embed = self.batch_norm1(sub_node_embed)
                sub_edge_embed = self.batch_norm1(sub_edge_embed)

            x = self.fus1(x, sub_node_embed)
            embedding_list[sub_key] = [x.cpu(), sub_edge_embed.cpu()]

        return embedding_list


class MLPReadout(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        """
        out_dim: the final prediction dim, usually 1
        act: the final activation, if rating then None, if CTR then sigmoid
        """
        super(MLPReadout, self).__init__()
        self.layer1 = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.out_act = act

    def forward(self, x):
        ret = self.layer1(x)
        return ret


class MSHGAT(nn.Module):
    def __init__(self, opt, dropout=0.1):
        super(MSHGAT, self).__init__()
        self.hidden_size = opt.d_word_vec
        self.n_node = opt.user_size
        self.pos_dim = 8
        self.dropout = nn.Dropout(dropout)
        self.initial_feature = opt.initialFeatureSize

        self.hgnn = HGNN_ATT(self.initial_feature, self.hidden_size * 2, self.hidden_size, dropout=dropout)
        self.gnn = GraphNN(self.n_node, self.initial_feature, dropout=dropout)
        self.fus = Fusion(self.hidden_size + self.pos_dim)
        self.fus2 = Fusion(self.hidden_size)
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)
        self.decoder_attention1 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)
        self.decoder_attention2 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)

        self.linear2 = nn.Linear(self.hidden_size + self.pos_dim, self.n_node)
        self.embedding = nn.Embedding(self.n_node, self.initial_feature, padding_idx=0)
        self.reset_parameters()
        self.readout = MLPReadout(self.hidden_size, self.n_node, None)
        self.GRU = GRUNet(self.hidden_size, self.hidden_size, self.hidden_size, 1)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def pred(self, pred_logits):
        predictions = self.readout(pred_logits)
        return predictions

    def forward(self, input, input_timestamp, input_idx, graph, hypergraph_list):

        input = input[:, :-1]
        # print(input)
        # print(input_timestamp)
        input_timestamp = input_timestamp[:, :-1]
        hidden = self.dropout(self.gnn(graph))
        memory_emb_list = self.hgnn(hidden, hypergraph_list)
        # print(sorted(memory_emb_list.keys()))

        mask = (input == Constants.PAD)
        batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()
        order_embed = self.dropout(self.pos_embedding(batch_t))
        batch_size, max_len = input.size()

        zero_vec = torch.zeros_like(input)
        dyemb = torch.zeros(batch_size, max_len, self.hidden_size).cuda()
        cas_emb = torch.zeros(batch_size, max_len, self.hidden_size).cuda()
        # print("batch_size", batch_size)
        # print("max_len", max_len)
        h = self.GRU.init_hidden(batch_size*max_len)
        sub_emb_list = []
        cas_emb_list = []

        for ind, time in enumerate(sorted(memory_emb_list.keys())):
            if ind == 0:
                sub_input = torch.where(input_timestamp <= time, input, zero_vec)
                sub_emb = F.embedding(sub_input.cuda(), hidden.cuda())
                temp = sub_input == 0
                sub_cas = sub_emb.clone()
            else:
                cur = torch.where(input_timestamp <= time, input, zero_vec) - sub_input
                temp = cur == 0

                sub_cas = torch.zeros_like(cur)
                sub_cas[~temp] = 1
                sub_cas = torch.einsum('ij,i->ij', sub_cas, input_idx)
                sub_cas = F.embedding(sub_cas.cuda(), list(memory_emb_list.values())[ind - 1][1].cuda())
                sub_emb = F.embedding(cur.cuda(), list(memory_emb_list.values())[ind - 1][0].cuda())
                sub_input = cur + sub_input

            sub_cas[temp] = 0
            sub_emb[temp] = 0
            dyemb += sub_emb
            cas_emb += sub_cas

            if ind == len(memory_emb_list) - 1:
                sub_input = input - sub_input
                temp = sub_input == 0

                sub_cas = torch.zeros_like(sub_input)
                sub_cas[~temp] = 1
                sub_cas = torch.einsum('ij,i->ij', sub_cas, input_idx)
                sub_cas = F.embedding(sub_cas.cuda(), list(memory_emb_list.values())[ind - 1][1].cuda())
                sub_cas[temp] = 0
                sub_emb = F.embedding(sub_input.cuda(), list(memory_emb_list.values())[ind][0].cuda())
                sub_emb[temp] = 0

                dyemb += sub_emb
                casemb += sub_cas
            
            sub_cas_ = sub_cas.view(-1, sub_cas.size(-1))
            dy_emb_ = dyemb.view(-1, dyemb.size(-1))
            
            # dy_emb_list.append(dy_emb_)
            cas_emb_list.append(sub_cas_)
            
            # dy_emb = torch.stack(dy_emb_list, dim=1) 
            cas_emb = torch.stack(cas_emb_list, dim=1) 
            emb =  Fusion(dy_emb_,cas_emb)
        
        GRUoutput, h = self.GRU(emb, h)   
        dy_output = GRUoutput.sum(dim=1)  
        pred = self.pred(output)
        # print("pred.shape:", pred.size())
        # pred = self.pred(dyemb)
        # return pred.view(-1, pred.size(-1))
        return pred
