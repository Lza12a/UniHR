import torch
import torch.nn as nn
import math
from dgl.nn import GATv2Conv
import dgl
import dgl.function as fn

class CompGCNCov(nn.Module):
    """ The comp graph convolution layers, similar to https://github.com/malllabiisc/CompGCN"""
    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0., opn='corr'):
        super(CompGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.opn = opn
        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.rel_wt = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges: dgl.udf.EdgeBatch):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w[edge_dir.squeeze()]).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # msg = torch.bmm(edge_data.unsqueeze(1),
        #                 self.w.index_select(0, edge_dir.squeeze())).squeeze()  # [E, 1, in_c] @ [E, in_c, out_c]
        # first half edges are all in-directions, last half edges are out-directions.
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes: dgl.udf.NodeBatch):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a.real, a.imag
            r2, i2 = b.real, b.imag
            real = r1 * r2 - i1 * i2
            imag = r1 * i2 + i1 * r2
            return torch.complex(real, imag)

        def conj(a):
            a.imag = -a.imag
            return a

        def ccorr(a, b):
            return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a)), torch.fft.rfft(b)), a.shape[-1])

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.graph, x, rel_repr, edge_type, edge_norm):
        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        edge_type = edge_type.to(self.device)
        edge_norm = edge_norm.to(self.device)
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm
        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)  # [num_rel*2, num_base] @ [num_base, in_c]
        
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 3
        if self.bias is not None:
            x = x + self.bias
        x = self.bn(x)

        return self.act(x), torch.matmul(self.rel, self.w_rel)

class Global(nn.Module):
    def __init__(self, triple_train, graph_gat, graph, r, edge_norm, selected, ent_num, rel_num, dim, use_global, num_layers, heads, dropout, activation):
        self.triple_train = triple_train
        self.graph_gat = graph_gat
        self.graph = graph
        self.edge_type = r
        self.edge_norm = edge_norm
        self.selected = selected
        self.ent_num = ent_num
        self.rel_num = rel_num
        super(Global, self).__init__()
        self.layers0 = nn.ModuleList()
        self.layers1 = nn.ModuleList()
        for _ in range(num_layers):
            self.layers0.append(GATv2Conv(dim, dim // heads, num_heads=heads,allow_zero_in_degree=True, feat_drop=dropout))
            self.layers1.append(CompGCNCov(dim, dim, torch.tanh, bias = 'False', drop_rate = dropout, opn = 'corr'))
        self.num_layers = num_layers
        
        self.rel_node_emb = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.normal_(self.rel_node_emb, mean=0, std=0.02)
        self.W1 = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.normal_(self.W1, mean=0, std=0.02)
        self.W2 = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.normal_(self.W2, mean=0, std=0.02)
        self.W3 = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.normal_(self.W3, mean=0, std=0.02)

        self.special_embedding = nn.parameter.Parameter(torch.Tensor(2, dim))
        nn.init.normal_(self.special_embedding, mean=0, std=0.02)
        self.ent_embedding = nn.parameter.Parameter(torch.Tensor(ent_num, dim))
        nn.init.normal_(self.ent_embedding, mean=0, std=0.02)
        self.rel_embedding = nn.parameter.Parameter(torch.Tensor(rel_num*2+6, dim))
        nn.init.normal_(self.rel_embedding, mean=0, std=0.02)

        if activation == "gelu":
            self.activate = nn.GELU()
        elif activation == "relu":
            self.activate = nn.ReLU()
        elif activation == 'elu':
            self.activate = nn.ELU()
        elif activation == 'tanh':
            self.activate = nn.Tanh()
        self.use_global = use_global

    def forward(self):
        ent_embedding = self.ent_embedding
        rel_embedding = self.rel_embedding.to(ent_embedding.device)
        if self.use_global is True:
            x_r = torch.matmul(self.rel_embedding,self.rel_node_emb) # [num_rel*2,dim]
            fact_emb = self.get_fact_emb(self.triple_train,x_r,self.ent_embedding)
            ent_embedding = torch.cat((ent_embedding, x_r[:self.rel_num], fact_emb), dim = 0)  # embedding of entities
            for i in range(self.num_layers):
                tmp = self.layers0[i](self.graph_gat, ent_embedding).reshape(ent_embedding.shape[0], -1)
                tmp = self.activate(tmp)
                ent_embedding = ent_embedding + tmp
                tmp1, tmp2 = self.layers1[i](self.graph, ent_embedding, rel_embedding, self.edge_type, self.edge_norm)
                tmp1 = self.activate(tmp1)
                tmp2 = self.activate(tmp2)
                ent_embedding = ent_embedding + tmp1
                rel_embedding = rel_embedding + tmp2
        return torch.cat([self.special_embedding, rel_embedding[:self.rel_num], ent_embedding[:self.ent_num]], dim=0)
    
    def get_fact_emb(self, triple_train, rel_emb, ent_emb):
        # triple_train #[num_selected,3]
        fact_emb = []
        sub_emb = ent_emb[triple_train[:,0]]
        r_emb   = rel_emb[triple_train[:,1]]
        obj_emb = ent_emb[triple_train[:,2]] # [num_selected,dim]
        fact_emb = torch.matmul(sub_emb,self.W1) + torch.matmul(r_emb,self.W2) + torch.matmul(obj_emb,self.W3) # [num_selected,dim]
        return fact_emb


class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, bias: bool, use_node: bool) -> None:
        super().__init__()
        self.heads = heads
        self.use_node = use_node       

        if self.use_node is True:
            self.layer_s=nn.Linear(hidden_dim,hidden_dim)
            self.layer_r=nn.Linear(hidden_dim,hidden_dim)
            self.layer_o=nn.Linear(hidden_dim,hidden_dim)
            self.layer_a=nn.Linear(hidden_dim,hidden_dim)
            self.layer_v=nn.Linear(hidden_dim,hidden_dim)
        else:
            self.linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)


    def forward(self, x : torch.Tensor):
        shape = x.shape[:-1]

        if self.use_node is False:
            x = self.linear(x)
        else:
            device=x.device
            max_seq_len=x.size(1)
            mask_s = torch.tensor([1]+[0]*(max_seq_len-1)).to(device)
            mask_r = torch.tensor([0,1]+[0]*(max_seq_len-2)).to(device)
            mask_o = torch.tensor([0,0,1]+[0]*(max_seq_len-3)).to(device)
            mask_a = torch.tensor([0,0,0]+[1,0]*int(((max_seq_len-3)/2))).to(device)
            mask_v = torch.tensor([0,0,0]+[0,1]*int(((max_seq_len-3)/2))).to(device)

            x_s=self.layer_s(torch.mul(x,mask_s[:,None].expand(-1,x.size(-1))))
            x_r=self.layer_r(torch.mul(x,mask_r[:,None].expand(-1,x.size(-1))))
            x_o=self.layer_o(torch.mul(x,mask_o[:,None].expand(-1,x.size(-1))))
            x_a=self.layer_a(torch.mul(x,mask_a[:,None].expand(-1,x.size(-1))))
            x_v=self.layer_v(torch.mul(x,mask_v[:,None].expand(-1,x.size(-1))))
                            
            x=(x_s+x_r+x_o+x_a+x_v) 
      
        return x.reshape(*shape, self.heads, -1)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout_prob: float, use_edge: bool, remove_mask: bool, bias: bool, use_node: bool) -> None:
        super().__init__()
        assert hidden_dim % heads == 0
        self.dim = hidden_dim // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(hidden_dim, heads, bias, use_node)
        self.key = PrepareForMultiHeadAttention(hidden_dim, heads, bias, use_node)
        self.value = PrepareForMultiHeadAttention(hidden_dim, heads, True, use_node)
        self.pos = PrepareForMultiHeadAttention(hidden_dim, heads, True, use_node)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_edge = use_edge
        self.remove_mask = remove_mask
        self.scale = 1 / math.sqrt(self.dim)
        # trasformer-xl
        self.r_w_bias = nn.Parameter(torch.Tensor(heads, self.dim)) # u
        self.r_r_bias = nn.Parameter(torch.Tensor(heads, self.dim)) # v

    def get_mask(self, graph: torch.Tensor):
        return graph.unsqueeze(1).repeat(1, self.heads, 1, 1)

    def forward(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        graph: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor):
        # query/key/value: (batch, seq_len, hidden_dim)
        # graph: (batch, kinds, query, key)
        shape = query.shape[:-1]
        query = self.query(query)   # (batch, seq_len, head, hidden)
        key = self.key(key)         # (batch, seq_len, head, hidden)
        value = self.value(value)   # (batch, seq_len, head, hidden)
        seq_len = query.size(1)
        if self.use_edge is True:
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key) + torch.einsum("bqhd,bqkd->bhqk", query, edge_key) + torch.einsum("bkqd,bkhd->bhqk", edge_query, key) + torch.einsum("bkqd,bqkd->bqk", edge_query, edge_key).unsqueeze(1)
            scores = scores * self.scale
            mask = self.get_mask(graph)
            if self.remove_mask is True:
                for i in range(3,seq_len,2):
                    if i==3:
                        mask[:,:,i:(i+2),(i+2):]=False
                    elif i==(seq_len-2):
                        mask[:,:,i:(i+2),3:i]=False
                    else:
                        mask[:,:,i:(i+2),(i+2):]=False
                        mask[:,:,i:(i+2),3:i]=False     
            scores = scores.masked_fill(mask == 0, -100000)
            attn = self.softmax(scores)
            attn = self.dropout(attn)
            x = torch.einsum("bhqk,bkhd->bqhd", attn, value) + torch.einsum("bhqk,bqkd->bqhd", attn, edge_value)
            x = x.reshape(*shape, -1)
        else:
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
            scores *= self.scale
            mask = self.get_mask(graph)
            if self.remove_mask is True:
                for i in range(3,seq_len,2):
                    if i==3:
                        mask[:,:,i:(i+2),(i+2):]=False
                    elif i==(seq_len-2):
                        mask[:,:,i:(i+2),3:i]=False
                    else:
                        mask[:,:,i:(i+2),(i+2):]=False
                        mask[:,:,i:(i+2),3:i]=False  
            scores = scores.masked_fill(mask == 0, -100000)
            attn = self.softmax(scores)
            attn = self.dropout(attn)
            x = torch.einsum("bhqk,bkhd->bqhd", attn, value)
            x = x.reshape(*shape, -1)

        return self.output(x)  # (batch, query, hidden_dim)

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation) -> None:
        super().__init__()
        act = None
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        elif activation == 'tanh':
            act = nn.Tanh()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layer(x)
        
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout_prob: float, activation: str, use_edge: bool, remove_mask: bool, use_node: bool, bias=True, times=2) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, heads, dropout_prob, use_edge, remove_mask, bias, use_node)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, hidden_dim * times, hidden_dim, activation)

    def forward(self, x: torch.Tensor, graph: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor):
        attn = self.attention(query=x, key=x, value=x, graph=graph, edge_key=edge_key, edge_value=edge_value, edge_query=edge_query)
        x = self.norm_attention(x + self.dropout(attn))
        ff = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ff))
        return x

class Transformer(nn.Module):
    def __init__(self, triple_train, graph_gat, graph, r, edge_norm, selected,ent_num: int,rel_num: int, vocab_size: int, local_layers: int, global_layers: int, hidden_dim: int,
            local_heads: int, global_heads: int, use_global: bool, local_dropout: float, global_dropout: float, 
            decoder_activation: str, global_activation: str, use_edge: bool, remove_mask: bool, use_node: bool, bias=True, times=2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList()
        for _ in range(local_layers):
            self.layers.append(TransformerLayer(hidden_dim, local_heads, local_dropout, decoder_activation, use_edge, remove_mask, use_node, bias, times=times))
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(p=local_dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_act = nn.GELU()
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_bias = torch.nn.parameter.Parameter(torch.zeros(vocab_size))
        self.edge_query_embedding = nn.Embedding(14, hidden_dim // local_heads, padding_idx=0)
        self.edge_key_embedding = nn.Embedding(14, hidden_dim // local_heads, padding_idx=0)
        self.edge_value_embedding = nn.Embedding(14, hidden_dim // local_heads, padding_idx=0)
        self.init_params()
        self.globl = Global(triple_train, graph_gat, graph, r, edge_norm, selected, ent_num, rel_num, hidden_dim, use_global, global_layers, global_heads, global_dropout, global_activation)
    def init_params(self):
        for name, param in self.named_parameters():
            if "norm" in name:
                continue
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name or "att" in name:
                nn.init.normal_(param, mean=0, std=0.02)
            elif "embedding" in name:
                nn.init.normal_(param, mean=0, std=0.02)
            else:
                raise TypeError("Invalid Parameters")
    def forward(self, input_ids, input_mask, mask_position, mask_output, edge_labels):
        embedding = self.globl().to(input_ids.device)
        x = torch.nn.functional.embedding(input_ids, embedding)
        x = self.input_dropout(self.input_norm(x))
        edge_query = self.edge_query_embedding(edge_labels)
        edge_key = self.edge_key_embedding(edge_labels)
        edge_value = self.edge_value_embedding(edge_labels)

        for layer in self.layers:
            x = layer(x, input_mask, edge_key, edge_value, edge_query)
        x = x[torch.arange(x.shape[0]), mask_position]
        x = self.output_linear(x)  # x(batch_size, hiddem_dim)
        x = self.output_act(x)
        x = self.output_norm(x)
        y = torch.mm(x, embedding.transpose(0, 1)) + self.output_bias
        y = y.masked_fill(mask_output == 0, -100000)
        return y
