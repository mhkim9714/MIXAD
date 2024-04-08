import warnings
import torch
import torch.nn as nn
import numpy as np


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class GC(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(GC, self).__init__()
        self.cheb_k = cheb_k 
        self.weights = nn.Parameter(torch.FloatTensor(2*cheb_k*dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        x_g = []        
        support_set = []

        # Aggregate Neighbors
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
            support_set.extend(support_ks)
        # print(len(support_set))
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)

        # Transformation
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias
        return x_gconv
    

class STRGCCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(STRGCCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GC(dim_in+self.hidden_dim, 2*dim_out, cheb_k) 
        self.update = GC(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)

        z_r = torch.sigmoid(self.gate(input_and_state, supports)) 
        z,r = torch.split(z_r, self.hidden_dim, dim=-1) 
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    

class STRGC_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(STRGC_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num 
        self.input_dim = dim_in 
        self.num_layers = num_layers 
        self.strgc_cells = nn.ModuleList()
        self.strgc_cells.append(STRGCCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.strgc_cells.append(STRGCCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x 
        output_hidden = []

        for i in range(self.num_layers): 
            state = init_state[i] 
            inner_states = []

            for t in range(seq_length): 
                state = self.strgc_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)

            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        return current_inputs, output_hidden 
    
    def init_hidden(self, batch_size): 
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.strgc_cells[i].init_hidden_state(batch_size)) 
        return init_states


class STRGC_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(STRGC_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in 
        self.num_layers = num_layers 
        self.strgc_cells = nn.ModuleList()
        self.strgc_cells.append(STRGCCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.strgc_cells.append(STRGCCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports): 
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt 
        output_hidden = []

        for i in range(self.num_layers): 
            state = self.strgc_cells[i](current_inputs, init_state[i], supports) 
            output_hidden.append(state)
            current_inputs = state

        return current_inputs, output_hidden


class MIXAD(nn.Module):
    def __init__(self, args):
        super(MIXAD, self).__init__()
        self.num_nodes = args.num_nodes 
        self.input_dim = args.input_dim
        self.rnn_units = args.rnn_units
        self.output_dim = args.output_dim 
        self.horizon = args.seq_len
        self.num_layers = args.num_rnn_layers
        self.cheb_k = args.max_diffusion_step
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning 
        
        self.mem_num = args.mem_num
        self.mem_dim = args.mem_dim
        self.memory = self.construct_memory()

        self.encoder = STRGC_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = STRGC_Decoder(self.num_nodes, self.output_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True) 
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True) 
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    
    def query_memory(self, h_t:torch.Tensor): 
        query = torch.matmul(h_t, self.memory['Wq']) 
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)
        value = torch.matmul(att_score, self.memory['Memory'])

        _, ind = torch.topk(att_score, k=2, dim=-1) 
        pos = self.memory['Memory'][ind[:, :, 0]] 
        neg = self.memory['Memory'][ind[:, :, 1]] 
        return value, query, pos, neg, att_score
    
    def scaled_laplacian(self, node_embeddings1, node_embeddings2, is_eval=False):
        # Normalized graph Laplacian function.
        node_num = self.num_nodes
        learned_graph = torch.mm(node_embeddings1, node_embeddings2.transpose(0, 1))
        
        # Normalize the graph
        norm1 = torch.norm(node_embeddings1, p=2, dim=1, keepdim=True)
        norm2 = torch.norm(node_embeddings2, p=2, dim=1, keepdim=True)
        norm = torch.mm(norm1, norm2.transpose(0, 1))
        learned_graph = learned_graph / (norm + 1e-6)  # Adding a small epsilon to avoid division by zero
        learned_graph = (learned_graph + 1) / 2.
        learned_graph = torch.stack([learned_graph, 1-learned_graph], dim=-1)
       
        # Make the adj sparse
        if is_eval:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        else:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        adj = adj[:, :, 0].clone().reshape(node_num, -1)
        mask = torch.eye(node_num, node_num).bool().to(node_embeddings1.device)
        adj.masked_fill_(mask, 0)
       
        W = adj
        n = W.shape[0]
        d = torch.sum(W, axis=1)
        L = -W
        L[range(len(L)), range(len(L))] = d
        
        try:
            lambda_max = (L.max() - L.min())
        except Exception as e:
            print("eig error!!: {}".format(e))
            lambda_max = 1.0
       
        tilde = (2 * L / lambda_max - torch.eye(n).to(node_embeddings1.device))
        return adj, tilde
            
    def forward(self, x, batches_seen=None):
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        if self.train:
            adj1, learned_tilde1 = self.scaled_laplacian(node_embeddings1, node_embeddings2, is_eval=False)
            adj2, learned_tilde2 = self.scaled_laplacian(node_embeddings2, node_embeddings1, is_eval=False)
        else:
            adj1, learned_tilde1 = self.scaled_laplacian(node_embeddings1, node_embeddings2, is_eval=True)
            adj2, learned_tilde2 = self.scaled_laplacian(node_embeddings2, node_embeddings1, is_eval=True)
        supports = [learned_tilde1, learned_tilde2] 

        init_state = self.encoder.init_hidden(x.shape[0]) 
        h_en, state_en = self.encoder(x, init_state, supports) 
        h_t = h_en[:, -1, :, :] 

        h_att, query, pos, neg, att_score = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)
        
        labels = torch.flip(x, [1])
        ht_list = [h_t]*self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device) 
        out = []

        for t in range(self.horizon):
            h_de, ht_list = self.decoder(go, ht_list, supports)
            go = self.proj(h_de)
            out.append(go)

            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]

        output = torch.stack(out, dim=1)
        output = torch.flip(output, [1])

        h_att = torch.randn(x.shape[0], self.num_nodes, self.mem_dim).to(x.device)
        query = torch.randn(x.shape[0], self.num_nodes, self.mem_dim).to(x.device)
        pos = torch.randn(x.shape[0], self.num_nodes, self.mem_dim).to(x.device)
        neg = torch.randn(x.shape[0], self.num_nodes, self.mem_dim).to(x.device)
        att_score = torch.randn(x.shape[0], self.num_nodes, self.mem_num).to(x.device)

        return output, h_att, query, pos, neg, att_score, (adj1, adj2)

def count_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count += param.numel()
    return param_count