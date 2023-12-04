import torch
from torch import nn, einsum, broadcast_tensors
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def safe_div(num, den, eps = 1e-8):
    res = num.div(den.clamp(min = eps))
    res.masked_fill_(den == 0, 0.)
    return res

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x

def embedd_token(x, dims, layers):
    stop_concat = -len(dims)
    to_embedd = x[:, stop_concat:].long()
    for i,emb_layer in enumerate(layers):
        # the portion corresponding to `to_embedd` part gets dropped
        x = torch.cat([ x[:, :stop_concat], 
                        emb_layer( to_embedd[:, i] ) 
                      ], dim=-1)
        stop_concat = x.shape[-1]
    return x

# swish activation fallback

class Swish_(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

SiLU = nn.SiLU if hasattr(nn, 'SiLU') else Swish_

# helper classes

# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * self.scale

# global linear attention
#就是GAT的操作
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, mask = None):
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale #张量乘法

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b () () n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class GlobalLinearAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads, dim_head)
        self.attn2 = Attention(dim, heads, dim_head)

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, mask = None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x), self.norm_queries(queries)

        induced = self.attn1(queries, x, mask = mask)
        out     = self.attn2(x, induced)

        x =  out + res_x
        queries = induced + res_queries

        x = self.ff(x) + x
        return x, queries

# classes

class EGNN(nn.Module):
    def __init__(
        self,
        dim,
        edge_dim = 0,
        m_dim = 16,
        fourier_features = 0,
        num_nearest_neighbors = 0,
        dropout = 0.0,
        init_eps = 1e-3,
        norm_feats = False,
        norm_coors = False,
        norm_coors_scale_init = 1e-2,
        update_feats = True,
        update_coors = True,
        only_sparse_neighbors = False,
        valid_radius = float('inf'),
        m_pool_method = 'sum',
        soft_edges = False,
        coor_weights_clamp_value = None
    ):
        super().__init__()
        assert m_pool_method in {'sum', 'mean'}, 'pool method must be either sum or mean'
        assert update_feats or update_coors, 'you must update either features, coordinates, or both'

        self.fourier_features = fourier_features

        edge_input_dim = (fourier_features * 2) + (dim * 2) + edge_dim + 1
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_mlp = nn.Sequential(  #edge_mlp边线性变化，有点像做嵌入
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            dropout,
            SiLU(),  #激活函数
            nn.Linear(edge_input_dim * 2, m_dim),
            SiLU()
        )

        self.edge_gate = nn.Sequential(     #edge_gate有点像二分类
            nn.Linear(m_dim, 1),
            nn.Sigmoid()
        ) if soft_edges else None

        self.node_norm = nn.LayerNorm(dim) if norm_feats else nn.Identity() #层归一化
        self.coors_norm = CoorsNorm(scale_init = norm_coors_scale_init) if norm_coors else nn.Identity() #坐标归一化

        self.m_pool_method = m_pool_method #池化方法

        self.node_mlp = nn.Sequential( #点线性化，有点像做嵌入
            nn.Linear(dim + m_dim, dim * 2),
            dropout,
            SiLU(),
            nn.Linear(dim * 2, dim),
        ) if update_feats else None

        self.coors_mlp = nn.Sequential(#坐标线性化，有点像做嵌入
            nn.Linear(m_dim, m_dim * 4),
            dropout,
            SiLU(),
            nn.Linear(m_dim * 4, 1)
        ) if update_coors else None

        self.num_nearest_neighbors = num_nearest_neighbors  #最近邻居数量
        self.only_sparse_neighbors = only_sparse_neighbors  #稀疏邻居
        self.valid_radius = valid_radius  #有效范围，正无穷

        self.coor_weights_clamp_value = coor_weights_clamp_value

        self.init_eps = init_eps
        self.apply(self.init_)  #这个是用来初始化一些线性层，卷积层的的函数

    def init_(self, module):
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(self, feats, coors, edges = None, mask = None, adj_mat = None):
        b, n, d, device, fourier_features, num_nearest, valid_radius, only_sparse_neighbors = *feats.shape, feats.device, self.fourier_features, self.num_nearest_neighbors, self.valid_radius, self.only_sparse_neighbors

        if exists(mask):
            num_nodes = mask.sum(dim = -1)

        use_nearest = num_nearest > 0 or only_sparse_neighbors

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim = -1, keepdim = True)   #这块应该是做了|xi-xj| 距离信息也就是几何信息标量化

        i = j = n

        if use_nearest:
            ranking = rel_dist[..., 0].clone()   #去掉里面那一层了，表示i,j距离信息

            if exists(mask):
                rank_mask = mask[:, :, None] * mask[:, None, :]  #二维mask改成带bach形式的mask
                ranking.masked_fill_(~rank_mask, 1e5)   #b被mask取0了的距离信息取 1e5

            if exists(adj_mat): #转化为batch形式
                if len(adj_mat.shape) == 2:
                    adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

                if only_sparse_neighbors:
                    num_nearest = int(adj_mat.float().sum(dim = -1).max().item())
                    valid_radius = 0

                self_mask = rearrange(torch.eye(n, device = device, dtype = torch.bool), 'i j -> () i j')  #相当于单位矩阵

                adj_mat = adj_mat.masked_fill(self_mask, False).bool()  #原本的邻接矩阵自己和自己的部分要为1
                ranking.masked_fill_(self_mask, -1.)  #ranking相当于做mask任务和没有连接的点,自己和自己的距离是-1,相连接是0
                ranking.masked_fill_(adj_mat, 0.)
                if ranking.shape[1]<num_nearest:
                    num_nearest = ranking.shape[1]
                try:
                    nbhd_ranking, nbhd_indices = ranking.topk(num_nearest, dim = -1, largest = False)  #最小到大排序吧，按照距离无边>连接>自己和自己
                except:
                    print(coors)
                    print(rel_dist)
                    print(ranking.shape)
                    print(ranking)
                    exit()

            nbhd_mask = nbhd_ranking <= valid_radius

            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim = 2)
            rel_dist = batched_index_select(rel_dist, nbhd_indices, dim = 2)

            if exists(edges):
                edges = batched_index_select(edges, nbhd_indices, dim = 2)

            j = num_nearest

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        if use_nearest:
            feats_j = batched_index_select(feats, nbhd_indices, dim = 1)
        else:
            feats_j = rearrange(feats, 'b j d -> b () j d')

        feats_i = rearrange(feats, 'b i d -> b i () d')
        feats_i, feats_j = broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim = -1)   #非几何节点信息+几何信息标量化

        if exists(edges):
            edge_input = torch.cat((edge_input, edges), dim = -1)   #有边则也拼接

        m_ij = self.edge_mlp(edge_input)     #第一个不变性的公式,即使几何信息变化这里和也不会变

        if exists(self.edge_gate):
            m_ij = m_ij * self.edge_gate(m_ij)

        if exists(mask):
            mask_i = rearrange(mask, 'b i -> b i ()')

            if use_nearest:
                mask_j = batched_index_select(mask, nbhd_indices, dim = 1)
                mask = (mask_i * mask_j) & nbhd_mask
            else:
                mask_j = rearrange(mask, 'b j -> b () j')
                mask = mask_i * mask_j

        if exists(self.coors_mlp):
            coor_weights = self.coors_mlp(m_ij)
            coor_weights = rearrange(coor_weights, 'b i j () -> b i j')

            rel_coors = self.coors_norm(rel_coors)  #几何信息

            if exists(mask):
                coor_weights.masked_fill_(~mask, 0.)

            if exists(self.coor_weights_clamp_value):
                clamp_value = self.coor_weights_clamp_value
                coor_weights.clamp_(min = -clamp_value, max = clamp_value)

            coors_out = einsum('b i j, b i j c -> b i c', coor_weights, rel_coors) + coors  #第二个公式等变性
        else:
            coors_out = coors

        if exists(self.node_mlp):
            if exists(mask):
                m_ij_mask = rearrange(mask, '... -> ... ()')
                m_ij = m_ij.masked_fill(~m_ij_mask, 0.)

            if self.m_pool_method == 'mean':
                if exists(mask):
                    # masked mean
                    mask_sum = m_ij_mask.sum(dim = -2)
                    m_i = safe_div(m_ij.sum(dim = -2), mask_sum)
                else:
                    m_i = m_ij.mean(dim = -2)

            elif self.m_pool_method == 'sum':
                m_i = m_ij.sum(dim = -2)  #第三个公式求和

            normed_feats = self.node_norm(feats)
            node_mlp_input = torch.cat((normed_feats, m_i), dim = -1)
            node_out = self.node_mlp(node_mlp_input) + feats  #第四个公式
        else:
            node_out = feats

        return node_out, coors_out  #node_out第4个公式求出来的点，coors_out第二个公式求出来的边信息

class EGNN_Network(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        num_tokens = None,
        num_edge_tokens = None,
        num_positions = None,  #点的数量，也就是位置的数量
        edge_dim = 0,
        num_adj_degrees = None,
        adj_dim = 0,
        global_linear_attn_every = 1, #这里
        global_linear_attn_heads = 8,
        global_linear_attn_dim_head = 64,
        num_global_tokens = 4,
        coor_dim=3,
        **kwargs
    ):
        super().__init__()
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'
        self.num_positions = num_positions

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None    #点特征嵌入  词向量是21
        self.pos_emb = nn.Embedding(num_positions, dim) if exists(num_positions) else None #位置特征嵌入
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None  #边特征嵌入
        self.has_edges = edge_dim > 0

        self.num_adj_degrees = num_adj_degrees  #边的角度嵌入  个人感觉还是非几何信息
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None

        edge_dim = edge_dim if self.has_edges else 0
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        has_global_attn = global_linear_attn_every > 0    #全局的attention
        self.global_tokens = None
        if has_global_attn:
            self.global_tokens = nn.Parameter(torch.randn(num_global_tokens, dim))  #[4,dim]

        self.layers = nn.ModuleList([])
        for ind in range(depth):
            is_global_layer = has_global_attn and (ind % global_linear_attn_every) == 0
            print(is_global_layer)
            self.layers.append(nn.ModuleList([
                GlobalLinearAttention(dim = dim, heads = global_linear_attn_heads, dim_head = global_linear_attn_dim_head) if is_global_layer else None,
                EGNN(dim = dim, edge_dim = (edge_dim + adj_dim), norm_feats = True, **kwargs),
            ]))   #一层是类似GNN全局线性自注意力+EGNN也是聚合几何和非几何信息
        self.linear = nn.Linear(dim + coor_dim, 1)
        self.sigmoid = nn.Sigmoid

    def forward(
        self,
        feats,
        coors,
        adj_mat = None,
        edges = None,
        mask = None,
        return_coor_changes = False
    ):
        b, device = feats.shape[0], feats.device  #b是第1维，这里有点问题是

        if exists(self.token_emb):  #feats是(1, 1024)
            feats = self.token_emb(feats)   #点嵌入

        if exists(self.pos_emb):
            n = feats.shape[1]
            assert n <= self.num_positions, f'given sequence length {n} must be less than the number of positions {self.num_positions} set at init'
            pos_emb = self.pos_emb(torch.arange(n, device = device))
            feats += rearrange(pos_emb, 'n d -> () n d')

        if exists(edges) and exists(self.edge_emb):
            edges = self.edge_emb(edges)

        # create N-degrees adjacent matrix from 1st degree connections
        if exists(self.num_adj_degrees):
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            if exists(self.adj_emb):
                adj_emb = self.adj_emb(adj_indices)
                edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

        # setup global attention

        global_tokens = None
        if exists(self.global_tokens):
            global_tokens = repeat(self.global_tokens, 'n d -> b n d', b = b) #(4,133)

        # go through layers

        coor_changes = [coors]

        for global_attn, egnn in self.layers:
            if exists(global_attn):
                feats, global_tokens = global_attn(feats, global_tokens, mask = mask)

            feats, coors = egnn(feats, coors, adj_mat = adj_mat, edges = edges, mask = mask)
            coor_changes.append(coors)

        if return_coor_changes:
            return feats, coors, coor_changes

        atom=feats.shape[1]
        egnn_out=torch.cat([feats, coors], dim=-1)
        egnn_out=egnn_out.squeeze(0).sum(dim=0)/atom

        return egnn_out


class Network(nn.Module):


    def __init__(self,task_num):
        super().__init__()
        self.net = EGNN_Network(dim=133,
                    depth=3,
                    edge_dim=3,
                    num_nearest_neighbors=3,
                    coor_dim=3)
        self.linear=nn.Linear(136,task_num,bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, automs_index,feats_batch,edges_batch,coors_batch,adj_batch,mask_batch,batch,dataset_type):
        # Pass the input tensor through each of our operations
        egnn_outs = []

        N_now=0
        for i in range(batch):
            begin = automs_index[i][0]
            end = automs_index[i][1]

            N=end-begin

            N_pre = N_now
            N_now=N_pre+N*N
            feats = feats_batch[begin:end].unsqueeze(0)
            coors = coors_batch[begin:end].unsqueeze(0)
            edges = edges_batch[N_pre:N_now].reshape(N,N,-1).unsqueeze(0)

            adj = adj_batch[N_pre:N_now].reshape(-1,N)
            mask = mask_batch[begin:end].reshape(-1,N).bool()

            #egnn_out=None

            egnn_out = self.net(feats,coors, edges=edges, mask=mask,
                                adj_mat=adj)
            egnn_outs.append(egnn_out)
        egnn_outs = torch.stack(egnn_outs, dim=0)
        output=self.linear(egnn_outs)
        if dataset_type == 'classification':
            output = self.linear(egnn_outs)
            output = self.sigmoid(output)
        if dataset_type == 'regression':
            output = self.linear(egnn_outs)
        return output






