from helper import *
from model.hyper_conv_layer import HyperGCNConv


class GCNEncoder(torch.nn.Module):
    def __init__(self, edge_index, edge_type, num_rel, params=None, logger=None):
        super(GCNEncoder, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bce_loss = torch.nn.BCELoss()
        self.logger = logger

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.device = self.edge_index.device

        if self.p.score_func == 'transe':
            self.init_rel = get_param((num_rel, self.p.init_dim))
        else:
            self.init_rel = get_param((num_rel * 2, self.p.init_dim))

        if self.p.model == 'gen_rgcn':
            self.conv1 = HyperGCNConv(self.edge_index, self.edge_type, self.p.init_dim, self.p.gcn_dim, num_rel,
                                      act=self.act, params=self.p, logger=logger,
                                      gcn_filt_num=self.p.gcn_filt_num_layer1,
                                      base_num=self.p.base_num_layer1)
            self.conv2 = HyperGCNConv(self.edge_index, self.edge_type, self.p.gcn_dim, self.p.embed_dim, num_rel,
                                      act=self.act, params=self.p, logger=logger,
                                      gcn_filt_num=self.p.gcn_filt_num_layer2,
                                      base_num=self.p.base_num_layer2) if self.p.gcn_layer == 2 else None
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
        self.w_rel = get_param((self.p.init_dim, self.p.embed_dim))

    def forward_base(self, sub, rel, drop1, drop2, neg_ents=None):
        # init all relation embeddings
        r = self.init_rel \
            if self.p.score_func != 'transe' \
            else torch.cat([self.init_rel, -self.init_rel], dim=0)

        # Graph convolution layer 1
        ent_conv1, r = self.conv1(self.init_embed, rel_embed=r, sub=sub, rel=rel, neg_ents=neg_ents)
        ent_conv1 = drop1(ent_conv1)
        # Graph convolution layer 2 if corresponding hyper parameter is set
        ent_conv2, r = self.conv2(ent_conv1, rel_embed=r, sub=sub, rel=rel, neg_ents=neg_ents) \
            if self.p.gcn_layer == 2 \
            else (ent_conv1, r)
        ent_conv2 = drop2(ent_conv2) if self.p.gcn_layer == 2 else ent_conv1
        if self.p.model == 'gen_rgcn':
            r = r.mm(self.w_rel)
        # select embeddings of entities and relations involved in current batch
        sub_emb = torch.index_select(ent_conv2, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, ent_conv2

    def loss(self, pred, true_label):
        return self.bce_loss(pred, true_label)
