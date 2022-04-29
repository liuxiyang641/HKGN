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
            if self.p.model == 'hyper_gcn' and not self.p.hyper_rel:
                self.init_rel = get_param((num_rel * 2, self.p.embed_dim))
            else:
                self.init_rel = get_param((num_rel * 2, self.p.dz))
        # Hypernetworks for target relations
        if self.p.hyper_rel:
            self.w_rel = get_param((self.p.dz, self.p.embed_dim))

        if self.p.model == 'hyper_gcn':
            # layer 1
            self.conv1 = HyperGCNConv(self.edge_index, self.edge_type, self.p.init_dim, self.p.gcn_dim, num_rel,
                                      act=self.act, params=self.p, logger=logger,
                                      gcn_filt_num=self.p.gcn_filt_num_layer1,
                                      dx_size=self.p.dx_layer1, dy_size=self.p.dy_layer1)
            # layer 2
            if self.p.gcn_layer == 2:
                self.conv2 = HyperGCNConv(self.edge_index, self.edge_type, self.p.gcn_dim, self.p.embed_dim, num_rel,
                                          act=self.act, params=self.p, logger=logger,
                                          gcn_filt_num=self.p.gcn_filt_num_layer2,
                                          dx_size=self.p.dx_layer2, dy_size=self.p.dy_layer2)
            else:
                self.conv2 = None
        self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

    def forward_base(self, sub, rel, drop1, drop2, neg_ents=None):
        # init all relation embeddings
        r = self.init_rel \
            if self.p.score_func != 'transe' \
            else torch.cat([self.init_rel, -self.init_rel], dim=0)

        # Graph convolution layer 1
        ent_conv1 = self.conv1(self.init_embed, neg_ents=neg_ents)
        ent_conv1 = drop1(ent_conv1)
        # Graph convolution layer 2 if corresponding hyper parameter is set
        ent_conv2 = self.conv2(ent_conv1, neg_ents=neg_ents) if self.p.gcn_layer == 2 else ent_conv1
        ent_conv2 = drop2(ent_conv2) if self.p.gcn_layer == 2 else ent_conv1

        if self.p.model == 'hyper_gcn' and self.p.hyper_conv:
            r = r.mm(self.w_rel)

        # select embeddings of entities and relations involved in current batch
        sub_emb = torch.index_select(ent_conv2, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, ent_conv2

    def loss(self, pred, true_label):
        return self.bce_loss(pred, true_label)
