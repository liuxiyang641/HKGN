from model.gcn_encoder import *


class GCNTransE(GCNEncoder):
    def __init__(self, edge_index, edge_type, params=None, logger=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params, logger)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb + rel_emb

        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        score = torch.sigmoid(x)

        return score


class GCNDistMulti(GCNEncoder):
    def __init__(self, edge_index, edge_type, params=None, logger=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params, logger)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb

        x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score


class GCNConvE(GCNEncoder):
    def __init__(self, edge_index, edge_type, params=None, logger=None):
        super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params, logger)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        # now, keep the same size with initial entity embeddings
        if self.p.strategy == 'one_to_n_origin':
            self.bn2 = torch.nn.BatchNorm1d(self.p.init_dim)
        else:
            self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.layer1_drop = torch.nn.Dropout(self.p.layer1_drop)
        self.layer2_drop = torch.nn.Dropout(self.p.layer2_drop)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt

        # now, keep the same size with initial entity embeddings
        if self.p.strategy == 'one_to_n_origin':
            self.fc = torch.nn.Linear(self.flat_sz, self.p.init_dim)
        else:
            self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel, neg_ents=None):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.layer1_drop, self.layer2_drop, neg_ents)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        if self.p.strategy == 'one_to_n' or self.p.strategy == 'one_to_n_origin' or neg_ents is None:
            x = torch.mm(x, all_ent.transpose(1, 0))
            x += self.bias.expand_as(x)
        elif self.p.strategy == 'one_to_batch_n':
            x = torch.mm(x, all_ent[neg_ents].transpose(1, 0))
            x += self.bias[neg_ents]
        else:
            x = torch.mul(x.unsqueeze(1), all_ent[neg_ents]).sum(dim=-1)
            x += self.bias[neg_ents]
        # contain all scores including pos and neg
        score = torch.sigmoid(x)
        return score
