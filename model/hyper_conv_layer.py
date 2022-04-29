from helper import *


class HyperGCNConv(torch.nn.Module):
    def __init__(self, edge_index, edge_type, in_channels, out_channels, num_rels, act=lambda x: x, params=None,
                 logger=None, gcn_filt_num=None, base_num=None):
        super(self.__class__, self).__init__()
        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_rels = num_rels
        self.act = act
        self.device = None
        self.logger = logger
        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.device is None:
            self.device = self.edge_index.device
        self.num_ent = self.p.num_ent
        self.loop_index = torch.stack([torch.arange(self.num_ent), torch.arange(self.num_ent)]).to(self.device)
        self.loop_type = torch.full((self.num_ent,), 2 * self.num_rels, dtype=torch.long).to(self.device)

        self.edge_index = torch.cat([self.edge_index, self.loop_index], dim=1)
        self.edge_type = torch.cat([self.edge_type, self.loop_type], dim=0)

        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels))

        self.leaky_relu = nn.LeakyReLU(0.2)
        if self.p.experiment == 'gen_ctx_conv_global_wo_r':
            self.ker_size = self.p.gcn_ker_sz
            self.filter_num = gcn_filt_num
            self.w_conv_global = get_param((100, self.ker_size * self.ker_size * self.filter_num))
            self.rel_context = get_param((self.num_rels * 2 + 1, 100))
            # self.w_conv_global = get_param((self.num_rels * 2 + 1, self.filter_num, 1, self.ker_size, self.ker_size))
            if self.in_channels == 100:
                self.flatten_size = (10 - self.ker_size + 1) * (10 - self.ker_size + 1) * self.filter_num
            elif self.in_channels == 200:
                self.flatten_size = (10 - self.ker_size + 1) * (20 - self.ker_size + 1) * self.filter_num
            self.base_num = base_num
            self.w_rel_coff = get_param((2 * self.num_rels + 1, self.base_num))
            self.w_global = get_param((self.base_num, self.flatten_size * out_channels))
        elif self.p.experiment == 'gen_ctx_conv_global_wo_r_mem':
            self.ker_size = 3
            self.filter_num = gcn_filt_num
            # self.w_conv_global = get_param((100, self.ker_size * self.ker_size * self.filter_num))
            # self.rel_context = get_param((self.num_rels * 2 + 1, 100))
            self.w_conv_global = get_param((self.num_rels * 2 + 1, self.filter_num, 1, self.ker_size, self.ker_size))
            if self.in_channels == 100:
                self.flatten_size = (10 - self.ker_size + 1) * (10 - self.ker_size + 1) * self.filter_num
            elif self.in_channels == 200:
                self.flatten_size = (10 - self.ker_size + 1) * (20 - self.ker_size + 1) * self.filter_num
            self.base_num = base_num
            # self.w_rel_coff = get_param((2 * self.num_rels + 1, self.base_num))
            # self.w_global = get_param((self.base_num, self.flatten_size, out_channels))
            self.w_global = get_param((self.num_rels * 2 + 1, self.flatten_size, out_channels))

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.drop1 = torch.nn.Dropout(0.1)
        self.drop2 = torch.nn.Dropout(0.2)
        self.drop3 = torch.nn.Dropout(0.3)
        self.drop4 = torch.nn.Dropout(0.4)
        self.drop5 = torch.nn.Dropout(0.5)

        # num_edges = edge_index.size(1) // 2
        # self.edge_norm = self.compute_norm(self.edge_index, self.num_ent)

        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

        edge_type = self.edge_type.cpu().numpy()
        edge_index = self.edge_index.cpu().numpy()
        self.group_by_rel = ddict(list)
        for edge_id in range(edge_index.shape[1]):
            self.group_by_rel[edge_type[edge_id]].append(edge_index[:, edge_id])

        self.all_index = []
        self.all_edge_type = []
        for rel in self.group_by_rel.keys():
            edges_gb_rel = torch.LongTensor(self.group_by_rel.get(rel)).to(self.device).t()
            self.all_index.append(edges_gb_rel)
            self.all_edge_type.append(torch.full((1, edges_gb_rel.shape[1]), rel).to(self.device))
        self.all_index = torch.cat(self.all_index, dim=1)
        self.all_edge_type = torch.cat(self.all_edge_type, dim=1)
        self.edge_norm = self.compute_norm(self.all_index, self.num_ent)

    def forward(self, x, rel_embed, sub, rel, neg_ents=None):
        if self.p.experiment == 'gen_ctx_conv_global_wo_r':
            all_conv_weight = self.rel_context.mm(self.w_conv_global).reshape(
                (-1, self.filter_num, 1, self.ker_size, self.ker_size))
            # all_conv_weight = self.w_conv_global
            messages = self.message_wo_r_all_conv_global(x, all_conv_weight)
            # normalization
            edge_norm = self.compute_norm(self.edge_index, self.num_ent)
            all_messages = messages * edge_norm.view(-1, 1)
            all_messages = self.drop(all_messages)

            res = scatter_add(all_messages, self.edge_index[0, :], dim=0, dim_size=self.num_ent)
            out = res
            if self.p.bias:
                out = out + self.bias
            out = self.bn(out)

            return self.act(out), rel_embed
        elif self.p.experiment == 'gen_ctx_conv_global_wo_r_mem':
            # all_conv_weight = self.rel_context.mm(self.w_conv_global).reshape(
            #     (-1, self.filter_num, 1, self.ker_size, self.ker_size))
            # all_global_weight = torch.einsum('rb,bio->rio', self.w_rel_coff, self.w_global)
            all_conv_weight = self.w_conv_global
            all_global_weight = self.w_global

            all_messages = []
            for rel in self.group_by_rel.keys():
                messages = self.message_wo_r_conv_global(x, self.group_by_rel.get(rel), all_conv_weight[rel],
                                                         all_global_weight[rel])
                all_messages.append(messages)
            all_messages = torch.cat(all_messages, dim=0)
            # normalization
            all_messages = all_messages * self.edge_norm.view(-1, 1)
            all_messages = self.drop(all_messages)
            res = scatter_add(all_messages, self.all_index[0, :], dim=0, dim_size=self.num_ent)
            out = res

            if self.p.bias:
                out = out + self.bias
            out = self.bn(out)

            return self.act(out), rel_embed

    def message_wo_r_all_conv_global(self, x, all_conv_weight):
        conv_weight = all_conv_weight[self.edge_type]
        # edges_gb_rel: sub, rel, obj
        ent_emb = x[self.edge_index[1]]
        if ent_emb.shape[1] == 100:
            emb_h = ent_emb.reshape((-1, 1, 10, 10))
        else:
            emb_h = ent_emb.reshape((-1, 1, 10, 20))
        messages = F.conv2d(emb_h.view(1, emb_h.shape[0], emb_h.shape[2], emb_h.shape[3]),
                            conv_weight.view(emb_h.shape[0] * self.filter_num, 1, conv_weight.shape[3],
                                             conv_weight.shape[4]),
                            groups=emb_h.shape[0])
        messages = self.act(messages)
        messages = messages.reshape((ent_emb.shape[0], -1))
        w_rel_coff = self.w_rel_coff[self.edge_type]
        b_messages = torch.einsum('mi,bio->mbo', messages,
                                  self.w_global.reshape((self.base_num, self.flatten_size, self.out_channels)))
        messages = torch.einsum('mbo,mb->mo', b_messages, w_rel_coff)
        return messages

    def message_wo_r_conv_global(self, x, edge_gb_rel, conv_weight, global_weight):
        edges_gb_rel = torch.LongTensor(edge_gb_rel).to(self.device).t()
        ent_emb = x[edges_gb_rel[1, :]]
        if ent_emb.shape[1] == 100:
            tmp_h = ent_emb.reshape((-1, 1, 10, 10))
        else:
            tmp_h = ent_emb.reshape((-1, 1, 10, 20))
        messages = F.conv2d(tmp_h, conv_weight, stride=1, padding=0)
        messages = self.act(messages)

        messages = messages.reshape((ent_emb.shape[0], -1))
        messages = messages.mm(global_weight)

        return messages

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)
