from helper import *
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """
    Training Dataset class.
    Parameters
    ----------
    triples:	The triples used for training the model
    params:		Parameters for the experiments

    Returns
    -------
    A training Dataset class instance used by DataLoader
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
        self.strategy = self.p.strategy
        self.entities = np.arange(self.p.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])

        if self.p.strategy == 'one_to_n' or self.p.strategy == 'one_to_n_origin':
            trp_label = self.get_label(label)
            if self.p.lbl_smooth != 0.0:
                trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.p.num_ent)
            return triple, trp_label, None, None
        elif self.p.strategy == 'one_to_x':
            trp_label = self.get_label(label)
            if self.p.lbl_smooth != 0.0:
                trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.p.num_ent)
            neg_ent = torch.LongTensor(self.get_neg_ent(triple, label))
            return triple, trp_label, neg_ent
        elif self.p.strategy == 'one_to_batch_n':
            return triple, torch.IntTensor(label)
        elif self.p.strategy == 'nscaching':
            trp_label = self.get_label(label)
            if self.p.lbl_smooth != 0.0:
                trp_label = (1.0 - self.p.lbl_smooth) * trp_label + (1.0 / self.p.num_ent)
            return triple.unsqueeze(0).expand(label.shape[0], -1), \
                   trp_label.unsqueeze(0).expand(label.shape[0], -1), torch.IntTensor(label)
        else:
            raise NotImplementedError

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        trp_label = torch.stack([_[1] for _ in data], dim=0)
        # triple: (batch-size) * 3(sub, rel, -1) trp_label (batch-size) * num entity
        if not data[0][2] is None:  # one_to_x
            neg_ent = torch.stack([_[2] for _ in data], dim=0)
            return triple, trp_label, neg_ent
        else:
            return triple, trp_label

    @staticmethod
    def batch_collate_fn(data, p):
        def get_label(batch_graph_size, entid2idx, label):
            y = np.zeros([batch_graph_size], dtype=np.float32)
            label = np.int32(label)
            for e2 in label:
                y[entid2idx.get(e2)] = 1.0
            return torch.FloatTensor(y)

        triple = torch.stack([_[0] for _ in data], dim=0)
        sub_ent = triple[:, 0]
        pos_obj_ent = torch.cat([_[1] for _ in data], dim=0)
        batch_ent, indices = torch.sort(torch.unique(torch.cat([sub_ent, pos_obj_ent], dim=0)))
        batch_graph_size = batch_ent.shape[0]
        entid2idx = {ent_id: idx for idx, ent_id in enumerate(batch_ent.tolist())}
        trp_label = torch.stack([get_label(batch_graph_size, entid2idx, _[1]) for _ in data], dim=0)
        trp_label = (1.0 - p.lbl_smooth) * trp_label + (1.0 / p.num_ent)

        # triple: (batch-size) * 3(sub, rel, -1) trp_label (batch-size) * num entity
        return triple, trp_label, batch_ent

    @staticmethod
    def ns_collate_fn(data, p):
        triple = torch.cat([_[0] for _ in data], dim=0)
        trp_label = torch.cat([_[1] for _ in data], dim=0)
        pos_ent = torch.cat([_[2] for _ in data], dim=0)
        # triple: (batch-size) * 3(sub, rel, -1) trp_label (batch-size) * num entity
        return triple, trp_label, pos_ent

    def get_neg_ent(self, triple, label):
        def get(triple, label):
            # one pos + num_neg neg
            if self.p.strategy == 'one_to_x':
                pos_obj = triple[2]
                # pos_obj = label
                mask = np.ones([self.p.num_ent], dtype=np.bool)
                mask[label] = 0
                neg_ent = np.int32(np.random.choice(self.entities[mask], self.p.neg_num, replace=False)).reshape([-1])
                neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))
            else:
                pos_obj = label
                mask = np.ones([self.p.num_ent], dtype=np.bool)
                mask[label] = 0
                neg_ent = np.int32(
                    np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
                neg_ent = np.concatenate((pos_obj.reshape([-1]), neg_ent))

                if len(neg_ent) > self.p.neg_num:
                    import pdb
                    pdb.set_trace()

            return neg_ent

        neg_ent = get(triple, label)
        return neg_ent

    def get_label(self, label):
        if self.p.strategy == 'one_to_n' or self.p.strategy == 'one_to_n_origin':
            y = np.zeros([self.p.num_ent], dtype=np.float32)
            for e2 in label:
                y[e2] = 1.0
        elif self.p.strategy == 'one_to_x':  # positive and negative samples
            y = [1] + [0] * self.p.neg_num
        elif self.p.strategy == 'nscaching':
            y = [1] + [0] * self.p.neg_num
        else:
            raise NotImplementedError
        return torch.FloatTensor(y)


class TestDataset(Dataset):
    """
    Evaluation Dataset class.
    Parameters
    ----------
    triples:	The triples used for evaluating the model
    params:		Parameters for the experiments

    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """

    def __init__(self, triples, params):
        self.triples = triples
        self.p = params

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        triple, label = torch.LongTensor(ele['triple']), np.int32(ele['label'])
        label = self.get_label(label)

        return triple, label

    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)
        return triple, label

    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label:
            y[e2] = 1.0
        return torch.FloatTensor(y)
