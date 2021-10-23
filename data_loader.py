import os
import json
import torch
import pickle
import numpy as np
import torch.utils.data as data
from collections import Counter
from tqdm import tqdm

########################################################################
############################  For RHI-EOP ##############################
class MyDataset(data.Dataset):

    def __init__(self, file_name, opt, training=True):
        super().__init__()
        self.file_name = file_name
        self.processed_data_dir = opt['processed_data_dir']
        self.data_dir = os.path.join(opt['root'], self.file_name)
        self.hier1_rel_dir = os.path.join(opt['root'], opt['hier1_rel'])
        self.hier2_rel_dir = os.path.join(opt['root'], opt['hier2_rel'])
        self.rel_dir = os.path.join(opt['root'], opt['rel'])
        self.vec_dir = os.path.join(opt['root'], opt['vec'])
        self.max_length = opt['max_length']
        self.max_pos_length = opt['max_pos_length']
        self.training = training
        self.vec_save_dir = os.path.join(self.processed_data_dir, 'word_vec.npy')
        self.word2id_save_dir = os.path.join(self.processed_data_dir, 'word2id.json')
        self.init_rel()

        if not os.path.exists("./result"):
            os.mkdir("./result")
        if not os.path.exists(opt['save_dir']):
            os.mkdir(opt['save_dir'])

        if not os.path.exists("./_processed_data"):
            os.mkdir("./_processed_data")
        if not os.path.exists(self.processed_data_dir):
            os.mkdir(self.processed_data_dir)
            
        if os.path.exists(self.vec_save_dir) and os.path.exists(self.word2id_save_dir):
            self.word2id = json.load(open(self.word2id_save_dir))
        else:
            print("Extracting word2vec data")
            self.init_word()

        try:
            print("Trying to load processed data")
            self.data = pickle.load(open(os.path.join(self.processed_data_dir, self.file_name.split(".")[0]+'.pkl'), 'rb'))
            print("Load successfully")
        except:
            print("Processed data does not exist")
            self._preprocess()
        print("bag num:", self.__len__())

    def __len__(self):
        return len(self.data)

    def _preprocess(self):
        # Load files
        print("Loading data file...")
        ori_data = []
        with open(self.data_dir) as f:
            for line in f.readlines():
                line = line.strip().split()
                # print(line)
                line_data = {
                    "ent1": {"id": line[0], "name": line[2]},
                    "ent2": {"id": line[1], "name": line[3]},
                    "rel": self.rel2id[line[4]] if line[4] in self.rel2id else self.rel2id['NA'],
                    "rel_name": line[4],
                    "words": line[5:-1]
                }
                ori_data.append(line_data)
        print("Finish loading")
        # Sort data by entities and relations
        print("Sort data...")
        if self.training:
            ori_data.sort(key=lambda a: a['ent1']['id'] + '#' + a['ent2']['id'] + '#' + str(a['rel']))
        else:
            ori_data.sort(key=lambda a: a['ent1']['id'] + '#' + a['ent2']['id'])
        print("Finish sorting")
        # Pre-process data
        print("Pre-processing data...")
        last_bag = None
        self.data = []
        bag = {
            'word': [],
            'hier1_rel': [],
            'hier2_rel': [],
            'rel': [],
            'pos1': [],
            'pos2': [],
            'ent_order': [],
            'ent_pos': [],
            'index1': [],
            'index2': [],
            'ent1': [],
            'ent2': [],
            'mask': [],
            'length': []
        }
        for ins in tqdm(ori_data):
            if self.training:
                cur_bag = (ins['ent1']['id'], ins['ent2']['id'], ins['rel'])  # used for train
            else:
                cur_bag = (ins['ent1']['id'], ins['ent2']['id'])  # used for test

            if cur_bag != last_bag:
                if last_bag is not None:
                    self.data.append(bag)
                    bag = {
                        'word': [],
                        'hier1_rel': [],
                        'hier2_rel': [],
                        'rel': [],
                        'pos1': [],
                        'pos2': [],
                        'ent_order': [],
                        'ent_pos': [],
                        'index1': [],
                        'index2': [],
                        'ent1': [],
                        'ent2': [],
                        'mask': [],
                        'length': []
                    }
                last_bag = cur_bag

            # rel
            bag['rel'].append(ins['rel'])
            if ins['rel_name'] == 'NA':
                bag['hier1_rel'].append(self.hier1_rel2id['NA'])
                bag['hier2_rel'].append(self.hier2_rel2id['NA'])
            else:
                relation_split = ins['rel_name'].strip().split('/')
                first_relation = relation_split[1]
                second_relation = relation_split[2]
                if first_relation in self.hier1_rel2id:
                    bag['hier1_rel'].append(self.hier1_rel2id[first_relation])
                else:
                    bag['hier1_rel'].append(self.hier1_rel2id['NA'])

                temp_hier2_relation = '/' + first_relation + '/' + second_relation
                if temp_hier2_relation in self.hier2_rel2id:
                    bag['hier2_rel'].append(self.hier2_rel2id[temp_hier2_relation])
                else:
                    bag['hier2_rel'].append(self.hier2_rel2id['NA'])

            # word
            words = ins['words']
            _ids = [self.word2id[word] if word in self.word2id else self.word2id['[UNK]'] for word in words]
            _ids = _ids[:self.max_length]
            _ids.extend([self.word2id['[PAD]'] for _ in range(self.max_length - len(words))])
            bag['word'].append(_ids)

            # ent
            ent1 = ins['ent1']['name']
            ent2 = ins['ent2']['name']
            _ent1 = self.word2id[ent1] if ent1 in self.word2id else self.word2id['[UNK]']
            _ent2 = self.word2id[ent2] if ent2 in self.word2id else self.word2id['[UNK]']
            bag['ent1'].append(_ent1)
            bag['ent2'].append(_ent2)

            # pos
            p1 = words.index(ent1) if ent1 in words else 0
            p2 = words.index(ent2) if ent2 in words else 0
            p1 = p1 if p1 < self.max_length else self.max_length - 1
            p2 = p2 if p2 < self.max_length else self.max_length - 1
            bag['index1'].append(p1)
            bag['index2'].append(p2)
            if p1<p2:
                bag['ent_order'].append(0)
            else:
                bag['ent_order'].append(1)  # 逆序

            _ent_pos = p1 - p2 + self.max_pos_length
            if _ent_pos > 2 * self.max_pos_length:
                _ent_pos = 2 * self.max_pos_length
            if _ent_pos < 0:
                _ent_pos = 0
            bag['ent_pos'].append(_ent_pos)

            _pos1 = np.arange(self.max_length) - p1 + self.max_pos_length
            _pos2 = np.arange(self.max_length) - p2 + self.max_pos_length

            _pos1[_pos1 > 2 * self.max_pos_length] = 2 * self.max_pos_length
            _pos1[_pos1 < 0] = 0
            _pos2[_pos2 > 2 * self.max_pos_length] = 2 * self.max_pos_length
            _pos2[_pos2 < 0] = 0

            bag['pos1'].append(_pos1)
            bag['pos2'].append(_pos2)

            # mask
            p1, p2 = sorted((p1, p2))
            _mask = np.zeros(self.max_length, dtype=np.long)
            _mask[p2 + 1: len(words)] = 3
            _mask[p1 + 1: p2 + 1] = 2
            _mask[:p1 + 1] = 1
            _mask[len(words):] = 0
            bag['mask'].append(_mask)

            # sentence length
            _length = min(len(words), self.max_length)
            bag['length'].append(_length)

        # append the last bag
        if last_bag is not None:
            self.data.append(bag)

        print("Finish pre-processing")
        print("Storing processed files...")
        pickle.dump(self.data, open(os.path.join(self.processed_data_dir, self.file_name.split(".")[0]+'.pkl'), 'wb'))
        print("Finish storing")

    # 按照index取对应的bag
    def __getitem__(self, index):
        bag = self.data[index]
        word = torch.tensor(bag['word'], dtype=torch.long)
        pos1 = torch.tensor(bag['pos1'], dtype=torch.long)
        pos2 = torch.tensor(bag['pos2'], dtype=torch.long)
        ent_order = torch.tensor(bag['ent_order'], dtype=torch.long)
        ent_pos = torch.tensor(bag['ent_pos'], dtype=torch.long)
        index1 = torch.tensor(bag['index1'],  dtype=torch.long)
        index2 = torch.tensor(bag['index2'],  dtype=torch.long)
        ent1 = torch.tensor(bag['ent1'],  dtype=torch.long)
        ent2 = torch.tensor(bag['ent2'],  dtype=torch.long)
        mask = torch.tensor(bag['mask'], dtype=torch.long)
        length = torch.tensor(bag['length'], dtype=torch.long)
        hier1_rel = torch.tensor(bag['hier1_rel'],  dtype=torch.long)
        hier2_rel = torch.tensor(bag['hier2_rel'],  dtype=torch.long)
        rel = torch.tensor(bag['rel'],  dtype=torch.long)
        if self.training:
            hier1_rel = hier1_rel[0]
            hier2_rel = hier2_rel[0]
            rel = rel[0]
        else:
            # test 时， 存在同一实例，多标签现象，所以label向量可能含有多个 1 
            # hier1_rel
            hier1_rel_mul = torch.zeros(self.num_hier1_rel, dtype=torch.long)
            for i in set(hier1_rel):
                hier1_rel_mul[i] = 1
            hier1_rel = hier1_rel_mul

            # hier2_rel
            hier2_rel_mul = torch.zeros(self.num_hier2_rel, dtype=torch.long)
            for i in set(hier2_rel):
                hier2_rel_mul[i] = 1
            hier2_rel = hier2_rel_mul

            # rel
            rel_mul = torch.zeros(self.num_rel, dtype=torch.long)
            for i in set(rel):
                rel_mul[i] = 1
            rel = rel_mul
        return word, pos1, pos2, ent_order, ent_pos, index1, index2, ent1, ent2, mask, length, hier1_rel, hier2_rel, rel

    def init_word(self):
        f = open(self.vec_dir)
        num, dim = [int(x) for x in f.readline().split()[:2]]
        self.word2id = {}
        word_vec = np.zeros([num+2, dim], dtype=np.float32)
        for line in f.readlines():
            line = line.strip().split()
            word_vec[len(self.word2id)] = np.array(line[1:])
            self.word2id[line[0].lower()] = len(self.word2id)
        f.close()
        word_vec[len(self.word2id)] = np.random.randn(dim) / np.sqrt(dim)
        self.word2id['[UNK]'] = len(self.word2id)
        self.word2id['[PAD]'] = len(self.word2id)
        np.save(self.vec_save_dir, word_vec)
        json.dump(self.word2id, open(self.word2id_save_dir, 'w'))

    def init_rel(self):
        # hier1_rel
        f = open(self.hier1_rel_dir)
        self.hier1_rel2id = {}
        self.num_hier1_rel = int(f.readline().strip())
        for line in f.readlines():
            line = line.strip().split()
            self.hier1_rel2id[line[0]] = int(line[-1])

        # hier2_rel
        f = open(self.hier2_rel_dir)
        self.hier2_rel2id = {}
        self.num_hier2_rel = int(f.readline().strip())
        for line in f.readlines():
            line = line.strip().split()
            self.hier2_rel2id[line[0]] = int(line[-1])

        # rel
        f = open(self.rel_dir)
        self.rel2id = {}
        self.num_rel = int(f.readline().strip())
        for line in f.readlines():
            line = line.strip().split()
            self.rel2id[line[0]] = int(line[-1])

    def hier1_rel_num(self):
        return self.num_hier1_rel

    def hier2_rel_num(self):
        return self.num_hier2_rel

    def hier3_rel_num(self):
        return self.num_rel

    def loss_weight(self, message, key, num_rel):
        print(message)
        rel_ins = []
        for bag in self.data:
            rel_ins.extend(bag[key])
        stat = Counter(rel_ins)
        class_weight = torch.ones(num_rel, dtype=torch.float32)
        for k, v in stat.items():
            class_weight[k] = 1. / v**0.05
        if torch.cuda.is_available():
            class_weight = class_weight.cuda()
        return class_weight


# 将batch_size个样本bag，整理成一个batch
def my_collate_fn(X):
    X = list(zip(*X))
    word, pos1, pos2, ent_order, ent_pos, index1, index2, ent1, ent2, mask, length, hier1_rel, hier2_rel, rel = X
    scope = []
    ind = 0
    for w in word:
        scope.append((ind, ind + len(w)))
        ind += len(w)
    scope = torch.tensor(scope, dtype=torch.long)
    word = torch.cat(word, 0)
    pos1 = torch.cat(pos1, 0)
    pos2 = torch.cat(pos2, 0)
    ent_order = torch.cat(ent_order, 0)
    ent_pos = torch.cat(ent_pos, 0)
    index1 = torch.cat(index1, 0)
    index2 = torch.cat(index2, 0)
    mask = torch.cat(mask, 0)
    ent1 = torch.cat(ent1, 0)
    ent2 = torch.cat(ent2, 0)
    length = torch.cat(length, 0)
    hier1_rel = torch.stack(hier1_rel)
    hier2_rel = torch.stack(hier2_rel)
    rel = torch.stack(rel)
    return word, pos1, pos2, ent_order, ent_pos, index1, index2, ent1, ent2, mask, length, scope, hier1_rel, hier2_rel, rel


def my_data_loader(data_file, opt, shuffle, training=True, num_workers=2):
    dataset = MyDataset(data_file, opt, training)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=opt['batch_size'],
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=my_collate_fn,
                             drop_last=True)
    return loader