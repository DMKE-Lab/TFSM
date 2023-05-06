import numpy as np
import os
from collections import defaultdict
import pickle
import dgl
import torch

import json
import logging
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import random

def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.array(quadrupleList), np.asarray(times)


def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm


def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel_o = np.concatenate((rel + num_rels, rel))
    rel_s = np.concatenate((rel, rel + num_rels))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.edata['type_s'] = torch.LongTensor(rel_s)
    g.edata['type_o'] = torch.LongTensor(rel_o)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g


def id_json():
    # rels = set()
    # ents = set()
    rels = []
    ents = []
    with open('entity2id.txt', encoding='UTF-8') as fr:
        for line in fr:
            line = line.rstrip()
            ent = line.split('\t')[0]
            ent_id = int(line.split('\t')[1])
            # ents.add(ent)
            ents.append(ent)

    with open('relation2id.txt', encoding='UTF-8') as fr:
        for line in fr:
            line = line.rstrip()
            rel = line.split('\t')[0]
            rel_id = int(line.split('\t')[1])
            # rels.add(rel)
            rels.append(rel)

    relationid = {}
    # for idx, item in enumerate(list(rels)):
    for idx, item in enumerate(rels):
        relationid[item] = idx
    entid = {}
    for idx, item in enumerate(ents):
        entid[item] = idx

    json.dump(relationid, open('relation2ids', 'w'))
    json.dump(entid, open('ent2ids', 'w'))

def all_rels2json():
    quadruple =[]
    rel2quadruple = defaultdict(list)


    with open('train.txt', encoding='UTF-8') as fr:
        for line in fr:
            line = line.rstrip()
            head_id = int(line.split('\t')[0])
            rel_id = int(line.split('\t')[1])
            tail_id = int(line.split('\t')[2])
            time_id = int(line.split('\t')[3])
            quadruple = [head_id, rel_id, tail_id, time_id]
            rel2quadruple[rel_id].append(quadruple)
    with open('valid.txt', encoding='UTF-8') as fr:
        for line in fr:
            line = line.rstrip()
            head_id = int(line.split('\t')[0])
            rel_id = int(line.split('\t')[1])
            tail_id = int(line.split('\t')[2])
            time_id = int(line.split('\t')[3])
            quadruple = [head_id, rel_id, tail_id, time_id]
            rel2quadruple[rel_id].append(quadruple)
    with open('test.txt', encoding='UTF-8') as fr:
        for line in fr:
            line = line.rstrip()
            head_id = int(line.split('\t')[0])
            rel_id = int(line.split('\t')[1])
            tail_id = int(line.split('\t')[2])
            time_id = int(line.split('\t')[3])
            quadruple = [head_id, rel_id, tail_id, time_id]
            rel2quadruple[rel_id].append(quadruple)

    json.dump(rel2quadruple, open('./all_rels.json', 'w'))

def id2symbol():
    ent2id = json.load(open('./ent2ids'))
    rel2id = json.load(open('./relation2ids'))
    id2ent = {}
    id2rel = {}

    for key in ent2id.keys():
        id2ent[ent2id[key]] = key

    for key in rel2id.keys():
        id2rel[rel2id[key]] = key

    json.dump(id2ent, open('./id2ent.json', 'w'))
    json.dump(id2rel, open('./id2rel.json', 'w'))

def for_filtering():
    e1rel_e2 = defaultdict(list)
    train_tasks = json.load(open('./train_task.json'))
    dev_tasks = json.load(open('./dev_task.json'))
    test_tasks = json.load(open('./test_task.json'))
    few_quadruple = []
    # for _ in (train_tasks.values() + dev_tasks.values() + test_tasks.values()):
    for _ in (list(train_tasks.values()) + list(dev_tasks.values()) + list(test_tasks.values())):
        few_quadruple += _
    for quadruple in few_quadruple:
        e1,rel,e2,t = quadruple
        e1rel_e2[e1+rel].append(e2)

    json.dump(e1rel_e2, open('./e1rel_e2.json', 'w'))

def shuffle_ent2ids():
    ent2id = json.load(open('./ent2ids'))

    dict_key_ls = list(ent2id.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    i = 0
    for key in dict_key_ls:
        new_dic[key] = i
        i += 1
    print(i)
    json.dump(new_dic, open('./ent2ids_shuffle', 'w'))

def shuffle_rel2ids():
    rel2id = json.load(open('./relation2ids'))

    dict_key_ls = list(rel2id.keys())
    with_inv = []
    for i in dict_key_ls:
        with_inv.append(i)
        with_inv.append(i+'_inv')

    random.shuffle(with_inv)
    new_dic = {}
    i = 0
    for key in with_inv:
        new_dic[key] = i
        i += 1
    print(i)
    json.dump(new_dic, open('./rel2ids_shuffle', 'w'))


if __name__ == '__main__':
    id_json()
    all_rels2json()
    id2symbol()
    # for_filtering()
    # pass
