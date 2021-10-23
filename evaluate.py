import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import numpy as np
from Net import MyModel
from sklearn import metrics
from data_loader import my_data_loader
from config import config
from utils import AverageMeter


def valid(test_loader, model, opt):
    model.eval()
    avg_acc = AverageMeter()
    avg_pos_acc = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            if opt['model'] == 'GHE-LPC':
                word, pos1, pos2, ent_order, ent_pos, index1, index2, ent1, ent2, mask, length, scope, hier1_rel, hier2_rel, rel = data
                output, output_hier1_S, output_hier2_S, output_hier3_S, \
                    hier1_next_logits, hier2_next_logits, hier2_prev_logits, hier3_prev_logits = \
                        model(word, pos1, pos2, ent_order, ent_pos, index1, index2, ent1, ent2, mask, scope, length)

            output = torch.softmax(output, -1)
            label = rel.argmax(-1)
            _, pred = torch.max(output, -1)
            acc = (pred == label).sum().item() / label.shape[0]
            pos_total = (label != 0).sum().item()
            pos_correct = ((pred == label) & (label != 0)).sum().item()
            if pos_total > 0:
                pos_acc = pos_correct / pos_total
            else:
                pos_acc = 0
            # Log
            avg_acc.update(acc, 1)
            avg_pos_acc.update(pos_acc, 1)
            sys.stdout.write('\rstep: %d | acc: %f, pos_acc: %f'%(i, avg_acc.avg, avg_pos_acc.avg))
            sys.stdout.flush()
            y_true.append(rel[:, 1:])
            y_pred.append(output[:, 1:])
    y_true = torch.cat(y_true).detach().cpu()
    y_pred = torch.cat(y_pred).detach().cpu()
    return y_true, y_pred


def test(test_loader, opt, mode="standard", save2npy=True):
    print("=== Test ===")
    # Load model
    save_dir = opt['save_dir']
    if opt['model'] == 'GHE-LPC':
        model = MyModel(
            test_loader.dataset.vec_save_dir, 
            test_loader.dataset.hier1_rel_num(),
            test_loader.dataset.hier2_rel_num(),
            test_loader.dataset.hier3_rel_num(), 
            lambda_embed=opt['lambda_embed'], 
            use_ghe=opt['use_ghe'],
            hier_encoder_heads=opt['hier_encoder_heads']
            )
    
    if torch.cuda.is_available():
        model = model.cuda()
    # print(model)
    if opt['use_ghe']:
        print("Use Hierarchy Encoder to generate GHE!!")
    if opt['use_lpc']:
        print("Consider Local Probability Constaints!! alpha = ", opt['lpc_alpha'])

    fewrel100 = {} # test_loader.dataset.rel2id.keys()
    f = open("./data/rel100.txt", "r")
    content = f.readlines()
    for i in content:
        fewrel100[i.strip()] = 1
    f.close()
    fewrel200 = {} # test_loader.dataset.rel2id.keys()
    f = open("./data/rel200.txt", "r")
    content = f.readlines()
    for i in content:
        fewrel200[i.strip()] = 1
    f.close()

    rel2id_tmp = test_loader.dataset.rel2id
    id2rel = {v:k for k, v in rel2id_tmp.items()}
    
    state_dict = torch.load(os.path.join(save_dir, 'model.pth.tar'))['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)

    y_true_, y_pred_ = valid(test_loader, model, opt)
    y_true = y_true_.reshape(-1).numpy()
    y_pred = y_pred_.reshape(-1).numpy()
    y_true_ = y_true_.numpy()
    y_pred_ = y_pred_.numpy()
    # AUC
    auc = metrics.average_precision_score(y_true, y_pred)
    print("\n[TEST] auc: {}".format(auc))
    # P@N
    order = np.argsort(-y_pred)
    p100 = (y_true[order[:100]]).mean() * 100
    p200 = (y_true[order[:200]]).mean() * 100
    p300 = (y_true[order[:300]]).mean() * 100
    p500 = (y_true[order[:500]]).mean() * 100
    p1000 = (y_true[order[:1000]]).mean() * 100
    p2000 = (y_true[order[:2000]]).mean() * 100
    print("P@100: {0:.1f}, P@200: {1:.1f}, P@300: {2:.1f}, P@500: {3:.1f}, P@1000: {4:.1f}, P@2000: {5:.1f}, Mean: {6:.1f}".
          format(p100, p200, p300, p500, p1000, p2000, (p100 + p200 + p300 + p500 + p1000 + p2000) / 6))
    # PR
    order = np.argsort(y_pred)[::-1]
    correct = 0.
    total = y_true.sum()
    precision = []
    recall = []
    for i, o in enumerate(order):
        correct += y_true[o]
        precision.append(float(correct) / (i + 1))
        recall.append(float(correct) / total)
    precision = np.array(precision)
    recall = np.array(recall)

    
    if mode == "long-tail":
        # hits_k_100
        ss = 0
        ss10 = 0
        ss15 = 0
        ss20 = 0

        ss_rel = {}
        ss10_rel = {}
        ss15_rel = {}
        ss20_rel = {}

        for j, label in zip(y_pred_, y_true_):
            score = None
            num = None
            for ind, ll in enumerate(label):
                if ll > 0:
                    score = j[ind]
                    num = ind
                    break
            if num is None:
                continue
            if id2rel[num + 1] in fewrel100:
                ss += 1.0
                mx = 0
                for sc in j:
                    if sc > score:
                        mx = mx + 1
                if not num in ss_rel:
                    ss_rel[num] = 0
                    ss10_rel[num] = 0
                    ss15_rel[num] = 0
                    ss20_rel[num] = 0
                ss_rel[num] += 1.0
                if mx < 10:
                    ss10 += 1.0
                    ss10_rel[num] += 1.0
                if mx < 15:
                    ss15 += 1.0
                    ss15_rel[num] += 1.0
                if mx < 20:
                    ss20 += 1.0
                    ss20_rel[num] += 1.0
        print("hits_k_100:")
        print("micro")
        print(ss10 / ss * 100)
        print(ss15 / ss * 100)
        print(ss20 / ss * 100)
        print("macro")
        print((np.array([ss10_rel[i] / ss_rel[i] for i in ss_rel])).mean() * 100)
        print((np.array([ss15_rel[i] / ss_rel[i] for i in ss_rel])).mean() * 100)
        print((np.array([ss20_rel[i] / ss_rel[i] for i in ss_rel])).mean() * 100)

        # hits_k_200
        ss = 0
        ss10 = 0
        ss15 = 0
        ss20 = 0

        ss_rel = {}
        ss10_rel = {}
        ss15_rel = {}
        ss20_rel = {}

        for j, label in zip(y_pred_, y_true_):
            score = None
            num = None
            for ind, ll in enumerate(label):
                if ll > 0:
                    score = j[ind]
                    num = ind
                    break
            if num is None:
                continue
            if id2rel[num + 1] in fewrel200:
                ss += 1.0
                mx = 0
                for sc in j:
                    if sc > score:
                        mx = mx + 1
                if not num in ss_rel:
                    ss_rel[num] = 0
                    ss10_rel[num] = 0
                    ss15_rel[num] = 0
                    ss20_rel[num] = 0
                ss_rel[num] += 1.0
                if mx < 10:
                    ss10 += 1.0
                    ss10_rel[num] += 1.0
                if mx < 15:
                    ss15 += 1.0
                    ss15_rel[num] += 1.0
                if mx < 20:
                    ss20 += 1.0
                    ss20_rel[num] += 1.0
        print("hits_k_200:")
        print("micro")
        print(ss10 / ss * 100)
        print(ss15 / ss * 100)
        print(ss20 / ss * 100)
        print("macro")
        print((np.array([ss10_rel[i] / ss_rel[i] for i in ss_rel])).mean() * 100)
        print((np.array([ss15_rel[i] / ss_rel[i] for i in ss_rel])).mean() * 100)
        print((np.array([ss20_rel[i] / ss_rel[i] for i in ss_rel])).mean() * 100)

    if save2npy:
        print("Saving result")
        np.save(os.path.join(save_dir, 'precision-' + opt['model'] + '.npy'), precision)
        np.save(os.path.join(save_dir, 'recall-' + opt['model'] + '.npy'), recall)
    return y_true, y_pred


if __name__ == '__main__':
    opt = vars(config())
    if opt['model'] == 'GHE-LPC':
        test_loader = my_data_loader(opt['test'], opt, shuffle=False, training=False)
        if opt['pone']:
            print("#### pone: ####")
            test_loader_pone = my_data_loader("test_sort_pone.txt", opt, shuffle=False, training=False)
            y_true, y_pred = test(test_loader_pone, opt, save2npy=False)
        if opt['ptwo']:
            print("#### ptwo: ####")
            test_loader_ptwo = my_data_loader("test_sort_ptwo.txt", opt, shuffle=False, training=False)
            y_true, y_pred = test(test_loader_ptwo, opt, save2npy=False)
        if opt['pall']:
            print("#### pall: ####")
            test_loader_pall = my_data_loader("test_sort_pall.txt", opt, shuffle=False, training=False)
            y_true, y_pred = test(test_loader_pall, opt, save2npy=False)
        
    y_true, y_pred = test(test_loader, opt, mode="long-tail", save2npy=True)

    

