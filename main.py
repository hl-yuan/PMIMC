import os
import time
import torch
import logging
import sys
from Nmetrics import evaluate
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
import random
import load_data as loader
from network import Network
from loss import Cross_inscl_loss, Noise_robust_loss
from datasets import Data_Sampler, TrainDataset_Com, TrainDataset_All
from sklearn.cluster import KMeans
from utils import get_Similarity
import matplotlib

matplotlib.use('Agg')

def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

def pretrain(model, opt_pre, args, device, X_com, Y_com, X, Y):
    train_dataset = TrainDataset_Com(X_com, Y_com)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    t_progress = tqdm(range(args.pretrain_epochs), desc='Pretraining')

    for epoch in t_progress:
        tot_loss = 0.0
        loss_fn = torch.nn.MSELoss()
        for batch_idx, (xs, ys) in enumerate(train_loader):
            for v in range(args.V):
                xs[v] = torch.squeeze(xs[v]).to(device)

            opt_pre.zero_grad()
            zs, xrs = model(xs)
            loss_list = []
            for v in range(args.V):
                loss_value = loss_fn(xs[v], xrs[v])
                loss_list.append(loss_value)
            loss = sum(loss_list)
            loss.backward()
            opt_pre.step()
            tot_loss += loss.item()

    fea_emb = []
    for v in range(args.V):
        fea_emb.append([])

    all_dataset = TrainDataset_Com(X, Y)
    batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
    all_loader = torch.utils.data.DataLoader(dataset=all_dataset,
                                             batch_sampler=batch_sampler_all)
    with torch.no_grad():
        for batch_idx2, (xs2, _) in enumerate(all_loader):
            for v in range(args.V):
                xs2[v] = torch.squeeze(xs2[v]).to(device)
            zs2, xrs2 = model(xs2)
            for v in range(args.V):
                zs2[v] = zs2[v].cpu()
                fea_emb[v] = fea_emb[v] + zs2[v].tolist()

    for v in range(args.V):
        fea_emb[v] = torch.tensor(fea_emb[v])

    return fea_emb

def train_align(decoder_model, opt_align, args, device, X, Y, Miss_vecs, proto_Num, missindex, final_batch):
    train_dataset = TrainDataset_All(X, Y, Miss_vecs)
    batch_sampler = Data_Sampler(train_dataset, shuffle=True, batch_size=args.Batch_Rob, drop_last=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    time0 = time.time()
    AllPrototypes = [torch.tensor([]) for _ in range(args.V)]
    for batch_idx, (x, y, miss_vec) in enumerate(train_loader):
        opt_align.zero_grad()
        loss_fn = torch.nn.MSELoss().to(device)
        proto_Noiserobust = Noise_robust_loss().to(device)
        ins_contra = Cross_inscl_loss().to(device)

        loss_list_recon = []
        loss_list_ins = []
        loss_list_Rob = []
        Prototypes = [[] for _ in range(args.V)]


        for v in range(args.V):
            x[v] = torch.squeeze(x[v]).to(device)
            y[v] = torch.squeeze(y[v]).to(device)
            miss_vec[v] = torch.squeeze(miss_vec[v]).to(device)

        z, xr = decoder_model(x)
        for v in range(args.V):
            loss_list_recon.append(loss_fn(x[v][miss_vec[v] > 0], xr[v][miss_vec[v] > 0]))

        loss_recon = sum(loss_list_recon)

        for v1 in range(args.V):
            v2_start = v1 + 1
            for v2 in range(v2_start, args.V):
                align_index = []
                for i in range(x[0].shape[0]):
                    if miss_vec[v1][i] == 1 and miss_vec[v2][i] == 1:
                        align_index.append(i)

                z1 = z[v1][align_index]
                z2 = z[v2][align_index]
                l_inscontra = ins_contra(z1, z2)

                loss_list_ins.append(l_inscontra)

        loss_ins_cl = sum(loss_list_ins)

        for v1 in range(args.V):
            align_index = []
            for i in range(x[0].shape[0]):
                if miss_vec[v1][i] == 1:
                    align_index.append(i)
            Feature = z[v1][align_index]
            Pk = proto_Num[batch_idx]
            Pk = int(Pk)
            F = Feature[:, v1]
            size = F.size()[0]
            if Pk > size:
               Pk = size
            initial_prototypes = Feature[:Pk]

            max_iterations = 30
            tolerance = 1e-5
            for iteration in range(max_iterations):
                distances = torch.cdist(Feature, initial_prototypes)
                _, nearest_prototype_indices = torch.min(distances, dim=1)
                new_prototypes = torch.stack(
                    [Feature[nearest_prototype_indices == i].mean(dim=0) for i in range(Pk)])
                diff = torch.norm(new_prototypes - initial_prototypes, dim=1).max()
                if max_iterations >= 29:
                    initial_prototypes = Feature[:Pk]
                else:
                    initial_prototypes = new_prototypes
                if diff < tolerance:
                    break
            Prototypes[v1] = initial_prototypes
            initial_prototypes = initial_prototypes.to(device)
            AllPrototypes[v1] = AllPrototypes[v1].to(device)
            AllPrototypes[v1] = torch.cat((AllPrototypes[v1], initial_prototypes), dim=0)

        for v in range(args.V):
            AllPrototypes[v] = torch.tensor(AllPrototypes[v])

        for v1 in range(args.V):
            v2_start = v1 + 1
            for v2 in range(v2_start, args.V):
                prov1 = Prototypes[v1]
                prov2 = Prototypes[v2]
                l_Rob = proto_Noiserobust(prov1, prov2)
                loss_list_Rob.append(l_Rob)
        loss_pro_Rob = sum(loss_list_Rob)
        """total_loss"""
        loss_total = loss_recon + args.para_loss[0] * loss_ins_cl + args.para_loss[1] * loss_pro_Rob
        loss_total.backward()
        opt_align.step()

    fea_all = []
    for v in range(args.V):
        fea_all.append([])

    all_dataset = TrainDataset_Com(X, Y)
    batch_sampler_all = Data_Sampler(all_dataset, shuffle=False, batch_size=args.batch_size, drop_last=False)
    all_loader = torch.utils.data.DataLoader(dataset=all_dataset, batch_sampler=batch_sampler_all)
    with torch.no_grad():
        for batch_idx2, (xs2, _) in enumerate(all_loader):
            for v in range(args.V):  #
                xs2[v] = torch.squeeze(xs2[v]).to(device)
            zs2, xrs2 = decoder_model(xs2)
            for v in range(args.V):
                zs2[v] = zs2[v].cpu()
                fea_all[v] = fea_all[v] + zs2[v].tolist()

    for v in range(args.V):
        fea_all[v] = torch.tensor(fea_all[v])

    epoch_time = time.time() - time0
    all_dataset2 = TrainDataset_Com(fea_all, Y)
    batch_sampler_all2 = Data_Sampler(all_dataset2, shuffle=False, batch_size=final_batch, drop_last=False)
    all_loader2 = torch.utils.data.DataLoader(dataset=all_dataset2, batch_sampler=batch_sampler_all2)

    fea_final = []
    for v in range(args.V):
        fea_final.append([])
    for batch_idx, (xs, ys) in enumerate(all_loader2):
        for v in range(args.V):
            xs[v] = torch.squeeze(xs[v]).to(device)
            AllPrototypes[v] = torch.squeeze(AllPrototypes[v]).to(device)
        cossim_mat = []
        for v in range(args.V):
            sim_mat = get_Similarity(AllPrototypes[v], xs[v])
            sim_mat1 = get_Similarity(xs[v], xs[v])
            diag = torch.diag(sim_mat1)
            sim_diag = torch.diag_embed(diag)
            sim_mat1 = sim_mat1 - sim_diag
            for i in range(xs[0].shape[0]):
                if missindex[final_batch * batch_idx + i, v] == 0:
                    sim_mat1[:, i] = 0
            cossim_mat.append(sim_mat.t())
        for i in range(xs[0].shape[0]):
            for v in range(args.V):
                if missindex[final_batch * batch_idx + i, v] == 0:
                    vec_tmp = cossim_mat[v][i]
                    _, indices = torch.sort(vec_tmp, descending=True)
                    xs[v][i] = AllPrototypes[v][indices[0]]
        for v in range(args.V):
            fea_final[v] = fea_final[v] + xs[v].tolist()

    for v in range(args.V):
        fea_final[v] = torch.tensor(fea_final[v])

    return fea_final, epoch_time

"""  python main.py --i_d 0 --protorate 0.4 """
i_d = {
    0: "Caltech101_7",
    1: "LandUse_21",
    2: "Scene_15",
    3: "ALOI_100",
    4: "YouTubeFace10_4Views",
    5: "HandWritten",
    6: "EMNIST_digits_4Views",
    7: "AWA",
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
para_loss = [1e-4, 1e-2]
seed_everything(42)
parser = argparse.ArgumentParser(description='main_each_epoch')
parser.add_argument('--i_d', type=int, default='5')
parser.add_argument("--protorate", default=0.3, type=float)
parser.add_argument('--missrate', default=0.3, type=float)
args = parser.parse_args()
i_d = i_d[args.i_d]
print(i_d)
my_data_dic = loader.ALL_data
data_para = my_data_dic[i_d]
parser.add_argument('--dataset', default=data_para)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--Batch_Rob', default=256, type=int)
parser.add_argument('--lr_pre', default=0.0005, type=float)
parser.add_argument('--lr_align', default=0.0001, type=float)
parser.add_argument('--para_loss', default=para_loss, type=float)
parser.add_argument('--pretrain_epochs', default=1, type=int)
parser.add_argument('--ProtoRobs_epochs', default=1, type=int)
parser.add_argument('--final_batch', default=256, type=int)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--V", default=data_para['V'])
parser.add_argument("--K", default=data_para['K'])
parser.add_argument("--N", default=data_para['N'])
parser.add_argument("--view_dims", default=data_para['n_input'])
args = parser.parse_args()


def main():
    if not os.path.exists("./log/"):
        os.mkdir("./log/")
        if not os.path.exists('./log/' + str(data_para[1]) + '/'):
            os.mkdir('./log/' + str(data_para[1]) + '/')
    path = os.path.join("./log/" + str(data_para[1]) + '/')
    if not os.path.exists(path):
        os.makedirs(path)

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    current_time = current_time.replace(":", "_")
    log_filename = data_para[1] + "_protorate=" + str(
        args.protorate) + "_time=" + current_time + ".csv"

    df = pd.DataFrame(columns=['epoch', "acc", "nmi", "fscore", "ari", "recall", "precision"])
    df.to_csv(path + log_filename, index=False)

    fh = logging.FileHandler(os.path.join(path, log_filename))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


    print("--------missrate:{}--------".format(args.missrate))
    X, Y, missindex, X_com, Y_com, index_com, index_incom = loader.load_data(args.dataset, args.missrate)
    Miss_vecs = []
    for v in range(args.V):
        Miss_vecs.append(missindex[:, v])
    decoder_model = Network(args.V, args.view_dims, args.feature_dim).to(device)

    Protonum = len(X_com[0])
    n = args.N // args.Batch_Rob
    Num_C = (Protonum * args.protorate)
    c_num = Num_C // n
    proto_Num = [[] for _ in range(n)]
    for i in range(n):
        proto_Num[i] = c_num

    logging.info(args)
    logging.info("--------missrate:{}--------".format(args.missrate))
    logging.info("******** PreTraining begin ********")
    optimizer_pretrain = torch.optim.Adam(decoder_model.parameters(), lr=args.lr_pre)
    fea_emb = pretrain(decoder_model, optimizer_pretrain, args, device, X_com, Y_com, X, Y)
    optimizer_align = torch.optim.Adam(decoder_model.parameters(), lr=args.lr_align)

    logging.info("******** RobustTraining begin ********")
    acc_list, nmi_list, ari_list, fscore_list, recall_list, precision_list = [], [], [], [], [], []
    train_time = 0

    for epoch in range(args.ProtoRobs_epochs):

        fea_end, epoch_time = train_align(decoder_model, optimizer_align, args, device, X, Y, Miss_vecs, proto_Num,
                                          missindex, args.final_batch)
        train_time += epoch_time
        for v in range(args.V):
            fea_end[v] = fea_end[v].cpu()
        Labels = Y[0]
        estimator = KMeans(n_clusters=args.K)
        fea_cluster = fea_end[0]
        for i in range(1, len(fea_end)):
            fea_cluster = np.concatenate((fea_cluster, fea_end[i]), axis=1)

        estimator.fit(fea_cluster)
        pred_final = estimator.labels_

        acc, nmi, purity, fscore, precision, recall, ari = evaluate(Labels, pred_final)
        if epoch % 1 == 0:
            print(epoch, acc * 100, nmi * 100, fscore * 100, ari * 100, recall * 100, precision * 100)
            list = [epoch, acc * 100, nmi * 100, fscore * 100, ari * 100, recall * 100, precision * 100]

            data = pd.DataFrame([list])
            data.to_csv(path + log_filename, mode='a', header=False, index=False)

        acc_list.append(acc * 100)
        nmi_list.append(nmi * 100)
        fscore_list.append(fscore * 100)
        ari_list.append(ari * 100)
        recall_list.append(recall * 100)
        precision_list.append(precision * 100)


if __name__ == '__main__':
    main()
