import torch
import torch.nn.functional as F
from kmeans_gpu import kmeans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def NormalizeFeaTorch(features):
    rowsum = torch.tensor((features ** 2).sum(1))
    r_inv = torch.pow(rowsum, -0.5)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    normalized_feas = torch.mm(r_mat_inv, features)
    return normalized_feas



def get_Similarity(fea_mat1, fea_mat2):
    Sim_mat = F.cosine_similarity(fea_mat1.unsqueeze(1), fea_mat2.unsqueeze(0), dim=-1)
    return Sim_mat


def clustering(feature, cluster_num):

    predict_labels,  cluster_centers = kmeans(X=feature, num_clusters=cluster_num, distance='euclidean', device=device)
    return predict_labels.numpy(), cluster_centers




def euclidean_dist(x, y, root=False):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    if root:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

