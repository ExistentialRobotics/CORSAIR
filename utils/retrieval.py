from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist, squareform
import numpy as np


def compute_mAP(descriptors, gt_labels):
    """
    Compute mAP for retrieval, use subcategory as label
    descriptors: np array [n, dim]
    gt_labels: np array [n, ]
    """
    dists = squareform(pdist(descriptors, "cosine"))

    retrieval_out = dict(mAP_inst=[], mAP_cls={})
    for d, c in zip(dists, gt_labels):
        positive = gt_labels == c

        score = 100.0 * average_precision_score(positive, 2 - d)
        retrieval_out["mAP_inst"].append(score)
        retrieval_out["mAP_cls"][c] = retrieval_out["mAP_cls"].get(c, [])
        retrieval_out["mAP_cls"][c].append(score)
        # retrieval_out['AUC'].append(roc_auc_score(positive, 2-d))
    # compute per class avg
    retrieval_out["mAP_cls"] = [np.mean(v) for v in retrieval_out["mAP_cls"].values()]
    retrieval_out["mAP_inst"] = np.mean(np.array(retrieval_out["mAP_inst"]))
    return retrieval_out["mAP_inst"], retrieval_out["mAP_cls"]


def retrieval_dist(dists, threshold, table):
    stat = {}

    scores = []
    percision = []
    top1_error = []

    accept_range = int(len(dists) * threshold)
    # exclude itself
    rank_pd = np.argsort(dists, axis=1)
    rank_gt = np.argsort(table, axis=1)

    for d, p, g, t in zip(dists, rank_pd, rank_gt, table):
        # accept_range = max(min(accept_range_ref, np.sum(t<0.15)), 1)
        p = p[1 : accept_range + 1]
        g = g[:accept_range]
        positive = np.isin(p, g).astype(np.int32)

        if table[p[0], g[0]] == 200:
            top1_error.append(0)
        else:
            top1_error.append(table[p[0], g[0]])

        percision.append(100.0 * np.sum(positive) / accept_range)
        if np.sum(positive) == 0:
            score = 0
        else:
            score = 100.0 * average_precision_score(positive, 2 - d[p])
        scores.append(score)

    stat["mAP"] = np.mean(np.array(scores))
    stat["percision"] = np.mean(np.array(percision))
    stat["top1_error"] = np.mean(np.array(top1_error))
    return stat


def retrieval_eval(descriptors, threshold, table):
    """
    Evaluate retrieval using chamfer distance as label
    Input:
        descriptors: array of shape [n, dim]
        threshold: a float in (0, 1)
        table: array of shape [n, n]
    """
    stat = {}

    scores = []
    percision = []
    top1_error = []

    accept_range = int(len(descriptors) * threshold)
    dists = squareform(pdist(descriptors, "cosine"))
    # exclude itself
    rank_pd = np.argsort(dists, axis=1)
    rank_gt = np.argsort(table, axis=1)

    for d, p, g, t in zip(dists, rank_pd, rank_gt, table):
        # accept_range = max(min(accept_range_ref, np.sum(t<0.15)), 1)
        p = p[1 : accept_range + 1]
        g = g[:accept_range]
        positive = np.isin(p, g).astype(np.int32)
        percision.append(100.0 * np.sum(positive) / accept_range)

        # print(table[p[1], g[0]])

        if table[p[0], g[0]] == 200:
            top1_error.append(0)
        else:
            top1_error.append(table[p[0], g[0]])

        if np.sum(positive) == 0:
            score = 0
        else:
            score = 100.0 * average_precision_score(positive, 2 - d[p])
        scores.append(score)

    stat["mAP"] = np.mean(np.array(scores))
    stat["percision"] = np.mean(np.array(percision))
    stat["top1_error"] = np.mean(np.array(top1_error))

    return stat


def get_rank(descriptors, top_n=10):
    """
    Get top retrieved results. Itself is excluded.
    """

    dists = squareform(pdist(descriptors, "cosine"))
    rank = np.argsort(dists, 1)[:, 1 : top_n + 1]
    return rank


def scan2cad_retrieval_eval_dist(dists, table, best_match, pos_n):
    """
    Scan2cad retrieval using distance matrix
    """
    precision = []
    top1_error = []
    top1_predict = []
    gt = []

    pred_rank = np.argsort(dists, 1)
    gt_rank = np.argsort(table[best_match, :], 1)

    for g, p in zip(gt_rank, pred_rank):
        positive = np.isin(p[:pos_n], g[:pos_n]).astype(np.int32)
        precision.append(100.0 * np.sum(positive) / pos_n)

        top1_error.append(table[p[0], g[0]])

        top1_predict.append(p[0])
        gt.append(g[0])

    stat = {
        "precision": sum(precision) / len(precision),
        "top1_error": sum(top1_error) / len(top1_error),
        "top1_predict": top1_predict,
        "gt": gt,
    }

    return stat


def scan2cad_retrieval_eval(scan_feats, lib_feats, best_match, table, pos_n):
    """
    Scan2cad retrieval using descriptors
    """

    dists = np.linalg.norm(scan_feats[:, None, :] - lib_feats[None, :, :], ord=2, axis=2)

    return scan2cad_retrieval_eval_dist(dists, table, best_match, pos_n)
