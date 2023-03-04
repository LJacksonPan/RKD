import torch

__all__ = ['pdist']


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def recall(embeddings, labels, K=[]):
    D = pdist(embeddings, squared=True)
    knn_inds = D.topk(1 + max(K), dim=1, largest=False, sorted=True)[1][:, 1:]

    """
    Check if, knn_inds contain index of query image.
    """
    assert ((knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)).sum().item() == 0)

    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels

    recall_k = []

    for k in K:
        correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
    return recall_k


def aggreement(embeddings1, embeddings2, labels, K=[]):
    D1 = pdist(embeddings1, squared=True)
    knn_inds1 = D1.topk(1 + max(K), dim=1, largest=False, sorted=True)[1][:, 1:]
    D2 = pdist(embeddings2, squared=True)
    knn_inds2 = D2.topk(1 + max(K), dim=1, largest=False, sorted=True)[1][:, 1:]

    """
    Check if, knn_inds contain index of query image.
    """
    assert ((knn_inds1 == torch.arange(0, len(labels), device=knn_inds1.device).unsqueeze(1)).sum().item() == 0)
    assert ((knn_inds2 == torch.arange(0, len(labels), device=knn_inds2.device).unsqueeze(1)).sum().item() == 0)

    selected_labels1 = labels[knn_inds1.contiguous().view(-1)].view_as(knn_inds1)
    selected_labels2 = labels[knn_inds2.contiguous().view(-1)].view_as(knn_inds2)
    correct_labels = selected_labels1 == selected_labels2

    agreement = []

    for k in K:
        correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        agreement.append(correct_k)
    return agreement
