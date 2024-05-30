import numpy as np
import scipy
import scipy.spatial
from itertools import chain
import time
from sklearn.metrics.pairwise import cosine_similarity
from ipdb import set_trace





def ratk(image,text,align = 0):
    # sims = scipy.spatial.distance.cdist(image, text, 'euclidean')

    sims = scipy.spatial.distance.cdist(image, text, 'cosine')
    # set_trace()
    # sims = sims + 0.2*align

    # sims = align
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)


    for index in range(npts):
        inds = np.argsort(sims[index])
        # set_trace()
        ranks[index] = np.where(inds == index)[0][0]
        top1[index] = inds[0]

    # Compute metrics
    r1 = len(np.where(ranks < 1)[0]) / len(ranks)
    r5 =  len(np.where(ranks < 5)[0]) / len(ranks)
    r10 =  len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return_ranks = False
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return r1, r5, r10


def i2t5(images, captions, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # set_trace()
    sims = np.dot(images, captions.T)
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    # set_trace()
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i5(images, captions, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = np.dot(captions, images.T)
    # set_trace()
    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)