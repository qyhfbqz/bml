
from sklearn.neighbors import NearestNeighbors
from random import sample
import numpy as np


class HopkinsStatistic(object):
    """
    The Hopkins statistic (introduced by Brian Hopkins and John Gordon Skellam) 
    is a way of measuring the cluster tendency of a data set.
    """

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def fit(inputs, sample_ratio=0.1, n_jobs=-1):
        row, col = inputs.shape
        num_samples = int(row * sample_ratio)

        nbrs = NearestNeighbors(n_neighbors=2, n_jobs=n_jobs).fit(inputs)

        rands = sample(range(row), num_samples)

        ujd = []
        wjd = []
        min_x = np.amin(inputs, axis=0)
        max_x = np.amax(inputs, axis=0)
        for i in range(num_samples):
            random_vec = np.random.uniform(min_x, max_x, col).reshape(1, -1)
            u_dist, _ = nbrs.kneighbors(random_vec, 1, return_distance=True)
            ujd.append(u_dist[0][0])
            w_dist, _ = nbrs.kneighbors(inputs[i], 2, return_distance=True)
            wjd.append(w_dist[0][1])
        H = sum(ujd) / (sum(ujd) + sum(wjd))

        return H
