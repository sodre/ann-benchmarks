import os

import numpy as np
from securevector_core import HnswIndex

from ..base.module import BaseANN


class SecureVector(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = {"angular": "cosine", "euclidean": "euclidean"}[metric]
        self._M = method_param["M"]
        self._ef_construction = method_param["efConstruction"]
        self._ef = 10

    def fit(self, X):
        n, d = X.shape
        cache_dir = os.path.join("results", "securevector_indices", f"n{n}_d{d}_{self._metric}")
        path = os.path.join(
            cache_dir,
            f"M{self._M}_ef{self._ef_construction}.bin",
        )
        if os.path.exists(path):
            self._index = HnswIndex.load(path)  # load() runs optimize automatically
        else:
            self._index = HnswIndex(d, self._metric, self._M, self._ef_construction, n)
            self._index.insert_numpy(np.ascontiguousarray(X, dtype=np.float32))
            self._index.optimize()
            os.makedirs(cache_dir, exist_ok=True)
            self._index.save(path)

    def set_query_arguments(self, ef):
        self._ef = ef
        self.name = f"SecureVector(M={self._M}, ef_c={self._ef_construction}, ef={ef})"

    # -- Single query: prepared pattern (dtype conversion outside timer) --

    def prepare_query(self, q, n):
        self._q = np.ascontiguousarray(q, dtype=np.float32)
        self._n = n

    def run_prepared_query(self):
        self._results = self._index.search_numpy(self._q, self._n, self._ef)

    def get_prepared_query_results(self):
        return self._results

    # -- Batch query: entire loop in Rust --

    def prepare_batch_query(self, X, n):
        self._batch_X = np.ascontiguousarray(X, dtype=np.float32)
        self._batch_n = n

    def run_batch_query(self):
        self._batch_results = self._index.batch_search_numpy(
            self._batch_X, self._batch_n, self._ef
        )

    def get_batch_results(self):
        return self._batch_results

    # -- Fallback for non-prepared mode --

    def query(self, q, n):
        return self._index.search_numpy(np.ascontiguousarray(q, dtype=np.float32), n, self._ef)

    def __str__(self):
        return f"SecureVector(M={self._M}, ef_c={self._ef_construction}, ef={self._ef})"
