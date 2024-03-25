from ..base import BaseStreamingANN

from py_graph_anns import PyKnnGraph, NdArrayWithId

import numpy as np
import numpy.typing as npt


class GraphANNSStreamingANN(BaseStreamingANN):
    def setup(self, dtype, max_pts, ndims) -> None:
        if dtype != "float32":
            raise NotImplementedError("GraphANNS only supports float32")

        self.max_pts = max_pts
        self.ndims = ndims

        self.g = PyKnnGraph(
            max_pts,
            100,
            10,
            True,
            2,
            True,
            1,
        )

    def insert(self, X: np.array, ids: npt.NDArray[np.uint32]) -> None:
        # todo: could make this multithreaded if we pushed the batch down into
        # the rust layer
        for i in range(X.shape[0]):
            x = NdArrayWithId(ids[i], X[i])
            self.g.insert(x)

    def delete(self, ids: npt.NDArray[np.uint32]) -> None:
        for i in range(ids.shape[0]):
            self.g.delete(ids[i])

    def query(self, X: np.array, k: int) -> npt.NDArray[np.uint32]:
        results = []
        for i in range(X.shape[0]):
            x = NdArrayWithId(0, X[i])
            results.append(self.g.query(x, k))
        self.res = np.array(results, dtype=np.uint32)
