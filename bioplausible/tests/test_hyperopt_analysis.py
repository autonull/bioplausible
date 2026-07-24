import unittest

import numpy as np

from bioplausible.hyperopt.analysis import encode_configs, reduce_dimensions


class TestAnalysis(unittest.TestCase):
    def test_encode_configs(self):
        configs = [
            {"lr": 0.01, "layers": 2, "opt": "sgd"},
            {"lr": 0.001, "layers": 4, "opt": "adam"},
            {"lr": 0.05, "layers": 2, "opt": "sgd", "new_param": 100},
        ]

        matrix = encode_configs(configs)

        # Check shape
        # 3 configs
        # Num features: lr, layers, new_param = 3
        # Cat features: opt (sgd, adam) = 2 columns (one-hot)
        # Total = 5
        self.assertEqual(matrix.shape[0], 3)
        self.assertGreaterEqual(matrix.shape[1], 4)

    def test_reduce_dimensions_pca(self):
        # Create random data
        X = np.random.rand(10, 5)

        reduced = reduce_dimensions(X, method="pca", n_components=2)

        self.assertEqual(reduced.shape, (10, 2))

    def test_reduce_dimensions_tsne(self):
        # Create random data
        X = np.random.rand(10, 5)

        reduced = reduce_dimensions(X, method="tsne", n_components=2)

        self.assertEqual(reduced.shape, (10, 2))

    def test_empty_input(self):
        matrix = encode_configs([])
        self.assertEqual(matrix.size, 0)

        reduced = reduce_dimensions(matrix)
        self.assertEqual(reduced.size, 0)


if __name__ == "__main__":
    unittest.main()
