import unittest

import torch

from utils.normalization import Normalizer


class NormalizerTest(unittest.TestCase):
    def test_normalize_and_inverse(self):
        normalizer = Normalizer(2)
        data = torch.tensor([[1.0, 3.0], [3.0, 7.0]])
        normalized = normalizer(data, accumulate=True)

        torch.testing.assert_close(normalizer.mean, torch.tensor([[2.0, 5.0]]))
        torch.testing.assert_close(normalizer.std, torch.tensor([[1.0, 2.0]]))
        torch.testing.assert_close(
            normalized, torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        )
        torch.testing.assert_close(normalizer.inverse(normalized), data)


if __name__ == "__main__":
    unittest.main()
