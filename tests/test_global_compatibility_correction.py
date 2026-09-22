import unittest

import torch

from ignn.modules.GlobalCompatibilityCorrection import (
    GlobalCompatibilityCorrection,
)


class GlobalCompatibilityCorrectionTests(unittest.TestCase):
    def test_global_correction_is_finite_and_differentiable(self):
        matrix = torch.tensor([[.2, .8], [.7, .3]])
        module = GlobalCompatibilityCorrection(matrix)
        logits = torch.randn(5, 2, requires_grad=True)
        indices = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
        adjacency = torch.sparse_coo_tensor(
            indices, torch.ones(5), (5, 5)).coalesce()
        labels = torch.tensor([0, 1, 0, 1, 0])
        train_mask = torch.tensor([True, True, False, False, False])

        output = module(logits, adjacency, labels, train_mask)
        output.sum().backward()

        self.assertEqual(output.shape, logits.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(
            module.compatibility().sum(dim=1), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
