import unittest
from pathlib import Path
import shutil

import torch
from torch import nn
from torch_geometric.data import Data

from ignn.configs import INConf
from ignn.models import IGNN
from ignn.modules.SpectralFeatureDecoupling import SpectralFeatureDecoupling


def copy_shared_parameters(source, target):
    state = target.state_dict()
    for name, value in source.state_dict().items():
        mapped = name.replace(".nei_rel_learn.", ".nei_rel_learn.base.")
        if mapped in state and state[mapped].shape == value.shape:
            state[mapped] = value.clone()
    target.load_state_dict(state)


class SpectralFeatureDecouplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_dir = Path(__file__).parent / ".sfd_cache"
        shutil.rmtree(cls.cache_dir, ignore_errors=True)
        cls.cache_dir.mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cache_dir, ignore_errors=True)

    def test_initial_output_matches_concat_exactly(self):
        torch.manual_seed(7)
        base = nn.Sequential(nn.Dropout(0.0), nn.Linear(12, 4), nn.ReLU())
        module = SpectralFeatureDecoupling(base, n_hops=2, width=4)
        x = torch.randn(9, 12)
        module.eval()
        torch.testing.assert_close(module(x), base(x), rtol=0, atol=0)

    def test_signed_channel_gates_and_gradient(self):
        module = SpectralFeatureDecoupling(nn.Linear(12, 3), n_hops=2, width=4)
        with torch.no_grad():
            module.detail_gates[0].fill_(2.0)
            module.detail_gates[1].fill_(-2.0)
        coeff = module.coefficients()
        self.assertTrue(torch.all(coeff[0] > 0))
        self.assertTrue(torch.all(coeff[1] < 0))
        module(torch.randn(5, 12)).sum().backward()
        self.assertGreater(module.detail_gates.grad.abs().sum().item(), 0)

    def test_full_model_cpu_cuda_cache_and_multilayer(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for fast, layers, hops in [(False, 1, 1), (False, 2, 3), (True, 1, 3), (True, 2, 1)]:
                with self.subTest(device=device, fast=fast, layers=layers, hops=hops):
                    params = dict(in_feats=5, h_feats=8, n_clusters=3, IN="IN-SN", n_hops=hops,
                                  n_layers=layers, fast=fast, agg_type="gcn_incep")
                    base = IGNN(RN="concat", **params).to(device).eval()
                    model = IGNN(RN="spectral_decoupling", **params).to(device).eval()
                    copy_shared_parameters(base, model)
                    data = Data(x=torch.randn(12, 5), y=torch.arange(12) % 3,
                                edge_index=torch.stack((torch.arange(12), torch.arange(12).roll(1)))).to(device)
                    config = INConf(
                        f"spectral_{device}_{fast}_{layers}_{hops}", hops,
                        fast=fast, symm_norm=True, save_dir=str(self.cache_dir))
                    expected = base.classifier(
                        base(data.edge_index, data.x, config, device))
                    actual = model.classifier(
                        model(data.edge_index, data.x, config, device))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    actual.square().mean().backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                        if p.grad is not None))

    def test_raw_mode_preserves_none_fusion_shape_and_initial_logits(self):
        params = dict(in_feats=5, h_feats=8, n_clusters=3, IN="IN-SN", n_hops=3,
                      n_layers=1, fast=False, agg_type="gcn_incep")
        base = IGNN(RN="none", **params).eval()
        model = IGNN(RN="spectral_decoupling_raw", **params).eval()
        copy_shared_parameters(base, model)
        data = Data(x=torch.randn(12, 5), y=torch.arange(12) % 3,
                    edge_index=torch.stack((torch.arange(12), torch.arange(12).roll(1))))
        config = INConf("spectral_raw_test", 3, fast=False, symm_norm=True)
        expected = base.classifier(base(data.edge_index, data.x, config, "cpu"))
        actual = model.classifier(model(data.edge_index, data.x, config, "cpu"))
        self.assertEqual(model.classifier[1].in_features, 32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
