import unittest

import torch

from pointwise_model import initialise, BodyInterfaceDecoder, POINTS, TYPES, FEATURES
from model import augment, body_loss


class ControlContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(120)
        self.points = torch.randn(3, TYPES * POINTS, FEATURES)
        self.valid = torch.ones(3, TYPES * POINTS, dtype=torch.bool)
        self.valid[0, POINTS:] = False
        self.summary = torch.rand(3, 13)
        self.summary[:, :8] /= self.summary[:, :8].sum(-1, keepdim=True)

    def test_shared_initialisation_not_trained_weights(self):
        model, shared = initialise()
        torch.manual_seed(42)
        original = BodyInterfaceDecoder("separate")
        for key, value in shared.items():
            self.assertTrue(torch.equal(value, original.state_dict()[key]))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 25509)
        other, _ = initialise()
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, other.state_dict()[key]))

    def test_no_interpoint_exchange(self):
        model, _ = initialise()
        captured = []
        handle = model.blocks[-1].register_forward_hook(lambda m, a, o: captured.append(o.detach().clone()))
        model(self.points, self.valid, self.summary)
        changed = self.points.clone()
        changed[2, 4] += torch.linspace(-1, 1, FEATURES)
        model(changed, self.valid, self.summary)
        other = torch.arange(TYPES * POINTS) != 4
        self.assertTrue(torch.equal(captured[0][2, other], captured[1][2, other]))
        self.assertGreater((captured[0][2, 4] - captured[1][2, 4]).abs().max().item(), 1e-5)
        handle.remove()

    def test_masked_empty_sets_and_permutations(self):
        model, _ = initialise()
        value = model(self.points, self.valid, self.summary)
        altered = self.points.clone()
        altered[~self.valid] = 1e5
        torch.testing.assert_close(value, model(altered, self.valid, self.summary))
        idx = torch.cat([torch.randperm(POINTS) + i * POINTS for i in range(TYPES)])
        torch.testing.assert_close(value, model(self.points[:, idx], self.valid[:, idx], self.summary), atol=2e-6, rtol=2e-6)

    def test_all_parameters_receive_finite_gradients(self):
        model, _ = initialise()
        target = torch.randint(1, 80, (3, 8)).float()
        loss = body_loss(model(self.points, self.valid, self.summary), target, torch.ones(5))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_same_d4_stream_and_unchanged_nonspatial_inputs(self):
        a = augment(self.points, torch.Generator().manual_seed(420001))
        b = augment(self.points, torch.Generator().manual_seed(420001))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(a[:, :, 4:], self.points[:, :, 4:]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
