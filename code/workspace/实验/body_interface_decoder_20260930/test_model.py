import unittest
import numpy as np
import torch

from model import BodyInterfaceDecoder, body_loss, augment, POINTS, TYPES, FEATURES
from representation import describe, targets, aggregate


class ContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def values(self):
        p = torch.randn(3,TYPES*POINTS,FEATURES)
        v = torch.ones(3,TYPES*POINTS,dtype=torch.bool)
        v[0,POINTS:] = False
        g = torch.rand(3,13)
        g[:,:8] = g[:,:8]/g[:,:8].sum(1,keepdim=True)
        return p,v,g

    def test_parameter_and_initial_state_match(self):
        torch.manual_seed(42); a = BodyInterfaceDecoder("separate")
        torch.manual_seed(42); b = BodyInterfaceDecoder("coupled")
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

    def test_empty_interfaces_and_padding(self):
        p,v,g = self.values()
        for mode in ("separate","coupled"):
            net = BodyInterfaceDecoder(mode).eval()
            a = net(p,v,g)
            altered = p.clone(); altered[~v] = 1e5
            b = net(altered,v,g)
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a,b)

    def test_within_type_permutation(self):
        p,v,g = self.values()
        idx = torch.cat([torch.randperm(POINTS)+i*POINTS for i in range(TYPES)])
        for mode in ("separate","coupled"):
            net = BodyInterfaceDecoder(mode).eval()
            torch.testing.assert_close(net(p,v,g),net(p[:,idx],v[:,idx],g),atol=2e-6,rtol=2e-6)

    def test_attention_contract(self):
        p,v,g = self.values()
        # Inspect the first block output; final readout legitimately sees all sets.
        for mode in ("separate","coupled"):
            net = BodyInterfaceDecoder(mode)
            captured = []
            handle = net.blocks[0].register_forward_hook(lambda m,a,o:captured.append(o.detach().clone()))
            net(p,v,g)
            changed = p.clone(); changed[2,2*POINTS:] += 3
            net(changed,v,g)
            difference = (captured[0][2,:POINTS]-captured[1][2,:POINTS]).abs().max().item()
            self.assertEqual(difference,0.) if mode == "separate" else self.assertGreater(difference,1e-7)
            handle.remove()

    def test_gradients_and_loss(self):
        p,v,g = self.values()
        target = torch.ones(3,8)
        for mode in ("separate","coupled"):
            net = BodyInterfaceDecoder(mode)
            loss = body_loss(net(p,v,g),target,torch.ones(5))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in net.parameters()))

    def test_d4_transforms_only_spatial_fields(self):
        p,_,_ = self.values()
        transformed = augment(p,torch.Generator().manual_seed(71))
        torch.testing.assert_close(p[:,:,4:],transformed[:,:,4:])
        for i in (0,2):
            torch.testing.assert_close(p[:,:,i:i+2].square().sum(-1),transformed[:,:,i:i+2].square().sum(-1))

    def test_descriptors_and_exact_count_evaluation(self):
        image = np.zeros((10,10),np.float32)
        pred = np.full((10,10),1,np.uint8)
        pred[2:5,2:5] = 3; pred[6:8,6:9] = 5; pred[1,2:5] = 2
        probs = np.eye(8,dtype=np.float32)[pred].transpose(2,0,1)*.992+.001
        d,l,p = describe(image,probs,np.zeros((16,10,10),np.float32))
        self.assertEqual(len(d["points"]),2)
        self.assertTrue(d["valid"][:,0].all())
        gt = pred.copy(); gt[2,2] = 4; gt[7,8] = 2
        counts,outside = targets(l,p,gt)
        assigned = np.array([4,1])  # zero-based oil classes => cluster,corner
        dense = p.copy()
        dense[l>0] = assigned[l[l>0]-1]+3
        self.assertTrue(np.array_equal(dense>=3,p>=3))
        self.assertTrue(np.array_equal(dense[p<3],p[p<3]))
        for grid,pix,tgt in (("native",dense,gt),("canonical512",dense[1::2,1::2],gt[1::2,1::2])):
            actual = np.bincount((tgt.astype(np.int64)*8+pix).ravel(),minlength=64).reshape(8,8)
            np.testing.assert_array_equal(aggregate(counts[grid],assigned,outside[grid]),actual)
        # Changing targets cannot change descriptors: describe accepts no target.
        other = describe(image,probs,np.zeros((16,10,10),np.float32))[0]
        for key in d:
            np.testing.assert_array_equal(d[key],other[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
