import numpy as np
import torch
import unittest
from Tensor import Tensor

x_init = np.random.randn(1,3).astype(np.float32)
W_init = np.random.randn(3,3).astype(np.float32)
m_init = np.random.randn(1,3).astype(np.float32)

class TestAutograd(unittest.TestCase):
    def test_backward_pass(self):
        def test_autograd():
            x = Tensor(x_init)
            W = Tensor(W_init)
            m = Tensor(m_init)
            out = x.dot(W).relu()
            #out = out.logsoftmax()
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.data, x.grad, W.grad

        def test_pytorch():
            x = torch.tensor(x_init, requires_grad=True)
            W = torch.tensor(W_init, requires_grad=True)
            m = torch.tensor(m_init)
            out = x.matmul(W).relu()
            #out = torch.nn.functional.log_softmax(out, dim=1)
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.detach().numpy(), x.grad, W.grad
    
        for x,y in zip(test_autograd(), test_pytorch()):
            if x != y:
                assert True, "Test Failed!"


TestAutograd().test_backward_pass()
