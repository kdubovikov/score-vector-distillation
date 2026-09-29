"""Independent value/gradient oracles for the historical forward-KL recipe."""
import math

import pytest
import torch

from score_vector_distillation.config import DistillationLoss
from score_vector_distillation.losses import centered_mse_loss, forward_kl_loss
from score_vector_distillation.training.model import _DISTILLATION_LOSSES


def test_direction_reduction_gradient_and_detached_teacher():
    student = torch.tensor([[.2, -.1, .4], [.1, .3, -.2]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([[.3, .2, -.1], [-.1, .1, .4]], dtype=torch.float64, requires_grad=True)
    tau = .1
    ps = torch.softmax(student.detach()/tau, -1)
    pt = torch.softmax(teacher.detach()/tau, -1)
    expected = tau**2 * sum(float(p * math.log(p/q)) for p,q in zip(pt.flatten(), ps.flatten()))/2
    actual = forward_kl_loss(student, teacher)
    assert float(actual.detach()) == pytest.approx(expected, abs=1e-14)
    actual.backward()
    torch.testing.assert_close(student.grad, tau*(ps-pt)/2)
    assert teacher.grad is None
    assert _DISTILLATION_LOSSES[DistillationLoss.FORWARD_KL_TAU_0P1] is forward_kl_loss


def test_offsets_identity_and_single_candidate():
    torch.manual_seed(11)
    s = torch.randn(3, 32, dtype=torch.float64, requires_grad=True)
    t = torch.randn_like(s)
    torch.testing.assert_close(forward_kl_loss(s,t), forward_kl_loss(s+7,t-3))
    identical = forward_kl_loss(s, s)
    torch.testing.assert_close(identical, torch.zeros_like(identical), atol=1e-14, rtol=0)
    grad = torch.autograd.grad(identical,s)[0]
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-14, rtol=0)
    one = forward_kl_loss(s[:,:1],t[:,:1])
    torch.testing.assert_close(one, torch.zeros_like(one), atol=0, rtol=0)


def test_high_temperature_scale_and_gradient():
    s = torch.tensor([[.1, .3, -.2]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[.2, -.1, .4]], dtype=torch.float64)
    kl = forward_kl_loss(s,t,temperature=1000)
    mse = centered_mse_loss(s,t)
    torch.testing.assert_close(2*kl,mse,rtol=2e-4,atol=1e-9)
    torch.testing.assert_close(2*torch.autograd.grad(kl,s,retain_graph=True)[0],
                               torch.autograd.grad(mse,s)[0],rtol=1e-3,atol=1e-7)


@pytest.mark.parametrize('temperature',[0,-1,float('nan'),float('inf')])
def test_reject_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        forward_kl_loss(torch.zeros(1,3),torch.zeros(1,3),temperature=temperature)
