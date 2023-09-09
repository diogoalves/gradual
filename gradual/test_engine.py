import torch
import numpy as np

from engine import Value, Tensor

def test_sanity_check_Value():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    z = z.exp()
    q = z.relu() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    z = z.exp()
    q = z.relu() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    # assert ymg.data == ypt.data.item()
    assert np.allclose(ymg.data, ypt.data.item()) , f'ymg.data={ymg.data} ypt.data.item={ypt.data.item()}'
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops_Value():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

def test_sanity_check_Tensor():
    x = Tensor([-4.0])
    z = 2 * x + 2 + x
    z = z.exp()
    q = z.relu() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    z = z.exp()
    q = z.relu() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert np.allclose(ymg.data, ypt.data.item()), f'ymg.data={ymg.data} ypt.data.item()={ypt.data.item()}'
    # backward pass went well
    assert np.allclose(xmg.grad, xpt.grad.item()), f'xmg.grad={xmg.grad} xpt.grad.item()={xpt.grad.item()}'

def test_more_ops_Tensor():

    a = Tensor([-4.0])
    b = Tensor([2.0])
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

def test_Tensor_arrays():

    x_dut = Tensor([2.0, 0.0])
    w_dut = Tensor([-3.0, 1.0])
    b_dut = Tensor([6.8813735])
    n_dut = (x_dut*w_dut).sum() + b_dut
    o_dut = n_dut.tanh()
    o_dut.backward()

    x_ref = torch.Tensor([2.0, 0.0])           ; x_ref.requires_grad = True
    w_ref = torch.Tensor([-3.0, 1.0])          ; w_ref.requires_grad = True
    b_ref = torch.Tensor([6.8813735])          ; b_ref.requires_grad = True
    n_ref = (x_ref*w_ref).sum() + b_ref
    o_ref = torch.tanh(n_ref)
    o_ref.backward()

    assert np.allclose(o_dut.data, o_ref.data.numpy()), f'o_dut.data={o_dut.data}, o_ref.data={o_ref.data.numpy()}'
    assert np.allclose(x_dut.grad, x_ref.grad.numpy()), f'x_dut.grad={x_dut.grad}, x_ref.grad={x_ref.grad.numpy()}'
    assert np.allclose(w_dut.grad, w_ref.grad.numpy()), f'w_dut.grad={w_dut.grad}, w_ref.grad={w_ref.grad.numpy()}'
    assert np.allclose(b_dut.grad, b_ref.grad.numpy()), f'b_dut.grad={b_dut.grad}, b_ref.grad={b_ref.grad.numpy()}'
 
