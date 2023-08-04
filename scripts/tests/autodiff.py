"""Testing the autodifferentiation engine."""
import subprocess
import os
import torch

cwd = os.getcwd()
my_impl = "./autodiff"

torch_res = []


def get_res(code):
    """Get the result from running my implementation."""
    output_bytes = subprocess.check_output(my_impl + f" {code}",
                                           shell=True, cwd=cwd)
    output_str = output_bytes.decode('utf-8')
    # alphebetical order, first is letter of the variable,
    # second is the value third is the grad
    output_lines = output_str.strip().split('\n')
    my_impl_res = [[line.split()[0], *map(float, line.split()[1:])]
                   for line in output_lines]
    return my_impl_res


def add_val(val, name):
    """Add the val and grad of a torch.Tensor."""
    torch_res.append([name, val.item(), val.grad.item()])


def assert_with_mes(mine, torch):
    """Assert equality with a message."""
    if isinstance(mine, str) or isinstance(torch, str):
        try:
            assert mine == torch
            return 0
        except AssertionError:
            return 1
    else:
        try:
            assert round(mine, 6) == round(torch, 6)
            return 0
        except AssertionError:
            return 1


def check_res(my_res, torch_res, show=False):
    """Check if the two results are equal."""
    assert len(my_res) == len(torch_res), "Length is not equal"
    check = 0;
    for mres, tres in zip(my_res, torch_res):
        if assert_with_mes(mres[0], tres[0]):
            print_diff(mres, tres, "names")
            check = 1

        if assert_with_mes(mres[1], tres[1]):
            print_diff(mres, tres, "values")
            check = 1


        if assert_with_mes(mres[2], tres[2]):
            print_diff(mres, tres, "grad")
            check = 1

        if show and not check:
            print(f"Clear: name: {mres[0]} val: {mres[1]} grad: {mres[2]}")
            print(f"Torch: name: {tres[0]} val: {tres[1]} grad: {tres[2]}")
    return check


def test_against_cn(code, show=False):
    """Do the test."""
    my_impl_res = get_res(code)
    check = check_res(my_impl_res, torch_res, show)
    if not check:
        print_pass(code)


def print_diff(mres, tres, name):
    """Print what is different and relevant all variable info."""
    print(f"Different {name}")
    print(f"Mine:  name: {mres[0]} | val: {mres[1]} | grad: {mres[2]}")
    print(f"Torch: name: {tres[0]} | val: {tres[1]} | grad: {tres[2]}")


def print_pass(code):
    """Print the test that was passed."""
    print(f"Passed test: {code}")


def do_test_one():
    """Test general things one."""
    global torch_res
    torch_res = []

    a = torch.Tensor([-2.0])
    a.requires_grad = True
    b = torch.Tensor([3.0])
    b.requires_grad = True
    c = a * b
    d = a + b
    e = c * d
    f = a - e
    g = torch.tanh(f)
    c.retain_grad()
    d.retain_grad()
    e.retain_grad()
    f.retain_grad()
    g.retain_grad()
    g.backward()

    add_val(a, 'a')
    add_val(b, 'b')
    add_val(c, 'c')
    add_val(d, 'd')
    add_val(e, 'e')
    add_val(f, 'f')
    add_val(g, 'g')

    test_against_cn("1")


def do_test_two():
    """Test general things two."""
    global torch_res
    torch_res = []

    a = torch.Tensor([-4.0])
    a.requires_grad = True
    b = torch.Tensor([2.0])
    b.requires_grad = True
    c = a + b
    c.retain_grad()
    d = (a * b) + b
    c += c + 1  # -3
    c += 1 + c + (-a)  # -3+=2, -1
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e.relu()
    c.retain_grad()
    d.retain_grad()
    e.retain_grad()
    f.retain_grad()
    f.backward()
    add_val(a, 'a')
    add_val(b, 'b')
    add_val(c, 'c')
    add_val(d, 'd')
    add_val(e, 'e')
    add_val(f, 'f')
    test_against_cn("2")


def do_test_pow():
    """Test raising things to a power."""
    global torch_res
    torch_res = []
    a = torch.Tensor([5.0])
    a.requires_grad = True
    b = torch.Tensor([10.0])
    b.requires_grad = True
    c = a**b
    c = c**2
    c.retain_grad()
    c.backward()
    add_val(a, 'a')
    add_val(b, 'b')
    add_val(c, 'c')
    test_against_cn("pow")


def do_test_on_itself():
    """Test adding variables to themselves."""
    global torch_res
    torch_res = []
    a = torch.Tensor([3.0])
    a.requires_grad = True
    b = torch.Tensor([7.0])
    b.requires_grad = True
    c = a + b
    c.retain_grad()
    c += 2
    c += 2
    c *= a
    c -= b
    d = 5
    d -= c
    d += d
    d.retain_grad()
    d.backward()
    add_val(a, 'a')
    add_val(b, 'b')
    add_val(c, 'c')
    add_val(d, 'd')
    test_against_cn("on_itself")


def do_test_tanh():
    """Test tanh function."""
    global torch_res
    torch_res = []

    x1 = torch.Tensor([2.0])
    x1.requires_grad = True
    x2 = torch.Tensor([-.0])
    x2.requires_grad = True

    w1 = torch.Tensor([-3.0])
    w1.requires_grad = True
    w2 = torch.Tensor([1.0])
    w2.requires_grad = True

    b = torch.Tensor([7.0])
    b.requires_grad = True

    n = x1*w1 + x2*w2 + b

    o = torch.tanh(n)
    o.backward()
    add_val(x1, "x1")
    add_val(w1, "w1")
    add_val(x2, "x2")
    add_val(w2, "w2")
    test_against_cn("tanh")


def do_test_relu():
    """Test relu function."""
    global torch_res
    torch_res = []

    a = torch.Tensor([10.0])
    a.requires_grad = True
    b = torch.Tensor([5.0])
    b.requires_grad = True
    c = a * b
    d = torch.relu(c)
    c.retain_grad()
    d.retain_grad()
    d.backward()
    add_val(a, 'a')
    add_val(b, 'b')
    add_val(c, 'c')
    add_val(d, 'd')

    test_against_cn("relu")


def do_test_sigmoid():
    """Test sigmoid function."""
    global torch_res
    torch_res = []

    a = torch.Tensor([0.3])
    a.requires_grad = True
    b = torch.Tensor([0.5])
    b.requires_grad = True
    c = torch.Tensor([-1.0])
    c.requires_grad = True
    d = (a + b) * c
    e = d * a
    f = torch.sigmoid(e)
    d.retain_grad()
    e.retain_grad()
    f.retain_grad()
    f.backward()
    add_val(a, 'a')
    add_val(b, 'b')
    add_val(c, 'c')
    add_val(d, 'd')
    add_val(e, 'e')
    add_val(f, 'f')

    test_against_cn("sigmoid")


def do_test_leaky_relu():
    """Test against clear net leaky relu."""
    global torch_res
    torch_res = []

    leaky_relu = torch.nn.LeakyReLU(0.1)

    a = torch.Tensor([72])
    a.requires_grad = True
    b = torch.Tensor([38])
    b.requires_grad = True
    c = torch.Tensor([-10.0])
    c.requires_grad = True
    d = (a + b) * c
    e = d * a
    f = leaky_relu(e)
    d.retain_grad()
    e.retain_grad()
    f.retain_grad()
    f.backward()
    add_val(a, 'a')
    add_val(b, 'b')
    add_val(c, 'c')
    add_val(d, 'd')
    add_val(e, 'e')
    add_val(f, 'f')

    test_against_cn("leaky_relu")


do_test_one()
do_test_two()
do_test_pow()
do_test_on_itself()
do_test_tanh()
do_test_relu()
do_test_sigmoid()
do_test_leaky_relu()
