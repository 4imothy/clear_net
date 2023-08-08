"""Testing the autodifferentiation engine."""
import subprocess
import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")


cwd = os.getcwd()
my_impl = "./convolution"
poss_values = [0.10290608318034533, 0.8051580508692876,
               0.39055048005351034, 0.7739175926400883,
               0.24730207704015073, 0.7987075645399935,
               0.24602568871407338, 0.6268407447350659,
               0.4646505260697441, 0.20524882983167547,
               0.5031590491750169, 0.2550151936024112,
               0.3354895253780905, 0.6825483746871936,
               0.6204572461588524, 0.6487941004544666,
               0.742795723261874, 0.8436721618301802,
               0.0433154872324607, 0.42621935359557017]
poss_kernel_elements = [2, 6, 7, 8, 4, 0, 6, 4, 2, 0, 9, 7,
                        5, 9, 8, 8, 4, 6, 0, 2, 4, 7, 6, 1,
                        7, 5, 2, 9, 6, 7, 8]


def get_res(code):
    """Get the result from running my implementation."""
    output_bytes = subprocess.check_output(my_impl + f" {code}",
                                           shell=True, cwd=cwd)
    output_str = output_bytes.decode('utf-8')
    parts = output_str.split()
    elements = []
    for part in parts:
        elements.append(float(part))
    rows, cols = map(int, elements[:2])
    elements = elements[2:]
    data = torch.tensor(elements, dtype=torch.float32)
    matrix = data.view(rows, cols)
    return matrix


def compare_and_print(cn_mat, torch_mat, name, atol=1e-7):
    """Compare the two matrices and print results."""
    elementwise_comparison = np.isclose(cn_mat, torch_mat, atol=atol)
    if np.all(elementwise_comparison):
        print(f"Pass: {name}")
    else:
        print(f"Fail: {name}")
        print("Differences:")
        diff_indices = np.where(~elementwise_comparison)
        for i, j in zip(*diff_indices):
            torch_val = torch_mat[i, j]
            cn_val = cn_mat[i, j]
            print(
                f"Different at ({i}, {j}): torch: {torch_val}, cn: {cn_val}")


def test_same_zeroes():
    """Test same correlation with result being all zeroes."""
    cn_res = get_res("same_zeroes")
    input_matrix = torch.zeros(15, 15)

    poss_idx = 0
    for i in range(15):
        for j in range(15):
            input_matrix[i, j] = poss_values[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_values)
    kernel = torch.zeros(3, 3)
    output_matrix = torch.nn.functional.conv2d(input_matrix.unsqueeze(
        0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding='same')
    output_matrix = output_matrix.squeeze()
    compare_and_print(cn_res, output_matrix, "same zeroes")


def test_same_identity():
    """Test same correlation with identity kernel."""
    cn_res = get_res("same_identity")
    dim = 10
    input_matrix = torch.zeros(dim, dim)

    poss_idx = 0
    for i in range(dim):
        for j in range(dim):
            input_matrix[i, j] = poss_values[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_values)
    kernel = torch.zeros(3, 3)
    kernel[1][1] = 1
    output_matrix = torch.nn.functional.conv2d(input_matrix.unsqueeze(
        0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding='same')
    output_matrix = output_matrix.squeeze()
    compare_and_print(cn_res, output_matrix, "same identity")


def test_same_guassian_blur_3x3():
    """Test same correlation with guassian blur filter."""
    cn_res = get_res("same_guassian_blur_3")
    dim = 20
    input_matrix = torch.zeros(dim, dim)

    poss_idx = 0
    for i in range(dim):
        for j in range(dim):
            input_matrix[i, j] = poss_values[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_values)
    kernel = torch.zeros(3, 3)
    kernel[0][0] = 1 / 16
    kernel[0][1] = 2 / 16
    kernel[0][2] = 1 / 16
    kernel[1][0] = 2 / 16
    kernel[1][1] = 4 / 16
    kernel[1][2] = 2 / 16
    kernel[2][0] = 1 / 16
    kernel[2][1] = 2 / 16
    kernel[2][2] = 1 / 16
    output_matrix = torch.nn.functional.conv2d(input_matrix.unsqueeze(
        0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding='same')
    output_matrix = output_matrix.squeeze()
    compare_and_print(cn_res, output_matrix, "same guassian 3x3")


def test_same_guassian_blur_5x5():
    """Test same correlation with guassian blur filter."""
    cn_res = get_res("same_guassian_blur_5")
    dim = 20
    input_matrix = torch.zeros(dim, dim)

    poss_idx = 0
    for i in range(dim):
        for j in range(dim):
            input_matrix[i, j] = poss_values[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_values)
    k_size = 5
    kernel = torch.zeros(k_size, k_size)
    kernel[0][0] = 1
    kernel[0][1] = 4
    kernel[0][2] = 7
    kernel[0][3] = 4
    kernel[0][4] = 1
    kernel[1][0] = 4
    kernel[1][1] = 16
    kernel[1][2] = 26
    kernel[1][3] = 16
    kernel[1][4] = 4
    kernel[2][0] = 7
    kernel[2][1] = 26
    kernel[2][2] = 41
    kernel[2][3] = 26
    kernel[2][4] = 7
    kernel[3][0] = 4
    kernel[3][1] = 16
    kernel[3][2] = 26
    kernel[3][3] = 16
    kernel[3][4] = 4
    kernel[4][0] = 1
    kernel[4][1] = 4
    kernel[4][2] = 7
    kernel[4][3] = 4
    kernel[4][4] = 1

    kernel = kernel / kernel.sum()  # Normalize the kernel
    output_matrix = torch.nn.functional.conv2d(input_matrix.unsqueeze(
        0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding='same')
    output_matrix = output_matrix.squeeze()
    compare_and_print(cn_res, output_matrix, "same guassian 5x5", 1e-3)


def test_same_even_dim_kernel():
    """Test same correlation with an even kernel."""
    cn_res = get_res("same_even_kernel")
    dim = 20
    input_matrix = torch.zeros(dim, dim)
    poss_idx = 0
    for i in range(dim):
        for j in range(dim):
            input_matrix[i, j] = poss_values[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_values)
    kernel = torch.zeros(4, 4)
    poss_idx = 0
    for i in range(4):
        for j in range(4):
            kernel[i][j] = poss_kernel_elements[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_kernel_elements)

    output_matrix = torch.nn.functional.conv2d(input_matrix.unsqueeze(
        0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding='same')
    output_matrix = output_matrix.squeeze()
    compare_and_print(cn_res, output_matrix, "same even kernel")


test_same_zeroes()
test_same_identity()
test_same_guassian_blur_3x3()
test_same_guassian_blur_5x5()
test_same_even_dim_kernel()
