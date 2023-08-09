"""Testing the autodifferentiation engine."""
import subprocess
import os
import numpy as np
import scipy as sp


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
    matrix = np.array(elements).reshape(rows, cols)
    return matrix


def compare_and_print(cn_mat, sp_mat, name, atol=1e-7):
    """Compare the two matrices and print results."""
    elementwise_comparison = np.isclose(cn_mat, sp_mat, atol=atol)
    if np.all(elementwise_comparison):
        print(f"Pass: {name}")
    else:
        print(f"Fail: {name}")
        print("Differences:")
        diff_indices = np.where(~elementwise_comparison)
        for i, j in zip(*diff_indices):
            sp_val = sp_mat[i, j]
            cn_val = cn_mat[i, j]
            print(
                f"Different at ({i}, {j}): scipy: {sp_val}, cn: {cn_val}")


def test_same_zeros():
    """Test same correlation with result being all zeros."""
    cn_res = get_res("same_zeros")
    input_matrix = np.zeros((15, 15))

    poss_idx = 0
    for i in range(15):
        for j in range(15):
            input_matrix[i, j] = poss_values[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_values)
    kernel = np.zeros((3, 3))
    output_matrix = sp.signal.correlate2d(input_matrix, kernel, mode='same')
    compare_and_print(cn_res, output_matrix, "same zeros")


def test_same_identity():
    """Test same correlation with identity kernel."""
    cn_res = get_res("same_identity")
    dim = 10
    input_matrix = np.zeros((dim, dim))

    poss_idx = 0
    for i in range(dim):
        for j in range(dim):
            input_matrix[i, j] = poss_values[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_values)
    kernel = np.zeros((3, 3))
    kernel[1][1] = 1
    output_matrix = sp.signal.correlate2d(input_matrix, kernel, mode='same')
    compare_and_print(cn_res, output_matrix, "same identity")


def test_same_guassian_blur_3x3():
    """Test same correlation with guassian blur filter."""
    cn_res = get_res("same_guassian_blur_3")
    dim = 20
    input_matrix = np.zeros((dim, dim))

    poss_idx = 0
    for i in range(dim):
        for j in range(dim):
            input_matrix[i, j] = poss_values[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_values)
    kernel = np.zeros((3, 3))
    kernel[0][0] = 1 / 16
    kernel[0][1] = 2 / 16
    kernel[0][2] = 1 / 16
    kernel[1][0] = 2 / 16
    kernel[1][1] = 4 / 16
    kernel[1][2] = 2 / 16
    kernel[2][0] = 1 / 16
    kernel[2][1] = 2 / 16
    kernel[2][2] = 1 / 16
    output_matrix = sp.signal.correlate2d(input_matrix, kernel, mode='same')
    compare_and_print(cn_res, output_matrix, "same guassian 3x3")


def test_same_guassian_blur_5x5():
    """Test same correlation with guassian blur filter."""
    cn_res = get_res("same_guassian_blur_5")
    dim = 20
    input_matrix = np.zeros((dim, dim))

    poss_idx = 0
    for i in range(dim):
        for j in range(dim):
            input_matrix[i, j] = poss_values[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_values)
    k_size = 5
    kernel = np.zeros((k_size, k_size))
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
    output_matrix = sp.signal.correlate2d(input_matrix, kernel, mode='same')
    compare_and_print(cn_res, output_matrix, "same guassian 5x5", 1e-3)


def do_test_with_default_elements(code, input_dim, krows, kcols, mode,
                                  print_name):
    """Do the test on the given example."""
    cn_res = get_res(code)
    input_matrix = np.zeros((input_dim, input_dim))
    poss_idx = 0
    for i in range(input_dim):
        for j in range(input_dim):
            input_matrix[i, j] = poss_values[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_values)
    kernel = np.zeros((krows, kcols))
    poss_idx = 0
    for i in range(krows):
        for j in range(kcols):
            kernel[i][j] = poss_kernel_elements[poss_idx]
            poss_idx = (poss_idx + 1) % len(poss_kernel_elements)

    output_matrix = sp.signal.correlate2d(input_matrix, kernel, mode=mode)
    compare_and_print(cn_res, output_matrix, print_name)


# tests with predefined elements
test_same_zeros()
test_same_identity()
test_same_guassian_blur_3x3()
test_same_guassian_blur_5x5()

# tests with elements from a known list
do_test_with_default_elements("same_even_kernel", 20, 4, 4, 'same',
                              "same even kernel")
do_test_with_default_elements("same_rect", 30, 5, 3, 'same', "same rect")
do_test_with_default_elements("full_7x7", 30, 7, 7, 'full', "full 7x7")
do_test_with_default_elements("full_even", 15, 4, 4, 'full',
                              "full even kernel")
do_test_with_default_elements("full_rect", 30, 4, 7, 'full', "full rect")
do_test_with_default_elements("valid_7x7", 11, 7, 7, 'valid', "valid 7x7")
