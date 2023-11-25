"""Testing the autodifferentiation engine."""
import os
import numpy as np
import scipy as sp
from tests import (
    input_elem,
    matrix_elem,
    get_mat_res,
    compare_matrices
)


script_path = os.path.abspath(__file__)
script_name = os.path.splitext(os.path.basename(script_path))[0]
my_impl = "./" + script_name


def test_same_zeros():
    """Test same correlation with result being all zeros."""
    cn_res = get_mat_res(my_impl, "same_zeros")
    input_matrix = np.zeros((15, 15))

    poss_idx = 0
    for i in range(15):
        for j in range(15):
            input_matrix[i, j] = input_elem[poss_idx]
            poss_idx = (poss_idx + 1) % len(input_elem)
    kernel = np.zeros((3, 3))
    output_matrix = sp.signal.correlate2d(input_matrix, kernel, mode='same')
    compare_matrices(cn_res, output_matrix, "same zeros")


def test_same_identity():
    """Test same correlation with identity kernel."""
    cn_res = get_mat_res(my_impl, "same_identity")
    dim = 10
    input_matrix = np.zeros((dim, dim))

    poss_idx = 0
    for i in range(dim):
        for j in range(dim):
            input_matrix[i, j] = input_elem[poss_idx]
            poss_idx = (poss_idx + 1) % len(input_elem)
    kernel = np.zeros((3, 3))
    kernel[1][1] = 1
    output_matrix = sp.signal.correlate2d(input_matrix, kernel, mode='same')
    compare_matrices(cn_res, output_matrix, "same identity")


def test_same_guassian_blur_3x3():
    """Test same correlation with guassian blur filter."""
    cn_res = get_mat_res(my_impl, "same_guassian_blur_3")
    dim = 20
    input_matrix = np.zeros((dim, dim))

    poss_idx = 0
    for i in range(dim):
        for j in range(dim):
            input_matrix[i, j] = input_elem[poss_idx]
            poss_idx = (poss_idx + 1) % len(input_elem)
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
    compare_matrices(cn_res, output_matrix, "same guassian 3x3")


def test_same_guassian_blur_5x5():
    """Test same correlation with guassian blur filter."""
    cn_res = get_mat_res(my_impl, "same_guassian_blur_5")
    dim = 20
    input_matrix = np.zeros((dim, dim))

    poss_idx = 0
    for i in range(dim):
        for j in range(dim):
            input_matrix[i, j] = input_elem[poss_idx]
            poss_idx = (poss_idx + 1) % len(input_elem)
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
    compare_matrices(cn_res, output_matrix, "same guassian 5x5", 1e-3)


def do_test_with_default_elements(code, input_rows, input_cols, krows, kcols,
                                  mode, print_name):
    """Do the test on the given example."""
    cn_res = get_mat_res(my_impl, code)
    input_matrix = np.zeros((input_rows, input_cols))
    poss_idx = 0
    for i in range(input_rows):
        for j in range(input_cols):
            input_matrix[i, j] = input_elem[poss_idx]
            poss_idx = (poss_idx + 1) % len(input_elem)
    kernel = np.zeros((krows, kcols))
    poss_idx = 0
    for i in range(krows):
        for j in range(kcols):
            kernel[i][j] = matrix_elem[poss_idx]
            poss_idx = (poss_idx + 1) % len(matrix_elem)

    output_matrix = sp.signal.correlate2d(input_matrix, kernel, mode=mode)
    compare_matrices(cn_res, output_matrix, print_name)


# tests with predefined elements
test_same_zeros()
test_same_identity()
test_same_guassian_blur_3x3()
test_same_guassian_blur_5x5()

# tests with elements from a known list
do_test_with_default_elements("same_even_kernel", 20, 20, 4, 4, 'same',
                              "same even kernel")
do_test_with_default_elements("same_rect", 30, 30, 5, 3, 'same', "same rect")
do_test_with_default_elements("full_7x7", 30, 30, 7, 7, 'full', "full 7x7")
do_test_with_default_elements("full_even", 15, 15, 4, 4, 'full',
                              "full even kernel")
do_test_with_default_elements("full_rect", 30, 30, 4, 7, 'full', "full rect")
do_test_with_default_elements("valid_7x7", 11, 11, 7, 7, 'valid', "valid 7x7")
do_test_with_default_elements(
    "valid_rect", 23, 23, 1, 6, 'valid', "valid rect")
do_test_with_default_elements("valid_rect_input", 10, 20, 4, 4, 'valid',
                              "valid rectangular input")
