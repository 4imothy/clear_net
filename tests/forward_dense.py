import os
import numpy as np
from tests import (
    input_elem,
    matrix_elem,
    get_vec_res,
    compare_vectors
)


script_path = os.path.abspath(__file__)
script_name = os.path.splitext(os.path.basename(script_path))[0]
my_impl = "./" + script_name


def do_test_with_default(code, in_dim, out_dim, print_name):
    cn_res = get_vec_res(my_impl, code)

    input_vec = np.zeros(in_dim)
    poss_idx = 0
    for i in range(in_dim):
        input_vec[i] = input_elem[poss_idx]
        poss_idx = (poss_idx + 1) % len(input_elem)

    weights = np.zeros((in_dim, out_dim))
    poss_idx = 0
    for i in range(in_dim):
        for j in range(out_dim):
            weights[i][j] = matrix_elem[poss_idx]
            poss_idx = (poss_idx + 1) % len(matrix_elem)

    output_vec = np.dot(input_vec, weights)
    compare_vectors(cn_res, output_vec, print_name)


def do_test(code, in_dim, out_dim, print_name, input_elem, mat_elem):
    cn_res = get_vec_res(my_impl, code)

    input_vec = np.zeros(in_dim)
    poss_idx = 0
    for i in range(in_dim):
        input_vec[i] = input_elem[poss_idx]
        poss_idx = (poss_idx + 1) % len(input_elem)

    weights = np.zeros((in_dim, out_dim))
    poss_idx = 0
    for i in range(in_dim):
        for j in range(out_dim):
            weights[i, j] = matrix_elem[poss_idx]
            poss_idx = (poss_idx + 1) % len(matrix_elem)

    output_vec = np.dot(input_vec, weights)
    compare_vectors(cn_res, output_vec, print_name)


zeros = [0, 0, 0, 0, 0, 0]
do_test("same_zeros", 15, 15, "same zeros", zeros, matrix_elem)
do_test("up_zeros", 10, 15, "up zeros", zeros, matrix_elem)
do_test("down_zeros", 15, 7, "down zeros", zeros, matrix_elem)
do_test_with_default("10_10", 10, 10, "10 to 10")
do_test_with_default("1000_1000", 1000, 1000, "1000 to 1000")
do_test_with_default("1_5", 1, 5, "1 to 5")
do_test_with_default("5_1", 5, 1, "5 to 1")
do_test_with_default("15_40", 15, 40, "15 to 40")
do_test_with_default("40_15", 40, 15, "40 to 15")
