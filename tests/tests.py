import subprocess
import numpy as np
import os

cwd = os.getcwd()

input_elem = [0.10290608318034533, 0.8051580508692876,
              0.39055048005351034, 0.7739175926400883,
              0.24730207704015073, 0.7987075645399935,
              0.24602568871407338, 0.6268407447350659,
              0.4646505260697441, 0.20524882983167547,
              0.5031590491750169, 0.2550151936024112,
              0.3354895253780905, 0.6825483746871936,
              0.6204572461588524, 0.6487941004544666,
              0.742795723261874, 0.8436721618301802,
              0.0433154872324607, 0.42621935359557017]
matrix_elem = [2, 6, 7, 8, 4, 0, 6, 4, 2, 0, 9, 7,
               5, 9, 8, 8, 4, 6, 0, 2, 4, 7, 6, 1,
               7, 5, 2, 9, 6, 7, 8]


def get_mat_res(my_impl, code):
    """Get the result from running my implementation."""
    output_bytes = subprocess.check_output(my_impl + f" {code}",
                                           shell=True, cwd=cwd)
    output_str = output_bytes.decode('utf-8')
    elements = [float(part) for part in output_str.split()]
    rows, cols = map(int, elements[:2])
    elements = elements[2:]
    matrix = np.array(elements).reshape(rows, cols)
    return matrix


def get_vec_res(my_impl, code):
    """Get the result from running my implementation for vectors."""
    output_bytes = subprocess.check_output(my_impl + f" {code}",
                                           shell=True, cwd=cwd)
    output_str = output_bytes.decode('utf-8')
    elements = [float(part) for part in output_str.split()]

    vector = np.array(elements[1:])

    return vector


def compare_matrices(cn_mat, sp_mat, name, atol=1e-7):
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


def compare_vectors(cn_vec, sp_vec, name, atol=1e-7):
    """Compare the two matrices and print results."""
    elementwise_comparison = np.isclose(cn_vec, sp_vec, atol=atol)
    if np.all(elementwise_comparison):
        print(f"Pass: {name}")
    else:
        print(f"Fail: {name}")
        print("Differences:")
        diff_indices = np.where(~elementwise_comparison)
        for i in zip(*diff_indices):
            sp_val = sp_vec[i]
            cn_val = cn_vec[i]
            print(
                f"Different at ({i}): scipy: {sp_val}, cn: {cn_val}")
