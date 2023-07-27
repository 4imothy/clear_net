#define CLEAR_NET_IMPLEMENTATION
#include "../../clear_net.h"
#include <stdio.h>
#include <time.h>

#define WARM_UP_TIME 1
#define NUM_ITERS 100
#define ISPOW2(V_) (ceil(log2(V_)) == floor(log2(V_)))
typedef void (*MulFunc)(Matrix, Matrix, Matrix, size_t);

void naive(Matrix dest, Matrix left, Matrix right, size_t pointless) {
    // ensure that the inner dimensions are equal
    CLEAR_NET_ASSERT(left.ncols == right.nrows);
    size_t inner = left.ncols;
    // assert that destination has the outer dimensions
    CLEAR_NET_ASSERT(dest.nrows == left.nrows);
    CLEAR_NET_ASSERT(dest.ncols == right.ncols);
    // iterate over outer size
    for (size_t i = 0; i < left.nrows; ++i) {
        for (size_t j = 0; j < right.ncols; ++j) {
            MAT_GET(dest, i, j) = 0;
            // iterater over the inner size
            for (size_t k = 0; k < inner; ++k) {
                MAT_GET(dest, i, j) +=
                    MAT_GET(left, i, k) * MAT_GET(right, k, j);
            }
        }
    }
}

void better_cache_use(Matrix dest, Matrix left, Matrix right,
                      size_t pointless) {
    CLEAR_NET_ASSERT(left.ncols == right.nrows);
    size_t n = left.ncols;
    CLEAR_NET_ASSERT(dest.nrows == left.nrows);
    CLEAR_NET_ASSERT(dest.ncols == right.ncols);
    mat_fill(dest, 0);

    for (size_t i = 0; i < dest.nrows; ++i) {
        for (size_t k = 0; k < n; ++k) {
            for (size_t j = 0; j < dest.ncols; ++j) {
                MAT_GET(dest, i, j) +=
                    MAT_GET(left, i, k) * MAT_GET(right, k, j);
            }
        }
    }
}

void bench(Matrix dest, Matrix left, Matrix right, MulFunc func, char *name) {
    double start = (double)clock() / CLOCKS_PER_SEC;
    double end = start;

    printf("Benching %s...\n", name);
    printf("Warming up for %d seconds...\n", WARM_UP_TIME);
    size_t len = left.nrows;
    while (end - start < WARM_UP_TIME) {
        func(dest, left, right, len);
        end = (double)clock() / CLOCKS_PER_SEC;
    }

    double total_time = 0;

    for (size_t i = 0; i < NUM_ITERS; ++i) {
        start = (double)clock() / CLOCKS_PER_SEC;
        func(dest, left, right, len);
        end = (double)clock() / CLOCKS_PER_SEC;
        total_time += end - start;
    }
    printf("on average the %s solution took: %f seconds\n", name,
           total_time / (double)NUM_ITERS);
}

Matrix matrix_addition(Matrix dest, Matrix a, Matrix b) {
    CLEAR_NET_ASSERT(dest.nrows == a.nrows);
    CLEAR_NET_ASSERT(dest.ncols == a.ncols);
    CLEAR_NET_ASSERT(dest.nrows == b.nrows);
    CLEAR_NET_ASSERT(dest.ncols == b.ncols);

    for (size_t i = 0; i < dest.nrows; ++i) {
        for (size_t j = 0; j < dest.ncols; ++j) {
            MAT_GET(dest, i, j) = MAT_GET(a, i, j) + MAT_GET(b, i, j);
        }
    }

    return dest;
}

Matrix matrix_subtraction(Matrix dest, Matrix a, Matrix b) {
    CLEAR_NET_ASSERT(dest.nrows == a.nrows);
    CLEAR_NET_ASSERT(dest.ncols == a.ncols);
    CLEAR_NET_ASSERT(dest.nrows == b.nrows);
    CLEAR_NET_ASSERT(dest.ncols == b.ncols);

    for (size_t i = 0; i < dest.nrows; ++i) {
        for (size_t j = 0; j < dest.ncols; ++j) {
            MAT_GET(dest, i, j) = MAT_GET(a, i, j) - MAT_GET(b, i, j);
        }
    }

    return dest;
}

void strassen(Matrix dest, Matrix srcA, Matrix srcB, size_t length) {
    if (length == 2) {
        better_cache_use(dest, srcA, srcB, 0);
        return;
    }

    size_t len = length / 2;

    Matrix a11 = alloc_mat(len, len), a12 = alloc_mat(len, len),
           a21 = alloc_mat(len, len), a22 = alloc_mat(len, len),
           b11 = alloc_mat(len, len), b12 = alloc_mat(len, len),
           b21 = alloc_mat(len, len), b22 = alloc_mat(len, len),
           c11 = alloc_mat(len, len), c12 = alloc_mat(len, len),
           c21 = alloc_mat(len, len), c22 = alloc_mat(len, len),
           m1 = alloc_mat(len, len), m2 = alloc_mat(len, len),
           m3 = alloc_mat(len, len), m4 = alloc_mat(len, len),
           m5 = alloc_mat(len, len), m6 = alloc_mat(len, len),
           m7 = alloc_mat(len, len), temp1 = alloc_mat(len, len),
           temp2 = alloc_mat(len, len);

    /* Divide matrix into four parts */
    for (size_t i = 0; i < len; ++i) {
        for (size_t j = 0; j < len; ++j) {
            MAT_GET(a11, i, j) = MAT_GET(srcA, i, j);
            MAT_GET(a12, i, j) = MAT_GET(srcA, i, j + len);
            MAT_GET(a21, i, j) = MAT_GET(srcA, i + len, j);
            MAT_GET(a22, i, j) = MAT_GET(srcA, i + len, j + len);

            MAT_GET(b11, i, j) = MAT_GET(srcB, i, j);
            MAT_GET(b12, i, j) = MAT_GET(srcB, i, j + len);
            MAT_GET(b21, i, j) = MAT_GET(srcB, i + len, j);
            MAT_GET(b22, i, j) = MAT_GET(srcB, i + len, j + len);
        }
    }

    /* Calculate seven formulas of strassen Algorithm */
    strassen(m1, matrix_addition(temp1, a11, a22),
             matrix_addition(temp2, b11, b22), len);
    strassen(m2, matrix_addition(temp1, a21, a22), b11, len);
    strassen(m3, a11, matrix_subtraction(temp1, b12, b22), len);
    strassen(m4, a22, matrix_subtraction(temp1, b21, b11), len);
    strassen(m5, matrix_addition(temp1, a11, a12), b22, len);
    strassen(m6, matrix_subtraction(temp1, a21, a11),
             matrix_addition(temp2, b11, b12), len);
    strassen(m7, matrix_subtraction(temp1, a12, a22),
             matrix_addition(temp2, b21, b22), len);

    /* Merge the answer of matrix dest */
    /* c11 = m1 + m4 - m5 + m7 = m1 + m4 - (m5 - m7) */
    matrix_subtraction(c11, matrix_addition(temp1, m1, m4),
                       matrix_subtraction(temp2, m5, m7));
    matrix_addition(c12, m3, m5);
    matrix_addition(c21, m2, m4);
    matrix_addition(c22, matrix_subtraction(temp1, m1, m2),
                    matrix_addition(temp2, m3, m6));

    /* Store the answer of matrix multiplication */
    for (size_t i = 0; i < len; ++i) {
        for (size_t j = 0; j < len; ++j) {
            MAT_GET(dest, i, j) = MAT_GET(c11, i, j);
            MAT_GET(dest, i, j + len) = MAT_GET(c12, i, j);
            MAT_GET(dest, i + len, j) = MAT_GET(c21, i, j);
            MAT_GET(dest, i + len, j + len) = MAT_GET(c22, i, j);
        }
    }
    dealloc_mat(&a11);
    dealloc_mat(&a12);
    dealloc_mat(&a21);
    dealloc_mat(&a22);

    dealloc_mat(&b11);
    dealloc_mat(&b12);
    dealloc_mat(&b21);
    dealloc_mat(&b22);

    dealloc_mat(&c11);
    dealloc_mat(&c12);
    dealloc_mat(&c21);
    dealloc_mat(&c22);

    dealloc_mat(&m1);
    dealloc_mat(&m2);
    dealloc_mat(&m3);
    dealloc_mat(&m4);
    dealloc_mat(&m5);
    dealloc_mat(&m6);
    dealloc_mat(&m7);
    dealloc_mat(&temp1);
    dealloc_mat(&temp2);
}

int main(void) {
    srand(0);
    size_t louter = pow(2, 10);
    size_t router = pow(2, 8);
    size_t inner = pow(2, 5);

    size_t left_nrows = louter;
    size_t inner_dim = inner;
    size_t right_ncols = router;

    Matrix left = alloc_mat(left_nrows, inner_dim);
    Matrix right = alloc_mat(inner_dim, right_ncols);
    Matrix dest = alloc_mat(left_nrows, right_ncols);
    mat_rand(left, -20, 20);
    mat_rand(right, -20, 20);

    printf("%zu x %zu * %zu x %zu\n", left_nrows, inner_dim, inner_dim,
           right_ncols);
    bench(dest, left, right, naive, "naive");
    bench(dest, left, right, better_cache_use, "better cache use");
    bench(dest, left, right, strassen, "strassen");
}
