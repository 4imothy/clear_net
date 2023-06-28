#include <stdio.h>
#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"

#define BITS_PER_NUM 2

// a full adder with carry in and carry out
int main() {
    size_t num_values = (1 << BITS_PER_NUM);
    size_t num_combinations = num_values * num_values * num_values;
    // A, B, and sum have BITS_PER_NUM, cin and cout are each one bit
    size_t num_cols = BITS_PER_NUM * 3 + 2;
    Matrix data = alloc_mat(num_combinations, num_cols);
    Matrix input = {
        .elements = &MAT_GET(data, 0, 0),
        .nrows = data.nrows,
        // 2 input numbers and one cin bit
        .ncols = 2 * BITS_PER_NUM + 1,
        .stride = data.stride,
    };
    Matrix output = {
        .elements = &MAT_GET(data, 0, BITS_PER_NUM + 1),
        .nrows = data.nrows,
        // output number and the cout
        .ncols = BITS_PER_NUM + 1,
        .stride = data.stride,
    };

    for (size_t i = 0; i < num_combinations; ++i) {
        size_t a = i / num_values;
        size_t b = i % num_values;
        // used in cout of output
        size_t res = a + b;
        size_t cin = i % num_values;

        for (size_t j = 0; j < BITS_PER_NUM; ++j) {
            MAT_GET(input, i, j) = (a >> j) & 1;
            MAT_GET(input, i, j + BITS_PER_NUM) = (b >> j) & 1;
            MAT_GET(input, i, j) = (cin >> j) & 1;
        }
    }

    for (size_t i = 0; i < num_combinations; ++i) {
        size_t a = i / num_values;
        size_t b = i % num_values;
        size_t cin;
        if (i > num_combinations / 2 - 1) {
            cin = 1;
        } else {
            cin = 0;
        }
        for (size_t k = 0; k < BITS_PER_NUM; ++k) {
            MAT_GET(input, i, k) = (a >> k) & 1;
            MAT_GET(input, i, k + BITS_PER_NUM) = (b >> k) & 1;
            MAT_GET(input, i, k + 2 * BITS_PER_NUM) = cin;
        }
    }
    MAT_PRINT(input);

    return 0;
}
