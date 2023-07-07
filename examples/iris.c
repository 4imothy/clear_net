#include <stdio.h>
#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"
#include <stdbool.h>
#include <string.h>

// clang-format off
// sepal length (cm), sepal width (cm), petal length (cm), petal width (cm), target
float test_values[] = {
    5.1, 3.5, 1.4, 0.2, 0, 
    4.9, 3.0, 1.4, 0.2, 0, 
    4.7, 3.2, 1.3, 0.2, 0, 
    4.6, 3.1, 1.5, 0.2, 0, 
    5.0, 3.6, 1.4, 0.2, 0, 
    5.4, 3.9, 1.7, 0.4, 0, 
    4.6, 3.4, 1.4, 0.3, 0, 
    5.0, 3.4, 1.5, 0.2, 0, 
    4.4, 2.9, 1.4, 0.2, 0, 
    4.9, 3.1, 1.5, 0.1, 0, 
    5.4, 3.7, 1.5, 0.2, 0, 
    4.8, 3.4, 1.6, 0.2, 0, 
    4.8, 3.0, 1.4, 0.1, 0, 
    4.3, 3.0, 1.1, 0.1, 0, 
    5.8, 4.0, 1.2, 0.2, 0, 
    5.7, 4.4, 1.5, 0.4, 0, 
    5.4, 3.9, 1.3, 0.4, 0, 
    5.1, 3.5, 1.4, 0.3, 0, 
    5.7, 3.8, 1.7, 0.3, 0, 
    5.1, 3.8, 1.5, 0.3, 0, 
    5.4, 3.4, 1.7, 0.2, 0, 
    5.1, 3.7, 1.5, 0.4, 0, 
    4.6, 3.6, 1.0, 0.2, 0, 
    5.1, 3.3, 1.7, 0.5, 0, 
    4.8, 3.4, 1.9, 0.2, 0, 
    5.0, 3.0, 1.6, 0.2, 0, 
    5.0, 3.4, 1.6, 0.4, 0, 
    5.2, 3.5, 1.5, 0.2, 0, 
    5.2, 3.4, 1.4, 0.2, 0, 
    4.7, 3.2, 1.6, 0.2, 0, 
    4.8, 3.1, 1.6, 0.2, 0, 
    5.4, 3.4, 1.5, 0.4, 0, 
    5.2, 4.1, 1.5, 0.1, 0, 
    5.5, 4.2, 1.4, 0.2, 0, 
    4.9, 3.1, 1.5, 0.1, 0, 
    5.0, 3.2, 1.2, 0.2, 0, 
    5.5, 3.5, 1.3, 0.2, 0, 
    4.9, 3.1, 1.5, 0.1, 0, 
    4.4, 3.0, 1.3, 0.2, 0, 
    5.1, 3.4, 1.5, 0.2, 0, 
    5.0, 3.5, 1.3, 0.3, 0, 
    4.5, 2.3, 1.3, 0.3, 0, 
    4.4, 3.2, 1.3, 0.2, 0, 
    5.0, 3.5, 1.6, 0.6, 0, 
    5.1, 3.8, 1.9, 0.4, 0, 
    7.0, 3.2, 4.7, 1.4, 1, 
    6.4, 3.2, 4.5, 1.5, 1, 
    6.9, 3.1, 4.9, 1.5, 1, 
    5.5, 2.3, 4.0, 1.3, 1, 
    6.5, 2.8, 4.6, 1.5, 1, 
    5.7, 2.8, 4.5, 1.3, 1, 
    6.3, 3.3, 4.7, 1.6, 1, 
    4.9, 2.4, 3.3, 1.0, 1, 
    6.6, 2.9, 4.6, 1.3, 1, 
    5.2, 2.7, 3.9, 1.4, 1, 
    5.0, 2.0, 3.5, 1.0, 1, 
    5.9, 3.0, 4.2, 1.5, 1, 
    6.0, 2.2, 4.0, 1.0, 1, 
    6.1, 2.9, 4.7, 1.4, 1, 
    5.6, 2.9, 3.6, 1.3, 1, 
    6.7, 3.1, 4.4, 1.4, 1, 
    5.6, 3.0, 4.5, 1.5, 1, 
    5.8, 2.7, 4.1, 1.0, 1, 
    6.2, 2.2, 4.5, 1.5, 1, 
    5.6, 2.5, 3.9, 1.1, 1, 
    5.9, 3.2, 4.8, 1.8, 1, 
    6.1, 2.8, 4.0, 1.3, 1, 
    6.3, 2.5, 4.9, 1.5, 1, 
    6.1, 2.8, 4.7, 1.2, 1, 
    6.4, 2.9, 4.3, 1.3, 1, 
    6.6, 3.0, 4.4, 1.4, 1, 
    6.8, 2.8, 4.8, 1.4, 1, 
    6.7, 3.0, 5.0, 1.7, 1, 
    6.0, 2.9, 4.5, 1.5, 1, 
    5.7, 2.6, 3.5, 1.0, 1, 
    5.5, 2.4, 3.8, 1.1, 1, 
    5.5, 2.4, 3.7, 1.0, 1, 
    5.8, 2.7, 3.9, 1.2, 1, 
    6.0, 2.7, 5.1, 1.6, 1, 
    5.4, 3.0, 4.5, 1.5, 1, 
    6.0, 3.4, 4.5, 1.6, 1, 
    6.7, 3.1, 4.7, 1.5, 1, 
    6.3, 2.3, 4.4, 1.3, 1, 
    5.6, 3.0, 4.1, 1.3, 1, 
    5.5, 2.5, 4.0, 1.3, 1, 
    5.5, 2.6, 4.4, 1.2, 1, 
    6.1, 3.0, 4.6, 1.4, 1, 
    5.8, 2.6, 4.0, 1.2, 1, 
    5.0, 2.3, 3.3, 1.0, 1, 
    5.6, 2.7, 4.2, 1.3, 1, 
    6.3, 3.3, 6.0, 2.5, 2, 
    5.8, 2.7, 5.1, 1.9, 2, 
    7.1, 3.0, 5.9, 2.1, 2, 
    6.3, 2.9, 5.6, 1.8, 2, 
    6.5, 3.0, 5.8, 2.2, 2, 
    7.6, 3.0, 6.6, 2.1, 2, 
    4.9, 2.5, 4.5, 1.7, 2, 
    7.3, 2.9, 6.3, 1.8, 2, 
    6.7, 2.5, 5.8, 1.8, 2, 
    7.2, 3.6, 6.1, 2.5, 2, 
    6.5, 3.2, 5.1, 2.0, 2, 
    6.4, 2.7, 5.3, 1.9, 2, 
    6.8, 3.0, 5.5, 2.1, 2, 
    5.7, 2.5, 5.0, 2.0, 2, 
    5.8, 2.8, 5.1, 2.4, 2, 
    6.4, 3.2, 5.3, 2.3, 2, 
    6.5, 3.0, 5.5, 1.8, 2, 
    7.7, 3.8, 6.7, 2.2, 2, 
    7.7, 2.6, 6.9, 2.3, 2, 
    6.0, 2.2, 5.0, 1.5, 2, 
    6.9, 3.2, 5.7, 2.3, 2, 
    5.6, 2.8, 4.9, 2.0, 2, 
    7.7, 2.8, 6.7, 2.0, 2, 
    6.3, 2.7, 4.9, 1.8, 2, 
    6.7, 3.3, 5.7, 2.1, 2, 
    7.2, 3.2, 6.0, 1.8, 2, 
    6.2, 2.8, 4.8, 1.8, 2, 
    6.1, 3.0, 4.9, 1.8, 2, 
    6.4, 2.8, 5.6, 2.1, 2, 
    7.2, 3.0, 5.8, 1.6, 2, 
    7.4, 2.8, 6.1, 1.9, 2, 
    7.9, 3.8, 6.4, 2.0, 2, 
    6.4, 2.8, 5.6, 2.2, 2, 
    6.3, 2.8, 5.1, 1.5, 2, 
    6.1, 2.6, 5.6, 1.4, 2, 
    7.7, 3.0, 6.1, 2.3, 2, 
    6.3, 3.4, 5.6, 2.4, 2, 
    6.4, 3.1, 5.5, 1.8, 2, 
    6.0, 3.0, 4.8, 1.8, 2, 
    6.9, 3.1, 5.4, 2.1, 2, 
    6.7, 3.1, 5.6, 2.4, 2, 
    6.9, 3.1, 5.1, 2.3, 2, 
    5.8, 2.7, 5.1, 1.9, 2, 
    6.8, 3.2, 5.9, 2.3, 2, 
    6.7, 3.3, 5.7, 2.5, 2, 
};
// clang-format on
float validation_values[] = {
    4.8, 3.0, 1.4, 0.3, 0, 5.1, 3.8, 1.6, 0.2, 0, 4.6, 3.2, 1.4, 0.2, 0,
    5.3, 3.7, 1.5, 0.2, 0, 5.0, 3.3, 1.4, 0.2, 0, 5.7, 3.0, 4.2, 1.2, 1,
    5.7, 2.9, 4.2, 1.3, 1, 6.2, 2.9, 4.3, 1.3, 1, 5.1, 2.5, 3.0, 1.1, 1,
    5.7, 2.8, 4.1, 1.3, 1, 6.7, 3.0, 5.2, 2.3, 2, 6.3, 2.5, 5.0, 1.9, 2,
    6.5, 3.0, 5.2, 2.0, 2, 6.2, 3.4, 5.4, 2.3, 2, 5.9, 3.0, 5.1, 1.8, 2};

float fix_output(float out) { return roundf(out * 2); }

int main(int argc, char *argv[]) {
    bool print = true;

    // Loop through command-line arguments
    for (int i = 1; i < argc; i++) {
        // Check if the argument is the "-b" flag
        if (strcmp(argv[i], "-b") == 0) {
            print = false;
            break;
        }
    }

    srand(0);
    size_t data_cols = 5;
    size_t input_dim = 4;
    size_t output_dim = 1;
    size_t val_size = 15;
    size_t test_size = 150 - val_size;
    Matrix test = mat_form(test_size, data_cols, data_cols, test_values);
    Matrix input =
        mat_form(test_size, input_dim, test.stride, &MAT_GET(test, 0, 0));
    Matrix target = mat_form(test_size, output_dim, test.stride,
                             &MAT_GET(test, 0, input_dim));
    for (size_t i = 0; i < target.nrows; ++i) {
        for (size_t j = 0; j < target.ncols; ++j) {
            MAT_GET(target, i, j) = MAT_GET(target, i, j) / 2;
        }
    }
    Matrix val = mat_form(val_size, data_cols, data_cols, validation_values);
    size_t shape[] = {input_dim, output_dim};
    Net net = alloc_net(shape, ARR_LEN(shape));
    net_rand(net, -1, 1);
    size_t num_epochs = 10000;
    float error;
    float error_break = 0.01;
    for (size_t i = 0; i < num_epochs; ++i) {
        net_backprop(net, input, target);
        error = net_errorf(net, input, target);
        if (i % (num_epochs / 5) == 0) {
            if (print) {
                printf("Cost at %zu is %f\n", i, error);
            }
        }
        if (error < error_break) {
            if (print) {
                printf("Less than %f error at %zu\n", error_break, i);
            }
            break;
        }
    }
    if (print) {
        net_print_results(net, input, target, &fix_output);
        Matrix val_input =
            mat_form(val_size, input_dim, val.stride, &MAT_GET(val, 0, 0));
        Matrix val_target = mat_form(val_size, output_dim, val.stride,
                                     &MAT_GET(val, 0, input_dim));
        for (size_t i = 0; i < val_target.nrows; ++i) {
            for (size_t j = 0; j < val_target.ncols; ++j) {
                MAT_GET(val_target, i, j) = MAT_GET(val_target, i, j) / 2;
            }
        }
        net_print_results(net, val_input, val_target, &fix_output);
        net_save_to_file("model", net);
        dealloc_net(&net);
        net = alloc_net_from_file("model");
        printf("After loading file\n");
        net_print_results(net, val_input, val_target, &fix_output);
    }
    return 0;
}
