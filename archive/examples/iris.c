#include <stdio.h>
#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"
#include <stdbool.h>
#include <string.h>

CLEAR_NET_DEFINE_HYPERPARAMETERS

// sepal length (cm), sepal width (cm), petal length (cm), petal width (cm),
// target
// clang-format off
float train_values[] = {
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

float validation_values[] = {
    4.8, 3.0, 1.4, 0.3, 0,
    5.1, 3.8, 1.6, 0.2, 0,
    4.6, 3.2, 1.4, 0.2, 0,
    5.3, 3.7, 1.5, 0.2, 0,
    5.0, 3.3, 1.4, 0.2, 0,
    5.7, 3.0, 4.2, 1.2, 1,
    5.7, 2.9, 4.2, 1.3, 1,
    6.2, 2.9, 4.3, 1.3, 1,
    5.1, 2.5, 3.0, 1.1, 1,
    5.7, 2.8, 4.1, 1.3, 1,
    6.7, 3.0, 5.2, 2.3, 2,
    6.3, 2.5, 5.0, 1.9, 2,
    6.5, 3.0, 5.2, 2.0, 2,
    6.2, 3.4, 5.4, 2.3, 2,
    5.9, 3.0, 5.1, 1.8, 2
};
// clang-format on

int main(int argc, char *argv[]) {
    bool print = true;

    // Loop through command-line arguments
    if (argc > 1 && strcmp(argv[1], "-b") == 0) {
        print = false;
    }

    srand(0);
    size_t data_cols = 5;
    size_t input_dim = 4;
    size_t output_dim = data_cols - input_dim;
    size_t val_size = 15;
    size_t train_size = 150 - val_size;
    Matrix train =
        cn_form_matrix(train_size, data_cols, data_cols, train_values);
    Matrix input = cn_form_matrix(train_size, input_dim, train.stride,
                                  &MAT_AT(train, 0, 0));
    Matrix target = cn_form_matrix(train_size, output_dim, train.stride,
                                   &MAT_AT(train, 0, input_dim));
    for (size_t i = 0; i < target.nrows; ++i) {
        MAT_AT(target, i, 0) /= 2;
    }

    Matrix val =
        cn_form_matrix(val_size, data_cols, data_cols, validation_values);
    Matrix val_input =
        cn_form_matrix(val_size, input_dim, val.stride, &MAT_AT(val, 0, 0));
    Matrix val_target = cn_form_matrix(val_size, output_dim, val.stride,
                                       &MAT_AT(val, 0, input_dim));
    for (size_t i = 0; i < val_size; ++i) {
        MAT_AT(val_target, i, 0) /= 2;
    }
    cn_default_hparams();
    cn_set_rate(0.02);
    cn_with_momentum(0.9);
    Net net = cn_alloc_vani_net(input_dim);
    cn_alloc_dense_layer(&net, Sigmoid, 1);
    cn_randomize_net(&net, -1, 1);
    size_t num_epochs = 10000;
    float loss;
    float error_break = 0.01;
    size_t i;
    size_t batch_size = 45;
    cn_shuffle_vani_input(&input, &target);
    Matrix batch_in;
    Matrix batch_tar;
    CLEAR_NET_ASSERT(train_size % batch_size == 0);
    for (i = 0; i < num_epochs; ++i) {
        for (size_t batch_num = 0; batch_num < train_size / batch_size;
             ++batch_num) {
            cn_get_batch_vani(&batch_in, &batch_tar, input, target, batch_num,
                              batch_size);
            cn_learn_vani(&net, batch_in, batch_tar);
        }
        loss = cn_loss_vani(&net, input, target);
        if (loss < error_break) {
            break;
        }
        if (i % (num_epochs / 5) == 0 && print) {
            printf("Average loss: %f\n", loss);
        }
    }
    if (print) {
        printf("Final loss at %zu : %g\n", i, loss);
        cn_print_vani_results(net, input, target);
        char *file = "model";
        cn_save_net_to_file(net, file);
        cn_dealloc_net(&net);
        printf("After loading from file\n");
        net = cn_alloc_net_from_file(file);
        cn_print_vani_results(net, input, target);
        printf("On validation set\n");
        cn_print_vani_results(net, val_input, val_target);
        cn_dealloc_net(&net);
    }
    return 0;
}
