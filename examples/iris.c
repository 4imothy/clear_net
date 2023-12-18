#include "../lib/clear_net.h"
#include <string.h>
#include <stdbool.h>
#include <stdio.h>

#define data cn.data

// sepal length (cm), sepal width (cm), petal length (cm), petal width (cm),
// target
// clang-format off

scalar train_values[] = {
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

scalar validation_values[] = {
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
    ulong data_cols = 5;
    ulong input_dim = 4;
    ulong output_dim = data_cols - input_dim;
    ulong val_size = 15;
    ulong train_size = 150 - val_size;
    Vector *train_in = data.allocVectors(train_size, input_dim);
    Vector *train_tar = data.allocVectors(train_size, output_dim);

    for (ulong i = 0; i < train_size; ++i) {
        train_in[i] = data.formVector(input_dim, &train_values[data_cols * i]);
        train_tar[i] = data.formVector(
            output_dim, &train_values[(data_cols * i) + input_dim]);
        VEC_AT(train_tar[i], 0) /= 2;
    }

    CNData *io_train_in = data.allocDataFromVectors(train_in, train_size);
    CNData *io_train_tar = data.allocDataFromVectors(train_tar, train_size);

    HParams *hp = cn.allocDefaultHParams();
    cn.setRate(hp, 0.02);
    cn.withMomentum(hp, 0.9);
    Net *net = cn.allocVanillaNet(hp, input_dim);
    cn.allocDenseLayer(net, SIGMOID, 1);
    cn.randomizeNet(net, -1, 1);

    ulong num_epochs = 1000;
    float loss;
    float error_break = 0.01;
    ulong batch_size = 45;
    ulong i;
    data.shuffleDatas(io_train_in, io_train_tar);
    CNData *batch_in = data.allocEmptyData();
    CNData *batch_tar = data.allocEmptyData();
    CLEAR_NET_ASSERT(train_size % batch_size == 0);

    for (i = 0; i < num_epochs; ++i) {
        for (ulong batch_num = 0; batch_num < train_size / batch_size;
             ++batch_num) {
            data.setBatch(io_train_in, io_train_tar, batch_num, batch_size,
                          batch_in, batch_tar);
            cn.lossVanilla(net, batch_in, batch_tar);
            cn.backprop(net);
        }
        loss = cn.lossVanilla(net, io_train_in, io_train_tar);
        if (loss < error_break) {
            printf("Loss < %f at epoch %zu\n", error_break, i);
            break;
        }
        if (i % (num_epochs / 5) == 0 && print) {
            printf("Average loss: %f\n", loss);
        }
    }

    cn.printVanillaPredictions(net, io_train_in, io_train_tar);

    char *file = "model";
    cn.saveNet(net, file);
    cn.deallocNet(net);
    net = cn.allocNetFromFile(file);

    Vector *val_in = data.allocVectors(val_size, input_dim);
    Vector *val_tar = data.allocVectors(val_size, output_dim);

    for (ulong i = 0; i < val_size; ++i) {
        val_in[i] =
            data.formVector(input_dim, &validation_values[data_cols * i]);
        val_tar[i] = data.formVector(
            output_dim, &validation_values[(data_cols * i) + input_dim]);
        VEC_AT(val_tar[i], 0) /= 2;
    }

    CNData *io_val_in = data.allocDataFromVectors(val_in, val_size);
    CNData *io_val_tar = data.allocDataFromVectors(val_tar, val_size);
    cn.printVanillaPredictions(net, io_val_in, io_val_tar);
    cn.deallocNet(net);
}
