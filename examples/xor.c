#include "../lib/clear_net.h"
#include <stdio.h>

#define la cn.la

int main(void) {
    srand(0);
    HParams hp = cn.defaultHParams();
    cn.setRate(&hp, 10);
    Net *net = cn.allocVanillaNet(hp, 2);
    cn.allocDenseLayer(net, Sigmoid, 2);
    cn.allocDenseLayer(net, Sigmoid, 1);
    cn.randomizeNet(net, -1, 1);
    cn.printNet(net, "net");
    Matrix data = la.allocMatrix(4, 3);
    for (ulong i = 0; i < 2; ++i) {
        for (ulong j = 0; j < 2; ++j) {
            ulong row = i * 2 + j;
            MAT_AT(data, row, 0) = i;
            MAT_AT(data, row, 1) = j;
            MAT_AT(data, row, 2) = i ^ j;
        }
    }

    la.printMatrix(&data, "data");

    Matrix input =
        la.formMatrix(data.nrows, 2, data.stride, &MAT_AT(data, 0, 0));
    Matrix target = la.formMatrix(data.nrows, 1, data.stride,
                                  &MAT_AT(data, 0, data.ncols - 1));
    scalar loss;
    ulong num_epochs = 10000;

    for (ulong i = 0; i < num_epochs; ++i) {
        loss = cn.learnVanilla(net, input, target);
        if (i % 100 == 0) {
            printf("Average loss: %f\n", loss);
        }
    }
    printf("Final loss: %g\n", loss);
    char* file = "model";
    cn.printVanillaPredictions(net, input, target);
    cn.saveNet(net, file);
    cn.deallocNet(net);
    printf("after loading\n");
    net = cn.allocNetFromFile("model");
    cn.printVanillaPredictions(net, input, target);
    la.deallocMatrix(&data);
    return 0;
}
