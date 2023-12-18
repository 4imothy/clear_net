#include "../lib/clear_net.h"
#include <stdio.h>

#define data cn.data

int main(void) {
    srand(0);
    ulong num = 4;
    Vector *input = data.allocVectors(num, 2);
    Vector *target = data.allocVectors(num, 1);
    for (ulong i = 0; i < 2; ++i) {
        for (ulong j = 0; j < 2; ++j) {
            ulong row = i * 2 + j;
            VEC_AT(input[row], 0) = i;
            VEC_AT(input[row], 1) = j;
            VEC_AT(target[row], 0) = i ^ j;
        }
    }

    HParams *hp = cn.allocDefaultHParams();
    cn.setRate(hp, 10);
    Net *net = cn.allocVanillaNet(hp, 2);
    cn.allocDenseLayer(net, SIGMOID, 2);
    cn.allocDenseLayer(net, SIGMOID, 1);
    cn.randomizeNet(net, -1, 1);
    cn.printNet(net, "before");

    scalar loss;
    ulong num_epochs = 10000;

    CNData *io_input = data.allocDataFromVectors(input, num);
    CNData *io_target = data.allocDataFromVectors(target, num);

    data.shuffleDatas(io_input, io_target);

    for (ulong i = 0; i < num_epochs; ++i) {
        loss = cn.lossVanilla(net, io_input, io_target);
        cn.backprop(net);
    }
    printf("Final loss: %f\n", loss);
    cn.printNet(net, "trained");
    char *file = "model";
    cn.printVanillaPredictions(net, io_input, io_target);
    cn.saveNet(net, file);
    cn.deallocNet(net);
    printf("after loading\n");
    net = cn.allocNetFromFile(file);
    cn.printVanillaPredictions(net, io_input, io_target);
    data.deallocData(io_input);
    data.deallocData(io_target);
    cn.deallocNet(net);
    return 0;
}
