#include "../lib/clear_net.h"
#include <stdio.h>

#define data cn.data

// a full adder with carry in and carry out
int main(void) {
    srand(0);
    // clang-format off
    scalar all[] = {
        // a  b  cin  sum  cout
        0,  0,  0,   0,   0,
        0,  0,  1,   1,   0,
        0,  1,  0,   1,   0,
        0,  1,  1,   0,   1,
        1,  0,  0,   1,   0,
        1,  0,  1,   0,   1,
        1,  1,  0,   0,   1,
        1,  1,  1,   1,   1,
    };
    // clang-format on

    // 2^3
    ulong num_combinations = 8;
    // a, b, cin
    ulong num_inputs = 3;
    // sum, cout
    ulong num_outputs = 2;
    ulong stride = 5;
    Vector *inputs = data.allocVectors(num_combinations, num_inputs);
    Vector *targets = data.allocVectors(num_combinations, num_outputs);

    for (ulong i = 0; i < num_combinations; ++i) {
        inputs[i] = data.formVector(inputs->nelem, &all[i * stride]);
        targets[i] =
            data.formVector(targets->nelem, &all[(i * stride) + num_inputs]);
    }

    CNData *io_inputs = data.allocDataFromVectors(inputs, num_combinations);
    CNData *io_targets = data.allocDataFromVectors(targets, num_combinations);

    ulong num_epochs = 20000;

    HParams *hp = cn.allocDefaultHParams();

    cn.setLeaker(hp, 1);
    cn.withMomentum(hp, 0.9);
    Net *net = cn.allocVanillaNet(hp, 3);
    cn.allocDenseLayer(net, TANH, 3);
    cn.allocDenseLayer(net, LEAKYRELU, 8);
    cn.allocDenseLayer(net, SIGMOID, num_outputs);
    cn.randomizeNet(net, -1, 1);
    cn.printNet(net, "net");
    scalar loss;
    for (ulong i = 0; i < num_epochs; ++i) {
        loss = cn.lossVanilla(net, io_inputs, io_targets);
        if (i % (num_epochs / 10) == 0) {
            printf("Average loss: %g\n", loss);
        }
        cn.backprop(net);
    }
    printf("Final loss: %g\n", loss);
    cn.printVanillaPredictions(net, io_inputs, io_targets);
    char *file = "model";
    cn.saveNet(net, file);
    cn.deallocNet(net);
    net = cn.allocNetFromFile(file);
    printf("after loading\n");
    cn.printVanillaPredictions(net, io_inputs, io_targets);
    cn.deallocNet(net);
    return 0;
}
