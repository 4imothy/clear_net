#include "../lib/clear_net.h"
#include <stdio.h>

#define la cn.la

// a full adder with carry in and carry out
int main(void) {
    srand(0);
    // clang-format off
    scalar data[] = {
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

    // 2^3
    ulong num_combinations = 8;
    // a, b, cin
    ulong num_inputs = 3;
    // sum, cout
    ulong num_outputs = 2;
    ulong stride = 5;
    Matrix input = la.formMatrix(num_combinations, num_inputs, stride, data);
    Matrix target =
        la.formMatrix(num_combinations, num_outputs, stride, &data[num_inputs]);
    la.shuffleMatrixRows(&input, &target);
    ulong num_epochs = 20000;

    HParams *hp = cn.allocDefaultHParams();

    cn.setLeaker(hp, 1);
    cn.withMomentum(hp, 0.9);
    Net *net = cn.allocVanillaNet(hp, 3);
    cn.allocDenseLayer(net, Tanh, 3);
    cn.allocDenseLayer(net, LeakyReLU, 8);
    cn.allocDenseLayer(net, Sigmoid, num_outputs);
    cn.randomizeNet(net, -1, 1);
    cn.printNet(net, "net");
    scalar loss;
    for (ulong i = 0; i < num_epochs; ++i) {
        loss = cn.lossVanilla(net, input, target);
        if (i % (num_epochs / 10) == 0) {
            printf("Average loss: %g\n", loss);
        }
        cn.backprop(net);
    }
    printf("Final loss: %g\n", loss);
    cn.printVanillaPredictions(net, input, target);
    char *file = "model";
    cn.saveNet(net, file);
    cn.deallocNet(net);
    net = cn.allocNetFromFile(file);
    printf("after loading\n");
    cn.printVanillaPredictions(net, input, target);
    cn.deallocNet(net);
    return 0;
}
