#include "../lib/clear_net.h"
#include <stdio.h>

#define la cn.la

// a full adder with carry in and carry out
int main(void) {
    srand(0);
    // clang-format off
    float data[] = {
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
    la.shuffleVanillaInput(&input, &target);
    ulong num_epochs = 20000;

    HParams hp = cn.defaultHParams();

    cn.setLeaker(&hp, 1);
    Net *net = cn.allocVanillaNet(hp, 3);
    // cn_with_momentum(0.9);
    cn.allocDenseLayer(net, Tanh, 3);
    cn.allocDenseLayer(net, LeakyReLU, 8);
    cn.allocDenseLayer(net, Sigmoid, num_outputs);
    cn.randomizeNet(net, -1, 1);
    cn.printNet(net, "net");
    float loss;
    for (ulong i = 0; i < num_epochs; ++i) {
        loss = cn.learnVanilla(net, input, target);
        if (i % (num_epochs / 10) == 0) {
            printf("Average loss: %g\n", loss);
        }
    }
    printf("Final loss: %g\n", loss);

    Vector out_store = la.allocVector(target.ncols);
    for (ulong i = 0; i < input.nrows; ++i) {
        Vector in = la.formVector(input.ncols, &MAT_AT(input, i, 0));
        la.printVector(&in, "in");
        cn.predictDense(net, in, &out_store);
        la.printVector(&out_store, "out");
    }

    cn.deallocNet(net);
    la.deallocVector(&out_store);

    return 0;
}
