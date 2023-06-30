#include <stdio.h>
#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"

#define BITS_PER_NUM 1

// a full adder with carry in and carry out
int main() {
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
    // clang-format on

    // 2^3
    size_t num_combinations = 8;
    // a, b, cin
    size_t num_inputs = 3;
    // sum, cout
    size_t num_outputs = 2;
    Matrix input = mat_form(num_combinations, num_inputs, 5, data);
    Matrix target =
        mat_form(num_combinations, num_outputs, 5, &data[num_inputs]);

    size_t num_epochs = 20000;
    size_t shape[] = {num_inputs, 3, 8, num_outputs};
    Net net = alloc_net(shape, ARR_LEN(shape));
    net_rand(net, -1, 1);
    float error_break = 0.01;
    float error;
    for (size_t i = 0; i < num_epochs; ++i) {
        error = net_errorf(net, input, target);
        net_backprop(net, input, target);
        if (i % (num_epochs / 5) == 0) {
            printf("Cost at %zu: %f\n", i, error);
        }
		if (error < error_break) {
		  printf("Error less than %f at %zu\n", error_break, i);
		  break;
		}
    }
    net_print_results(net, input, target);
    return 0;
}
