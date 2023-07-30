#define CLEAR_NET_IMPLEMENTATION
#define CLEAR_NET_ACT_HIDDEN Sigmoid
#include "../clear_net.h"
#define BITS_PER_NUM 1

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
    size_t num_combinations = 8;
    // a, b, cin
    size_t num_inputs = 3;
    // sum, cout
    size_t num_outputs = 2;
    size_t stride = 5;
    Matrix input = matrix_form(num_combinations, num_inputs, stride, data);
    Matrix target =
        matrix_form(num_combinations, num_outputs, stride, &data[num_inputs]);
    size_t num_epochs = 20000;
    size_t shape[] = {num_inputs, 3, 8, num_outputs};
    size_t nlayers = sizeof(shape) / sizeof(*shape);
    Net net = alloc_net(shape, nlayers);
    printf("%zu\n", net.nlayers);
    net_randomize(net, -1, 1);
    float error_break = 0.01f;
    float loss;
    for (size_t i = 0; i < num_epochs; ++i) {
        loss = net_learn(&net, input, target);
        printf("Average loss: %g\n", loss);
    }
    printf("Final loss: %g\n", loss);
    net_print_results(net, input, target);
    /* char *name = "model"; */
    /* net_save_to_file(name, net); */
    /* dealloc_net(&net, 0); */
    /* net = alloc_net_from_file(name); */
    /* printf("After loading file\n"); */
    /* net_print_results(net, input, target, &roundf); */
    /* dealloc_net(&net, 1); */

    return 0;
}
