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
    Matrix input = cn_form_matrix(num_combinations, num_inputs, stride, data);
    Matrix target =
        cn_form_matrix(num_combinations, num_outputs, stride, &data[num_inputs]);
    size_t num_epochs = 20000;
    size_t shape[] = {num_inputs, 3, 8, num_outputs};
    size_t nlayers = sizeof(shape) / sizeof(*shape);
    Net net = cn_alloc_net(shape, nlayers);
    cn_randomize_net(net, -1, 1);
    cn_print_net(net, "before");
    float loss;
    for (size_t i = 0; i < num_epochs; ++i) {
        loss = cn_learn(&net, input, target);
        printf("Average loss: %g\n", loss);
    }
    printf("Final loss: %g\n", loss);
    cn_print_net(net, "final");
    cn_print_net_results(net, input, target);
    char *name = "model";
    cn_save_net_to_file(net, name);
    cn_dealloc_net(&net, 0);
    net = cn_alloc_net_from_file(name);
    printf("After loading file\n");
    cn_print_net_results(net, input, target);
    cn_dealloc_net(&net, 1);

    return 0;
}
