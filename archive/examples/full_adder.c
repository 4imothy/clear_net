#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"

CLEAR_NET_DEFINE_HYPERPARAMETERS

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
    cn_default_hparams();
    Net net = cn_alloc_vani_net(3);
    cn_with_momentum(0.9);
    cn_set_neg_scale(1);
    cn_alloc_dense_layer(&net, Tanh, 3);
    cn_alloc_dense_layer(&net, LeakyReLU, 8);
    cn_alloc_dense_layer(&net, Sigmoid, num_outputs);
    cn_randomize_net(&net, -1, 1);
    cn_print_net(net, "net");
    float loss;
    for (size_t i = 0; i < num_epochs; ++i) {
        loss = cn_learn_vani(&net, input, target);
        if (i % (num_epochs / 10) == 0) {
            printf("Average loss: %g\n", loss);
        }
    }
    printf("Final loss: %g\n", loss);
    cn_print_vani_results(net, input, target);
    char *name = "model";
    cn_save_net_to_file(net, name);
    cn_dealloc_net(&net);
    net = cn_alloc_net_from_file(name);
    printf("After loading file\n");
    cn_print_vani_results(net, input, target);
    cn_dealloc_net(&net);

    return 0;
}
