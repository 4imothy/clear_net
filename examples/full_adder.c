#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"

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
    NetConfig hparams = cn_init_net_conf();
    cn_with_momentum(&hparams, 0.1);
    Net net = cn_init_net(hparams);
    cn_set_neg_scale(1);
    cn_alloc_dense_layer(&net, num_inputs, 3, Tanh);
    cn_alloc_dense_layer(&net, 3, 8, LeakyReLU);
    cn_alloc_dense_layer(&net, 8, num_outputs, Sigmoid);
    cn_randomize_net(net, -1, 1);
    cn_print_net(net, "net");
    float loss;
    for (size_t i = 0; i < num_epochs; ++i) {
        loss = cn_learn(&net, input, target);
        if (i % (num_epochs / 10) == 0) {
            printf("Average loss: %g\n", loss);
        }
    }
    printf("Final loss: %g\n", loss);
    cn_print_net_results(net, input, target);
    char *name = "model";
    cn_save_net_to_file(net, name);
    cn_dealloc_net(&net);
    net = cn_alloc_net_from_file(name);
    printf("After loading file\n");
    cn_print_net_results(net, input, target);
    cn_dealloc_net(&net);

    return 0;
}
