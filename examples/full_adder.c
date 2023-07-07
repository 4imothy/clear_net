#define CLEAR_NET_IMPLEMENTATION
#define CLEAR_NET_ACT_OUTPUT ELU
#define CLEAR_NET_ACT_HIDDEN ELU
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
    float error_break = 0.01f;
    float error;
    for (size_t i = 0; i < num_epochs; ++i) {
        net_backprop(net, input, target);
        error = net_errorf(net, input, target);
        if (i % (num_epochs / 5) == 0) {
            printf("Cost at %zu: %f\n", i, error);
        }
        if (error < error_break) {
            printf("Error less than %f at %zu\n", error_break, i);
            break;
        }
    }
    //	float do_nothing(float x){ return x;}
    net_print_results(net, input, target, &roundf);
    net_save_to_file("model", net);
    dealloc_net(&net);
    net = alloc_net_from_file("model");
    printf("After loading file\n");
    net_print_results(net, input, target, &roundf);
    dealloc_net(&net);

    return 0;
}
