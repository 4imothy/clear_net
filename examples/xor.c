#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"

int main() {
    srand(0);
    Matrix data = alloc_mat(4, 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            size_t row = i * 2 + j;
            MAT_GET(data, row, 0) = i;
            MAT_GET(data, row, 1) = j;
            MAT_GET(data, row, 2) = i ^ j;
        }
    }
    Matrix input = mat_form(data.nrows, 2, data.stride, &MAT_GET(data, 0, 0));
    Matrix output = mat_form(data.nrows, 1, data.stride, &MAT_GET(data, 0, input.ncols));

    size_t shape[] = {2, 2, 1};
    size_t layers = ARR_LEN(shape);
    Net net = alloc_net(shape, layers);
    // TODO try this with the 0,5 with the combatting to vanishing gradient
    net_rand(net, -1, 1);

    size_t num_epochs = 20000;
    float error;
	float error_break = 0.001;
    for (size_t i = 0; i < num_epochs; ++i) {
        net_backprop(net, input, output);
        error = net_errorf(net, input, output);
        if (i % (num_epochs / 5) == 0) {
            printf("Cost at %zu: %f\n", i, error);
        }
        if (error <= error_break) {
		  printf("Less than: %f error at epoch %zu\n", error_break, i);
		  break;
        }

    }
	net_print_results(net, input, output);
	
    dealloc_net(&net);
    return 0;
}
