#define CLEAR_NET_ACT_HIDDEN Sigmoid
#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"

int main(void) {
    srand(0);
    // size_t shape[] = {2, 8, 7, 5, 1};
    // size_t shape[] = {2, 25, 50, 1};
    size_t shape[] = {2, 2, 1};
    size_t nlayers = sizeof((shape)) / sizeof((*shape));
    Net net = cn_alloc_net(shape, nlayers);
    cn_randomize_net(net, -1, 1);
    Matrix data = cn_alloc_matrix(4, 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            size_t row = i * 2 + j;
            MAT_AT(data, row, 0) = i;
            MAT_AT(data, row, 1) = j;
            MAT_AT(data, row, 2) = i ^ j;
        }
    }
    Matrix input = cn_form_matrix(data.nrows, 2, data.stride, &MAT_AT(data, 0, 0));
    Matrix target = cn_form_matrix(data.nrows, 1, data.stride,
                                &MAT_AT(data, 0, data.ncols - 1));
    float loss;
    size_t num_epochs = 10000;
    for (size_t i = 0; i < num_epochs; ++i) {
        loss = cn_learn(&net, input, target);
        if (i % 100 == 0) {
            printf("Average loss: %g\n", loss);
        }
    }
    printf("Final loss: %g\n", loss);
    cn_print_net_results(net, input, target);
    char* file = "model";
    cn_save_net_to_file(net, file);
    cn_dealloc_net(&net, 0);
    net = cn_alloc_net_from_file(file);
    cn_print_net_results(net, input, target);
    cn_dealloc_net(&net, 1);
    cn_dealloc_matrix(&data);
}
