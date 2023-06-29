#define CLEAR_NET_IMPLEMENTATION
#include "../clear_net.h"

int main() {
    srand(0);
    Matrix t = alloc_mat(4, 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            size_t row = i * 2 + j;
            MAT_GET(t, row, 0) = i;
            MAT_GET(t, row, 1) = j;
            MAT_GET(t, row, 2) = i ^ j;
        }
    }
    Matrix input = mat_form(t.nrows, 2, t.stride, &MAT_GET(t, 0, 0));
    Matrix output = mat_form(t.nrows, 1, t.stride, &MAT_GET(t, 0, input.ncols));

    size_t shape[] = {2, 2, 1};
    size_t layers = ARR_LEN(shape);
    Net net = alloc_net(shape, layers);
    // TODO try this with the 0,5 with the combatting to vanishing gradient
    net_rand(net, -1, 1);

    size_t num_epochs = 20000;
    float error;
    for (size_t i = 0; i < num_epochs; ++i) {
        net_backprop(net, input, output);
        error = net_errorf(net, input, output);
        if (error < 0.001) {
            printf("Error bound met at: %zu", i);
            return 0;
        }
        if (i % (num_epochs / 5) == 0) {
            printf("Cost at %zu: %f\n", i, error);
        }
    }
    printf("Final Cost: %f\n", net_errorf(net, input, output));

    for (size_t i = 0; i < output.nrows; ++i) {
        Matrix in = mat_row(input, i);
        MAT_PRINT(in);
        mat_copy(NET_INPUT(net), in);
        net_forward(net);
        MAT_PRINT(mat_row(output, i));
        MAT_PRINT(NET_OUTPUT(net));
        printf("----------\n");
    }

    dealloc_net(&net);

    return 0;
}
