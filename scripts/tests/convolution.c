#define CLEAR_NET_IMPLEMENTATION
#include "../../clear_net.h"
#include <string.h>

void print_results(Matrix mat) {
    printf("%zu %zu", mat.nrows, mat.ncols);
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            printf(" %f ", MAT_AT(mat, i, j));
        }
    }
}

const float poss_elements[] = {
    0.10290608318034533, 0.8051580508692876,  0.39055048005351034,
    0.7739175926400883,  0.24730207704015073, 0.7987075645399935,
    0.24602568871407338, 0.6268407447350659,  0.4646505260697441,
    0.20524882983167547, 0.5031590491750169,  0.2550151936024112,
    0.3354895253780905,  0.6825483746871936,  0.6204572461588524,
    0.6487941004544666,  0.742795723261874,   0.8436721618301802,
    0.0433154872324607,  0.42621935359557017};
const size_t poss_elements_len = 20;

const float poss_kernel_elements[] = {2, 6, 7, 8, 4, 0, 6, 4, 2, 0, 9,
                                      7, 5, 9, 8, 8, 4, 6, 0, 2, 4, 7,
                                      6, 1, 7, 5, 2, 9, 6, 7, 8};
const size_t poss_kernel_elem_len =
    sizeof(poss_kernel_elements) / sizeof(*poss_kernel_elements);

void fill_matrix(Matrix *mat, const float *elements, size_t elem_len) {
    size_t poss_id = 0;
    for (size_t i = 0; i < mat->nrows; ++i) {
        for (size_t j = 0; j < mat->ncols; ++j) {
            MAT_AT(*mat, i, j) = elements[poss_id];
            poss_id = (poss_id + 1) % elem_len;
        }
    }
}

void do_test_with_default_elements(Net *net, Padding padding, Activation act, const size_t input_rows, const size_t input_cols, size_t krows, size_t kcols) {
    cn_alloc_conv_layer(net, padding, act, 1, 1, input_rows, input_cols, krows, kcols);
    fill_matrix(&net->layers[0].data.conv.filters[0].kernels[0], poss_kernel_elements,
                    poss_kernel_elem_len);
    float input[input_rows * input_cols];
    Matrix minput = cn_form_matrix(input_rows, input_cols, input_cols, input);
    fill_matrix(&minput, poss_elements, poss_elements_len);
    cn_forward_conv(&net->layers[0].data.conv, &minput);
    print_results(net->layers[0].data.conv.outputs[0]);
}

int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    // Make it so the activation does nothing
    cn_default_hparams();
    Net net = cn_init_net();
    Activation default_act = LeakyReLU;
    cn_set_neg_scale(1);
    srand(0);
    if (strcmp(argv[1], "same_zeros") == 0) {
        const size_t dim = 15;
        cn_alloc_conv_layer(&net, Same, default_act, 1, 1, dim, dim, 3, 3);
        float elem[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        net.layers[0].data.conv.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        cn_forward_conv(&net.layers[0].data.conv, &minput);
        print_results(net.layers[0].data.conv.outputs[0]);
    }

    else if (strcmp(argv[1], "same_identity") == 0) {
        const size_t dim = 10;
        cn_alloc_conv_layer(&net, Same, default_act, 1, 1, dim, dim, 3, 3);
        float elem[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
        net.layers[0].data.conv.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        cn_forward_conv(&net.layers[0].data.conv, &minput);
        print_results(net.layers[0].data.conv.outputs[0]);
    }

    else if (strcmp(argv[1], "same_guassian_blur_3") == 0) {
        const size_t dim = 20;
        cn_alloc_conv_layer(&net, Same, default_act, 1, 1, dim, dim, 3, 3);
        float elem[] = {0.0625, 0.1250, 0.0625, 0.1250, 0.25,
                        0.1250, 0.0625, 0.1250, 0.0625};
        net.layers[0].data.conv.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        cn_forward_conv(&net.layers[0].data.conv, &minput);
        print_results(net.layers[0].data.conv.outputs[0]);
    }

    else if (strcmp(argv[1], "same_guassian_blur_5") == 0) {
        const size_t dim = 20;
        cn_alloc_conv_layer(&net, Same, default_act, 1, 1, dim, dim, 5, 5);
        float elem[] = {0.0037, 0.0147, 0.0256, 0.0147, 0.0037, 0.0147, 0.0586,
                        0.0952, 0.0586, 0.0147, 0.0256, 0.0952, 0.1502, 0.0952,
                        0.0256, 0.0147, 0.0586, 0.0952, 0.0586, 0.0147, 0.0037,
                        0.0147, 0.0256, 0.0147, 0.0037};
        net.layers[0].data.conv.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        cn_forward_conv(&net.layers[0].data.conv, &minput);
        print_results(net.layers[0].data.conv.outputs[0]);
    }

    else if (strcmp(argv[1], "same_even_kernel") == 0) {
        do_test_with_default_elements(&net, Same, default_act, 20, 20, 4, 4);
    }

    else if (strcmp(argv[1], "same_rect") == 0) {
        do_test_with_default_elements(&net, Same, default_act, 30, 30, 5, 3);
    }

    else if (strcmp(argv[1], "full_7x7") == 0) {
        do_test_with_default_elements(&net, Full, default_act, 30, 30, 7, 7);
    }

    else if (strcmp(argv[1], "full_even") == 0) {
        do_test_with_default_elements(&net, Full, default_act, 15, 15, 4, 4);
    }

    else if (strcmp(argv[1], "full_rect") == 0) {
        do_test_with_default_elements(&net, Full, default_act, 30, 30, 4, 7);
    }

    else if (strcmp(argv[1], "valid_7x7") == 0) {
        do_test_with_default_elements(&net, Valid, default_act, 11, 11, 7, 7);
    }

    else if (strcmp(argv[1], "valid_rect") == 0) {
        do_test_with_default_elements(&net, Valid, default_act, 23, 23, 1, 6);
    }

    else if (strcmp(argv[1], "valid_rect_input") == 0) {
        do_test_with_default_elements(&net, Valid, default_act, 10, 20, 4, 4);
    }
}
