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

void do_test_with_default_elements(Padding padding, Activation act, const size_t input_rows, const size_t input_cols, size_t krows, size_t kcols) {
    ConvolutionalLayer cl = cn_init_convolutional_layer(padding, act, input_rows, input_cols, krows, kcols);
    cn_alloc_filter(&cl, 1);
    fill_matrix(&cl.filters[0].kernels[0], poss_kernel_elements,
                    poss_kernel_elem_len);
    float input[input_rows * input_cols];
    Matrix minput = cn_form_matrix(input_rows, input_cols, input_cols, input);
    fill_matrix(&minput, poss_elements, poss_elements_len);
    cn_correlate_layer(&cl, &minput, 1);
    print_results(cl.outputs[0]);
}

int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    srand(0);
    if (strcmp(argv[1], "same_zeros") == 0) {
        const size_t dim = 15;
        ConvolutionalLayer cl = cn_init_convolutional_layer(Same, Sigmoid, 15, 15, 3, 3);
        cn_alloc_filter(&cl, 1);
        float elem[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        cl.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        cn_correlate_layer(&cl, &minput, 1);
        print_results(cl.outputs[0]);
    }

    else if (strcmp(argv[1], "same_identity") == 0) {
        const size_t dim = 10;
        ConvolutionalLayer cl = cn_init_convolutional_layer(Same, Sigmoid, 10, 10, 3, 3);
        cn_alloc_filter(&cl, 1);
        float elem[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
        cl.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        cn_correlate_layer(&cl, &minput, 1);
        print_results(cl.outputs[0]);
    }

    else if (strcmp(argv[1], "same_guassian_blur_3") == 0) {
        const size_t dim = 20;
        ConvolutionalLayer cl = cn_init_convolutional_layer(Same, Sigmoid, dim, dim, 3, 3);
        cn_alloc_filter(&cl, 1);
        float elem[] = {0.0625, 0.1250, 0.0625, 0.1250, 0.25,
                        0.1250, 0.0625, 0.1250, 0.0625};
        cl.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        cn_correlate_layer(&cl, &minput, 1);
        print_results(cl.outputs[0]);
    }

    else if (strcmp(argv[1], "same_guassian_blur_5") == 0) {
        const size_t dim = 20;
        ConvolutionalLayer cl = cn_init_convolutional_layer(Same, Sigmoid, dim, dim, 5, 5);
        cn_alloc_filter(&cl, 1);
        float elem[] = {0.0037, 0.0147, 0.0256, 0.0147, 0.0037, 0.0147, 0.0586,
                        0.0952, 0.0586, 0.0147, 0.0256, 0.0952, 0.1502, 0.0952,
                        0.0256, 0.0147, 0.0586, 0.0952, 0.0586, 0.0147, 0.0037,
                        0.0147, 0.0256, 0.0147, 0.0037};
        cl.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        cn_correlate_layer(&cl, &minput, 1);
        print_results(cl.outputs[0]);
    }

    else if (strcmp(argv[1], "same_even_kernel") == 0) {
        do_test_with_default_elements(Same, Sigmoid, 20, 20, 4, 4);
    }

    else if (strcmp(argv[1], "same_rect") == 0) {
        do_test_with_default_elements(Same, Sigmoid, 30, 30, 5, 3);
    }

    else if (strcmp(argv[1], "full_7x7") == 0) {
        do_test_with_default_elements(Full, Sigmoid, 30, 30, 7, 7);
    }

    else if (strcmp(argv[1], "full_even") == 0) {
        do_test_with_default_elements(Full, Sigmoid, 15, 15, 4, 4);
    }

    else if (strcmp(argv[1], "full_rect") == 0) {
        do_test_with_default_elements(Full, Sigmoid, 30, 30, 4, 7);
    }

    else if (strcmp(argv[1], "valid_7x7") == 0) {
        do_test_with_default_elements(Valid, Sigmoid, 11, 11, 7, 7);
    }

    else if (strcmp(argv[1], "valid_rect") == 0) {
        do_test_with_default_elements(Valid, Sigmoid, 23, 23, 1, 6);
    }

    else if (strcmp(argv[1], "valid_rect_input") == 0) {
        do_test_with_default_elements(Valid, Sigmoid, 10, 20, 4, 4);
    }

    else if (strcmp(argv[1], "multi_output") == 0) {
        size_t n_cls = 5;
        size_t in_dim = 20;

        ConvolutionalLayer *cls = CLEAR_NET_ALLOC(n_cls * sizeof(ConvolutionalLayer));
        cls[0] = cn_init_convolutional_layer(Valid, Sigmoid, in_dim, in_dim, 5,5);
        cn_alloc_filter(&cls[0], 3);
        cn_alloc_filter(&cls[0], 3);
        cn_alloc_filter(&cls[0], 3);
        _cn_randomize_convolutional_layer(&cls[0], -1, 1);

        cls[1] = cn_init_convolutional_layer(Valid, Sigmoid, cls[0].output_nrows, cls[0].output_ncols, 5, 5);
        cn_alloc_filter(&cls[1], 2);
        cn_alloc_filter(&cls[1], 2);
        _cn_randomize_convolutional_layer(&cls[1], -1, 1);

        cls[2] = cn_init_convolutional_layer(Valid, Sigmoid, cls[1].output_nrows, cls[1].output_ncols, 7, 7);
        cn_alloc_filter(&cls[2], 4);
        cn_alloc_filter(&cls[2], 4);
        cn_alloc_filter(&cls[2], 4);
        _cn_randomize_convolutional_layer(&cls[2], -1, 1);
        float input[in_dim * in_dim];
        Matrix minput = cn_form_matrix(in_dim, in_dim, in_dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);

        cn_correlate_layer(&cls[0], &minput, 1);
        cn_correlate_layer(&cls[1], cls[0].outputs, cls[0].nfilters);
        cn_correlate_layer(&cls[2], cls[1].outputs, cls[1].nfilters);
        printf("zero output\n");
        for (size_t i = 0; i < cls[0].nfilters; ++i) {
            cn_print_matrix(cls[0].outputs[i], "output");
        }

        printf("one output\n");
        for (size_t i = 0; i < cls[1].nfilters; ++i) {
            cn_print_matrix(cls[1].outputs[i], "output");
        }

        printf("two output\n");
        for (size_t i = 0; i < cls[2].nfilters; ++i) {
            cn_print_matrix(cls[2].outputs[i], "output");
        }
    }

    else if (strcmp(argv[1], "mnist_structure") == 0) {
        size_t n_cls = 5;
        size_t in_dim = 28;
        ConvolutionalLayer *cls = CLEAR_NET_ALLOC(n_cls * sizeof(ConvolutionalLayer));
        cls[0] = cn_init_convolutional_layer(Valid, Sigmoid, in_dim, in_dim, 8, 8);
        cn_alloc_filter(&cls[0], 3);
        cn_alloc_filter(&cls[0], 3);
        cn_alloc_filter(&cls[0], 3);
        cn_alloc_filter(&cls[0], 3);
        cn_alloc_filter(&cls[0], 3);
        _cn_randomize_convolutional_layer(&cls[0], -1, 1);

        PoolingLayer pooler = cn_alloc_pooling_layer(Average, cls[0].nfilters, cls[0].output_nrows, cls[0].output_ncols, 4, 4);
        float input[in_dim * in_dim];
        Matrix minput = cn_form_matrix(in_dim, in_dim, in_dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);

        cn_correlate_layer(&cls[0], &minput, 1);
        for (size_t i = 0; i < cls[0].nfilters; ++i) {
            cn_print_matrix(cls[0].outputs[i], "convlayer");
        }
        cn_pool_layer(&pooler, cls[0].outputs, cls[0].nfilters);

        for (size_t i = 0; i < pooler.noutput; ++i) {
            cn_print_matrix(pooler.outputs[i], "pooler");
        }

        GlobalPoolingLayer gpooler = cn_alloc_global_pooling_layer(Max, pooler.noutput);
        cn_global_pool_layer(&gpooler, pooler.outputs, pooler.noutput);

        _cn_print_vector(gpooler.output, "gpooler");
    }
}
