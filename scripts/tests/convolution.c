// TODO add multi input/output test
// TODO make an add filter method to the ConvolutionalLayer, create the layer
// first then add filter method,
// like the new way to create a neural net
// TODO test it running computations on the gradient store
#include <stddef.h>
#define CLEAR_NET_IMPLEMENTATION
#include "../../clear_net.h"
#include <string.h>

typedef struct ConvolutionalLayer ConvolutionalLayer;
typedef struct Filter Filter;

typedef enum {
    Same,
    Valid,
    Full,
} Padding;

struct Filter {
    Matrix *kernels;
    size_t nkernels;
    Matrix biases;
    Matrix output;
    Activation act;
    size_t gs_id;
};

struct ConvolutionalLayer {
    Filter *filters;
    size_t nfilters;
    Padding padding;
};

float padded_mat_at(Matrix mat, long row, long col) {
    if (row < 0 || col < 0 || row >= (long)mat.nrows ||
        col >= (long)mat.ncols) {
        return 0;
    }
    return MAT_AT(mat, row, col);
}

float correlation(Matrix kern, Matrix *input, long top_left_row,
                  long top_left_col) {
    float res = 0;
    for (size_t i = 0; i < kern.nrows; ++i) {
        for (size_t j = 0; j < kern.ncols; ++j) {
            res += MAT_AT(kern, i, j) *
                   padded_mat_at(*input, top_left_row + i, top_left_col + j);
        }
    }
    return res;
}

void forward(ConvolutionalLayer *layer, Matrix **input) {
    float res;
    size_t row_padding;
    size_t col_padding;
    for (size_t i = 0; i < layer->nfilters; ++i) {
        for (size_t j = 0; j < layer->filters[i].output.nrows; ++j) {
            for (size_t k = 0; k < layer->filters[i].output.ncols; ++k) {
                for (size_t l = 0; l < layer->filters[i].nkernels; ++l) {
                    switch (layer->padding) {
                    case Same:
                        row_padding =
                            (layer->filters[i].kernels[l].nrows - 1) / 2;
                        col_padding =
                            (layer->filters[i].kernels[l].ncols - 1) / 2;
                        break;
                    case Full:
                        row_padding =
                            layer->filters[i].kernels[l].nrows - 1;
                        col_padding =
                            layer->filters[i].kernels[l].ncols - 1;
                        break;
                    case Valid:
                        row_padding = 0;
                        col_padding = 0;
                        break;
                    }
                    long top_left_row = (long)j - row_padding;
                    long top_left_col = (long)k - col_padding;
                    res = correlation(layer->filters[i].kernels[l], input[i],
                                      top_left_row, top_left_col);
                    MAT_AT(layer->filters[i].output, j, k) += res;
                }
            }
        }
    }
}

Filter alloc_filter(size_t kernel_nrows, size_t kernel_ncols, size_t nkernels, size_t input_nrows,
                    size_t input_ncols, Padding padding) {
    Filter filter;
    filter.nkernels = nkernels;
    filter.kernels = CLEAR_NET_ALLOC(filter.nkernels * sizeof(Matrix));
    for (size_t i = 0; i < filter.nkernels; ++i) {
        filter.kernels[i] = cn_alloc_matrix(kernel_nrows, kernel_ncols);
    }
    size_t output_nrows;
    size_t output_ncols;
    switch(padding) {
    case Same:
        output_nrows = input_nrows;
        output_ncols = input_ncols;
        break;
    case Full:
        output_nrows = input_nrows + kernel_nrows - 1;
        output_ncols = input_ncols + kernel_ncols - 1;
        break;
    case Valid:
        output_nrows = input_nrows - kernel_nrows + 1;
        output_ncols = input_ncols - kernel_ncols + 1;
        break;
    }
    filter.biases = cn_alloc_matrix(output_nrows, output_ncols);
    filter.output = cn_alloc_matrix(output_nrows, output_ncols);
    filter.act = Sigmoid;
    filter.gs_id = 0;
    return filter;
}

ConvolutionalLayer alloc_convolutional_layer(Filter *filters, size_t nfilters, Padding padding) {
    return (ConvolutionalLayer){
        .filters = filters,
        .nfilters = nfilters,
        .padding = padding,
    };
}

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

void do_test_with_default_elements(const size_t input_rows, const size_t input_cols, size_t krows, size_t kcols, Padding padding) {
         size_t nfilters = 1;
         Filter filter = alloc_filter(krows, kcols, 1, input_rows, input_cols, padding);
        fill_matrix(&filter.kernels[0], poss_kernel_elements,
                    poss_kernel_elem_len);
        ConvolutionalLayer cl = alloc_convolutional_layer(&filter, nfilters, padding);
        float input[input_rows * input_cols];
        Matrix minput = cn_form_matrix(input_rows, input_cols, input_cols, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        Matrix *list = &minput;
        forward(&cl, &list);
        print_results(cl.filters[0].output);
}

int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    srand(0);
    if (strcmp(argv[1], "same_zeros") == 0) {
        Padding padding = Same;
        size_t nfilters = 1;
        const size_t dim = 15;
        Filter filter = alloc_filter(3, 3, 1, dim, dim, padding);
        float elem[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        filter.kernels[0].elements = elem;
        ConvolutionalLayer cl = alloc_convolutional_layer(&filter, nfilters, padding);
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        Matrix *list = &minput;
        forward(&cl, &list);
        print_results(cl.filters[0].output);
    }

    else if (strcmp(argv[1], "same_identity") == 0) {
        Padding padding = Same;
        size_t nfilters = 1;
        const size_t dim = 10;
        Filter filter = alloc_filter(3, 3, 1, dim, dim, Same);
        float elem[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
        filter.kernels[0].elements = elem;
        ConvolutionalLayer cl = alloc_convolutional_layer(&filter, nfilters, padding);
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        Matrix *list = &minput;
        forward(&cl, &list);
        print_results(cl.filters[0].output);
    }

    else if (strcmp(argv[1], "same_guassian_blur_3") == 0) {
        Padding padding = Same;
        size_t nfilters = 1;
        const size_t dim = 20;
        Filter filter = alloc_filter(3, 3, 1, dim, dim, padding);
        float elem[] = {0.0625, 0.1250, 0.0625, 0.1250, 0.25,
                        0.1250, 0.0625, 0.1250, 0.0625};
        filter.kernels[0].elements = elem;
        ConvolutionalLayer cl = alloc_convolutional_layer(&filter, nfilters, padding);
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        Matrix *list = &minput;
        forward(&cl, &list);
        print_results(cl.filters[0].output);
    }

    else if (strcmp(argv[1], "same_guassian_blur_5") == 0) {
        Padding padding = Same;
        size_t nfilters = 1;
        const size_t dim = 20;
        Filter filter = alloc_filter(5, 5, 1, dim, dim, padding);
        float elem[] = {0.0037, 0.0147, 0.0256, 0.0147, 0.0037, 0.0147, 0.0586,
                        0.0952, 0.0586, 0.0147, 0.0256, 0.0952, 0.1502, 0.0952,
                        0.0256, 0.0147, 0.0586, 0.0952, 0.0586, 0.0147, 0.0037,
                        0.0147, 0.0256, 0.0147, 0.0037};
        filter.kernels[0].elements = elem;
        ConvolutionalLayer cl = alloc_convolutional_layer(&filter, nfilters, padding);
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        Matrix *list = &minput;
        forward(&cl, &list);
        print_results(cl.filters[0].output);
    }

    else if (strcmp(argv[1], "same_even_kernel") == 0) {
        do_test_with_default_elements(20, 20, 4,4, Same);
    }

    else if (strcmp(argv[1], "same_rect") == 0) {
        do_test_with_default_elements(30, 30, 5, 3, Same);
    }

    else if (strcmp(argv[1], "full_7x7") == 0) {
        do_test_with_default_elements(30, 30, 7, 7, Full);
    }

    else if (strcmp(argv[1], "full_even") == 0) {
        do_test_with_default_elements(15, 15, 4, 4, Full);
    }

    else if (strcmp(argv[1], "full_rect") == 0) {
        do_test_with_default_elements(30, 30, 4, 7, Full);
    }

    else if (strcmp(argv[1], "valid_7x7") == 0) {
        do_test_with_default_elements(11, 11, 7, 7, Valid);
    }

    else if (strcmp(argv[1], "valid_rect") == 0) {
        do_test_with_default_elements(23, 23, 1, 6, Valid);
    }

    else if (strcmp(argv[1], "valid_rect_input") == 0) {
        do_test_with_default_elements(10, 20, 4, 4, Valid);
    }
}
