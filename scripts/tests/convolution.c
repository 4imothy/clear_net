// TODO add multi input/output test
// TODO test it running computations on the gradient store
// - Can do this by arg 2 being to run on the gradient store or normal math
// - Do this after
// TODO make a function that reshapes the 3d output of a convolutional layer to
// a vector
// Global Max Pooling, take each output of the layer and take the greatest
// element and pass that to the dense layers via a vector
// TODO make the outputs owned by the layer struct not by each Filter

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
};

struct ConvolutionalLayer {
    Filter *filters;
    size_t nfilters;
    Padding padding;
    size_t input_nrows;
    size_t input_ncols;
    size_t output_nrows;
    size_t output_ncols;
    size_t k_nrows;
    size_t k_ncols;
};

// Each layer has its own list of outputs which all have the same dimension
void alloc_filter(ConvolutionalLayer *c_layer, size_t nkernels) {
    Filter filter;
    filter.nkernels = nkernels;
    // TODO check if this can be a reference that is allocated
    filter.kernels = CLEAR_NET_ALLOC(nkernels * sizeof(Matrix));
    for (size_t i = 0; i < filter.nkernels; ++i) {
        filter.kernels[i] = cn_alloc_matrix(c_layer->k_nrows, c_layer->k_ncols);
    }
    filter.biases = cn_alloc_matrix(c_layer->output_nrows, c_layer->output_ncols);
    filter.output = cn_alloc_matrix(c_layer->output_nrows, c_layer->output_ncols);
    filter.act = Sigmoid;
    // TODO check if this can be reference
    c_layer->filters = CLEAR_NET_REALLOC(c_layer->filters, c_layer->nfilters + 1 * sizeof(Filter));
    c_layer->filters[c_layer->nfilters] = filter;
    c_layer->nfilters++;
}


ConvolutionalLayer init_convolutional_layer(Padding padding, size_t input_nrows, size_t input_ncols, size_t kernel_nrows, size_t kernel_ncols) {
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
    return (ConvolutionalLayer){
        .filters = NULL,
        .nfilters = 0,
        .padding = padding,
        .input_nrows = input_nrows,
        .input_ncols = input_ncols,
        .output_nrows = output_nrows,
        .output_ncols = output_ncols,
        .k_nrows = kernel_nrows,
        .k_ncols = kernel_ncols,
    };
}

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
    // TODO for the actual net will need to apply biases and activations
}


void _forward(GradientStore *gs, ConvolutionalLayer *layer, Matrix **input) {
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
    // TODO for the actual net will need to apply biases and activations
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

void do_test_with_default_elements(Padding padding, const size_t input_rows, const size_t input_cols, size_t krows, size_t kcols) {
    ConvolutionalLayer cl = init_convolutional_layer(padding, input_rows, input_cols, krows, kcols);
    alloc_filter(&cl, 1);
    fill_matrix(&cl.filters[0].kernels[0], poss_kernel_elements,
                    poss_kernel_elem_len);
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
        const size_t dim = 15;
        ConvolutionalLayer cl = init_convolutional_layer(Same, 15, 15, 3, 3);
        alloc_filter(&cl, 1);
        float elem[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        cl.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        Matrix *list = &minput;
        forward(&cl, &list);
        print_results(cl.filters[0].output);
    }

    else if (strcmp(argv[1], "same_identity") == 0) {
        const size_t dim = 10;
        ConvolutionalLayer cl = init_convolutional_layer(Same, 10, 10, 3, 3);
        alloc_filter(&cl, 1);
        float elem[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
        cl.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        Matrix *list = &minput;
        forward(&cl, &list);
        print_results(cl.filters[0].output);
    }

    else if (strcmp(argv[1], "same_guassian_blur_3") == 0) {
        const size_t dim = 20;
        ConvolutionalLayer cl = init_convolutional_layer(Same, dim, dim, 3, 3);
        alloc_filter(&cl, 1);
        float elem[] = {0.0625, 0.1250, 0.0625, 0.1250, 0.25,
                        0.1250, 0.0625, 0.1250, 0.0625};
        cl.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        Matrix *list = &minput;
        forward(&cl, &list);
        print_results(cl.filters[0].output);
    }

    else if (strcmp(argv[1], "same_guassian_blur_5") == 0) {
        const size_t dim = 20;
        ConvolutionalLayer cl = init_convolutional_layer(Same, dim, dim, 5, 5);
        alloc_filter(&cl, 1);
        float elem[] = {0.0037, 0.0147, 0.0256, 0.0147, 0.0037, 0.0147, 0.0586,
                        0.0952, 0.0586, 0.0147, 0.0256, 0.0952, 0.1502, 0.0952,
                        0.0256, 0.0147, 0.0586, 0.0952, 0.0586, 0.0147, 0.0037,
                        0.0147, 0.0256, 0.0147, 0.0037};
        cl.filters[0].kernels[0].elements = elem;
        float input[dim * dim] = {0};
        Matrix minput = cn_form_matrix(dim, dim, dim, input);
        fill_matrix(&minput, poss_elements, poss_elements_len);
        Matrix *list = &minput;
        forward(&cl, &list);
        print_results(cl.filters[0].output);
    }

    else if (strcmp(argv[1], "same_even_kernel") == 0) {
        do_test_with_default_elements(Same, 20, 20, 4, 4);
    }

    else if (strcmp(argv[1], "same_rect") == 0) {
        do_test_with_default_elements(Same, 30, 30, 5, 3);
    }

    else if (strcmp(argv[1], "full_7x7") == 0) {
        do_test_with_default_elements(Full, 30, 30, 7, 7);
    }

    else if (strcmp(argv[1], "full_even") == 0) {
        do_test_with_default_elements(Full, 15, 15, 4, 4);
    }

    else if (strcmp(argv[1], "full_rect") == 0) {
        do_test_with_default_elements(Full, 30, 30, 4, 7);
    }

    else if (strcmp(argv[1], "valid_7x7") == 0) {
        do_test_with_default_elements(Valid, 11, 11, 7, 7);
    }

    else if (strcmp(argv[1], "valid_rect") == 0) {
        do_test_with_default_elements(Valid, 23, 23, 1, 6);
    }

    else if (strcmp(argv[1], "valid_rect_input") == 0) {
        do_test_with_default_elements(Valid, 10, 20, 4, 4);
    }
}
