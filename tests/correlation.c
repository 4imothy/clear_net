#include "tests.h"
#include "net_test.h"
#include "../lib/autodiff.h"

#define LEN(ptr) sizeof(ptr) / sizeof(*ptr)

scalar poss_elements[] = {
    0.10290608318034533, 0.8051580508692876,
    0.39055048005351034, 0.7739175926400883,
    0.24730207704015073, 0.7987075645399935,
    0.24602568871407338, 0.6268407447350659,
    0.4646505260697441, 0.20524882983167547,
    0.5031590491750169,  0.2550151936024112,
    0.3354895253780905,  0.6825483746871936,
    0.6204572461588524, 0.6487941004544666,
    0.742795723261874,   0.8436721618301802,
    0.0433154872324607,  0.42621935359557017};

const ulong poss_elements_len = LEN(poss_elements);

float poss_kernel_elements[] = {2, 6, 7, 8, 4, 0, 6, 4, 2, 0, 9,
                                      7, 5, 9, 8, 8, 4, 6, 0, 2, 4, 7,
                                      6, 1, 7, 5, 2, 9, 6, 7, 8};
const size_t poss_kernel_elem_len = LEN(poss_kernel_elements);

const ulong nchannels = 1;

void printResults(CompGraph *cg, UMat mat) {
    printf("%zu %zu", mat.nrows, mat.ncols);
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            printf(" %f ", getVal(cg, MAT_AT(mat, i, j)));
        }
    }
}

void do_test(Padding padding, Activation act, const ulong input_nrows, const ulong input_ncols, ulong kern_nrows, ulong kern_ncols, scalar *kern_pool, ulong kern_pool_len, scalar *input_pool, ulong input_pool_len) {
    HParams *hp = allocDefaultHParams();
    setLeaker(hp, 1);
    Net *net = allocConvNet(hp, input_nrows, input_ncols, nchannels);
    allocConvLayer(net, padding, act , 1, kern_nrows, kern_ncols);
    Mat kern = net->layers[0].data.conv.filters[0].kernels[0];
    ulong cur_id = 0;
    for (ulong i = 0; i < kern.nrows; ++i) {
        for (ulong j = 0; j < kern.ncols; ++j) {

            setVal(net->cg, MAT_ID(kern, i, j), kern_pool[cur_id]);
            cur_id = (cur_id + 1) % kern_pool_len;

        }
    }

    UMat input = allocUMat(input_nrows, input_ncols);
    cur_id = 0;
    for (ulong i = 0; i < input.nrows; ++i) {
        for (ulong j = 0; j < input.ncols; ++j) {
            MAT_AT(input, i, j) = initLeafScalar(net->cg, input_pool[cur_id]);
            cur_id = (cur_id + 1) % input_pool_len;
        }
    }
    forwardConv(net->cg, &net->layers[0].data.conv, &input, net->hp.leaker);


    printResults(net->cg, net->layers[0].data.conv.outputs[0]);
}

int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    Activation default_act = LeakyReLU;
    srand(0);
    if (strequal(argv[1], "same_zeros")) {
        const size_t dim = 15;
        float elem[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        ulong elem_len = LEN(elem);
        do_test(Same, default_act, dim, dim, 3,3, elem, elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "same_identity")) {
        float elem[] = {0, 0, 0,
                        0, 1, 0,
                        0, 0, 0};
        ulong elem_len = LEN(elem);
        const size_t dim = 10;
        do_test(Same, default_act, dim, dim, 3, 3, elem, elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "same_guassian_blur_3")) {
        const size_t dim = 20;
        float elem[] = {0.0625, 0.1250, 0.0625,
                        0.1250, 0.25, 0.1250,
                        0.0625, 0.1250, 0.0625};
        ulong elem_len = LEN(elem);
        do_test(Same, default_act, dim, dim, 3,3, elem, elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "same_guassian_blur_5")) {
        const size_t dim = 20;
        float elem[] = {0.0037, 0.0147, 0.0256, 0.0147, 0.0037,
                        0.0147, 0.0586, 0.0952, 0.0586, 0.0147,
                        0.0256, 0.0952, 0.1502, 0.0952, 0.0256,
                        0.0147, 0.0586, 0.0952, 0.0586, 0.0147,
                        0.0037, 0.0147, 0.0256, 0.0147, 0.0037};
        ulong elem_len = LEN(elem);
        do_test(Same, default_act, dim, dim, 5, 5, elem, elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "same_even_kernel")) {
        do_test(Same, default_act, 20, 20, 4, 4, poss_kernel_elements, poss_kernel_elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "same_rect")) {
        do_test(Same, default_act, 30, 30, 5, 3, poss_kernel_elements, poss_kernel_elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "full_7x7")) {
        do_test(Full, default_act, 30, 30, 7, 7, poss_kernel_elements, poss_kernel_elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "full_even")) {
        do_test(Full, default_act, 15, 15, 4, 4, poss_kernel_elements, poss_kernel_elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "full_rect")) {
        do_test(Full, default_act, 30, 30, 4, 7, poss_kernel_elements, poss_kernel_elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "valid_7x7")) {
        do_test(Valid, default_act, 11, 11, 7, 7, poss_kernel_elements, poss_kernel_elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "valid_rect")) {
        do_test(Valid, default_act, 23, 23, 1, 6, poss_kernel_elements, poss_kernel_elem_len, poss_elements, poss_elements_len);
    }

    else if (strequal(argv[1], "valid_rect_input")) {
        do_test(Valid, default_act, 10, 20, 4, 4, poss_kernel_elements, poss_kernel_elem_len, poss_elements, poss_elements_len);
    }
}
