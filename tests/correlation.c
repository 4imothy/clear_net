#include "tests.h"

const ulong nchannels = 1;
const Activation default_act = LEAKYRELU;

void do_test(Padding padding, ulong input_nrows, ulong input_ncols,
             ulong kern_nrows, ulong kern_ncols, scalar *kern_pool,
             ulong kern_pool_len, scalar *input_pool, ulong input_pool_len) {
    HParams *hp = allocDefaultHParams();
    setLeaker(hp, 1);
    Net *net = allocConvNet(hp, input_nrows, input_ncols, nchannels);
    allocConvLayer(net, default_act, padding, 1, kern_nrows, kern_ncols);
    Mat kern = net->layers[0].in.conv.filters[0].kernels[0];
    fill_mat(net->cg, &kern, kern_pool, kern_pool_len);

    UMat input = allocUMat(input_nrows, input_ncols);
    set_umat(net->cg, &input, input_pool, input_pool_len);
    forwardConv(net->cg, &net->layers[0].in.conv, &input, net->hp.leaker);

    printMatResults(net->cg, net->layers[0].in.conv.outputs[0]);
    deallocNet(net);
}

int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    srand(0);
    if (strequal(argv[1], "same_zeros")) {
        const size_t dim = 15;
        scalar elem[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        ulong elem_len = LEN(elem);
        do_test(SAME, dim, dim, 3, 3, elem, elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "same_identity")) {
        scalar elem[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
        ulong elem_len = LEN(elem);
        const size_t dim = 10;
        do_test(SAME, dim, dim, 3, 3, elem, elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "same_guassian_blur_3")) {
        const size_t dim = 20;
        scalar elem[] = {0.0625, 0.1250, 0.0625, 0.1250, 0.25,
                        0.1250, 0.0625, 0.1250, 0.0625};
        ulong elem_len = LEN(elem);
        do_test(SAME, dim, dim, 3, 3, elem, elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "same_guassian_blur_5")) {
        const size_t dim = 20;
        scalar elem[] = {0.0037, 0.0147, 0.0256, 0.0147, 0.0037, 0.0147, 0.0586,
                        0.0952, 0.0586, 0.0147, 0.0256, 0.0952, 0.1502, 0.0952,
                        0.0256, 0.0147, 0.0586, 0.0952, 0.0586, 0.0147, 0.0037,
                        0.0147, 0.0256, 0.0147, 0.0037};
        ulong elem_len = LEN(elem);
        do_test(SAME, dim, dim, 5, 5, elem, elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "same_even_kernel")) {
        do_test(SAME, 20, 20, 4, 4, matrix_elem, matrix_elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "same_rect")) {
        do_test(SAME, 30, 30, 5, 3, matrix_elem, matrix_elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "full_7x7")) {
        do_test(FULL, 30, 30, 7, 7, matrix_elem, matrix_elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "full_even")) {
        do_test(FULL, 15, 15, 4, 4, matrix_elem, matrix_elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "full_rect")) {
        do_test(FULL, 30, 30, 4, 7, matrix_elem, matrix_elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "valid_7x7")) {
        do_test(VALID, 11, 11, 7, 7, matrix_elem, matrix_elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "valid_rect")) {
        do_test(VALID, 23, 23, 1, 6, matrix_elem, matrix_elem_len, input_elem,
                input_elem_len);
    }

    else if (strequal(argv[1], "valid_rect_input")) {
        do_test(VALID, 10, 20, 4, 4, matrix_elem, matrix_elem_len, input_elem,
                input_elem_len);
    }
}
