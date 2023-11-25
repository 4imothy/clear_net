#include "tests.h"

const Activation default_act = LeakyReLU;

void do_test(ulong input_nelem, ulong out_nelem, scalar *mat_pool,
             ulong mat_pool_len, scalar *input_pool, ulong input_pool_len) {
    HParams *hp = allocDefaultHParams();
    setLeaker(hp, 1);
    Net *net = allocVanillaNet(hp, input_nelem);
    allocDenseLayer(net, default_act, out_nelem);
    Mat mat = net->layers[0].data.dense.weights;
    fill_mat(net->cg, &mat, mat_pool, mat_pool_len);

    UVec input = allocUVec(input_nelem);
    set_uvec(net->cg, &input, input_pool, input_pool_len);
    forwardDense(net->cg, &net->layers[0].data.dense, input, net->hp.leaker);

    printVecResults(net->cg, net->layers[0].data.dense.output);
    deallocNet(net);
}

int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    srand(0);
    if (strequal(argv[1], "same_zeros")) {
        ulong in_dim = 15;
        scalar elem[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        ulong elem_len = LEN(elem);
        do_test(in_dim, in_dim, elem, elem_len, input_elem, input_elem_len);
    }

    else if (strequal(argv[1], "up_zeros")) {
        scalar elem[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        ulong elem_len = LEN(elem);
        ulong in_dim = 10;
        ulong out_dim = 15;
        do_test(in_dim, out_dim, elem, elem_len, input_elem, input_elem_len);
    }

    else if (strequal(argv[1], "down_zeros")) {
        scalar elem[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        ulong elem_len = LEN(elem);
        ulong in_dim = 15;
        ulong out_dim = 7;
        do_test(in_dim, out_dim, elem, elem_len, input_elem, input_elem_len);
    }

    else if (strequal(argv[1], "10_10")) {
        do_test(10, 10, matrix_elem, matrix_elem_len, input_elem, input_elem_len);
    }

    else if (strequal(argv[1], "1000_1000")) {
        do_test(1000, 1000, matrix_elem, matrix_elem_len, input_elem, input_elem_len);
    }

    else if (strequal(argv[1], "1_5")) {
        do_test(1, 5, matrix_elem, matrix_elem_len, input_elem, input_elem_len);
    }

    else if (strequal(argv[1], "5_1")) {
        do_test(5, 1, matrix_elem, matrix_elem_len, input_elem, input_elem_len);
    }

    else if (strequal(argv[1], "15_40")) {
        do_test(15, 40, matrix_elem, matrix_elem_len, input_elem, input_elem_len);
    }

    else if (strequal(argv[1], "40_15")) {
        do_test(40, 15, matrix_elem, matrix_elem_len, input_elem, input_elem_len);
    }


    return 0;
}
