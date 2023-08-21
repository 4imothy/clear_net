#include <stdio.h>
#include <string.h>
#define CLEAR_NET_IMPLEMENTATION
#include "../../clear_net.h"

int main(int argc, char *argv[]) {
    CLEAR_NET_ASSERT(argc == 2);
    cn_default_hparams();
    Net net = cn_init_net();
    float eps = 0.000001;
    if (strcmp(argv[1], "dense") == 0) {
        size_t din = 5;
        size_t dout = 3;
        cn_alloc_dense_layer(&net, Sigmoid, din, 10);
        cn_alloc_secondary_dense_layer(&net, Tanh, dout);
        cn_randomize_net(&net, -1 , 1);
        Vector in = cn_alloc_vector(din);
        _cn_randomize_vector(&in, -1 , 1);
        Vector fout = cn_predict_vani(&net, in);

        GradientStore *gs = &net.computation_graph;
        gs->length = 1;
        _cn_copy_net_params(&net, gs);
        in.gs_id = gs->length;
        _cn_copy_vector_params(gs, in);
        Vector gsout = _cn_predict_vani(&net, gs, in);
        CLEAR_NET_ASSERT(gsout.nelem == fout.nelem);
        for (size_t i = 0; i < dout; ++i) {
            CLEAR_NET_ASSERT(GET_NODE(VEC_ID(gsout, i)).num - VEC_AT(fout, i) < eps);
        }
        printf("Pass: Dense Forward\n");
    } else if (strcmp(argv[1], "conv") == 0) {
        size_t in_width = 20;
        size_t in_height = 20;
        cn_alloc_conv_layer(&net, Valid, Sigmoid, 1, 3, in_height, in_width, 9 , 9);
        cn_alloc_secondary_conv_layer(&net, Valid, Sigmoid, 5, 5, 5);
        cn_alloc_secondary_conv_layer(&net, Valid, Sigmoid, 1, 5, 5);
        cn_randomize_net(&net, -1, 1);
        Matrix in = cn_alloc_matrix(in_height, in_width);
        _cn_randomize_matrix(&in, -1, 1);
        LAData fout = cn_predict_conv(&net, &in);

        GradientStore *gs = &net.computation_graph;
        gs->length = 1;
        _cn_copy_net_params(&net, gs);
        in.gs_id = gs->length;
        _cn_copy_matrix_params(gs, in);
        LAData gsout = _cn_predict_conv(&net, gs, &in);
        Matrix fes = fout.data.mat;
        Matrix gses = gsout.data.mat;
        CLEAR_NET_ASSERT(gses.nrows == fes.nrows);
        CLEAR_NET_ASSERT(gses.ncols == fes.ncols);
        for (size_t i = 0; i < fes.nrows; ++i) {
            for (size_t j = 0; j < fes.ncols; ++j) {
                CLEAR_NET_ASSERT(MAT_AT(fes, i, j) - GET_NODE(MAT_ID(gses, i, j)).num < eps);
            }
        }
        printf("Pass: convolutional\n");
    } else if (strcmp(argv[1], "with_pooling") == 0) {
        size_t in_width = 28;
        size_t in_height = 28;
        cn_alloc_conv_layer(&net, Valid, Sigmoid, 1, 3, in_height, in_width, 9 , 9);
        cn_alloc_secondary_conv_layer(&net, Valid, Sigmoid, 5, 5, 5);
        cn_alloc_pooling_layer(&net, Average, 4, 4);
        cn_alloc_secondary_conv_layer(&net, Valid, Sigmoid, 10, 3, 3);
        cn_alloc_global_pooling_layer(&net, Max);
        cn_randomize_net(&net, -1, 1);
        Matrix in = cn_alloc_matrix(in_height, in_width);
        LAData fout = cn_predict_conv(&net, &in);

        GradientStore *gs = &net.computation_graph;
        gs->length = 1;
        _cn_copy_net_params(&net, gs);
        in.gs_id = gs->length;
        _cn_copy_matrix_params(gs, in);
        LAData gsout = _cn_predict_conv(&net, gs, &in);
        Vector fes = fout.data.vec;
        Vector gses = gsout.data.vec;
        CLEAR_NET_ASSERT(gses.nelem == fes.nelem);
        for (size_t i = 0; i < fes.nelem; ++i) {
            CLEAR_NET_ASSERT(VEC_AT(fes, i) - GET_NODE(VEC_ID(gses, i)).num < eps);
        }
        printf("Pass: convolutional with pooling\n");
    }

    return 0;
}
