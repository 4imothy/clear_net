#include "clear_net.h"
#include "la.h"

IOData *formDataFromVectors(Vector *vectors, ulong nelem) {
    IOData* data = CLEAR_NET_ALLOC(sizeof(IOData));
    data->type = VecList;
    data->nelem = nelem;
    data->nchannels = 1;
    data->data.vec_list = vectors;
    return data;
}

IOData *formDataFromMatrices(Matrix *matrices, ulong nelem) {
    IOData* data = CLEAR_NET_ALLOC(sizeof(IOData));
    data->type = MatList;
    data->nelem = nelem;
    data->nchannels = 1;
    data->data.mat_list = matrices;
    return data;
}

IOData *formDataFromMultiChannelMatrices(Matrix **multi_matrices, ulong nelem, ulong nchannels) {
    IOData* data = CLEAR_NET_ALLOC(sizeof(IOData));
    data->type = MultiMatList;
    data->nelem = nelem;
    data->nchannels = nchannels;
    data->data.multi_mat_list = multi_matrices;
    return data;
}

void deallocIOData(IOData *data) {
    for (ulong i= 0 ; i < data->nelem; ++i) {
        switch(data->type) {
        case(MatList): {
            deallocMatrix(&data->data.mat_list[i]);
            break;
        }
        case(VecList): {
            deallocVector(&data->data.vec_list[i]);
            break;
        }
        case(MultiMatList):
            for (ulong j = 0 ; j < data->nelem; ++j) {
                deallocMatrix(&data->data.multi_mat_list[i][j]);
            }
            break;
        }
    }

    CLEAR_NET_DEALLOC(data);
}

Matrix formMatrix(ulong nrows, ulong ncols, ulong stride, scalar *elements) {
    return (Matrix){
        .nrows = nrows, .ncols = ncols, .stride = stride, .elem = elements};
}

Matrix allocMatrix(ulong nrows, ulong ncols) {
    scalar *elem = CLEAR_NET_ALLOC(nrows * ncols * sizeof(scalar));
    CLEAR_NET_ASSERT(elem);
    return formMatrix(nrows, ncols, ncols, elem);
}

void deallocMatrix(Matrix *mat) {
    CLEAR_NET_DEALLOC(mat->elem);
    mat->elem = NULL;
    mat->nrows = 0;
    mat->ncols = 0;
    mat->stride = 0;
}

void printMatrix(Matrix *mat, char *name) {
    printf("%s = [\n", name);
    for (ulong i = 0; i < mat->nrows; ++i) {
        printf("    ");
        for (ulong j = 0; j < mat->ncols; ++j) {
            printf("%f ", MAT_AT(*mat, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void shuffleMatrixRows(Matrix *input, Matrix *target) {
    CLEAR_NET_ASSERT(input->nrows == target->nrows);
    scalar t;
    for (ulong i = 0; i < input->nrows; ++i) {
        ulong j = i + rand() % (input->nrows - i);
        if (i != j) {
            for (ulong k = 0; k < input->ncols; ++k) {
                t = MAT_AT(*input, i, k);
                MAT_AT(*input, i, k) = MAT_AT(*input, j, k);
                MAT_AT(*input, j, k) = t;
            }
            for (ulong k = 0; k < target->ncols; ++k) {
                t = MAT_AT(*target, i, k);
                MAT_AT(*target, i, k) = MAT_AT(*target, j, k);
                MAT_AT(*target, j, k) = t;
            }
        }
    }
}

Vector formVector(ulong nelem, scalar *elem) {
    return (Vector){
        .nelem = nelem,
        .elem = elem,
    };
}

Vector allocVector(ulong nelem) {
    scalar *elem = CLEAR_NET_ALLOC(nelem * sizeof(scalar));
    CLEAR_NET_ASSERT(elem);
    return formVector(nelem, elem);
}

void deallocVector(Vector *vec) {
    CLEAR_NET_DEALLOC(vec->elem);
    vec->nelem = 0;
}

void printVectorInline(Vector *vec) {
    for (ulong i = 0; i < vec->nelem; ++i) {
        printf("%f ", VEC_AT(*vec, i));
    }
}

void printVector(Vector *vec, char *name) {
    printf("%s = [\n", name);
    printf("    ");
    printVectorInline(vec);
    printf("\n]\n");
}

void setBatchFromMatrix(Matrix all_input, Matrix all_target, ulong batch_num,
                        ulong batch_size, Matrix *batch_in, Matrix *batch_tar) {
    *batch_in = formMatrix(batch_size, all_input.ncols, all_input.stride,
                           &MAT_AT(all_input, batch_num * batch_size, 0));
    *batch_tar = formMatrix(batch_size, all_target.ncols, all_target.stride,
                            &MAT_AT(all_target, batch_num * batch_size, 0));
}
