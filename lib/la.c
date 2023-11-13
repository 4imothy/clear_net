#include "clear_net.h"

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
    printf("start\n");
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
