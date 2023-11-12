#include <stdio.h>
#include "clear_net.h"
#include "autodiff.h"
#include "net.h"

Matrix formMatrix(long nrows, long ncols, long stride,
                      scalar *elements) {
    return (Matrix){
        .nrows = nrows,
        .ncols = ncols,
        .stride = stride,
        .elem = elements
    };
}

// TODO make this use formMatrix
Matrix allocMatrix(ulong nrows, ulong ncols) {
    Matrix mat;
    mat.nrows = nrows;
    mat.ncols = ncols;
    mat.stride = ncols;
    mat.elem = CLEAR_NET_ALLOC(nrows * ncols * sizeof(*mat.elem));
    CLEAR_NET_ASSERT(mat.elem != NULL);
    return mat;
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

_cn_names const cn = {
    .ad = {
        .allocCompGraph = allocCompGraph,
        .deallocCompGraph = deallocCompGraph,
        .initLeafScalar = initLeafScalar,
        .getVal = getVal,
        .getGrad = getGrad,
        .add = add,
        .sub = sub,
        .mul = mul,
        .raise = raise,
        .relu = relu,
        .leakyRelu = leakyRelu,
        .htan = htan,
        .sigmoid = sigmoid,
        .elu = elu,
        .backprop = backprop,
    },
    .mat = {
        .deallocMatrix = deallocMatrix,
        .allocMatrix = allocMatrix,
        .formMatrix = formMatrix,
        .printMatrix = printMatrix,
    },
    .allocConvNet = allocConvNet,
    .allocVanillaNet = allocVanillaNet,
    .allocDenseLayer = allocDenseLayer,
    .randomizeNet = randomizeNet,
    .deallocNet = deallocNet,
    .learnVanilla = learnVanilla,
    .printNet = printNet,
};
