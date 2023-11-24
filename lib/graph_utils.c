#include "graph_utils.h"
#include "autodiff.h"
#include "clear_net.h"

Mat createMat(CompGraph *cg, ulong nrows, ulong ncols, ulong *offset) {
    Mat mat = (Mat){
        .start_id = *offset,
        .nrows = nrows,
        .ncols = ncols,
    };
    for (ulong i = 0; i < nrows; ++i) {
        for (ulong j = 0; j < ncols; ++j) {
            initLeafScalar(cg, 0);
        }
    }
    *offset += nrows * ncols;
    return mat;
}

void zeroMat(Mat *mat) {
    mat->nrows = 0;
    mat->ncols = 0;
    mat->start_id = 0;
}

void printMat(CompGraph *cg, Mat *mat, char *name) {
    printf("%s = [\n", name);
    for (ulong i = 0; i < mat->nrows; ++i) {
        printf("    ");
        for (ulong j = 0; j < mat->ncols; ++j) {
            printf("%f ", getVal(cg, MAT_ID(*mat, i, j)));
        }
        printf("\n");
    }
    printf("]\n");
}

void randomizeMat(CompGraph *cg, Mat *mat, scalar lower, scalar upper) {
    for (ulong i = 0; i < mat->nrows; ++i) {
        for (ulong j = 0; j < mat->ncols; ++j) {
            setValRand(cg, MAT_ID(*mat, i, j), lower, upper);
        }
    }
}

void applyMatGrads(CompGraph *cg, Mat *mat, HParams *hp) {
    for (ulong i = 0; i < mat->nrows; ++i) {
        for (ulong j = 0; j < mat->ncols; ++j) {
            // printf("before: %f\n", getVal(cg, MAT_ID(*mat, i, j)));
            _applyGrad(cg, MAT_ID(*mat, i, j), hp);
            // printf("after: %f\n", getVal(cg, MAT_ID(*mat, i, j)));
        }
    }
}

Vec createVec(CompGraph *cg, ulong nelem, ulong *offset) {
    Vec vec = (Vec){
        .nelem = nelem,
        .start_id = *offset,
    };
    for (ulong i = 0; i < nelem; ++i) {
        initLeafScalar(cg, 0);
    }
    *offset += nelem;
    return vec;
}

void zeroVec(Vec *vec) {
    vec->nelem = 0;
    vec->start_id = 0;
}

void printVec(CompGraph *cg, Vec *vec, char *name) {
    printf("%s = [\n", name);
    printf("    ");
    for (ulong i = 0; i < vec->nelem; ++i) {
        printf("%f ", getVal(cg, VEC_ID(*vec, i)));
    }
    printf("\n]\n");
}

void randomizeVec(CompGraph *cg, Vec *vec, scalar lower, scalar upper) {
    for (ulong i = 0; i < vec->nelem; ++i) {
        setValRand(cg, VEC_ID(*vec, i), lower, upper);
    }
}

void applyVecGrads(CompGraph *cg, Vec *vec, HParams *hp) {
    for (ulong i = 0; i < vec->nelem; ++i) {
        _applyGrad(cg, VEC_ID(*vec, i), hp);
    }
}

UMat allocUMat(ulong nrows, ulong ncols) {
    UMat umat;
    umat.nrows = nrows;
    umat.ncols = ncols;
    umat.stride = ncols;
    umat.elem = CLEAR_NET_ALLOC(nrows * ncols * sizeof(*umat.elem));
    CLEAR_NET_ASSERT(umat.elem != NULL);
    return umat;
}

void deallocUMat(UMat *umat) {
    CLEAR_NET_DEALLOC(umat->elem);
    umat->elem = NULL;
    umat->nrows = 0;
    umat->ncols = 0;
}

UVec allocUVec(ulong nelem) {
    UVec uvec;
    uvec.nelem = nelem;
    uvec.elem = CLEAR_NET_ALLOC(nelem * sizeof(*uvec.elem));
    CLEAR_NET_ASSERT(uvec.elem != NULL);
    return uvec;
}

void deallocUVec(UVec *uvec) {
    CLEAR_NET_DEALLOC(uvec->elem);
    uvec->nelem = 0;
    uvec->elem = NULL;
}

UMatList allocUMatList(ulong nrows, ulong ncols, ulong nchannels) {
    UMatList list;
    list.nelem = nchannels;
    list.mats = CLEAR_NET_ALLOC(nchannels * sizeof(UMat));
    CLEAR_NET_ASSERT(list.mats != NULL);
    for (ulong i = 0; i < nchannels; ++i) {
        list.mats[i] = allocUMat(nrows, ncols);
    }

    return list;
}

void deallocUMatList(UMatList *list) {
    for (ulong i = 0; i < list->nelem; ++i) {
        deallocUMat(&list->mats[i]);
    }
    CLEAR_NET_DEALLOC(list->mats);
    list->nelem = 0;
}

void deallocUData(UData *data) {
    switch (data->type) {
    case (UVector):
        deallocUVec(&data->data.vec);
        break;
    case (UMatrix):
        deallocUMat(&data->data.mat);
        break;
    case (UMatrixList):
        deallocUMatList(&data->data.mat_list);
        break;
    }
}
