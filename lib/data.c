#include "clear_net.h"
#include "data.h"

CNData *allocEmptyData(void) {
    CNData *data = CLEAR_NET_ALLOC(sizeof(CNData));
    data->nelem = 0;
    data->nchannels = 0;
    return data;
}

void fillDataWithVectors(CNData *d, Vector *vectors, ulong nelem) {
    d->type = VECTORS;
    d->nelem = nelem;
    d->nchannels = 1;
    d->in.vectors = vectors;
}

CNData *allocDataFromVectors(Vector *vectors, ulong nelem) {
    CNData* data = CLEAR_NET_ALLOC(sizeof(CNData));
    fillDataWithVectors(data, vectors, nelem);
    return data;
}


void fillDataWithMatrices(CNData *d, Matrix *matrices, ulong nelem) {
    d->type = MATRICES;
    d->nelem = nelem;
    d->nchannels = 1;
    d->in.matrices = matrices;
}

CNData *allocDataFromMatrices(Matrix *matrices, ulong nelem) {
    CNData* data = CLEAR_NET_ALLOC(sizeof(CNData));
    fillDataWithMatrices(data, matrices,  nelem);
    return data;
}

void fillDataWithMultiMatrices(CNData *d, Matrix **multi_matrices, ulong nelem, ulong nchannels) {
    d->type = MULTIMATRICES;
    d->nelem = nelem;
    d->nchannels = nchannels;
    d->in.multi_matrices = multi_matrices;
}

CNData *allocDataFromMultiChannelMatrices(Matrix **multi_matrices, ulong nelem, ulong nchannels) {
    CNData* data = CLEAR_NET_ALLOC(sizeof(CNData));
    fillDataWithMultiMatrices(data, multi_matrices, nelem, nchannels);
    return data;
}

void deallocData(CNData *data) {
    switch(data->type) {
    case(VECTORS):
        deallocVectors(data->in.vectors, data->nelem);
        return;
    case(MATRICES):
        deallocMatrices(data->in.matrices, data->nelem);
        return;
    case(MULTIMATRICES):
        deallocMultiMatrices(data->in.multi_matrices, data->nelem, data->nchannels);
        return;
    }
    data->nelem = 0;
    data->nchannels = 0;
    CLEAR_NET_DEALLOC(data);
}

void printData(CNData *d) {
    switch(d->type) {
    case(VECTORS):
        printVectors(d->in.vectors, d->nelem);
        return;
    case(MATRICES):
        printMatrices(d->in.matrices, d->nelem);
        return;
    case(MULTIMATRICES):
        printMultiMatrices(d->in.multi_matrices, d->nelem, d->nchannels);
        return;
    }
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

Matrix *allocMatrices(ulong count, ulong nrows, ulong ncols) {
    Matrix *list = CLEAR_NET_ALLOC(count * sizeof(Matrix));
    for (ulong i =0; i < count; ++i) {
        list[i] = allocMatrix(nrows, ncols);
    }

    return list;
}

void deallocMatrices(Matrix *list, ulong count) {
    for (ulong i = 0; i < count; ++i) {
        deallocMatrix(&list[i]);
    }
    CLEAR_NET_DEALLOC(list);
}

void printMatrices(Matrix *list, ulong count) {
    for (ulong i = 0;  i < count; ++i) {
        printf("%zu", i);
        printMatrix(&list[i], "");
    }
}

void swapMatrices(Matrix *list, ulong one, ulong two) {
    Matrix t = list[one];
    list[one] = list[two];
    list[two] = t;
}

Matrix **allocMultiMatrices(ulong count, ulong nchannels, ulong nrows, ulong ncols) {
    Matrix **list = CLEAR_NET_ALLOC(count * sizeof(Matrix*));

    for (ulong i = 0; i < count; ++i) {
        list[i] = allocMatrices(nchannels, nrows, ncols);
    }

    return list;
}

void deallocMultiMatrices(Matrix **list, ulong count, ulong nchannels) {
    for (ulong i = 0; i < count; ++i) {
        deallocMatrices(list[i], nchannels);
    }
    CLEAR_NET_DEALLOC(list);
}

void printMultiMatrices(Matrix **list, ulong count, ulong nchannels) {
    for (ulong i = 0;  i < count; ++i) {
        printMatrices(list[i], nchannels);
    }
}

void swapMultiMatrices(Matrix **list, ulong one, ulong two) {
    Matrix *t = list[one];
    list[one] = list[two];
    list[two] = t;
}

Vector *allocVectors(ulong count, ulong nelem) {
    Vector *list = CLEAR_NET_ALLOC(count * sizeof(Vector));

    for (ulong i = 0; i < count; ++i) {
        list[i] = allocVector(nelem);
    }

    return list;
}

void deallocVectors(Vector *list, ulong count) {
    for (ulong i = 0; i < count; ++i) {
        deallocVector(&list[i]);
    }
    CLEAR_NET_DEALLOC(list);
}

void printVectors(Vector *list, ulong count) {
    for (ulong i = 0;  i < count; ++i) {
        printf("%zu", i);
        printVector(&list[i], "");
    }
}

void swapVectors(Vector *list, ulong one, ulong two) {
    Vector t = list[one];
    list[one] = list[two];
    list[two] = t;
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

void swapInData(CNData *d, ulong one, ulong two) {
    switch(d->type) {
    case(VECTORS):
        swapVectors(d->in.vectors, one, two);
        break;
    case(MATRICES):
        swapMatrices(d->in.matrices, one, two);
        break;
    case(MULTIMATRICES):
        swapMultiMatrices(d->in.multi_matrices, one, two);
        break;
    }
}

void shuffleDatas(CNData *input, CNData *target) {
    CLEAR_NET_ASSERT(input->nelem == target->nelem);
    for (ulong i = 0; i < input->nelem; ++i) {
        ulong j = i + rand() % (input->nelem - i);
        swapInData(input, i, j);
        swapInData(target, i, j);
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

void batchData(CNData *all, CNData *batch, ulong batch_num, ulong batch_size) {
    switch(all->type) {
    case(VECTORS):
        fillDataWithVectors(batch, &all->in.vectors[batch_num * batch_size], batch_size);
        return;
    case(MATRICES):
        fillDataWithMatrices(batch, &all->in.matrices[batch_num * batch_size], batch_size);
        return;
    case(MULTIMATRICES):
        fillDataWithMultiMatrices(batch, &all->in.multi_matrices[batch_num * batch_size], batch_size, all->nchannels);
        return;
    }
}

void setBatch(CNData *all_input, CNData *all_target, ulong batch_num, ulong batch_size, CNData *batch_in, CNData *batch_tar) {
    batchData(all_input, batch_in, batch_num, batch_size);
    batchData(all_target, batch_tar, batch_num, batch_size);
}
