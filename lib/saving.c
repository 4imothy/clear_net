#include "saving.h"

#define FWRITE(ptr, nitems, fp) fwrite(ptr, sizeof(*ptr), nitems, fp);
#define FREAD(ptr, nitems, fp) fread(ptr, sizeof(*ptr), nitems, fp);

void saveHParams(HParams *hp, FILE *fp) {
    FWRITE(&hp->rate, 1, fp);
    FWRITE(&hp->leaker, 1, fp);
    FWRITE(&hp->beta, 1, fp);
    FWRITE(&hp->momentum, 1, fp);
}

HParams *allocHParamsFromFile(FILE *fp) {
    HParams *hp = CLEAR_NET_ALLOC(sizeof(HParams));
    FREAD(&hp->rate, 1, fp);
    FREAD(&hp->leaker, 1, fp);
    FREAD(&hp->beta, 1, fp);
    FREAD(&hp->momentum, 1, fp);
    return hp;
}

void saveMat(CompGraph *cg, Mat *mat, FILE *fp) {
    for (ulong i = 0; i < mat->nrows; ++i) {
        for (ulong j = 0; j < mat->ncols; ++j) {
            scalar val = getVal(cg, MAT_ID(*mat, i, j));
            FWRITE(&val, 1, fp);
        }
    }
}

void loadMatFromFile(CompGraph *cg, Mat *mat, FILE *fp) {
    for (ulong i = 0; i < mat->nrows; ++i) {
        for (ulong j = 0; j < mat->ncols; ++j) {
            scalar val;
            FREAD(&val, 1, fp);
            setVal(cg, MAT_ID(*mat, i, j), val);
        }
    }
}

void saveVec(CompGraph *cg, Vec *vec, FILE *fp) {
    for (ulong i = 0; i < vec->nelem; ++i) {
        scalar val = getVal(cg, VEC_ID(*vec, i));
        FWRITE(&val, 1, fp);
    }
}

void loadVecFromFile(CompGraph *cg, Vec *vec, FILE *fp) {
    for (ulong i = 0; i < vec->nelem; ++i) {
        scalar val;
        FREAD(&val, 1, fp);
        setVal(cg, VEC_ID(*vec, i), val);
    }
}

void saveDenseLayer(CompGraph *cg, DenseLayer *layer, FILE *fp) {
    FWRITE(&layer->act, 1, fp);
    FWRITE(&layer->weights.ncols, 1, fp);
    saveMat(cg, &layer->weights, fp);
    saveVec(cg, &layer->biases, fp);
}

void allocDenseLayerFromFile(Net *net, FILE *fp) {
    Activation act;
    FREAD(&act, 1, fp);
    ulong out_dim;
    FREAD(&out_dim, 1, fp);
    allocDenseLayer(net, act, out_dim);
    loadMatFromFile(net->cg, &net->layers[net->nlayers - 1].in.dense.weights,
                    fp);
    loadVecFromFile(net->cg, &net->layers[net->nlayers - 1].in.dense.biases,
                    fp);
}

void saveConvLayer(CompGraph *cg, ConvolutionalLayer *layer, FILE *fp) {
    FWRITE(&layer->act, 1, fp);
    FWRITE(&layer->padding, 1, fp);
    FWRITE(&layer->nfilters, 1, fp);
    FWRITE(&layer->k_nrows, 1, fp);
    FWRITE(&layer->k_ncols, 1, fp);
    for (ulong i = 0; i < layer->nfilters; ++i) {
        for (ulong j = 0; j < layer->nimput; ++j) {
            saveMat(cg, &layer->filters[i].kernels[j], fp);
        }
        saveMat(cg, &layer->filters[i].biases, fp);
    }
}

void allocConvLayerFromFile(Net *net, FILE *fp) {
    Activation act;
    FREAD(&act, 1, fp);
    Padding padding;
    FREAD(&padding, 1, fp);
    ulong nfilters;
    FREAD(&nfilters, 1, fp);
    ulong k_nrows;
    FREAD(&k_nrows, 1, fp);
    ulong k_ncols;
    FREAD(&k_ncols, 1, fp);

    allocConvLayer(net, act, padding, nfilters, k_nrows, k_ncols);
    ConvolutionalLayer *layer = &net->layers[net->nlayers - 1].in.conv;
    for (ulong i= 0 ; i < layer->nfilters; ++i) {
        for (ulong j = 0; j < layer->nimput; ++j) {
            loadMatFromFile(net->cg, &layer->filters[i].kernels[j], fp);
        }
        loadMatFromFile(net->cg, &layer->filters[i].biases, fp);
    }

}

void savePoolingLayer(PoolingLayer *pool, FILE *fp) {
    FWRITE(&pool->strat, 1, fp);
    FWRITE(&pool->k_nrows, 1, fp);
    FWRITE(&pool->k_ncols, 1, fp);
}

void allocPoolingLayerFromFile(Net *net, FILE *fp) {
    Pooling strat;
    FREAD(&strat, 1, fp);
    ulong k_nrows;
    FREAD(&k_nrows, 1, fp);
    ulong k_ncols;
    FREAD(&k_ncols, 1, fp);
    allocPoolingLayer(net, strat, k_nrows, k_ncols);
}

void saveGlobalPoolingLayer(GlobalPoolingLayer *layer, FILE *fp) {
    FWRITE(&layer->strat, 1, fp);
}

void allocGlobalPoolingLayerFromFile(Net *net, FILE *fp) {
    Pooling strat;
    FREAD(&strat, 1, fp);
    allocGlobalPoolingLayer(net, strat);
}

void saveNet(Net *net, char *path) {
    FILE *fp = fopen(path, "wb");
    FWRITE(&net->input.type, 1, fp);
    FWRITE(&net->nlayers, 1, fp);
    saveHParams(&net->hp, fp);

    switch (net->input.type) {
    case (UVEC): // vanilla net
        FWRITE(&net->input.in.vec.nelem, 1, fp);
        break;
    case (UMAT): // convolutional net
        FWRITE(&net->input.in.mat.nrows, 1, fp);
        FWRITE(&net->input.in.mat.ncols, 1, fp);
        break;
    case (UMATLIST): // convolutional net with many channels to start
        FWRITE(&net->input.nchannels, 1, fp);
        FWRITE(&net->input.in.mats->nrows, 1, fp);
        FWRITE(&net->input.in.mats->ncols, 1, fp);
        break;
    }

    for (ulong i = 0; i < net->nlayers; ++i) {
        FWRITE(&net->layers[i].type, 1, fp);
        switch (net->layers[i].type) {
        case (DENSE):
            saveDenseLayer(net->cg, &net->layers[i].in.dense, fp);
            break;
        case (CONV):
            saveConvLayer(net->cg, &net->layers[i].in.conv, fp);
            break;
        case (POOL):
            savePoolingLayer(&net->layers[i].in.pool, fp);
            break;
        case (GLOBPOOL):
            saveGlobalPoolingLayer(&net->layers[i].in.glob_pool, fp);
            break;
        }
    }
    fclose(fp);
}

Net *allocNetFromFile(char *path) {
    FILE *fp = fopen(path, "rb");
    UType in_type;
    FREAD(&in_type, 1, fp);
    ulong nlayers;
    FREAD(&nlayers, 1, fp);
    HParams *hp = allocHParamsFromFile(fp);
    Net *net;
    switch (in_type) {
    case (UVEC): {
        ulong in_dim;
        FREAD(&in_dim, 1, fp);
        net = allocVanillaNet(hp, in_dim);
        break;
    }
    case (UMAT): {
        ulong in_nrows;
        ulong in_ncols;
        FREAD(&in_nrows, 1, fp);
        FREAD(&in_ncols, 1, fp);
        net = allocConvNet(hp, in_nrows, in_ncols, 1);
        break;
    }
    case (UMATLIST): {
        ulong nchannels;
        ulong in_nrows;
        ulong in_ncols;
        FREAD(&nchannels, 1, fp);
        FREAD(&in_nrows, 1, fp);
        FREAD(&in_ncols, 1, fp);
        net = allocConvNet(hp, in_nrows, in_ncols, nchannels);
        break;
    }
    }

    LayerType type;
    for (ulong i = 0; i < nlayers; ++i) {
        FREAD(&type, 1, fp);
        switch (type) {
        case (DENSE):
            allocDenseLayerFromFile(net, fp);
            break;
        case (CONV):
            allocConvLayerFromFile(net, fp);
            break;
        case (POOL):
            allocPoolingLayerFromFile(net, fp);
            break;
        case (GLOBPOOL):
            allocGlobalPoolingLayerFromFile(net, fp);
            break;
        }
    }

    fclose(fp);

    return net;
}
