#include "autodiff.h"
#include "clear_net.h"
#include "graph_utils.h"
#include "la.h"

// TODO to save space can not alloc for any storing index matirx as the number
// of elements between each element should be the same just store the stride
// again

typedef struct {
    Mat weights;
    Vec biases;
    Activation act;
    UVec output;
} DenseLayer;

typedef enum {
    Same,
    Valid,
    Full,
} Padding;

typedef struct {
    Mat *kernels;
    Mat biases;
} Filter;

typedef struct {
    Filter *filters;
    ulong nfilters;
    UMat *outputs;
    Padding padding;
    Activation act;
    ulong nimput;
    ulong input_nrows;
    ulong input_ncols;
    ulong output_nrows;
    ulong output_ncols;
    ulong k_nrows;
    ulong k_ncols;
} ConvolutionalLayer;

typedef enum {
    Max,
    Average,
} Pooling;

typedef struct {
    UMat *outputs;
    Pooling strat;
    ulong noutput;
    ulong k_nrows;
    ulong k_ncols;
    ulong output_nrows;
    ulong output_ncols;
} PoolingLayer;

typedef struct {
    UVec output;
    Pooling strat;
} GlobalPoolingLayer;

typedef union {
    DenseLayer dense;
    ConvolutionalLayer conv;
    PoolingLayer pool;
    GlobalPoolingLayer glob_pool;
} LayerData;

typedef enum {
    Dense,
    Conv,
    Pool,
    GlobPool,
} LayerType;

typedef struct {
    LayerType type;
    LayerData data;
} Layer;

struct Net {
    ulong nlayers;
    Layer *layers;
    CompGraph *cg;
    ulong nparams;
    UType output_type;
    UData input;
    HParams hp;
};

ulong activate(CompGraph *cg, ulong x, Activation act, scalar leaker) {
    switch (act) {
    case ReLU:
        return relu(cg, x);
    case Sigmoid:
        return sigmoid(cg, x);
    case Tanh:
        return htan(cg, x);
    case LeakyReLU:
        return leakyRelu(cg, x, leaker);
    case ELU:
        return elu(cg, x, leaker);
    }
}

void addLayer(Net *net, Layer layer) {
    net->layers = CLEAR_NET_REALLOC(net->layers,
                                    (net->nlayers + 1) * sizeof(*net->layers));
    net->layers[net->nlayers] = layer;
    net->nlayers++;
}

void allocDenseLayer(Net *net, Activation act, ulong dim_out) {
    ulong dim_input;
    if (net->nlayers != 0) {
        Layer player = net->layers[net->nlayers - 1];
        CLEAR_NET_ASSERT(player.type == Dense || player.type == GlobPool);
        if (player.type == Dense) {
            dim_input = player.data.dense.output.nelem;
        } else {
            dim_input = player.data.glob_pool.output.nelem;
        }
    } else {
        CLEAR_NET_ASSERT(net->input.type == UVector);
        dim_input = net->input.data.vec.nelem;
    }

    DenseLayer layer;
    layer.act = act;
    ulong offset = net->nparams + 1;
    layer.weights = createMat(net->cg, dim_input, dim_out, &offset);
    layer.biases = createVec(net->cg, dim_out, &offset);

    layer.output = allocUVec(dim_out);

    Layer l;
    l.type = Dense;
    l.data.dense = layer;

    addLayer(net, l);
    net->output_type = UVector;
    net->nparams = offset - 1;
}

void deallocDenseLayer(DenseLayer *layer) {
    deallocUVec(&layer->output);
    zeroMat(&layer->weights);
    zeroVec(&layer->biases);
}

void printDenseLayer(CompGraph *cg, DenseLayer *dense, size_t index) {
    printf("Layer #%zu: Dense\n", index);
    printMat(cg, &dense->weights, "weight matrix");
    printVec(cg, &dense->biases, "bias vector");
}

void randomizeDenseLayer(CompGraph *cg, DenseLayer *layer, scalar lower,
                         scalar upper) {
    randomizeMat(cg, &layer->weights, lower, upper);
    randomizeVec(cg, &layer->biases, lower, upper);
}

UVec forwardDense(CompGraph *cg, DenseLayer *layer, UVec input, scalar leaker) {
    for (ulong i = 0; i < layer->weights.ncols; ++i) {
        ulong res = initLeafScalar(cg, 0);
        for (ulong j = 0; j < input.nelem; ++j) {
            res = add(cg, res,
                      mul(cg, MAT_ID(layer->weights, j, i), VEC_AT(input, j)));
        }
        res = add(cg, res, VEC_ID(layer->biases, i));
        res = activate(cg, res, layer->act, leaker);
        VEC_AT(layer->output, i) = res;
    }
    return layer->output;
}

void applyDenseGrads(CompGraph *cg, DenseLayer *layer, HParams *hp) {
    applyMatGrads(cg, &layer->weights, hp);
    applyVecGrads(cg, &layer->biases, hp);
}

void allocConvLayer(Net *net, Padding padding, Activation act, ulong noutput,
                    ulong kernel_nrows, ulong kernel_ncols) {
    ulong input_nrows;
    ulong input_ncols;
    ulong nimput;
    if (net->nlayers == 0) {
        CLEAR_NET_ASSERT(net->input.type == UMatrix ||
                         net->input.type == UMatrixList);
        if (net->input.type == UMatrix) {
            input_nrows = net->input.data.mat.nrows;
            input_ncols = net->input.data.mat.ncols;
            nimput = 1;
        } else {
            input_nrows = net->input.data.mat_list.mats->nrows;
            input_ncols = net->input.data.mat_list.mats->ncols;
            nimput = net->input.data.mat_list.nelem;
        }
    } else {
        Layer player = net->layers[net->nlayers - 1];
        CLEAR_NET_ASSERT(player.type == Conv || player.type == Pool);
        if (player.type == Conv) {
            input_nrows = player.data.conv.output_nrows;
            input_nrows = player.data.conv.output_ncols;
            nimput = player.data.conv.nfilters;
        } else {
            input_nrows = player.data.pool.output_nrows;
            input_ncols = player.data.pool.output_ncols;
            nimput = player.data.pool.noutput;
        }
    }

    ulong output_nrows;
    ulong output_ncols;
    switch (padding) {
    case Same:
        output_nrows = input_nrows;
        output_ncols = input_ncols;
        break;
    case Full:
        output_nrows = input_nrows + kernel_nrows - 1;
        output_ncols = input_ncols + kernel_ncols - 1;
        break;
    case Valid:
        output_nrows = input_nrows - kernel_nrows + 1;
        output_ncols = input_ncols - kernel_ncols + 1;
        break;
    }

    CLEAR_NET_ASSERT(output_nrows > 0 && output_ncols > 0);

    ConvolutionalLayer layer;
    layer.nimput = nimput;
    layer.nfilters = noutput;
    layer.padding = padding;
    layer.act = act;
    layer.input_nrows = input_nrows;
    layer.input_ncols = input_ncols;
    layer.output_nrows = output_nrows;
    layer.output_ncols = output_ncols;
    layer.k_nrows = kernel_nrows;
    layer.k_ncols = kernel_ncols;

    layer.filters = CLEAR_NET_ALLOC(layer.nfilters * sizeof(*layer.filters));
    layer.outputs = CLEAR_NET_ALLOC(layer.nfilters * sizeof(*layer.outputs));

    size_t offset = net->nparams + 1;
    for (ulong i = 0; i < layer.nfilters; ++i) {
        Filter filter;
        filter.kernels = CLEAR_NET_ALLOC(nimput * sizeof(*filter.kernels));
        for (ulong j = 0; j < nimput; ++j) {
            filter.kernels[j] =
                createMat(net->cg, layer.k_nrows, layer.k_ncols, &offset);
        }
        filter.biases = createMat(net->cg, output_nrows, output_ncols, &offset);
        layer.filters[i] = filter;
        layer.outputs[i] = allocUMat(output_nrows, output_ncols);
    }
    net->nparams = offset - 1;

    Layer l;
    l.type = Conv;
    l.data.conv = layer;
    addLayer(net, l);
    net->output_type = UMatrix;
}

void deallocConvLayer(ConvolutionalLayer *layer) {
    for (ulong i = 0; i < layer->nfilters; ++i) {
        Filter *filter = &layer->filters[i];
        for (ulong j = 0; j < layer->nimput; ++j) {
            zeroMat(&filter->kernels[j]);
        }
        zeroMat(&filter->biases);
        deallocUMat(&layer->outputs[i]);
    }
    CLEAR_NET_DEALLOC(layer->outputs);
    layer->filters = NULL;
    layer->outputs = NULL;
    layer->nfilters = 0;
    layer->padding = 0;
    layer->act = 0;
    layer->nimput = 0;
    layer->input_nrows = 0;
    layer->input_ncols = 0;
    layer->output_nrows = 0;
    layer->output_ncols = 0;
    layer->k_nrows = 0;
    layer->k_ncols = 0;
}

void randomizeConvLayer(CompGraph *cg, ConvolutionalLayer *layer, scalar lower,
                        scalar upper) {
    for (ulong i = 0; i < layer->nfilters; ++i) {
        for (ulong j = 0; j < layer->nimput; ++j) {
            randomizeMat(cg, &layer->filters[i].kernels[j], lower, upper);
        }
        randomizeMat(cg, &layer->filters[i].biases, lower, upper);
    }
}

ulong correlate(CompGraph *cg, Mat kern, UMat input, ulong top_left_row,
                ulong top_left_col) {
    ulong res = initLeafScalar(cg, 0);
    ulong rows = kern.nrows;
    ulong col = kern.ncols;
    for (ulong i = 0; i < rows; ++i) {
        for (ulong j = 0; j < col; ++j) {
            ulong r = top_left_row + i;
            ulong c = top_left_col + i;
            if (r >= 0 && c >= 0 && r < input.nrows && c < input.ncols) {
                ulong val = mul(cg, MAT_AT(input, r, c), MAT_ID(kern, i, j));
                res = add(cg, res, val);
            }
        }
    }
    return res;
}

void setPadding(Padding padding, ulong k_nrows, ulong k_ncols,
                ulong *row_padding, ulong *col_padding) {
    switch (padding) {
    case Same:
        *row_padding = (k_nrows - 1) / 2;
        *col_padding = (k_ncols - 1) / 2;
        break;
    case Full:
        *row_padding = k_nrows - 1;
        *col_padding = k_ncols - 1;
        break;
    case Valid:
        *row_padding = 0;
        *col_padding = 0;
        break;
    }
}

UMat *forwardConv(CompGraph *cg, ConvolutionalLayer *layer, UMat *input,
                  scalar leaker) {
    for (ulong i = 0; i < layer->nfilters; ++i) {
        for (ulong j = 0; i < layer->output_nrows; ++j) {
            for (ulong k = 0; k < layer->output_ncols; ++k) {
                MAT_AT(layer->outputs[i], j, k) = initLeafScalar(cg, 0);
            }
        }
    }
    ulong row_padding;
    ulong col_padding;

    setPadding(layer->padding, layer->k_nrows, layer->k_ncols, &row_padding,
               &col_padding);

    for (ulong i = 0; i < layer->nimput; ++i) {
        for (ulong j = 0; j < layer->nfilters; ++j) {
            for (ulong k = 0; layer->output_nrows; ++k) {
                for (ulong l = 0; l < layer->output_ncols; ++l) {
                    ulong top_left_row = k - row_padding;
                    ulong top_left_col = l - col_padding;
                    ulong res = correlate(cg, layer->filters[j].kernels[i],
                                          input[i], top_left_row, top_left_col);
                    MAT_AT(layer->outputs[j], k, l) =
                        add(cg, MAT_AT(layer->outputs[j], k, l), res);
                }
            }
        }
    }

    for (ulong i = 0; i < layer->nfilters; ++i) {
        for (ulong j = 0; j < layer->output_nrows; ++j) {
            for (ulong k = 0; k < layer->output_ncols; ++k) {
                MAT_AT(layer->outputs[i], j, k) =
                    add(cg, MAT_ID(layer->filters[i].biases, j, k),
                        MAT_AT(layer->outputs[i], j, k));
                MAT_AT(layer->outputs[i], j, k) = activate(
                    cg, MAT_AT(layer->outputs[i], j, k), layer->act, leaker);
            }
        }
    }

    return layer->outputs;
}

void applyConvGrads(CompGraph *cg, ConvolutionalLayer *layer, HParams *hp) {
    for (ulong i = 0; i < layer->nfilters; ++i) {
        for (ulong j = 0; j < layer->nimput; ++j) {
            applyMatGrads(cg, &layer->filters[i].kernels[j], hp);
        }
        applyMatGrads(cg, &layer->filters[i].biases, hp);
    }
}

void allocPoolingLayer(Net *net, Pooling strat, ulong kernel_nrows,
                       ulong kernel_ncols) {
    CLEAR_NET_ASSERT(net->layers[net->nlayers - 1].type == Conv);
    PoolingLayer pooler;
    pooler.strat = strat;
    pooler.k_nrows = kernel_nrows;
    pooler.k_ncols = kernel_ncols;
    pooler.output_nrows =
        net->layers[net->nlayers - 1].data.conv.output_nrows / kernel_nrows;
    pooler.output_ncols =
        net->layers[net->nlayers - 1].data.conv.output_ncols / kernel_ncols;
    ulong nimput = net->layers[net->nlayers - 1].data.conv.nfilters;
    pooler.outputs = CLEAR_NET_ALLOC(nimput * sizeof(Mat));
    for (ulong i = 0; i < nimput; ++i) {
        pooler.outputs[i] = allocUMat(pooler.output_nrows, pooler.output_ncols);
    }
    pooler.noutput = nimput;

    Layer l;
    l.type = Pool;
    l.data.pool = pooler;
    addLayer(net, l);
    net->output_type = UMatrix;
}

void deallocPoolingLayer(PoolingLayer *layer) {
    layer->strat = 0;
    layer->k_nrows = 0;
    layer->k_ncols = 0;
    layer->output_nrows = 0;
    layer->output_ncols = 0;
    for (ulong i = 0; i < layer->noutput; ++i) {
        deallocUMat(&layer->outputs[i]);
    }
    CLEAR_NET_DEALLOC(layer->outputs);
    layer->noutput = 0;
}

UMat *poolLayer(CompGraph *cg, PoolingLayer *pooler, UMat *input) {
    for (ulong i = 0; i < pooler->noutput; ++i) {
        for (ulong j = 0; j < input[i].nrows; j += pooler->k_nrows) {
            for (ulong k = 0; k < input[i].ncols; k += pooler->k_ncols) {
                scalar max_store = -1 * FLT_MAX;
                ulong max_id = 0;
                ulong avg_id = initLeafScalar(cg, 0);
                ulong cur;
                ulong nelem = pooler->k_nrows * pooler->k_ncols;
                for (ulong l = 0; l < pooler->k_nrows; ++l) {
                    for (ulong m = 0; m < pooler->k_ncols; ++m) {
                        cur = MAT_AT(*input, j + l, k + m);
                        switch (pooler->strat) {
                        case (Max):
                            if (getVal(cg, cur) > max_store) {
                                max_store = getVal(cg, cur);
                                max_id = cur;
                            }
                            break;
                        case (Average):
                            avg_id = add(cg, avg_id, cur);
                        }
                    }
                }
                switch (pooler->strat) {
                case (Max):
                    MAT_AT(pooler->outputs[i], j / pooler->k_nrows,
                           k / pooler->k_ncols) = max_id;
                    break;
                case (Average): {
                    ulong coef = initLeafScalar(cg, 1 / (scalar)nelem);
                    MAT_AT(pooler->outputs[i], j / pooler->k_nrows,
                           k / pooler->k_ncols) = mul(cg, avg_id, coef);
                    break;
                }
                }
            }
        }
    }
    return pooler->outputs;
}

void createGlobalPoolingLayer(Net *net, Pooling strat) {
    CLEAR_NET_ASSERT(net->nlayers > 0);
    CLEAR_NET_ASSERT(net->layers[net->nlayers - 1].type == Conv ||
                     net->layers[net->nlayers - 1].type == Pool);

    // TODO this doesn't handle case where previous layer is Pooling
    GlobalPoolingLayer gpooler = (GlobalPoolingLayer){
        .strat = strat,
        .output = allocUVec(net->layers[net->nlayers - 1].data.conv.nfilters),
    };
    Layer l;
    l.type = GlobPool;
    l.data.glob_pool = gpooler;
    addLayer(net, l);
    net->output_type = UVector;
}

void deallocGlobalPoolingLayer(GlobalPoolingLayer *layer) {
    deallocUVec(&layer->output);
    layer->strat = 0;
}

UVec globalPoolLayer(CompGraph *cg, GlobalPoolingLayer *pooler, UMat *input) {
    for (ulong i = 0; i < pooler->output.nelem; ++i) {
        scalar max_store = -1 * FLT_MAX;
        ulong max_id;
        ulong avg_id = initLeafScalar(cg, 0);
        ulong nelem = input[i].nrows * input[i].ncols;
        for (ulong j = 0; j < input[i].nrows; ++j) {
            for (ulong k = 0; k < input[i].ncols; ++k) {
                ulong cur = MAT_AT(input[i], j, k);
                switch (pooler->strat) {
                case (Max):
                    if (getVal(cg, cur) > max_store) {
                        max_store = getVal(cg, cur);
                        max_id = cur;
                    }
                    break;
                case (Average):
                    avg_id = add(cg, avg_id, cur);
                    break;
                }
            }
        }
        switch (pooler->strat) {
        case (Max):
            VEC_AT(pooler->output, i) = max_id;
            break;
        case (Average): {
            ulong coef = initLeafScalar(cg, 1 / (scalar)nelem);
            VEC_AT(pooler->output, i) = mul(cg, avg_id, coef);
            break;
        }
        }
    }
    return pooler->output;
}

HParams defaultHParams(void) {
    return (HParams){
        .rate = 0.1,
        .leaker = 0.1,
    };
}

void setRate(HParams *hp, scalar rate) { hp->rate = rate; }

void setLeaker(HParams *hp, scalar leaker) { hp->leaker = leaker; }

void withMomentum(HParams *hp, scalar beta) {
    hp->momentum = true;
    hp->beta = beta;
}

Net *allocNet(HParams hp) {
    Net *net = CLEAR_NET_ALLOC(sizeof(Net));
    net->layers = NULL;
    net->nlayers = 0;
    net->nparams = 0;
    net->hp = hp;
    net->cg = allocCompGraph(0);
    return net;
}

Net *allocVanillaNet(HParams hp, ulong input_nelem) {
    CLEAR_NET_ASSERT(input_nelem != 0);
    UData input;
    input.type = UVector;
    input.data.vec = allocUVec(input_nelem);
    Net *net = allocNet(hp);
    net->input = input;
    return net;
}

Net *allocConvNet(HParams hp, ulong input_nrows, ulong input_ncols,
                  ulong nchannels) {
    CLEAR_NET_ASSERT(input_nrows != 0 && input_ncols != 0 && nchannels != 0);
    UData input;
    if (nchannels == 0) {
        input.type = UMatrix;
        input.data.mat = allocUMat(input_nrows, input_ncols);
    } else {
        input.type = UMatrixList;
        input.data.mat_list =
            allocUMatList(input_nrows, input_ncols, nchannels);
    }

    Net *net = allocNet(hp);
    net->input = input;
    return net;
}

void printNet(Net *net, char *name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < net->nlayers; ++i) {
        Layer layer = net->layers[i];
        switch (layer.type) {
        case (Dense):
            printDenseLayer(net->cg, &layer.data.dense, i);
            break;
        case (Conv):
            // TODO
            break;
        case (Pool):
            // TODO
            break;
        case (GlobPool):
            // TODO
            break;
        }
    }
    printf("]\n");
}

void deallocNet(Net *net) {
    for (ulong i = 0; i < net->nlayers; ++i) {
        Layer layer = net->layers[i];
        switch (layer.type) {
        case (Dense):
            deallocDenseLayer(&layer.data.dense);
            break;
        case (Conv):
            deallocConvLayer(&layer.data.conv);
            break;
        case (Pool):
            deallocPoolingLayer(&layer.data.pool);
            break;
        case (GlobPool):
            deallocGlobalPoolingLayer(&layer.data.glob_pool);
            break;
        }
    }
    deallocCompGraph(net->cg);
    CLEAR_NET_DEALLOC(net->layers);
    deallocUData(&net->input);
    CLEAR_NET_DEALLOC(net);
}

void randomizeNet(Net *net, scalar lower, scalar upper) {
    for (ulong i = 0; i < net->nlayers; ++i) {
        Layer layer = net->layers[i];
        if (layer.type == Dense) {
            randomizeDenseLayer(net->cg, &layer.data.dense, lower, upper);
        } else if (layer.type == Conv) {
            randomizeConvLayer(net->cg, &layer.data.conv, lower, upper);
        }
    }
}

UVec _predictDense(Net *net, CompGraph *cg, UVec prev) {
    for (ulong i = 0; i < net->nlayers; ++i) {
        prev =
            forwardDense(cg, &net->layers[i].data.dense, prev, net->hp.leaker);
    }
    return prev;
}

Vector *predictDense(Net *net, Vector input, Vector *store) {
    CLEAR_NET_ASSERT(net->input.data.vec.nelem == input.nelem);
    CompGraph *cg = net->cg;

    for (ulong i = 0; i < input.nelem; ++i) {
        VEC_AT(net->input.data.vec, i) = getSize(cg);
        initLeafScalar(cg, VEC_AT(input, i));
    }
    UVec prediction = _predictDense(net, cg, net->input.data.vec);
    CLEAR_NET_ASSERT(store->nelem == prediction.nelem);
    for (ulong i = 0; i < store->nelem; ++i) {
        VEC_AT(*store, i) = getVal(cg, VEC_AT(prediction, i));
    }

    return store;
}

void applyNetGrads(CompGraph *cg, Net *net) {
    for (size_t i = 0; i < net->nlayers; ++i) {
        if (net->layers[i].type == Dense) {
            applyDenseGrads(cg, &net->layers[i].data.dense, &net->hp);
        } else if (net->layers[i].type == Conv) {
            applyConvGrads(cg, &net->layers[i].data.conv, &net->hp);
        }
    }
}

scalar learnVanilla(Net *net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    CLEAR_NET_ASSERT(net->input.data.vec.nelem == input.ncols);
    ulong train_size = input.nrows;
    CompGraph *cg = net->cg;
    setSize(cg, net->nparams + 1);

    scalar total_loss = 0;
    for (ulong i = 0; i < input.ncols; ++i) {
        VEC_AT(net->input.data.vec, i) = getSize(cg) + i;
    }

    Vec target_vec;
    target_vec.nelem = target.ncols;
    for (ulong i = 0; i < train_size; ++i) {
        for (ulong j = 0; j < input.ncols; ++j) {
            initLeafScalar(cg, MAT_AT(input, i, j));
        }
        UVec prediction = _predictDense(net, cg, net->input.data.vec);
        target_vec.start_id = getSize(cg);

        for (ulong j = 0; j < target.ncols; ++j) {
            initLeafScalar(cg, MAT_AT(target, i, j));
        }
        // start raising and stuff
        ulong raiser = initLeafScalar(cg, 2);
        ulong loss = initLeafScalar(cg, 0);
        for (ulong j = 0; j < target_vec.nelem; ++j) {
            loss = add(
                cg, loss,
                raise(cg, sub(cg, VEC_AT(prediction, j), VEC_ID(target_vec, j)),
                      raiser));
        }
        backprop(cg, loss, net->hp.leaker);
        total_loss += getVal(cg, loss);
        setSize(cg, net->nparams + 1);
    }
    applyNetGrads(cg, net);
    resetGrads(cg, net->nparams);

    return total_loss / train_size;
}

void printVanillaPredictions(Net *net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    printf("Input | Net Output | Target \n");
    scalar loss = 0;

    Vector out = allocVector(target.ncols);

    for (ulong i = 0; i < input.nrows; ++i) {
        Vector in = formVector(input.ncols, &MAT_AT(input, i, 0));
        predictDense(net, in, &out);
        Vector tar = formVector(target.ncols, &MAT_AT(target, i, 0));
        for (size_t j = 0; j < out.nelem; ++j) {
            loss += powf(VEC_AT(out, j) - VEC_AT(tar, j), 2);
        }
        printVectorInline(&in);
        printf("| ");
        printVectorInline(&out);
        printf("| ");
        printVectorInline(&tar);
        printf("\n");
    }
    printf("Average Loss: %f\n", loss / input.nrows);
    deallocVector(&out);
}
