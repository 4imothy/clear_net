#include "net.h"
#include "autodiff.h"
#include "data.h"
#include "graph_utils.h"
#include "net_types.h"
#include <float.h>

// FUTURE to save space can not alloc for any storing index matirx as the number
// of elements between each element should be the same just store the stride
// again
// TODO input can always be a list of matrices indexed the normal way, indexing at 0 would be same as input.mat

ulong activate(CompGraph *cg, ulong x, Activation act, scalar leaker) {
    switch (act) {
    case RELU:
        return relu(cg, x);
    case SIGMOID:
        return sigmoid(cg, x);
    case TANH:
        return htan(cg, x);
    case LEAKYRELU:
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
        CLEAR_NET_ASSERT(player.type == DENSE || player.type == GLOBPOOL);
        if (player.type == DENSE) {
            dim_input = player.in.dense.output.nelem;
        } else {
            dim_input = player.in.glob_pool.output.nelem;
        }
    } else {
        CLEAR_NET_ASSERT(net->input.type == UVEC);
        dim_input = net->input.in.vec.nelem;
    }

    DenseLayer layer;
    layer.act = act;
    ulong offset = net->nparams + 1;
    layer.weights = createMat(net->cg, dim_input, dim_out, &offset);
    layer.biases = createVec(net->cg, dim_out, &offset);

    layer.output = allocUVec(dim_out);

    Layer l;
    l.type = DENSE;
    l.in.dense = layer;

    addLayer(net, l);
    net->output_type = UVEC;
    net->nparams = offset - 1;
}

void deallocDenseLayer(DenseLayer *layer) {
    deallocUVec(&layer->output);
    zeroMat(&layer->weights);
    zeroVec(&layer->biases);
}

void printDenseLayer(CompGraph *cg, DenseLayer *dense, ulong index) {
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
    applyMatGrads(cg, &layer->weights, hp->rate, hp->momentum, hp->beta);
    applyVecGrads(cg, &layer->biases, hp->rate, hp->momentum, hp->beta);
}

void allocConvLayer(Net *net, Activation act, Padding padding, ulong noutput,
                    ulong kernel_nrows, ulong kernel_ncols) {
    ulong input_nrows;
    ulong input_ncols;
    ulong nimput;
    if (net->nlayers == 0) {
        CLEAR_NET_ASSERT(net->input.type == UMAT ||
                         net->input.type == UMATLIST);
        nimput = net->input.nchannels;
        if (net->input.type == UMAT) {
            input_nrows = net->input.in.mat.nrows;
            input_ncols = net->input.in.mat.ncols;
        } else {
            input_nrows = net->input.in.mats->nrows;
            input_ncols = net->input.in.mats->ncols;
        }
    } else {
        Layer player = net->layers[net->nlayers - 1];
        CLEAR_NET_ASSERT(player.type == CONV || player.type == POOL);
        if (player.type == CONV) {
            input_nrows = player.in.conv.output_nrows;
            input_ncols = player.in.conv.output_ncols;
            nimput = player.in.conv.nfilters;
        } else {
            input_nrows = player.in.pool.output_nrows;
            input_ncols = player.in.pool.output_ncols;
            nimput = player.in.pool.noutput;
        }
    }

    ulong output_nrows;
    ulong output_ncols;
    switch (padding) {
    case SAME:
        output_nrows = input_nrows;
        output_ncols = input_ncols;
        break;
    case FULL:
        output_nrows = input_nrows + kernel_nrows - 1;
        output_ncols = input_ncols + kernel_ncols - 1;
        break;
    case VALID:
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

    ulong offset = net->nparams + 1;
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
    l.type = CONV;
    l.in.conv = layer;
    addLayer(net, l);
    net->output_type = UMAT;
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

void printConvLayer(CompGraph *cg, ConvolutionalLayer *layer, ulong id) {
    printf("Layer #%zu Convolutional\n", id);
    for (ulong i = 0; i < layer->nfilters; ++i) {
        for (ulong j = 0; j < layer->nimput; ++j) {
            printf("Kernel #%zu", j);
            printMat(cg, &layer->filters[i].kernels[j], "");
        }
        printMat(cg, &layer->filters[i].biases, "filter bias");
    }
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

ulong correlate(CompGraph *cg, Mat kern, UMat input, long top_left_row,
                long top_left_col) {
    ulong res = initLeafScalar(cg, 0);
    for (long i = 0; i < (long)kern.nrows; ++i) {
        for (long j = 0; j < (long)kern.ncols; ++j) {
            long r = top_left_row + i;
            long c = top_left_col + j;
            if (r >= 0 && c >= 0 && r < (long)input.nrows &&
                c < (long)input.ncols) {
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
    case SAME:
        *row_padding = (k_nrows - 1) / 2;
        *col_padding = (k_ncols - 1) / 2;
        break;
    case FULL:
        *row_padding = k_nrows - 1;
        *col_padding = k_ncols - 1;
        break;
    case VALID:
        *row_padding = 0;
        *col_padding = 0;
        break;
    }
}

UMat *forwardConv(CompGraph *cg, ConvolutionalLayer *layer, UMat *input,
                  scalar leaker) {
    for (ulong i = 0; i < layer->nfilters; ++i) {
        for (ulong j = 0; j < layer->output_nrows; ++j) {
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
            for (long k = 0; k < (long)layer->output_nrows; ++k) {
                for (long l = 0; l < (long)layer->output_ncols; ++l) {
                    long top_left_row = k - row_padding;
                    long top_left_col = l - col_padding;
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
            applyMatGrads(cg, &layer->filters[i].kernels[j], hp->rate,
                          hp->momentum, hp->beta);
        }
        applyMatGrads(cg, &layer->filters[i].biases, hp->rate, hp->momentum,
                      hp->beta);
    }
}

void allocPoolingLayer(Net *net, Pooling strat, ulong kernel_nrows,
                       ulong kernel_ncols) {
    CLEAR_NET_ASSERT(net->layers[net->nlayers - 1].type == CONV);
    PoolingLayer pooler;
    pooler.strat = strat;
    pooler.k_nrows = kernel_nrows;
    pooler.k_ncols = kernel_ncols;
    pooler.output_nrows =
        net->layers[net->nlayers - 1].in.conv.output_nrows / kernel_nrows;
    pooler.output_ncols =
        net->layers[net->nlayers - 1].in.conv.output_ncols / kernel_ncols;
    pooler.noutput = net->layers[net->nlayers - 1].in.conv.nfilters;
    pooler.outputs = CLEAR_NET_ALLOC(pooler.noutput * sizeof(UMat));
    for (ulong i = 0; i < pooler.noutput; ++i) {
        pooler.outputs[i] = allocUMat(pooler.output_nrows, pooler.output_ncols);
    }

    Layer l;
    l.type = POOL;
    l.in.pool = pooler;
    addLayer(net, l);
    net->output_type = UMAT;
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

void printPoolingLayer(PoolingLayer *layer, ulong id) {
    printf("Layer #%zu ", id);
    switch (layer->strat) {
    case (MAX):
        printf("Max ");
        break;
    case (AVERAGE):
        printf("Average ");
        break;
    }
    printf("Pooling Layer\n");
}

UMat *forwardPool(CompGraph *cg, PoolingLayer *pooler, UMat *input) {
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
                        cur = MAT_AT(input[i], j + l, k + m);
                        switch (pooler->strat) {
                        case (MAX):
                            if (getVal(cg, cur) > max_store) {
                                max_store = getVal(cg, cur);
                                max_id = cur;
                            }
                            break;
                        case (AVERAGE):
                            avg_id = add(cg, avg_id, cur);
                        }
                    }
                }
                switch (pooler->strat) {
                case (MAX):
                    MAT_AT(pooler->outputs[i], j / pooler->k_nrows,
                           k / pooler->k_ncols) = max_id;
                    break;
                case (AVERAGE): {
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

void allocGlobalPoolingLayer(Net *net, Pooling strat) {
    CLEAR_NET_ASSERT(net->nlayers > 0);
    CLEAR_NET_ASSERT(net->layers[net->nlayers - 1].type == CONV ||
                     net->layers[net->nlayers - 1].type == POOL);

    GlobalPoolingLayer gpooler;
    if (net->layers[net->nlayers - 1].type == CONV) {
        gpooler = (GlobalPoolingLayer){
            .strat = strat,
            .output =
                allocUVec(net->layers[net->nlayers - 1].in.conv.nfilters)};
    } else {
        gpooler = (GlobalPoolingLayer){
            .strat = strat,
            .output = allocUVec(net->layers[net->nlayers - 1].in.pool.noutput),
        };
    }
    Layer l;
    l.type = GLOBPOOL;
    l.in.glob_pool = gpooler;
    addLayer(net, l);
    net->output_type = UVEC;
}

void deallocGlobalPoolingLayer(GlobalPoolingLayer *layer) {
    deallocUVec(&layer->output);
    layer->strat = 0;
}

void printGlobalPoolingLayer(GlobalPoolingLayer *layer, ulong id) {
    printf("Layer #%zu: ", id);
    switch (layer->strat) {
    case (MAX):
        printf("Max ");
        break;
    case (AVERAGE):
        printf("Average ");
        break;
    }
    printf("Global Pooling Layer\n");
}

UVec forwardGlobPool(CompGraph *cg, GlobalPoolingLayer *pooler, UMat *input) {
    for (ulong i = 0; i < pooler->output.nelem; ++i) {
        scalar max_store = -1 * FLT_MAX;
        ulong max_id;
        ulong avg_id = initLeafScalar(cg, 0);
        ulong nelem = input[i].nrows * input[i].ncols;
        for (ulong j = 0; j < input[i].nrows; ++j) {
            for (ulong k = 0; k < input[i].ncols; ++k) {
                ulong cur = MAT_AT(input[i], j, k);
                switch (pooler->strat) {
                case (MAX):
                    if (getVal(cg, cur) > max_store) {
                        max_store = getVal(cg, cur);
                        max_id = cur;
                    }
                    break;
                case (AVERAGE):
                    avg_id = add(cg, avg_id, cur);
                    break;
                }
            }
        }
        switch (pooler->strat) {
        case (MAX):
            VEC_AT(pooler->output, i) = max_id;
            break;
        case (AVERAGE): {
            ulong coef = initLeafScalar(cg, 1 / (scalar)nelem);
            VEC_AT(pooler->output, i) = mul(cg, avg_id, coef);
            break;
        }
        }
    }
    return pooler->output;
}

HParams *allocDefaultHParams(void) {
    HParams *hp = CLEAR_NET_ALLOC(sizeof(HParams));
    hp->rate = 0.1;
    hp->leaker = 0.1;
    hp->momentum = false;
    hp->beta = 0.9;
    return hp;
}

void setRate(HParams *hp, scalar rate) { hp->rate = rate; }

void setLeaker(HParams *hp, scalar leaker) { hp->leaker = leaker; }

void withMomentum(HParams *hp, scalar beta) {
    hp->momentum = true;
    hp->beta = beta;
}

Net *allocNet(HParams *hp) {
    Net *net = CLEAR_NET_ALLOC(sizeof(Net));
    net->layers = NULL;
    net->nlayers = 0;
    net->nparams = 0;
    net->hp = *hp;
    net->cg = allocCompGraph(0);
    return net;
}

Net *allocVanillaNet(HParams *hp, ulong input_nelem) {
    CLEAR_NET_ASSERT(input_nelem != 0);
    UData input;
    input.type = UVEC;
    input.in.vec = allocUVec(input_nelem);
    Net *net = allocNet(hp);
    net->input = input;
    return net;
}

Net *allocConvNet(HParams *hp, ulong input_nrows, ulong input_ncols,
                  ulong nchannels) {
    CLEAR_NET_ASSERT(input_nrows != 0 && input_ncols != 0 && nchannels != 0);
    UData input;
    input.nchannels = nchannels;
    if (input.nchannels == 1) {
        input.type = UMAT;
        input.in.mat = allocUMat(input_nrows, input_ncols);
    } else {
        input.type = UMATLIST;
        input.in.mats = allocUMatList(input_nrows, input_ncols, nchannels);
    }

    Net *net = allocNet(hp);
    net->input = input;
    return net;
}

void printNet(Net *net, char *name) {
    printf("%s = [\n", name);
    for (ulong i = 0; i < net->nlayers; ++i) {
        Layer layer = net->layers[i];
        switch (layer.type) {
        case (DENSE):
            printDenseLayer(net->cg, &layer.in.dense, i);
            break;
        case (CONV):
            printConvLayer(net->cg, &layer.in.conv, i);
            break;
        case (POOL):
            printPoolingLayer(&layer.in.pool, i);
            break;
        case (GLOBPOOL):
            printGlobalPoolingLayer(&layer.in.glob_pool, i);
            break;
        }
    }
    printf("]\n");
}

void deallocNet(Net *net) {
    for (ulong i = 0; i < net->nlayers; ++i) {
        Layer layer = net->layers[i];
        switch (layer.type) {
        case (DENSE):
            deallocDenseLayer(&layer.in.dense);
            break;
        case (CONV):
            deallocConvLayer(&layer.in.conv);
            break;
        case (POOL):
            deallocPoolingLayer(&layer.in.pool);
            break;
        case (GLOBPOOL):
            deallocGlobalPoolingLayer(&layer.in.glob_pool);
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
        if (layer.type == DENSE) {
            randomizeDenseLayer(net->cg, &layer.in.dense, lower, upper);
        } else if (layer.type == CONV) {
            randomizeConvLayer(net->cg, &layer.in.conv, lower, upper);
        }
    }
}

UVec _predictVanilla(Net *net, UVec prev) {
    for (ulong i = 0; i < net->nlayers; ++i) {
        prev = forwardDense(net->cg, &net->layers[i].in.dense, prev,
                            net->hp.leaker);
    }
    return prev;
}

UData _predictConv(Net *net, UMat *prevs) {
    UVec uvec;
    scalar leaker = net->hp.leaker;
    CompGraph *cg = net->cg;
    for (ulong i = 0; i < net->nlayers; ++i) {
        Layer layer = net->layers[i];
        switch (layer.type) {
        case (DENSE):
            uvec = forwardDense(cg, &layer.in.dense, uvec, leaker);
            break;
        case (CONV):
            prevs = forwardConv(cg, &layer.in.conv, prevs, leaker);
            break;
        case (POOL):
            prevs = forwardPool(cg, &layer.in.pool, prevs);
            break;
        case (GLOBPOOL):
            uvec = forwardGlobPool(cg, &layer.in.glob_pool, prevs);
        }
    }

    UData data;
    data.type = net->output_type;
    data.nchannels = net->layers[net->nlayers - 1].in.conv.nfilters;
    switch (data.type) {
    case (UVEC):
        data.in.vec = uvec;
        break;
    case (UMAT):
        data.in.mat = *prevs;
        break;
    case (UMATLIST):
        data.in.mats = prevs;
        break;
    }
    return data;
}

// TODO use this function in loss and stuff
// sometimes just want to setup the indices though
// maybe do a setup indices and a setup with values
// seperate
void transferConvInput(Net *net, Matrix *in, ulong nchannels) {
    CompGraph *cg = net->cg;
    for (ulong i = 0; i < nchannels; ++i) {
        for (ulong j = 0; j < in->nrows; ++j) {
            for (ulong k = 0; k < in->ncols; ++k) {
                if (net->input.type == UMATLIST) {
                    MAT_AT(net->input.in.mats[i], j, k) = getSize(cg);
                } else {
                    MAT_AT(net->input.in.mat, j, k) = getSize(cg);
                }
                initLeafScalar(cg, MAT_AT(in[i], j, k));
            }
        }
    }
}

scalar lossVanilla(Net *net, CNData *input, CNData *target) {
    CLEAR_NET_ASSERT(input->nelem == target->nelem);
    CLEAR_NET_ASSERT(input->type == VECTORS && target->type == VECTORS);
    CLEAR_NET_ASSERT(net->input.in.vec.nelem == input->in.vectors->nelem);
    CompGraph *cg = net->cg;
    resetGrads(net->cg);
    ulong train_size = input->nelem;

    scalar total_loss = 0;
    for (ulong i = 0; i < input->in.vectors->nelem; ++i) {
        VEC_AT(net->input.in.vec, i) = getSize(cg) + i;
    }

    Vec target_vec;
    target_vec.nelem = target->in.vectors->nelem;
    for (ulong i = 0; i < train_size; ++i) {
        for (ulong j = 0; j < input->in.vectors->nelem; ++j) {
            initLeafScalar(cg, VEC_AT(input->in.vectors[i], j));
        }
        UVec prediction = _predictVanilla(net, net->input.in.vec);
        target_vec.start_id = getSize(cg);

        for (ulong j = 0; j < target->in.vectors->nelem; ++j) {
            initLeafScalar(cg, VEC_AT(target->in.vectors[i], j));
        }

        ulong raiser = initLeafScalar(cg, 2);
        ulong loss = initLeafScalar(cg, 0);
        for (ulong j = 0; j < target_vec.nelem; ++j) {
            loss = add(
                cg, loss,
                raise(cg, sub(cg, VEC_AT(prediction, j), VEC_ID(target_vec, j)),
                      raiser));
        }
        backward(cg, loss, net->hp.leaker);
        total_loss += getVal(cg, loss);
        setSize(cg, net->nparams + 1);
    }

    setSize(cg, net->nparams + 1);

    return total_loss / train_size;
}

scalar lossConv(Net *net, CNData *input, CNData *target) {
    CLEAR_NET_ASSERT(input->nelem == target->nelem);
    CLEAR_NET_ASSERT(input->type == MATRICES || input->type == MULTIMATRICES);
    if (net->input.type == UMATLIST) {
        CLEAR_NET_ASSERT(net->layers[0].in.conv.nimput == input->nchannels);
    }

    if (net->output_type == UVEC) {
        CLEAR_NET_ASSERT(target->type == VECTORS);
    } else {
        CLEAR_NET_ASSERT(target->type == MATRICES);
    }

    CompGraph *cg = net->cg;
    resetGrads(cg);
    scalar total_loss = 0;

    ulong offset = getSize(cg);

    // put these in a setup indices for matrix function
    // iterate over the list in this funciton

    ulong nchannels_in = net->layers[0].in.conv.nimput;

    ulong in_nrows;
    ulong in_ncols;
    if (nchannels_in > 1) {
        in_nrows = net->input.in.mats->nrows;
        in_ncols = net->input.in.mats->ncols;
    } else {
        in_nrows = net->input.in.mat.nrows;
        in_ncols = net->input.in.mat.ncols;
    }

    for (ulong i = 0; i < net->layers[0].in.conv.nimput; ++i) {
        for (ulong j = 0; j < in_nrows; ++j) {
            for (ulong k = 0; k < in_ncols; ++k) {
                if (nchannels_in > 1) {
                    MAT_AT(net->input.in.mats[i], j, k) = offset++;
                } else {
                    MAT_AT(net->input.in.mat, j, k) = offset++;
                }
            }
        }
    }
    for (ulong i = 0; i < input->nelem; ++i) {
        for (ulong j = 0; j < nchannels_in; ++j) {
            for (ulong k = 0; k < in_nrows; ++k) {
                for (ulong l = 0; l < in_ncols; ++l) {
                    if (nchannels_in > 1) {
                        initLeafScalar(
                                       cg, MAT_AT(input->in.multi_matrices[i][j], k, l));
                    } else {
                        initLeafScalar(cg, MAT_AT(input->in.matrices[i], k, l));
                    }
                }
            }
        }
        UData out;
        if (net->input.type == UMATLIST) {
            out = _predictConv(net, net->input.in.mats);
        } else {
            out = _predictConv(net, &net->input.in.mat);
        }
        size_t raiser = initLeafScalar(cg, 2);
        ulong loss = initLeafScalar(cg, 0);
        switch (out.type) {
        case (UVEC): {
            // TODO can be put in a function, create setupVecFromVector
            // do this there is a lot of target find with .start_id = getSize(cg) so put that in its own function
            Vec target_vec;
            target_vec.nelem = target->in.vectors->nelem;
            target_vec.start_id = getSize(cg);
            // TODO check if this can be done in one loop
            // initialize and backward on that point
            for (ulong j = 0; j < target_vec.nelem; ++j) {
                initLeafScalar(cg, VEC_AT(target->in.vectors[i], j));
            }

            for (ulong j = 0; j < target_vec.nelem; ++j) {
                loss = add(
                    cg, loss,
                    raise(cg,
                          sub(cg, VEC_AT(out.in.vec, j), VEC_ID(target_vec, j)),
                          raiser));
            }
            break;
        }
        case (UMAT): {
            Mat target_mat;
            target_mat.start_id = getSize(cg);
            target_mat.nrows = target->in.matrices->nrows;
            target_mat.ncols = target->in.matrices->ncols;
            for (ulong j = 0; j < target_mat.nrows; ++j) {
                for (ulong k = 0; k < target_mat.ncols; ++k) {
                    initLeafScalar(cg, MAT_AT(target->in.matrices[i], j, k));
                }
            }
            for (ulong j = 0; j < target_mat.nrows; ++j) {
                for (ulong k = 0; k < target_mat.ncols; ++k) {
                    loss = add(cg, loss,
                               raise(cg,
                                     sub(cg, MAT_AT(out.in.mat, j, k),
                                         MAT_ID(target_mat, j, k)),
                                     raiser));
                }
            }
            break;
        }
        case (UMATLIST): {
            CLEAR_NET_ASSERT(0 && "multi matrix output for convolutional net is not supported");
        }
        }
        backward(cg, loss, net->hp.leaker);
        total_loss += getVal(cg, loss);
        setSize(cg, net->nparams + 1);
    }

    setSize(cg, net->nparams + 1);

    return total_loss / input->nelem;
}

void backprop(Net *net) {
    for (ulong i = 0; i < net->nlayers; ++i) {
        if (net->layers[i].type == DENSE) {
            applyDenseGrads(net->cg, &net->layers[i].in.dense, &net->hp);
        } else if (net->layers[i].type == CONV) {
            applyConvGrads(net->cg, &net->layers[i].in.conv, &net->hp);
        }
    }
}

Vector *predictVanilla(Net *net, Vector input, Vector *store) {
    CLEAR_NET_ASSERT(net->input.in.vec.nelem == input.nelem);
    CompGraph *cg = net->cg;

    for (ulong i = 0; i < input.nelem; ++i) {
        VEC_AT(net->input.in.vec, i) = getSize(cg);
        initLeafScalar(cg, VEC_AT(input, i));
    }
    UVec prediction = _predictVanilla(net, net->input.in.vec);
    CLEAR_NET_ASSERT(store->nelem == prediction.nelem);
    for (ulong i = 0; i < store->nelem; ++i) {
        VEC_AT(*store, i) = getVal(cg, VEC_AT(prediction, i));
    }
    setSize(cg, net->nparams + 1);

    return store;
}

Vector *predictConvToVector(Net *net, Matrix *input, ulong nchannels,
                            Vector *store) {
    CLEAR_NET_ASSERT(net->input.type == UMATLIST || net->input.type == UMAT);
    CLEAR_NET_ASSERT(net->output_type == UVEC);
    ulong in_nrows;
    ulong in_ncols;
    CLEAR_NET_ASSERT(nchannels == net->layers[0].in.conv.nimput);
    if (net->input.type == UMATLIST) {
        in_nrows = net->input.in.mats->nrows;
        in_ncols = net->input.in.mats->ncols;
    } else {
        in_nrows = net->input.in.mat.nrows;
        in_ncols = net->input.in.mat.ncols;
    }
    CLEAR_NET_ASSERT(in_nrows == input->nrows);
    CLEAR_NET_ASSERT(in_ncols == input->ncols);

    transferConvInput(net, input, nchannels);

    UVec pred;
    if (net->input.type == UMATLIST) {
        pred = _predictConv(net, net->input.in.mats).in.vec;
    } else {
        pred = _predictConv(net, &net->input.in.mat).in.vec;
    }

    for (ulong i = 0; i < pred.nelem; ++i) {
        VEC_AT(*store, i) = getVal(net->cg, VEC_AT(pred, i));
    }

    setSize(net->cg, net->nparams + 1);

    return store;
}

Matrix *predictConvToMatrix(Net *net, Matrix *input, ulong nchannels,
                            Matrix *store) {
    CLEAR_NET_ASSERT(net->input.type == UMATLIST || net->input.type == UMAT);
    CLEAR_NET_ASSERT(net->output_type == UMAT);
    ulong in_nrows;
    ulong in_ncols;
    CLEAR_NET_ASSERT(nchannels == net->layers[0].in.conv.nimput);
    if (net->input.type == UMATLIST) {
        in_nrows = net->input.in.mats->nrows;
        in_ncols = net->input.in.mats->ncols;
    } else {
        in_nrows = net->input.in.mat.nrows;
        in_ncols = net->input.in.mat.ncols;
    }
    CLEAR_NET_ASSERT(in_nrows == input->nrows);
    CLEAR_NET_ASSERT(in_ncols == input->ncols);

    transferConvInput(net, input, nchannels);

    UMat pred;
    if (net->input.type == UMATLIST) {
        pred = _predictConv(net, net->input.in.mats).in.mat;
    } else {
        pred = _predictConv(net, &net->input.in.mat).in.mat;
    }

    for (ulong i = 0; i < pred.nrows; ++i) {
        for (ulong j = 0; j < pred.nrows; ++j) {
            MAT_AT(*store, i, j) = getVal(net->cg, MAT_AT(pred, i, j));
        }
    }

    setSize(net->cg, net->nparams + 1);

    return store;
}

void printVanillaPredictions(Net *net, CNData *input, CNData *target) {
    CLEAR_NET_ASSERT(input->nelem == target->nelem);
    printf("Input | Net Output | Target \n");
    scalar loss = 0;
    CompGraph *cg = net->cg;

    for (ulong i = 0; i < input->in.vectors->nelem; ++i) {
        VEC_AT(net->input.in.vec, i) = getSize(cg) + i;
    }

    for (ulong i = 0; i < input->nelem; ++i) {
        Vector in = input->in.vectors[i];
        Vector tar = target->in.vectors[i];

        for (ulong j = 0; j < in.nelem; ++j) {
            initLeafScalar(cg, VEC_AT(in, j));
        }
        UVec out = _predictVanilla(net, net->input.in.vec);

        for (ulong j = 0; j < out.nelem; ++j) {
            loss += pows(getVal(cg, VEC_AT(out, j)) - VEC_AT(tar, j), 2);
        }
        printVectorInline(&in);
        printf("| ");
        printUVecInline(cg, &out);
        printf("| ");
        printVectorInline(&tar);
        printf("\n");
        setSize(cg, net->nparams + 1);
    }
    printf("Average Loss: %f\n", loss / input->nelem);
}

void printConvPredictions(Net *net, CNData *input, CNData *target) {
    // TODO create a check CN input target function which has all the
    // asserts
    CLEAR_NET_ASSERT(input->nelem = target->nelem);
    // TODO this could be sent to its own function for setting up input indices

    ulong nchannels_in = net->layers[0].in.conv.nimput;
    ulong in_nrows;
    ulong in_ncols;

    if (nchannels_in > 1) {
        in_nrows = net->input.in.mats->nrows;
        in_ncols = net->input.in.mats->ncols;
    } else {
        in_nrows = net->input.in.mat.nrows;
        in_ncols = net->input.in.mat.ncols;
    }

    ulong offset = getSize(net->cg);
    for (ulong i = 0; i < nchannels_in; ++i) {
        for (ulong j = 0; j < in_nrows; ++j) {
            for (ulong k = 0; k < in_ncols; ++k) {
                if (net->input.type == UMATLIST) {
                    MAT_AT(net->input.in.mats[i], j, k) =
                        offset++;
                } else {
                    MAT_AT(net->input.in.mat, j, k) =
                        offset++;
                }
            }
        }
    }

    CompGraph *cg = net->cg;

    scalar loss = 0;
    Vector *v_tar;
    Matrix *m_tar;
    for (ulong i = 0; i < input->nelem; ++i) {
        Matrix *in;

        if (input->type == UMATLIST) {
            in = input->in.multi_matrices[i];
        } else {
            in = &input->in.matrices[i];
        }

        for (ulong i = 0; i < input->nchannels; ++i) {
            for (ulong j = 0; j < in->nrows; ++j) {
                for (ulong k = 0; k < in->ncols; ++k) {
                    initLeafScalar(cg, MAT_AT(in[i], j, k));
                }
            }
        }
        UData pred;
        if (net->input.type == UMATLIST) {
            pred = _predictConv(net, net->input.in.mats);
        } else {
            pred = _predictConv(net, &net->input.in.mat);
        }

        if (target->type == MATRICES) {
            m_tar = &target->in.matrices[i];
        } else {
            v_tar = &target->in.vectors[i];
        }

        if (pred.type == UVEC) {
            for (ulong j = 0; j < pred.in.vec.nelem; ++j) {
                loss += pows(
                    getVal(cg, VEC_AT(pred.in.vec, j)) - VEC_AT(*v_tar, j), 2);
            }

        } else {
            for (ulong j = 0; j < pred.in.mat.nrows; ++j) {
                for (ulong k = 0; k < pred.in.mat.nrows; ++k) {
                    loss += pows(getVal(cg, MAT_AT(pred.in.mat, j, k)) -
                                     MAT_AT(*m_tar, j, k),
                                 2);
                }
            }

        }

        for (ulong j = 0; j < input->nchannels; ++j) {
            if (input->type == MULTIMATRICES) {
                printf("Input %zu, Channel %zu", i, j);
                printMatrix(&input->in.multi_matrices[i][j], "");
            } else {
                printf("Input %zu", i);
                printMatrix(&input->in.matrices[i], "");
            }
        }
        printf("Output");
        if (net->output_type == UMAT) {
            printUMat(cg, &pred.in.mat, "");
        } else {
            printUVec(cg, &pred.in.vec, "");
        }

        printf("Target");
        if (target->type == MATRICES) {
            printMatrix(&target->in.matrices[i], "");
        } else {
            printVector(&target->in.vectors[i], "");
        }
        setSize(cg, net->nparams + 1);
    }

    printf("Total Loss: %f\n", loss / input->nelem);
}
