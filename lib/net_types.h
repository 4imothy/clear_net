#ifndef CN_NET_TYPES
#define CN_NET_TYPES

#include "clear_net.h"
#include "graph_utils.h"

struct HParams {
    scalar rate;
    scalar leaker;
    scalar beta;
    bool momentum;
};

typedef struct {
    Mat weights;
    Vec biases;
    Activation act;
    UVec output;
} DenseLayer;

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
    DENSE,
    CONV,
    POOL,
    GLOBPOOL,
} LayerType;

typedef struct {
    LayerType type;
    LayerData in;
} Layer;

struct Net {
    ulong nlayers;
    Layer *layers;
    UData input;
    CompGraph *cg;
    ulong nparams;
    UType output_type;
    HParams hp;
};

#endif // CN_NET_TYPES
