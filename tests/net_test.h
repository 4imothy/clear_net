#ifndef CN_NET_TEST
#define CN_NET_TEST
#include "../lib/clear_net.h"
#include "../lib/graph_utils.h"

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
    UData input;
    CompGraph *cg;
    ulong nparams;
    UType output_type;
    HParams hp;
};

HParams *allocDefaultHParams(void);
void setRate(HParams *hp, scalar rate);
void setLeaker(HParams *hp, scalar leaker);
void withMomentum(HParams *hp, scalar beta);
Net *allocVanillaNet(HParams *hp, ulong input_nelem);
Net *allocConvNet(HParams *hp, ulong input_nrows, ulong input_ncols,
                  ulong nchannels);
void randomizeNet(Net *net, scalar lower, scalar upper);
void allocDenseLayer(Net *net, Activation act, ulong dim_out);
void deallocNet(Net *net);
void printNet(Net *net, char *name);
Vector *predictVanilla(Net *net, Vector input, Vector *store);
void printVanillaPredictions(Net *net, Matrix input, Matrix target);
scalar lossVanilla(Net *net, Matrix input, Matrix target);
void backprop(Net *net);
void saveNet(Net *net, char *path);
Net *allocNetFromFile(char *path);
void allocConvLayer(Net *net, Padding padding, Activation act, ulong noutput,
                    ulong kernel_nrows, ulong kernel_ncols);
UMat *forwardConv(CompGraph *cg, ConvolutionalLayer *layer, UMat *input,
                  scalar leaker);

#endif // CN_NET_TEST
