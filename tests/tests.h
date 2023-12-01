#ifndef CN_TEST
#define CN_TEST

#include "../lib/autodiff.h"
#include "../lib/clear_net.h"
#include "../lib/graph_utils.h"
#include <string.h>

#define LEN(ptr) sizeof(ptr) / sizeof(*ptr)

scalar input_elem[] = {
    0.10290608318034533, 0.8051580508692876,  0.39055048005351034,
    0.7739175926400883,  0.24730207704015073, 0.7987075645399935,
    0.24602568871407338, 0.6268407447350659,  0.4646505260697441,
    0.20524882983167547, 0.5031590491750169,  0.2550151936024112,
    0.3354895253780905,  0.6825483746871936,  0.6204572461588524,
    0.6487941004544666,  0.742795723261874,   0.8436721618301802,
    0.0433154872324607,  0.42621935359557017};
const ulong input_elem_len = LEN(input_elem);

scalar matrix_elem[] = {2, 6, 7, 8, 4, 0, 6, 4, 2, 0, 9, 7, 5, 9, 8, 8,
                       4, 6, 0, 2, 4, 7, 6, 1, 7, 5, 2, 9, 6, 7, 8};
const size_t matrix_elem_len = LEN(matrix_elem);

int strequal(char *a, char *b) { return !(strcmp(a, b)); }

void printMatResults(CompGraph *cg, UMat mat) {
    printf("%zu %zu", mat.nrows, mat.ncols);
    for (size_t i = 0; i < mat.nrows; ++i) {
        for (size_t j = 0; j < mat.ncols; ++j) {
            printf(" %f ", getVal(cg, MAT_AT(mat, i, j)));
        }
    }
}

void printVecResults(CompGraph *cg, UVec vec) {
    printf("%zu ", vec.nelem);
    for (size_t i = 0; i < vec.nelem; ++i) {
        printf(" %f ", getVal(cg, VEC_AT(vec, i)));
    }
}

void fill_mat(CompGraph *cg, Mat *mat, scalar *elem, ulong elem_len) {
    ulong cur_id = 0;
    for (ulong i = 0; i < mat->nrows; ++i) {
        for (ulong j = 0; j < mat->ncols; ++j) {
            setVal(cg, MAT_ID(*mat, i, j), elem[cur_id]);
            cur_id = (cur_id + 1) % elem_len;
        }
    }
}

void set_umat(CompGraph *cg, UMat *umat, scalar *elem, ulong elem_len) {
    ulong cur_id = 0;
    for (ulong i = 0; i < umat->nrows; ++i) {
        for (ulong j = 0; j < umat->ncols; ++j) {
            MAT_AT(*umat, i, j) = initLeafScalar(cg, elem[cur_id]);
            cur_id = (cur_id + 1) % elem_len;
        }
    }
}

void set_uvec(CompGraph *cg, UVec *uvec, scalar *elem, ulong elem_len) {
    ulong cur_id = 0;
    for (ulong i = 0; i < uvec->nelem; ++i) {
        VEC_AT(*uvec, i) = initLeafScalar(cg, elem[cur_id]);
        cur_id = (cur_id + 1) % elem_len;
    }
}

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
void allocConvLayer(Net *net, Activation act, Padding padding, ulong noutput,
                    ulong kernel_nrows, ulong kernel_ncols);
UMat *forwardConv(CompGraph *cg, ConvolutionalLayer *layer, UMat *input,
                  scalar leaker);
UVec forwardDense(CompGraph *cg, DenseLayer *layer, UVec input, scalar leaker);

#endif // CN_TEST
