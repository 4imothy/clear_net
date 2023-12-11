#ifndef CLEAR_NET
#define CLEAR_NET
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define MAT_AT(mat, r, c) (mat).elem[(r) * (mat).stride + (c)]
#define VEC_AT(vec, i) (vec).elem[(i)]

// TODO move all headers out of this file except for the strictly necessary
// TODO support for pooling layers being the first in a net
// TODO rewrite the compgraph to pass around Scalar type each of which has its values index in the graph, so the graph only stores scalars, or try a Scalar type which only has the index and value and a Parameter type which has all the other stuff

// TODO create a python script to automate documentation creating, read a
// comment above the function declaration in the clear_net.h thing

// TODO test for saving and loading the net

// FUTURE have a train and use mode which is a bool passed to a wrapper for the autodiff functions
//   1. Return a void pointer but the scalar res will be stack allocated so this won't really work
//   2. Return a wrapper class around a ulong and scalar *.s (scalar) or *.u (ulong)

// FUTURE make a nice error interface to replace assertions of 0
// FUTURE make a data loading library that works well with clear_net, this will
// load images and put the class as the folder, so change the python script to
// respect this structure

typedef float scalar;
typedef unsigned long ulong;
typedef struct CompGraph CompGraph;
typedef struct Net Net;
typedef struct HParams HParams;
typedef struct CNData CNData;

typedef enum {
    SIGMOID,
    RELU,
    TANH,
    LEAKYRELU,
    ELU,
} Activation;

typedef enum {
    MAX,
    AVERAGE,
} Pooling;

typedef struct {
    scalar *elem;
    ulong stride;
    ulong nrows;
    ulong ncols;
} Matrix;

typedef struct {
    scalar *elem;
    ulong nelem;
} Vector;

typedef enum {
    SAME,
    VALID,
    FULL,
} Padding;

typedef struct {
    struct {
        CompGraph *(*allocCompGraph)(ulong max_length);
        void (*deallocCompGraph)(CompGraph *cg);
        ulong (*initLeafScalar)(CompGraph *cg, scalar num);
        void (*resetGrads)(CompGraph *cg, ulong count);
        void (*applyGrad)(CompGraph *cg, ulong x);
        scalar (*getVal)(CompGraph *cg, ulong x);
        scalar (*getGrad)(CompGraph *cg, ulong x);
        void (*setVal)(CompGraph *cg, ulong x, scalar num);
        void (*setValRand)(CompGraph *cg, ulong x, scalar lower, scalar upper);
        ulong (*add)(CompGraph *cg, ulong left, ulong right);
        ulong (*sub)(CompGraph *cg, ulong left, ulong right);
        ulong (*mul)(CompGraph *cg, ulong left, ulong right);
        ulong (*raise)(CompGraph *cg, ulong to_raise, ulong pow);
        ulong (*relu)(CompGraph *cg, ulong x);
        ulong (*leakyRelu)(CompGraph *cg, ulong x, scalar leaker);
        ulong (*htan)(CompGraph *cg, ulong x);
        ulong (*sigmoid)(CompGraph *cg, ulong x);
        ulong (*elu)(CompGraph *cg, ulong x, scalar leaker);
        void (*backward)(CompGraph *cg, ulong last, scalar leaker);
    } ad;
    struct {
        Matrix (*allocMatrix)(ulong nrows, ulong ncols);
        Matrix (*formMatrix)(ulong nrows, ulong ncols, ulong stride,
                             scalar *elements);
        void (*deallocMatrix)(Matrix *mat);
        void (*printMatrix)(Matrix *mat, char *name);
        Vector (*allocVector)(ulong nelem);
        Vector (*formVector)(ulong nelem, scalar *elem);
        void (*printVector)(Vector *vec, char *name);
        void (*deallocVector)(Vector *vec);
        void (*shuffleDatas)(CNData *input, CNData *target);
        void (*setBatchFromMatrix)(Matrix all_input, Matrix all_target,
                                   ulong batch_num, ulong batch_size,
                                   Matrix *batch_in, Matrix *batch_tar);
        CNData *(*allocDataFromVectors)(Vector *vectors, ulong nelem);
        CNData *(*allocDataFromMatrices)(Matrix *matrices, ulong nelem);
        CNData *(*allocDataFromMultiChannelMatrices)(Matrix **multi_matrices,
                                                     ulong nelem,
                                                     ulong nchannels);
        CNData *(*allocEmptyData)(void);
        void (*deallocData)(CNData *data);
        Vector *(*allocVectors)(ulong count, ulong nelem);
        Matrix *(*allocMatrices)(ulong count, ulong nrows, ulong ncols);
        Matrix **(*allocMultiMatrices)(ulong count, ulong nchannels,
                                       ulong nrows, ulong ncols);
        void (*deallocVectors)(Vector *list, ulong count);
        void (*deallocMatrices)(Matrix *list, ulong count);
        void (*deallocMultiMatrices)(Matrix **list, ulong count,
                                     ulong nchannels);
        void (*printVectors)(Vector *list, ulong count);
        void (*printMatrices)(Matrix *list, ulong count);
        void (*printMultiMatrices)(Matrix **list, ulong count, ulong nchannels);
        void (*printData)(CNData *d);
        void (*setBatch)(CNData *all_input, CNData *all_target, ulong batch_num,
                         ulong batch_size, CNData *batch_in, CNData *batch_tar);
    } data;
    HParams *(*allocDefaultHParams)(void);
    void (*setRate)(HParams *hp, scalar rate);
    void (*withMomentum)(HParams *hp, scalar beta);
    void (*setLeaker)(HParams *hp, scalar leaker);
    void (*randomizeNet)(Net *net, scalar lower, scalar upper);
    Net *(*allocVanillaNet)(HParams *hp, ulong input_nelem);
    Net *(*allocConvNet)(HParams *hp, ulong input_nrows, ulong input_ncols,
                         ulong nchannels);
    void (*allocDenseLayer)(Net *net, Activation act, ulong dim_out);
    void (*allocConvLayer)(Net *net, Activation act, Padding padding,
                           ulong noutput, ulong kernel_nrows,
                           ulong kernel_ncols);
    void (*allocPoolingLayer)(Net *net, Pooling strat, ulong kernel_nrows,
                              ulong kernel_ncols);
    void (*allocGlobalPoolingLayer)(Net *net, Pooling strat);
    void (*deallocNet)(Net *net);
    void (*printNet)(Net *net, char *name);
    Vector *(*predictVanilla)(Net *net, Vector input, Vector *store);
    Vector *(*predictConvToVector)(Net *net, Matrix *input, ulong nchannels,
                                   Vector *store);
    Matrix *(*predictConvToMatrix)(Net *net, Matrix *input, ulong nchannels,
                                   Matrix *store);
    void (*printVanillaPredictions)(Net *net, CNData *input, CNData *target);
    void (*printConvPredictions)(Net *net, CNData *input, CNData *target);
    scalar (*lossVanilla)(Net *net, CNData *input, CNData *target);
    scalar (*lossConv)(Net *net, CNData *input, CNData *target);
    void (*backprop)(Net *net);
    void (*saveNet)(Net *net, char *path);
    Net *(*allocNetFromFile)(char *path);
} _cn_names;

extern _cn_names const cn;

#ifndef CLEAR_NET_ALLOC
#define CLEAR_NET_ALLOC malloc
#endif // CLEAR_NET_ALLOC
#ifndef CLEAR_NET_REALLOC
#define CLEAR_NET_REALLOC realloc
#endif // CLEAR_NET_REALLOC
#ifndef CLEAR_NET_DEALLOC
#define CLEAR_NET_DEALLOC free
#endif // CLEAR_NET_MALLOC
#ifndef CLEAR_NET_ASSERT
#define CLEAR_NET_ASSERT assert
#endif // CLEAR_NET_ASSERT

#endif // CLEAR_NET
