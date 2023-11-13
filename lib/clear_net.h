#ifndef CLEAR_NET
#define CLEAR_NET
#include "stdlib.h"

#define MAT_AT(mat, r, c) (mat).elem[(r) * (mat).stride + (c)]
#define VEC_AT(vec, i) (vec).elem[(i)]

// TODO change the internal structure right now have to declared in name.c, name.h, clear_net.h, clear_net.c
//   would be nice to store the output struct in the header file and extern it in the C file, then take that struct and access it in the clear_net.h can do that without exposing it I think
// TODO move the Matrix and Vector to la.c and la.h
// TODO need to do stochastic gradient descent stuff
// TODO a print net results function
// TODO need to save and load a model

typedef float scalar;
typedef unsigned long ulong;
typedef struct CompGraph CompGraph;
typedef struct Net Net;
typedef enum {
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU,
    ELU,
} Activation;

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

typedef struct {
    scalar rate;
    scalar leaker;
} HParams;

typedef struct {
    struct {
        CompGraph* (*allocCompGraph)(ulong max_length);
        void (*deallocCompGraph)(CompGraph *cg);
        ulong (*initLeafScalar)(CompGraph *cg, scalar num);
        void (*resetGrads)(CompGraph *cg, ulong count);
        void (*applyGrad)(CompGraph *cg, ulong x, scalar rate);
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
        void (*backprop)(CompGraph *cg, ulong last, scalar leaker);
    } ad;
    struct {
        Matrix (*allocMatrix)(ulong nrows, ulong ncols);
        void (*deallocMatrix)(Matrix *mat);
        Matrix (*formMatrix)(ulong nrows, ulong ncols, ulong stride, scalar *elements);
        void (*printMatrix)(Matrix *mat, char *name);
        Vector (*allocVector)(ulong nelem);
        Vector (*formVector)(ulong nelem, scalar *elem);
        void (*printVector)(Vector *vec, char *name);
        void (*deallocVector)(Vector *vec);
        void (*shuffleVanillaInput)(Matrix *input, Matrix *target);
    } la;
    HParams (*defaultHParams)(void);
    void (*setRate)(HParams *hp, scalar rate);
    void (*setLeaker)(HParams *hp, scalar leaker);
    void (*randomizeNet)(Net *net, scalar lower, scalar upper);
    Net* (*allocVanillaNet)(HParams hp, ulong input_nelem);
    Net* (*allocConvNet)(HParams hp, ulong input_nrows, ulong input_ncols, ulong nchannels);
    void (*allocDenseLayer)(Net *net, Activation act, ulong dim_out);
    void (*deallocNet)(Net *net);
    scalar (*learnVanilla)(Net *net, Matrix input, Matrix target);
    void (*printNet)(Net *net, char *name);
    Vector* (*predictDense)(Net *net, Vector input, Vector *store);
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
#include "assert.h"
#define CLEAR_NET_ASSERT assert
#endif // CLEAR_NET_ASSERT

#endif // CLEAR_NET
