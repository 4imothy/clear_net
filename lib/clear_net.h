#ifndef CLEAR_NET
#define CLEAR_NET
#include "stdlib.h"

#define MAT_AT(mat, r, c) (mat).elem[(r) * (mat).stride + (c)]
#define VEC_AT(vec, i) (vec).elem[(i)]

// TODO when doing examples should create a public matrix type for scalar types
// which is then used by the people
// TODO need to do stochastic gradient descent stuff
// TODO make a new macro which allocs and verifies that mem isn't null
// TODO hparam struct that is created and passed to training

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
    struct {
        CompGraph* (*allocCompGraph)(ulong max_size);
        void (*deallocCompGraph)(CompGraph* cg);
        ulong (*initLeafScalar)(CompGraph* cg, scalar num);
        scalar (*getVal)(CompGraph *cg, ulong x);
        scalar (*getGrad)(CompGraph *cg, ulong x);
        ulong (*add)(CompGraph *cg, ulong left, ulong right);
        ulong (*sub)(CompGraph *cg, ulong left, ulong right);
        ulong (*mul)(CompGraph *cg, ulong left, ulong right);
        ulong (*raise)(CompGraph *cg, ulong to_raise, ulong pow);
        ulong (*relu)(CompGraph *cg, ulong x);
        ulong (*leakyRelu)(CompGraph *cg, ulong x);
        ulong (*htan)(CompGraph *cg, ulong x);
        ulong (*sigmoid)(CompGraph *cg, ulong x);
        ulong (*elu)(CompGraph *cg, ulong x);
        void (*backprop)(CompGraph *cg, ulong last);
    } ad;
    struct {
        Matrix (*allocMatrix)(ulong nrows, ulong ncols);
        void (*deallocMatrix)(Matrix *mat);
        Matrix (*formMatrix)(long nrows, long ncols, long stride, scalar *elements);
        void (*printMatrix)(Matrix *mat, char *name);
    } mat;
    Net* (*allocVanillaNet)(ulong input_nelem);
    Net* (*allocConvNet)(ulong input_nrows, ulong input_ncols, ulong nchannels);
    void (*randomizeNet)(Net *net, scalar lower, scalar upper);
    void (*allocDenseLayer)(Net *net, Activation act, ulong dim_out);
    void (*deallocNet)(Net *net);
    scalar (*learnVanilla)(Net *net, Matrix input, Matrix target);
    void (*printNet)(Net *net, char *name);
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
#ifndef LEAKER
#define LEAKER 0.1 // TODO make this a hparam
#endif // LEAKER

#endif // CLEAR_NET
