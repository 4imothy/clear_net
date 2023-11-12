#ifndef CN_AUTODIFF
#define CN_AUTODIFF
#include "clear_net.h"

// TODO these should be moved to the C file
typedef enum {
    Add,
    Sub,
    Mul,
    Raise,
    Relu,
    LeakyRelu,
    Htan,
    Sig,
    Elu,
    None,
} Operation;

typedef struct {
    scalar num;
    scalar grad;
    Operation op;
    ulong prev_left;
    ulong prev_right;
} Scalar;

struct CompGraph {
    Scalar *vars;
    ulong size;
    ulong max_size;
};

CompGraph* allocCompGraph(ulong max_length);
void deallocCompGraph(CompGraph *cg);
ulong initLeafScalar(CompGraph *cg, scalar num);
void setVal(CompGraph *cg, ulong x, scalar num);
scalar getVal(CompGraph *cg, ulong x);
scalar getGrad(CompGraph *cg, ulong x);
void applyGrad(CompGraph *cg, ulong x, scalar rate);
// TODO this function should be public
void resetGrads(CompGraph *cg, ulong count);
ulong add(CompGraph *cg, ulong left, ulong right);
ulong sub(CompGraph *cg, ulong left, ulong right);
ulong mul(CompGraph *cg, ulong left, ulong right);
ulong raise(CompGraph *cg, ulong to_raise, ulong pow);
ulong relu(CompGraph *cg, ulong x);
ulong leakyRelu(CompGraph *cg, ulong x);
ulong htan(CompGraph *cg, ulong x);
ulong sigmoid(CompGraph *cg, ulong x);
ulong elu(CompGraph *cg, ulong x);
void backprop(CompGraph *cg, ulong last);

#endif // CN_AUTODIFF
