#ifndef AUTODIFF
#define AUTODIFF

typedef float scalar;
typedef unsigned long ulong;

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

typedef struct {
    Scalar *vars;
    ulong size;
    ulong max_size;
} CompGraph;

CompGraph allocCompGraph(ulong max_length);
void deallocCompGraph(CompGraph *cg);
ulong initLeafScalar(CompGraph *cg, scalar num);
void setVal(CompGraph *cg, ulong x, scalar num);
void applyGrad(CompGraph *cg, ulong x, scalar rate);
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

#endif // AUTODIFF
