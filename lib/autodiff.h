#ifndef CN_AUTODIFF
#define CN_AUTODIFF
#include "clear_net.h"
#include "scalar.h"
#include <stdbool.h>

scalar pows(scalar to_raise, scalar raiser);
scalar tanhs(scalar x);
CompGraph *allocCompGraph(ulong max_length);
void deallocCompGraph(CompGraph *cg);
ulong initLeafScalar(CompGraph *cg, scalar num);
void resetGrads(CompGraph *cg);
void applyGrad(CompGraph *cg, ulong x);
void applyGradWithHP(CompGraph *cg, ulong x, scalar rate, bool momentum,
                     scalar beta);
scalar getVal(CompGraph *cg, ulong x);
scalar getGrad(CompGraph *cg, ulong x);
void setVal(CompGraph *cg, ulong x, scalar num);
ulong add(CompGraph *cg, ulong left, ulong right);
ulong sub(CompGraph *cg, ulong left, ulong right);
ulong mul(CompGraph *cg, ulong left, ulong right);
ulong raise(CompGraph *cg, ulong to_raise, ulong pow);
ulong relu(CompGraph *cg, ulong x);
ulong leakyRelu(CompGraph *cg, ulong x, scalar leaker);
ulong htan(CompGraph *cg, ulong x);
ulong sigmoid(CompGraph *cg, ulong x);
ulong elu(CompGraph *cg, ulong x, scalar leaker);
void backward(CompGraph *cg, ulong last, scalar leaker);
ulong getSize(CompGraph *cg);
void setSize(CompGraph *cg, ulong size);
void setValRand(CompGraph *cg, ulong x, scalar lower, scalar upper);

#endif // CN_AUTODIFF
