#ifndef CN_NET
#define CN_NET
#include "clear_net.h"

HParams defaultHParams(void);
void setRate(HParams *hp, scalar rate);
void setLeaker(HParams *hp, scalar leaker);
void withMomentum(HParams *hp, scalar beta);
Net *allocVanillaNet(HParams hp, ulong input_nelem);
Net *allocConvNet(HParams hp, ulong input_nrows, ulong input_ncols,
                  ulong nchannels);
void randomizeNet(Net *net, scalar lower, scalar upper);
void allocDenseLayer(Net *net, Activation act, ulong dim_out);
void deallocNet(Net *net);
scalar learnVanilla(Net *net, Matrix input, Matrix target);
void printNet(Net *net, char *name);
Vector *predictDense(Net *net, Vector input, Vector *store);
void printVanillaPredictions(Net *net, Matrix input, Matrix target);
scalar lossVanilla(Net *net, Matrix input, Matrix target);

#endif // CN_NET
