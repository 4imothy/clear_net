#ifndef CN_NET
#define CN_NET
#include "clear_net.h"

Net* allocVanillaNet(ulong input_nelem);
Net* allocConvNet(ulong input_nrows, ulong input_ncols, ulong nchannels);
void randomizeNet(Net *net, scalar lower, scalar upper);
void allocDenseLayer(Net *net, Activation act, ulong dim_out);
void deallocNet(Net *net);
scalar learnVanilla(Net *net, Matrix input, Matrix target);
void printNet(Net *net, char *name);

#endif // CN_NET
