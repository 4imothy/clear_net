#ifndef CN_NET
#define CN_NET
#include "clear_net.h"

struct HParams {
    scalar rate;
    scalar leaker;
    scalar beta;
    bool momentum;
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
void printVanillaPredictions(Net *net, CNData *input, CNData *target);
scalar lossVanilla(Net *net, CNData *input, CNData* target);
void backprop(Net *net);
void saveNet(Net *net, char *path);
Net *allocNetFromFile(char *path);
void allocConvLayer(Net *net, Activation act, Padding padding, ulong noutput,
                    ulong kernel_nrows, ulong kernel_ncols);
void allocPoolingLayer(Net *net, Pooling strat, ulong kernel_nrows,
                       ulong kernel_ncols);
void allocGlobalPoolingLayer(Net *net, Pooling strat);
scalar lossConv(Net *net, CNData *input, CNData* target);

#endif // CN_NET
