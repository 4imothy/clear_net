#ifndef CN_NET
#define CN_NET
#include "autodiff.h"
#include "clear_net.h"
#include "data.h"
#include "graph_utils.h"

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
Vector *predictConvToVector(Net *net, Matrix *input, ulong nchannels,
                            Vector *store);
Matrix *predictConvToMatrix(Net *net, Matrix *input, ulong nchannels,
                            Matrix *store);
void printVanillaPredictions(Net *net, CNData *input, CNData *target);
void printConvPredictions(Net *net, CNData *input, CNData *target);
scalar lossVanilla(Net *net, CNData *input, CNData *target);
void backprop(Net *net);
void saveNet(Net *net, char *path);
Net *allocNetFromFile(char *path);
void allocConvLayer(Net *net, Activation act, Padding padding, ulong noutput,
                    ulong kernel_nrows, ulong kernel_ncols);
void allocPoolingLayer(Net *net, Pooling strat, ulong kernel_nrows,
                       ulong kernel_ncols);
void allocGlobalPoolingLayer(Net *net, Pooling strat);
scalar lossConv(Net *net, CNData *input, CNData *target);

#endif // CN_NET
