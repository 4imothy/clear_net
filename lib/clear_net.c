#include "clear_net.h"
#include "autodiff.h"
#include "la.h"
#include "net.h"

#define PAIR(func) .func = func

_cn_names const cn = {
    .ad =
        {
            PAIR(allocCompGraph),
            PAIR(deallocCompGraph),
            PAIR(initLeafScalar),
            PAIR(getVal),
            PAIR(getGrad),
            PAIR(applyGrad),
            PAIR(add),
            PAIR(sub),
            PAIR(mul),
            PAIR(raise),
            PAIR(relu),
            PAIR(leakyRelu),
            PAIR(htan),
            PAIR(sigmoid),
            PAIR(elu),
            PAIR(backprop),
            PAIR(setValRand),
        },
    .la =
        {
            PAIR(deallocMatrix),
            PAIR(allocMatrix),
            PAIR(formMatrix),
            PAIR(printMatrix),
            PAIR(allocVector),
            PAIR(formVector),
            PAIR(printVector),
            PAIR(deallocVector),
            PAIR(shuffleMatrixRows),
            PAIR(setBatchFromMatrix),
        },
    PAIR(defaultHParams),
    PAIR(setRate),
    PAIR(setLeaker),
    PAIR(withMomentum),
    PAIR(allocConvNet),
    PAIR(allocVanillaNet),
    PAIR(allocDenseLayer),
    PAIR(randomizeNet),
    PAIR(deallocNet),
    PAIR(learnVanilla),
    PAIR(printNet),
    PAIR(predictDense),
    PAIR(printVanillaPredictions),
    PAIR(lossVanilla),
    PAIR(saveModel),
    PAIR(allocNetFromFile),
};
