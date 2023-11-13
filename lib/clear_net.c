#include "clear_net.h"
#include "autodiff.h"
#include "la.h"
#include "net.h"

#define FUNC(func) .func = func

_cn_names const cn = {
    .ad =
        {
            FUNC(allocCompGraph),
            FUNC(deallocCompGraph),
            FUNC(initLeafScalar),
            FUNC(getVal),
            FUNC(getGrad),
            FUNC(applyGrad),
            FUNC(add),
            FUNC(sub),
            FUNC(mul),
            FUNC(raise),
            FUNC(relu),
            FUNC(leakyRelu),
            FUNC(htan),
            FUNC(sigmoid),
            FUNC(elu),
            FUNC(backprop),
            FUNC(setValRand),
        },
    .la =
        {
            FUNC(deallocMatrix),
            FUNC(allocMatrix),
            FUNC(formMatrix),
            FUNC(printMatrix),
            FUNC(allocVector),
            FUNC(formVector),
            FUNC(printVector),
            FUNC(deallocVector),
            FUNC(shuffleMatrixRows),
        },
    FUNC(defaultHParams),
    FUNC(setRate),
    FUNC(setLeaker),
    FUNC(withMomentum),
    FUNC(allocConvNet),
    FUNC(allocVanillaNet),
    FUNC(allocDenseLayer),
    FUNC(randomizeNet),
    FUNC(deallocNet),
    FUNC(learnVanilla),
    FUNC(printNet),
    FUNC(predictDense),
    FUNC(printVanillaPredictions),
};
