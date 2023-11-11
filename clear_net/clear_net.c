#include <stdio.h>
#include "clear_net.h"
#include "autodiff.h"

_cn_names const cn = {
    .ad = {
        .add = add,
        .sub = sub,
        .mul = mul,
        .raise = raise,
        .relu = relu,
        .leakyRelu = leakyRelu,
        .htan = htan,
        .sigmoid = sigmoid,
        .elu = elu,
        .backprop = backprop,
    }
};
