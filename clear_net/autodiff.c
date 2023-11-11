#include "autodiff.h"
#include "defines.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// TODO change powf to have one for floats and doubles in case scalar is changed
// to be a double

#define INITIAL_GRAPH_SIZE 50
#define NODE(id) (cg)->vars[(id)]
#define POS(cg, x) NODE(x).num > 0

ulong extendSize(ulong size) {
    return (size == 0 ? INITIAL_GRAPH_SIZE : size * 2);
}

CompGraph allocCompGraph(ulong max_size) {
    return (CompGraph){
        .vars = CLEAR_NET_ALLOC(max_size * sizeof(Scalar)),
        .size = 1,
        .max_size = max_size,
    };
}

void deallocCompGraph(CompGraph *cg) { CLEAR_NET_DEALLOC(cg->vars); }

void reallocGradientStore(CompGraph *cg, ulong new_size) {
    cg->vars = CLEAR_NET_REALLOC(cg->vars, new_size * sizeof(*cg->vars));
    cg->max_size = new_size;
}

Scalar createScalar(scalar num, ulong prev_left, ulong prev_right,
                    Operation op) {
    return (Scalar){
        .num = num,
        .grad = 0,
        .prev_left = prev_left,
        .prev_right = prev_right,
        .op = op,
    };
}

ulong initScalar(CompGraph *cg, scalar num, ulong prev_left, ulong prev_right,
                 Operation op) {
    if (cg->size >= cg->max_size) {
        cg->max_size = extendSize(cg->max_size);
        cg->vars = CLEAR_NET_REALLOC(cg->vars, cg->max_size * sizeof(Scalar));
        CLEAR_NET_ASSERT(cg->vars);
    }
    Scalar out = createScalar(num, prev_left, prev_right, op);
    cg->vars[cg->size] = out;
    cg->size++;
    return cg->size - 1;
}

ulong initLeafScalar(CompGraph *cg, scalar num) {
    return initScalar(cg, num, 0, 0, None);
}

void setVal(CompGraph *cg, ulong x, scalar num) {
    NODE(x).num = num;
}

void applyGrad(CompGraph *cg, ulong x, scalar rate) {
    NODE(x).num -= NODE(x).grad * rate;
}

ulong add(CompGraph *cg, ulong left, ulong right) {
    scalar val = NODE(left).num + NODE(right).num;
    size_t out = initScalar(cg, val, left, right, Add);
    return out;
}

void addBackprop(CompGraph *cg, Scalar *var) {
    NODE(var->prev_left).grad += var->grad;
    NODE(var->prev_right).grad += var->grad;
}

ulong sub(CompGraph *cg, ulong left, ulong right) {
    scalar val = NODE(left).num - NODE(right).num;
    size_t out = initScalar(cg, val, left, right, Sub);
    return out;
}

void subBackprop(CompGraph *cg, Scalar *var) {
    NODE(var->prev_left).grad += var->grad;
    NODE(var->prev_right).grad -= var->grad;
}

ulong mul(CompGraph *cg, ulong left, ulong right) {
    scalar val = NODE(left).num * NODE(right).num;
    size_t out = initScalar(cg, val, left, right, Mul);
    return out;
}

void mulBackprop(CompGraph *cg, Scalar *var) {
    NODE(var->prev_left).grad += var->grad * NODE(var->prev_right).num;
    NODE(var->prev_right).grad += var->grad * NODE(var->prev_left).num;
}

ulong raise(CompGraph *cg, ulong to_raise, ulong pow) {
    scalar val = powf(NODE(to_raise).num, NODE(pow).num);
    size_t out = initScalar(cg, val, to_raise, pow, Raise);
    return out;
}

void raiseBackprop(CompGraph *cg, Scalar *var) {
    scalar l_num = NODE(var->prev_left).num;
    scalar r_num = NODE(var->prev_right).num;
    NODE(var->prev_left).grad += r_num * powf(l_num, r_num - 1) * var->grad;
    NODE(var->prev_right).grad += logf(l_num) * powf(l_num, r_num) * var->grad;
}

ulong relu(CompGraph *cg, ulong x) {
    scalar val = POS(cg, x) > 0 ? NODE(x).num : 0;
    ulong out = initScalar(cg, val, x, 0, Relu);
    return out;
}

void reluBackprop(CompGraph *cg, Scalar *var) {
    if (var->num > 0) {
        NODE(var->prev_left).grad += var->grad;
    }
}

ulong leakyRelu(CompGraph *cg, ulong x) {
    scalar val = POS(cg, x) ? NODE(x).num : LEAKER * NODE(x).num;
    ulong out = initScalar(cg, val, x, 0, LeakyRelu);
    return out;
}

void leakyReluBackprop(CompGraph *cg, Scalar *var) {
    scalar change = var->num > 0 ? 1 : LEAKER;
    NODE(var->prev_left).grad += change * var->grad;
}

ulong htan(CompGraph *cg, ulong x) {
    scalar val = tanhf(NODE(x).num);
    ulong out = initScalar(cg, val, x, 0, Htan);
    return out;
}

void tanhBackprop(CompGraph *cg, Scalar *var) {
    NODE(var->prev_left).grad += (1 - powf(var->num, 2)) * var->grad;
}

ulong sigmoid(CompGraph *cg, ulong x) {
    scalar val = 1 / (1 + expf(-1 * (NODE(x).num)));
    ulong out = initScalar(cg, val, x, 0, Sig);
    return out;
}

void sigmoidBackprop(CompGraph *cg, Scalar *var) {
    NODE(var->prev_left).grad += var->num * (1 - var->num) * var->grad;
}

ulong elu(CompGraph *cg, ulong x) {
    scalar num = NODE(x).num;
    scalar val = num > 0 ? num : LEAKER * (expf(num) - 1);
    size_t out = initScalar(cg, val, x, 0, Elu);
    return out;
}

void eluBackprop(CompGraph *cg, Scalar *var) {
    scalar change = var->num > 0 ? 1 : var->num + LEAKER;
    NODE(var->prev_left).grad += change * var->grad;
}

void backprop(CompGraph *cg, ulong last) {
    NODE(last).grad = 1;
    Scalar *cur;
    for (ulong i = cg->size - 1; i > 0; --i) {
        cur = &NODE(i);
        switch (cur->op) {
        case Add:
            addBackprop(cg, cur);
            break;
        case Sub:
            subBackprop(cg, cur);
            break;
        case Mul:
            mulBackprop(cg, cur);
            break;
        case Raise:
            raiseBackprop(cg, cur);
            break;
        case Relu:
            reluBackprop(cg, cur);
            break;
        case LeakyRelu:
            leakyReluBackprop(cg, cur);
            break;
        case Elu:
            eluBackprop(cg, cur);
            break;
        case Htan:
            tanhBackprop(cg, cur);
            break;
        case Sig:
            sigmoidBackprop(cg, cur);
            break;
        case None:
            break;
        }
    }
}
