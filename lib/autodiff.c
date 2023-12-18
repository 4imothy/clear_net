#include "autodiff.h"
#include <math.h>

#define INITIAL_GRAPH_SIZE 50
#define NODE(id) (cg)->vars[(id)]
#define POS(cg, x) NODE(x).num > 0

typedef enum {
    ADD,
    SUB,
    MUL,
    RAISE,
    RECLU,
    LEAKYRECLU,
    HTAN,
    SIG,
    EXPLU,
    NONE,
} Operation;

typedef struct {
    scalar num;
    scalar grad;
    scalar store;
    Operation op;
    ulong prev_left;
    ulong prev_right;
} Scalar;

struct CompGraph {
    Scalar *vars;
    ulong size;
    ulong max_size;
};

ulong extendSize(ulong size) {
    return (size == 0 ? INITIAL_GRAPH_SIZE : size * 2);
}

CompGraph *allocCompGraph(ulong max_size) {
    CompGraph *cg = CLEAR_NET_ALLOC(sizeof(CompGraph));
    cg->size = 1;
    cg->max_size = max_size;
    cg->vars = CLEAR_NET_ALLOC(max_size * sizeof(Scalar));
    return cg;
}

void deallocCompGraph(CompGraph *cg) {
    CLEAR_NET_DEALLOC(cg->vars);
    CLEAR_NET_DEALLOC(cg);
}

void reallocGradientStore(CompGraph *cg, ulong new_size) {
    cg->vars = CLEAR_NET_REALLOC(cg->vars, new_size * sizeof(*cg->vars));
    cg->max_size = new_size;
}

ulong getSize(CompGraph *cg) { return cg->size; }

void setSize(CompGraph *cg, ulong size) { cg->size = size; }

Scalar createScalar(scalar num, ulong prev_left, ulong prev_right,
                    Operation op) {
    return (Scalar){
        .num = num,
        .grad = 0,
        .store = 0,
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
    return initScalar(cg, num, 0, 0, NONE);
}

void setVal(CompGraph *cg, ulong x, scalar num) { NODE(x).num = num; }

void setValRand(CompGraph *cg, ulong x, scalar lower, scalar upper) {
    setVal(cg, x, randRange(lower, upper));
}

scalar getVal(CompGraph *cg, ulong x) { return NODE(x).num; }

scalar getGrad(CompGraph *cg, ulong x) { return NODE(x).grad; }

void applyGrad(CompGraph *cg, ulong x) { NODE(x).num -= NODE(x).grad; }

void applyGradWithHP(CompGraph *cg, ulong x, scalar rate, bool momentum,
                     scalar beta) {
    scalar change = NODE(x).grad * rate;
    if (momentum) {
        NODE(x).store = (beta * NODE(x).store) + ((1 - beta) * change);
        change = NODE(x).store;
    }
    NODE(x).num -= change;
}

ulong add(CompGraph *cg, ulong left, ulong right) {
    scalar val = NODE(left).num + NODE(right).num;
    ulong out = initScalar(cg, val, left, right, ADD);
    return out;
}

void addBackward(CompGraph *cg, Scalar *var) {
    NODE(var->prev_left).grad += var->grad;
    NODE(var->prev_right).grad += var->grad;
}

ulong sub(CompGraph *cg, ulong left, ulong right) {
    scalar val = NODE(left).num - NODE(right).num;
    ulong out = initScalar(cg, val, left, right, SUB);
    return out;
}

void subBackward(CompGraph *cg, Scalar *var) {
    NODE(var->prev_left).grad += var->grad;
    NODE(var->prev_right).grad -= var->grad;
}

ulong mul(CompGraph *cg, ulong left, ulong right) {
    scalar val = NODE(left).num * NODE(right).num;
    ulong out = initScalar(cg, val, left, right, MUL);
    return out;
}

void mulBackward(CompGraph *cg, Scalar *var) {
    NODE(var->prev_left).grad += var->grad * NODE(var->prev_right).num;
    NODE(var->prev_right).grad += var->grad * NODE(var->prev_left).num;
}

ulong raise(CompGraph *cg, ulong to_raise, ulong pow) {
    scalar val = pows(NODE(to_raise).num, NODE(pow).num);
    ulong out = initScalar(cg, val, to_raise, pow, RAISE);
    return out;
}

void raiseBackward(CompGraph *cg, Scalar *var) {
    scalar l_num = NODE(var->prev_left).num;
    scalar r_num = NODE(var->prev_right).num;
    NODE(var->prev_left).grad += r_num * pows(l_num, r_num - 1) * var->grad;
    NODE(var->prev_right).grad += logf(l_num) * pows(l_num, r_num) * var->grad;
}

ulong relu(CompGraph *cg, ulong x) {
    scalar val = POS(cg, x) > 0 ? NODE(x).num : 0;
    ulong out = initScalar(cg, val, x, 0, RECLU);
    return out;
}

void reluBackward(CompGraph *cg, Scalar *var) {
    if (var->num > 0) {
        NODE(var->prev_left).grad += var->grad;
    }
}

ulong leakyRelu(CompGraph *cg, ulong x, scalar leaker) {
    scalar val = POS(cg, x) ? NODE(x).num : leaker * NODE(x).num;
    ulong out = initScalar(cg, val, x, 0, LEAKYRECLU);
    return out;
}

void leakyReluBackward(CompGraph *cg, Scalar *var, scalar leaker) {
    scalar change = var->num > 0 ? 1 : leaker;
    NODE(var->prev_left).grad += change * var->grad;
}

ulong htan(CompGraph *cg, ulong x) {
    scalar val = tanhs(NODE(x).num);
    ulong out = initScalar(cg, val, x, 0, HTAN);
    return out;
}

void tanhBackward(CompGraph *cg, Scalar *var) {
    NODE(var->prev_left).grad += (1 - pows(var->num, 2)) * var->grad;
}

ulong sigmoid(CompGraph *cg, ulong x) {
    scalar val = 1 / (1 + exps(-1 * (NODE(x).num)));
    ulong out = initScalar(cg, val, x, 0, SIG);
    return out;
}

void sigmoidBackward(CompGraph *cg, Scalar *var) {
    NODE(var->prev_left).grad += var->num * (1 - var->num) * var->grad;
}

ulong elu(CompGraph *cg, ulong x, scalar leaker) {
    scalar num = NODE(x).num;
    scalar val = num > 0 ? num : leaker * (exps(num) - 1);
    ulong out = initScalar(cg, val, x, 0, EXPLU);
    return out;
}

void eluBackward(CompGraph *cg, Scalar *var, scalar leaker) {
    scalar change = var->num > 0 ? 1 : var->num + leaker;
    NODE(var->prev_left).grad += change * var->grad;
}

void backward(CompGraph *cg, ulong last, scalar leaker) {
    NODE(last).grad = 1;
    Scalar *cur;
    for (ulong i = cg->size - 1; i > 0; --i) {
        cur = &NODE(i);
        switch (cur->op) {
        case ADD:
            addBackward(cg, cur);
            break;
        case SUB:
            subBackward(cg, cur);
            break;
        case MUL:
            mulBackward(cg, cur);
            break;
        case RAISE:
            raiseBackward(cg, cur);
            break;
        case RECLU:
            reluBackward(cg, cur);
            break;
        case LEAKYRECLU:
            leakyReluBackward(cg, cur, leaker);
            break;
        case EXPLU:
            eluBackward(cg, cur, leaker);
            break;
        case HTAN:
            tanhBackward(cg, cur);
            break;
        case SIG:
            sigmoidBackward(cg, cur);
            break;
        case NONE:
            break;
        }
    }
}

void resetGrads(CompGraph *cg) {
    for (ulong i = 0; i < cg->size; ++i) {
        NODE(i + 1).grad = 0;
    }
}
