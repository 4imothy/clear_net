/* TODO Should put some prefix before all functions */
#ifndef CLEAR_NET
#define CLEAR_NET

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// allow custom memory allocation strategies
#ifndef CLEAR_NET_ALLOC
#define CLEAR_NET_ALLOC malloc
#endif // CLEAR_NET_ALLOC
#ifndef CLEAR_NET_REALLOC
#define CLEAR_NET_REALLOC realloc
#endif // CLEAR_NET_REALLOC
// allow custom memory free strategies
#ifndef CLEAR_NET_DEALLOC
#define CLEAR_NET_DEALLOC free
#endif // CLEAR_NET_MALLOC
// allow custom assertion strategies
#ifndef CLEAR_NET_ASSERT
#include "assert.h" // assert
#define CLEAR_NET_ASSERT assert
#endif // CLEAR_NET_ASSERT

#ifndef CLEAR_NET_RATE
#define CLEAR_NET_RATE 0.5f
#endif
#ifndef CLEAR_NET_ACT_OUTPUT
#define CLEAR_NET_ACT_OUTPUT Sigmoid
#endif // CLEAR_NET_ACT_OUTPUT
#ifndef CLEAR_NET_ACT_HIDDEN
#define CLEAR_NET_ACT_HIDDEN Leaky_ReLU
#endif // CLEAR_NET_ACT_HIDDEN
#ifndef CLEAR_NET_ACT_NEG_SCALE
#define CLEAR_NET_ACT_NEG_SCALE 0.1f
#endif // CLEAR_NET_NEG_SCALE

#ifndef CLEAR_NET_MOMENTUM
#define CLEAR_NET_MOMENTUM 0
#endif // CLEAR_NET_MOMENTUM
#ifndef CLEAR_NET_MOMENTUM_BETA
#define CLEAR_NET_MOMENTUM_BETA 0.9
#endif // CLEAR_NET_MOMENTUM_BETA

#define GET_NODE(id) (nl)->vars[(id)]
#define NODE_LIST_INIT_LENGTH 10
#define EXTEND_LENGTH(len) ((len) == 0 ? NODE_LIST_INIT_LENGTH : ((len) * 2))

typedef struct VarNode VarNode;
typedef struct NodeList NodeList;

typedef void BackWardFunction(NodeList *nl, VarNode *var);

struct NodeList {
    VarNode *vars;
    size_t length;
    size_t max_length;
};

struct VarNode {
    float num;
    float grad;
    BackWardFunction *backward;
    size_t prev_left;
    size_t prev_right;
    size_t visited;
};

// TODO put all the function declarations here

#endif // CLEAR_NET

#ifdef CLEAR_NET_IMPLEMENTATION

NodeList alloc_node_list(size_t length) {
    return (NodeList){
        .vars = CLEAR_NET_ALLOC(length * sizeof(VarNode)),
        .length = length,
    };
}

void deolloc_node_list(NodeList *nl) { CLEAR_NET_DEALLOC(nl->vars); }

size_t init_var(NodeList *nl, float num, size_t prev_left, size_t prev_right,
                BackWardFunction *backward) {
    if (nl->length >= nl->max_length) {
        nl->max_length = EXTEND_LENGTH(nl->max_length);
        nl->vars =
            CLEAR_NET_REALLOC(nl->vars, nl->max_length * sizeof(VarNode));
        CLEAR_NET_ASSERT(nl->vars);
    }
    VarNode out = {
        .num = num,
        .grad = 0,
        .prev_left = prev_left,
        .prev_right = prev_right,
        .visited = 0,
        .backward = backward,
    };
    nl->vars[nl->length] = out;
    nl->length++;
    return nl->length - 1;
}

size_t init_leaf_var(NodeList *nl, float num) {
    return init_var(nl, num, 0, 0, NULL);
}

void add_backward(NodeList *nl, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->grad;
    GET_NODE(var->prev_right).grad += var->grad;
}

size_t add(NodeList *nl, size_t left, size_t right) {
    float val = GET_NODE(left).num + GET_NODE(right).num;
    size_t out = init_var(nl, val, left, right, add_backward);
    return out;
}

void subtract_backward(NodeList *nl, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->grad;
    GET_NODE(var->prev_right).grad -= var->grad;
}

size_t subtract(NodeList *nl, size_t left, size_t right) {
    float val = GET_NODE(left).num - GET_NODE(right).num;
    size_t out = init_var(nl, val, left, right, subtract_backward);
    return out;
}

void multiply_backward(NodeList *nl, VarNode *var) {
    GET_NODE(var->prev_left).grad += GET_NODE(var->prev_right).num * var->grad;
    GET_NODE(var->prev_right).grad += GET_NODE(var->prev_left).num * var->grad;
}

size_t multiply(NodeList *nl, size_t left, size_t right) {
    float val = GET_NODE(left).num * GET_NODE(right).num;
    size_t out = init_var(nl, val, left, right, multiply_backward);
    return out;
}

void raise_backward(NodeList *nl, VarNode *var) {
    float l_num = GET_NODE(var->prev_left).num;
    float r_num = GET_NODE(var->prev_right).num;
    GET_NODE(var->prev_left).grad += r_num * powf(l_num, r_num - 1) * var->grad;
    GET_NODE(var->prev_right).grad +=
        logf(l_num) * powf(l_num, r_num) * var->grad;
}

size_t raise(NodeList *nl, size_t to_raise, size_t pow) {
    float val = powf(to_raise, pow);
    size_t out = init_var(nl, val, to_raise, pow, raise_backward);
    return out;
}

void relu_backward(NodeList *nl, VarNode *var) {
    if (var->num > 0) {
        GET_NODE(var->prev_left).grad += var->grad;
    }
}

size_t relu(NodeList *nl, size_t x) {
    float val = GET_NODE(x).num > 0 ? GET_NODE(x).num : 0;
    size_t out = init_var(nl, val, x, 0, relu_backward);
    return out;
}

void tanh_backward(NodeList *nl, VarNode *var) {
    GET_NODE(var->prev_left).grad += (1 - powf(var->num, 2)) * var->grad;
}

size_t hyper_tan(NodeList *nl, size_t x) {
    float val = tanhf(GET_NODE(x).num);
    size_t out = init_var(nl, val, x, 0, tanh_backward);
    return out;
}

void sigmoid_backward(NodeList *nl, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->num * (1 - var->num) * var->grad;
}

size_t sigmoid(NodeList *nl, size_t x) {
    float val = 1 / (1 + expf(-GET_NODE(x).num));
    size_t out = init_var(nl, val, x, 0, sigmoid_backward);
    return out;
}

void topo(NodeList *nl, VarNode ***list, size_t a, size_t *i) {
    VarNode *var = &GET_NODE(a);
    var->visited = 1;

    if (var->prev_right != 0 && !(GET_NODE(var->prev_right).visited)) {
        topo(nl, list, var->prev_right, i);
    }
    if (var->prev_left != 0 && !(GET_NODE(var->prev_left).visited)) {
        topo(nl, list, var->prev_left, i);
    }
    (*list)[*i] = var;
    (*i)++;
}

void backward(NodeList *nl, size_t y) {
    GET_NODE(y).grad = 1;
    size_t count = 0;
    VarNode **list = CLEAR_NET_ALLOC(nl->length * sizeof(VarNode));
    topo(nl, &list, y, &count);
    for (size_t i = count; i > 0; --i) {
        if (list[i - 1]->backward != NULL) {
            list[i - 1]->backward(nl, list[i - 1]);
        }
    }
    CLEAR_NET_DEALLOC(list);
}

#endif // CLEAR_NET_IMPLEMENTATION
