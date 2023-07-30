/*
  TODO add a name space to things, cn_ for public _cn for private
  TODO Activation for: elu, Leak_Relu
  TODO momentum
  TODO stochastic gradient descent
  TODO save and load a net
 */
#ifndef CLEAR_NET
#define CLEAR_NET

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
#define CLEAR_NET_ACT_HIDDEN ReLU
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

#ifndef CLEAR_NET_PARAM_LIST_LENGTH
#define CLEAR_NET_PARAM_LIST_LENGTH 10
#endif // CLEAR_NET_PARAM_LIST_LENGTH

/* Declaration: Helpers */
float randf(void);

/* Declaration: Automatic Differentiation Engine */
typedef struct VarNode VarNode;
typedef struct NodeStore NodeStore;
typedef void BackWardFunction(NodeStore *nl, VarNode *var);

NodeStore alloc_node_store(size_t length);
void dealloc_node_store(NodeStore *nl);
size_t init_var(NodeStore *nl, float num, size_t prev_left, size_t prev_right,
                BackWardFunction *backward);
size_t init_leaf_var(NodeStore *nl, float num);
size_t add(NodeStore *nl, size_t left, size_t right);
void add_backward(NodeStore *nl, VarNode *var);
size_t subtract(NodeStore *nl, size_t left, size_t right);
void subtract_backward(NodeStore *nl, VarNode *var);
size_t multiply(NodeStore *nl, size_t left, size_t right);
void multiply_backward(NodeStore *nl, VarNode *var);
void relu_backward(NodeStore *nl, VarNode *var);
size_t reluv(NodeStore *nl, size_t x);
void tanh_backward(NodeStore *nl, VarNode *var);
size_t hyper_tanv(NodeStore *nl, size_t x);
void sigmoid_backward(NodeStore *nl, VarNode *var);
size_t sigmoidv(NodeStore *nl, size_t x);
void backward(NodeStore *nl, size_t y);

/* Declaration: Activation Functions */
typedef enum {
    Sigmoid,
    ReLU,
    Tanh,
} Activation;

/* Declaration: Linear Algebra */
typedef struct Matrix Matrix;

Matrix alloc_matrix(size_t nrows, size_t ncols);
void dealloc_matrix(Matrix* mat);
Matrix matrix_form(size_t nrows, size_t ncols, size_t stride,
                       float *elements);
void matrix_print(Matrix mat, char *name);

typedef struct Vector Vector;
Vector alloc_vector(size_t nelem);
void dealloc_vector(Vector* vec);
void vector_print(Vector vec, char *name);
void _vec_print_res(Vector vec);

/* Declaration: Net */
typedef struct Net Net;
typedef struct DenseLayer DenseLayer;

Net alloc_net(size_t *shape, size_t nlayers);
void dealloc_net(Net *net);
float net_learn(Net *net, Matrix input, Matrix target);
Vector _net_predict(Net *net, NodeStore *nl, Vector input);
Vector _net_predict_layer(DenseLayer layer, NodeStore *nl, Vector prev_output);
void net_randomize(Net net, float lower, float upper);
size_t _activate(NodeStore *nl, size_t id, Activation act);
void net_print(Net net, char *name);

#endif // CLEAR_NET

#ifdef CLEAR_NET_IMPLEMENTATION

/* Implement: Helpers */
float randf(void) { return (float)rand() / (float)RAND_MAX; }

/* Implement: Automatic Differentiation Engine */
#define CLEAR_NET_EXTEND_LENGTH_FUNCTION(len)               \
    ((len) == 0 ? CLEAR_NET_PARAM_LIST_LENGTH : ((len)*2))
#define GET_NODE(id) (ns)->vars[(id)]

struct NodeStore {
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

NodeStore alloc_node_store(size_t length) {
    return (NodeStore){
        .vars = CLEAR_NET_ALLOC(length * sizeof(VarNode)),
        .length = 1,
        .max_length = length,
    };
}

void dealloc_node_store(NodeStore *ns) { CLEAR_NET_DEALLOC(ns->vars); }

VarNode create_var(float num, size_t prev_left, size_t prev_right,
                   BackWardFunction *backward) {
    return (VarNode){
        .num = num,
        .grad = 0,
        .prev_left = prev_left,
        .prev_right = prev_right,
        .backward = backward,
    };
}

size_t init_var(NodeStore *ns, float num, size_t prev_left, size_t prev_right,
                BackWardFunction *backward) {
    if (ns->length >= ns->max_length) {
        ns->max_length = CLEAR_NET_EXTEND_LENGTH_FUNCTION(ns->max_length);
        ns->vars =
            CLEAR_NET_REALLOC(ns->vars, ns->max_length * sizeof(VarNode));
        CLEAR_NET_ASSERT(ns->vars);
    }
    VarNode out = create_var(num, prev_left, prev_right, backward);
    ns->vars[ns->length] = out;
    ns->length++;
    return ns->length - 1;
}

size_t init_leaf_var(NodeStore *ns, float num) {
    return init_var(ns, num, 0, 0, NULL);
}

void add_backward(NodeStore *ns, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->grad;
    GET_NODE(var->prev_right).grad += var->grad;
}

size_t add(NodeStore *ns, size_t left, size_t right) {
    float val = GET_NODE(left).num + GET_NODE(right).num;
    size_t out = init_var(ns, val, left, right, add_backward);
    return out;
}

void subtract_backward(NodeStore *ns, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->grad;
    GET_NODE(var->prev_right).grad -= var->grad;
}

size_t subtract(NodeStore *ns, size_t left, size_t right) {
    float val = GET_NODE(left).num - GET_NODE(right).num;
    size_t out = init_var(ns, val, left, right, subtract_backward);
    return out;
}

void multiply_backward(NodeStore *ns, VarNode *var) {
    GET_NODE(var->prev_left).grad += GET_NODE(var->prev_right).num * var->grad;
    GET_NODE(var->prev_right).grad += GET_NODE(var->prev_left).num * var->grad;
}

size_t multiply(NodeStore *ns, size_t left, size_t right) {
    float val = GET_NODE(left).num * GET_NODE(right).num;
    size_t out = init_var(ns, val, left, right, multiply_backward);
    return out;
}

void raise_backward(NodeStore *ns, VarNode *var) {
    float l_num = GET_NODE(var->prev_left).num;
    float r_num = GET_NODE(var->prev_right).num;
    GET_NODE(var->prev_left).grad += r_num * powf(l_num, r_num - 1) * var->grad;
    GET_NODE(var->prev_right).grad +=
        logf(l_num) * powf(l_num, r_num) * var->grad;
}

size_t raise(NodeStore *ns, size_t to_raise, size_t pow) {
    float val = powf(GET_NODE(to_raise).num, GET_NODE(pow).num);
    size_t out = init_var(ns, val, to_raise, pow, raise_backward);
    return out;
}

void relu_backward(NodeStore *ns, VarNode *var) {
    if (var->num > 0) {
        GET_NODE(var->prev_left).grad += var->grad;
    }
}

float relu(float x) {
    return x > 0 ? x : 0;
}

size_t reluv(NodeStore *ns, size_t x) {
    float val = relu(GET_NODE(x).num);
    size_t out = init_var(ns, val, x, 0, relu_backward);
    return out;
}

void tanh_backward(NodeStore *ns, VarNode *var) {
    GET_NODE(var->prev_left).grad += (1 - powf(var->num, 2)) * var->grad;
}

float hyper_tan(float x) {
    return tanhf(x);
}

size_t hyper_tanv(NodeStore *ns, size_t x) {
    float val = hyper_tan(GET_NODE(x).num);
    size_t out = init_var(ns, val, x, 0, tanh_backward);
    return out;
}

void sigmoid_backward(NodeStore *ns, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->num * (1 - var->num) * var->grad;
}

float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}

size_t sigmoidv(NodeStore *ns, size_t x) {
    float val = sigmoid(GET_NODE(x).num);
    size_t out = init_var(ns, val, x, 0, sigmoid_backward);
    return out;
}

void backward(NodeStore *ns, size_t y) {
    GET_NODE(y).grad = 1;
    VarNode var;
    for (size_t i = ns->length - 1; i > 0; --i) {
        var = GET_NODE(i);
        if (var.backward) {
            var.backward(ns, &var);
        }
    }
}

/* Implement: Linear Algebra */
#define MAT_ID(mat, r, c) (mat).ns_id + ((r) * (mat).stride) + (c)
#define MAT_AT(mat, r, c) (mat).elements[(r) * (mat).stride + (c)]
#define VEC_ID(vec, i) (vec).ns_id + (i)
#define VEC_AT(vec, i) (vec).elements[i]
#define MATRIX_PRINT(mat) matrix_print((mat), #mat)
#define VECTOR_PRINT(vec) vector_print((vec),#vec)

struct Matrix {
    float *elements;
    size_t ns_id;
    size_t stride;
    size_t nrows;
    size_t ncols;
};

Matrix alloc_matrix(size_t nrows, size_t ncols) {
    Matrix mat;
    mat.nrows = nrows;
    mat.ncols = ncols;
    mat.stride = ncols;
    mat.elements = CLEAR_NET_ALLOC(nrows * ncols * sizeof(*mat.elements));
    CLEAR_NET_ASSERT(mat.elements != NULL);
    return mat;
}

void dealloc_matrix(Matrix* mat) {
    CLEAR_NET_DEALLOC(mat->elements);
    mat->nrows = 0;
    mat->ncols = 0;
    mat->stride = 0;
    mat->ns_id = 0;
    mat->elements = NULL;
}

Matrix matrix_form(size_t nrows, size_t ncols, size_t stride,
                            float *elements) {
    return (Matrix){
        .ns_id = 0, .nrows = nrows, .ncols = ncols, .stride = stride, .elements = elements};
}

void matrix_print(Matrix mat, char *name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < mat.nrows; ++i) {
        printf("    ");
        for (size_t j = 0; j < mat.ncols; ++j) {
            printf("%f ", MAT_AT(mat, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

struct Vector {
    float *elements;
    size_t ns_id;
    size_t nelem;
};

Vector alloc_vector(size_t nelem) {
    Vector vec;
    vec.nelem = nelem;
    vec.elements = CLEAR_NET_ALLOC(nelem * sizeof(*vec.elements));
    CLEAR_NET_ASSERT(vec.elements != NULL);
    return vec;
}

void dealloc_vector(Vector* vec) {
    CLEAR_NET_DEALLOC(vec->elements);
    vec->nelem = 0;
    vec->ns_id = 0;
    vec->elements = NULL;
}

Vector vector_form(size_t nelem, float *elements) {
    return (Vector){
        .ns_id = 0,
        .nelem = nelem,
        .elements = elements,
    };
}


void vector_print(Vector vec, char *name) {
    printf("%s = [\n", name);
    printf("    ");
    for (size_t i = 0; i < vec.nelem; ++i) {
        printf("%f ", VEC_AT(vec, i));
    }
    printf("\n]\n");
}


/* Implement: Net */
struct DenseLayer {
    Matrix weights;
    Vector biases;
    Activation act;
    size_t *output_ns_ids;
    Vector output;
};

struct Net {
    DenseLayer *layers;
    NodeStore computation_graph;
    size_t nlayers;
    size_t nparam;
};

Net alloc_net(size_t *shape, size_t nlayers) {
    CLEAR_NET_ASSERT(nlayers != 0);

    Net net;
    net.nlayers = nlayers;
    net.layers = CLEAR_NET_ALLOC((nlayers - 1) * sizeof(DenseLayer));
    // Length calculation
    // | number of weights | biases
    // (shape[0] * shape[1]) + shape[1]
    size_t nparam = 1;
    for (size_t i = 0; i < nlayers - 1; ++i) {
        DenseLayer layer;
        if (i == nlayers - 2) {
            layer.act = CLEAR_NET_ACT_OUTPUT;
        } else {
            layer.act = CLEAR_NET_ACT_HIDDEN;
        }
        Matrix mat;
        mat.ns_id = nparam;
        mat.nrows = shape[i];
        mat.ncols = shape[i + 1];
        mat.stride = mat.ncols;
        mat.elements = CLEAR_NET_ALLOC(mat.nrows * mat.ncols * sizeof(*mat.elements));
        layer.weights = mat;
        nparam += (layer.weights.nrows * layer.weights.ncols);

        Vector vec;
        vec.ns_id = nparam;
        vec.nelem = shape[i + 1];
        vec.elements = CLEAR_NET_ALLOC(vec.nelem * sizeof(*vec.elements));
        layer.biases = vec;
        nparam += layer.biases.nelem;

        layer.output_ns_ids = CLEAR_NET_ALLOC(layer.biases.nelem * sizeof(*layer.output_ns_ids));
        layer.output = (Vector) {
            .elements = CLEAR_NET_ALLOC(layer.biases.nelem * sizeof(*layer.output.elements)),
            .nelem = layer.biases.nelem,
            .ns_id = 0,
        };

        net.layers[i] = layer;
    }
    net.nparam = nparam;
    net.computation_graph = alloc_node_store(net.nparam);
    return net;
}

void dealloc_net(Net *net) {
    for (size_t i = 0; i < net->nlayers; ++i) {
        dealloc_matrix(&net->layers[i].weights);
        dealloc_vector(&net->layers[i].biases);
        dealloc_vector(&net->layers[i].output);
        CLEAR_NET_DEALLOC(net->layers[i].output_ns_ids);
    }
    dealloc_node_store(&net->computation_graph);
}

size_t _activate(NodeStore *ns, size_t id, Activation act) {
    switch (act) {
    case ReLU:
        return reluv(ns, id);
    case Sigmoid:
        return sigmoidv(ns, id);
    case Tanh:
        return hyper_tanv(ns, id);
    }
}

float activate(float x, Activation act) {
    switch (act) {
    case ReLU:
        return relu(x);
    case Sigmoid:
        return sigmoid(x);
    case Tanh:
        return hyper_tan(x);
    }
}

void net_randomize(Net net, float lower, float upper) {
    for (size_t i = 0; i < net.nlayers - 1; ++i) {
        DenseLayer layer = net.layers[i];
        for (size_t j = 0; j < layer.weights.nrows; ++j) {
            for (size_t k = 0; k < layer.weights.ncols; ++k) {
                MAT_AT(layer.weights, j, k) =
                    randf() * (upper - lower) + lower;
            }
        }
        for (size_t j = 0; j < layer.biases.nelem; ++j) {
            VEC_AT(layer.biases, j) = randf() * (upper - lower) + lower;
        }
    }
}

void net_print(Net net, char *name) {
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < net.nlayers - 1; ++i) {
        DenseLayer layer = net.layers[i];
        snprintf(buf, sizeof(buf), "weight matrix: %zu", i);
        matrix_print(layer.weights, buf);
        snprintf(buf, sizeof(buf), "bias vector: %zu", i);
        vector_print(layer.biases, buf);
    }
}

Vector _net_predict_layer(DenseLayer layer, NodeStore *ns, Vector prev_output) {
    for (size_t i = 0; i < layer.weights.ncols; ++i) {
        size_t res = init_leaf_var(ns, 0);
        for (size_t j = 0; j < prev_output.nelem; ++j) {
            res = add(ns, res,
                      multiply(ns,
                               MAT_ID(layer.weights, j, i),
                               VEC_ID(prev_output, j)
                               ));
        }
        res = add(ns, res, VEC_ID(layer.biases, i));
        res = _activate(ns, res, layer.act);
        layer.output_ns_ids[i] = res;
    }

    Vector out = (Vector) {
        .ns_id = ns->length,
        .nelem = layer.weights.ncols,
    };

    for (size_t i = 0; i < layer.weights.ncols; ++i) {
        init_leaf_var(ns, 0);
        GET_NODE(VEC_ID(out, i)) = GET_NODE(layer.output_ns_ids[i]);
    }
    return out;
}

Vector _net_predict(Net *net, NodeStore *ns, Vector input) {
    CLEAR_NET_ASSERT(input.nelem == net->layers[0].weights.nrows);
    Vector guess = input;
    for (size_t i = 0; i < net->nlayers - 1; ++i) {
        guess = _net_predict_layer(net->layers[i], ns, guess);
     }

    return guess;
}

float net_learn_one_input(Net *net, NodeStore *ns, Matrix input, Matrix target) {
    Vector input_vec = (Vector) {
        .ns_id = ns->length,
        .nelem = input.ncols,
    };
    for (size_t i = 0; i < input_vec.nelem; ++i) {
        init_leaf_var(ns, MAT_AT(input, 0, i));
    }

    Vector prediction = _net_predict(net, ns, input_vec);
    Vector target_vec = (Vector) {
        .ns_id = ns->length,
        .nelem = target.ncols,
    };
    for (size_t i = 0; i < target.ncols; ++i) {
        init_leaf_var(ns, MAT_AT(target, 0, i));
    }
    size_t loss = init_leaf_var(ns, 0);
    for (size_t i = 0; i < target.ncols; ++i) {
        loss = add(ns, loss,
                   raise(ns,
                         subtract(ns,
                         VEC_ID(target_vec, i),
                         VEC_ID(prediction, i)),
                                  init_leaf_var(ns, 2)));
    }

    backward(ns, loss);

    return GET_NODE(loss).num;
}

float net_learn(Net *net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t train_size = input.nrows;
    net->computation_graph.length = 1;
    NodeStore *ns = &net->computation_graph;

    for (size_t i = 0; i < net->nlayers - 1; ++i) {
        for (size_t j = 0; j < net->layers[i].weights.nrows; ++j) {
            for (size_t k = 0; k < net->layers[i].weights.ncols; ++k) {
                init_leaf_var(ns, MAT_AT(net->layers[i].weights, j ,k));
            }
        }
        for (size_t j = 0; j < net->layers[i].biases.nelem; ++j) {
            init_leaf_var(ns, VEC_AT(net->layers[i].biases, j));
        }
    }

    Matrix one_input;
    Matrix one_target;
    float total_loss = 0;
    for (size_t i = 0; i < train_size; ++i) {
        one_input = matrix_form(1, input.ncols, input.stride, &MAT_AT(input, i, 0));
        one_target = matrix_form(1, target.ncols, target.stride, &MAT_AT(target, i, 0));
        total_loss += net_learn_one_input(net, ns, one_input, one_target);
        ns->length = net->nparam;
    }

    for (size_t i = 0; i < net->nlayers - 1; ++i) {
        for (size_t j = 0; j < net->layers[i].weights.nrows; ++j){
            for (size_t k = 0; k < net->layers[i].weights.ncols; ++k){
                MAT_AT(net->layers[i].weights, j, k) -= CLEAR_NET_RATE * GET_NODE(MAT_ID(net->layers[i].weights, j, k)).grad;
            }
        }
        for (size_t j = 0; j < net->layers[i].biases.nelem; ++j) {
            VEC_AT(net->layers[i].biases, j) -= CLEAR_NET_RATE * GET_NODE(VEC_ID(net->layers[i].biases, j)).grad;
        }
    }

    return total_loss / train_size;
}

Vector net_predict_layer(DenseLayer layer, Vector prev_output) {
    for (size_t i = 0; i < layer.weights.ncols; ++i) {
        float res = 0;
        for (size_t j = 0; j < prev_output.nelem; ++j) {
            res += MAT_AT(layer.weights, j, i) * VEC_AT(prev_output, j);
        }
        res += VEC_AT(layer.biases, i);
        res = activate(res, layer.act);
        layer.output.elements[i] = res;
    }
    return layer.output;
}

Vector net_predict(Net net, Vector input) {
    Vector guess = input;

    for (size_t i = 0; i < net.nlayers -1; ++i) {
        guess = net_predict_layer(net.layers[i], guess);
    }

    return guess;
}

void _vec_print_res(Vector vec) {
    for (size_t j = 0; j < vec.nelem; ++j) {
        printf("%f ", VEC_AT(vec, j));
    }
    printf("| ");
}

void net_print_results(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t size = input.nrows;
    printf("Input | Net Output | Target\n");
    for (size_t i = 0; i < size; ++i) {
        Vector in = vector_form(input.ncols, &MAT_AT(input, i, 0));
        Vector tar = vector_form(target.ncols, &MAT_AT(target, i, 0));
        Vector out = net_predict(net, in);
        _vec_print_res(in);
        _vec_print_res(out);
        _vec_print_res(tar);
        printf("\n");
    }
}



#endif // CLEAR_NET_IMPLEMENTATION
