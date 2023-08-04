/*
   Clear Net by Timothy Cronin

   To the extent possible under law, the person who associated CC0 with
   Clear Net has waived all copyright and related or neighboring rights
   to Clear Net.

   You should have received a copy of the CC0 legalcode along with this
   work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

   See end of file for full license.
*/
/*
  TODO make a print output function for easy examining of mnist, anything with many input dimensions
*/
/* Beginning */
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
#include "assert.h"
#define CLEAR_NET_ASSERT assert
#endif // CLEAR_NET_ASSERT
#ifndef CLEAR_NET_ACT_NEG_SCALE
#define CLEAR_NET_ACT_NEG_SCALE 0.1f
#endif // CLEAR_NET_NEG_SCALE

#ifndef CLEAR_NET_MOMENTUM
#define CLEAR_NET_MOMENTUM 0
#endif // CLEAR_NET_MOMENTUM
#ifndef CLEAR_NET_MOMENTUM_BETA
#define CLEAR_NET_MOMENTUM_BETA 0.9
#endif // CLEAR_NET_MOMENTUM_BETA

#ifndef CLEAR_NET_INITIAL_GRAPH_LENGTH
#define CLEAR_NET_INITIAL_GRAPH_LENGTH 10
#endif // CLEAR_NET_INITIAL_GRAPH_LENGTH

/* Declaration: Helpers */
float _cn_randf(void);
void _cn_fill_floats(float *ptr, size_t len, float val);

/* Declaration: Automatic Differentiation Engine */
typedef struct VarNode VarNode;
typedef struct GradientStore GradientStore;
typedef void BackWardFunction(GradientStore *nl, VarNode *var);

GradientStore cn_alloc_gradient_store(size_t length);
void cn_dealloc_gradient_store(GradientStore *nl);
size_t _cn_init_var(GradientStore *nl, float num, size_t prev_left,
                    size_t prev_right, BackWardFunction *backward);
size_t cn_init_leaf_var(GradientStore *nl, float num);
size_t cn_add(GradientStore *nl, size_t left, size_t right);
void _cn_add_backward(GradientStore *nl, VarNode *var);
size_t cn_subtract(GradientStore *nl, size_t left, size_t right);
void _cn_subtract_backward(GradientStore *nl, VarNode *var);
size_t cn_multiply(GradientStore *nl, size_t left, size_t right);
void _cn_multiply_backward(GradientStore *nl, VarNode *var);
size_t cn_raise(GradientStore *gs, size_t to_raise, size_t pow);
void _cn_raise_backward(GradientStore *gs, VarNode *var);
void relu_backward(GradientStore *nl, VarNode *var);
size_t cn_reluv(GradientStore *nl, size_t x);
void _cn_tanh_backward(GradientStore *nl, VarNode *var);
size_t cn_hyper_tanv(GradientStore *nl, size_t x);
void _cn_sigmoid_backward(GradientStore *nl, VarNode *var);
size_t cn_sigmoidv(GradientStore *nl, size_t x);
void _cn_leaky_relu_backward(GradientStore *nl, VarNode *var);
size_t cn_leaky_reluv(GradientStore *nl, size_t x);
void _cn_elu_backward(GradientStore *gs, VarNode *var);
size_t cn_eluv(GradientStore *gs, size_t x);
void cn_backward(GradientStore *nl, size_t y);

/* Declaration: Activation Functions */
typedef enum {
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU,
    ELU,
} Activation;

float cn_sigmoid(float x);
float cn_relu(float x);
float cn_hyper_tan(float x);
float cn_leaky_relu(float x);
float cn_elu(float x);

/* Declaration: Linear Algebra */
typedef struct Matrix Matrix;

Matrix cn_alloc_matrix(size_t nrows, size_t ncols);
void cn_dealloc_matrix(Matrix *mat);
Matrix cn_form_matrix(size_t nrows, size_t ncols, size_t stride,
                      float *elements);
void cn_print_matrix(Matrix mat, char *name);
void cn_shuffle_matrix_rows(Matrix mat);

typedef struct Vector Vector;
Vector _cn_alloc_vector(size_t nelem);
void _cn_dealloc_vector(Vector *vec);
Vector cn_form_vector(size_t nelem, float *elements);
void _cn_print_vector(Vector vec, char *name);
void _cn_print_vector_res(Vector vec);

/* Declaration: Net */
typedef struct NetConfig NetConfig;
typedef struct Net Net;
typedef struct DenseLayer DenseLayer;

Net cn_alloc_net(NetConfig net_conf);
void cn_dealloc_net(Net *net);
float cn_learn(Net *net, Matrix input, Matrix target);
Vector cn_predict(Net net, Vector input);
Vector cn_predict_layer(DenseLayer layer, Vector prev_output);
float _cn_find_grad(Net *net, GradientStore *gs, Vector input, Vector target);
Vector _cn_predict(Net *net, GradientStore *nl, Vector input);
Vector _cn_predict_layer(DenseLayer layer, GradientStore *nl,
                         Vector prev_output);
void cn_randomize_net(Net net, float lower, float upper);
size_t _cn_activate(GradientStore *nl, size_t id, Activation act);
float cn_error(Net net, Matrix input, Matrix target);
void cn_print_net(Net net, char *name);
void cn_print_net_results(Net net, Matrix input, Matrix target);
void cn_save_net_to_file(Net net, char *file_name);
Net cn_alloc_net_from_file(char *file_name);

#endif // CLEAR_NET

#ifdef CLEAR_NET_IMPLEMENTATION

/* Implement: Helpers */
float _cn_randf(void) { return (float)rand() / (float)RAND_MAX; }

void _cn_fill_floats(float *ptr, size_t len, float val) {
    for (size_t i = 0; i < len; ++i) {
        ptr[i] = val;
    }
}

/* Implement: Automatic Differentiation Engine */
#define CLEAR_NET_EXTEND_LENGTH_FUNCTION(len)                                  \
    ((len) == 0 ? CLEAR_NET_INITIAL_GRAPH_LENGTH : ((len)*2))
#define GET_NODE(id) (gs)->vars[(id)]

struct GradientStore {
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

GradientStore cn_alloc_gradient_store(size_t length) {
    return (GradientStore){
        .vars = CLEAR_NET_ALLOC(length * sizeof(VarNode)),
        .length = 1,
        .max_length = length,
    };
}

void cn_dealloc_gradient_store(GradientStore *gs) {
    CLEAR_NET_DEALLOC(gs->vars);
}

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

size_t _cn_init_var(GradientStore *gs, float num, size_t prev_left,
                    size_t prev_right, BackWardFunction *backward) {
    if (gs->length >= gs->max_length) {
        gs->max_length = CLEAR_NET_EXTEND_LENGTH_FUNCTION(gs->max_length);
        gs->vars =
            CLEAR_NET_REALLOC(gs->vars, gs->max_length * sizeof(VarNode));
        CLEAR_NET_ASSERT(gs->vars);
    }
    VarNode out = create_var(num, prev_left, prev_right, backward);
    gs->vars[gs->length] = out;
    gs->length++;
    return gs->length - 1;
}

size_t cn_init_leaf_var(GradientStore *gs, float num) {
    return _cn_init_var(gs, num, 0, 0, NULL);
}

void _cn_add_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->grad;
    GET_NODE(var->prev_right).grad += var->grad;
}

size_t cn_add(GradientStore *gs, size_t left, size_t right) {
    float val = GET_NODE(left).num + GET_NODE(right).num;
    size_t out = _cn_init_var(gs, val, left, right, _cn_add_backward);
    return out;
}

void _cn_subtract_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->grad;
    GET_NODE(var->prev_right).grad -= var->grad;
}

size_t cn_subtract(GradientStore *gs, size_t left, size_t right) {
    float val = GET_NODE(left).num - GET_NODE(right).num;
    size_t out = _cn_init_var(gs, val, left, right, _cn_subtract_backward);
    return out;
}

void _cn_multiply_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += GET_NODE(var->prev_right).num * var->grad;
    GET_NODE(var->prev_right).grad += GET_NODE(var->prev_left).num * var->grad;
}

size_t cn_multiply(GradientStore *gs, size_t left, size_t right) {
    float val = GET_NODE(left).num * GET_NODE(right).num;
    size_t out = _cn_init_var(gs, val, left, right, _cn_multiply_backward);
    return out;
}

void _cn_raise_backward(GradientStore *gs, VarNode *var) {
    float l_num = GET_NODE(var->prev_left).num;
    float r_num = GET_NODE(var->prev_right).num;
    GET_NODE(var->prev_left).grad += r_num * powf(l_num, r_num - 1) * var->grad;
    GET_NODE(var->prev_right).grad +=
        logf(l_num) * powf(l_num, r_num) * var->grad;
}

size_t cn_raise(GradientStore *gs, size_t to_cn_raise, size_t pow) {
    float val = powf(GET_NODE(to_cn_raise).num, GET_NODE(pow).num);
    size_t out = _cn_init_var(gs, val, to_cn_raise, pow, _cn_raise_backward);
    return out;
}

float cn_relu(float x) { return x > 0 ? x : 0; }

void cn_relu_backward(GradientStore *gs, VarNode *var) {
    if (var->num > 0) {
        GET_NODE(var->prev_left).grad += var->grad;
    }
}

size_t cn_reluv(GradientStore *gs, size_t x) {
    float val = cn_relu(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, cn_relu_backward);
    return out;
}

float cn_hyper_tan(float x) { return tanhf(x); }

void _cn_tanh_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += (1 - powf(var->num, 2)) * var->grad;
}

size_t cn_hyper_tanv(GradientStore *gs, size_t x) {
    float val = cn_hyper_tan(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_tanh_backward);
    return out;
}

float cn_sigmoid(float x) { return 1 / (1 + expf(-x)); }

void _cn_sigmoid_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->num * (1 - var->num) * var->grad;
}

size_t cn_sigmoidv(GradientStore *gs, size_t x) {
    float val = cn_sigmoid(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_sigmoid_backward);
    return out;
}

float cn_leaky_relu(float x) {
    return x >= 0 ? x : CLEAR_NET_ACT_NEG_SCALE * x;
}

void _cn_leaky_relu_backward(GradientStore *gs, VarNode *var) {
    float change = var->num >= 0 ? 1 : CLEAR_NET_ACT_NEG_SCALE;
    GET_NODE(var->prev_left).grad += change * var->grad;
}

size_t cn_leaky_reluv(GradientStore *gs, size_t x) {
    float val = cn_leaky_relu(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_leaky_relu_backward);
    return out;
}

float cn_elu(float x) {
    return x > 0 ? x : CLEAR_NET_ACT_NEG_SCALE * (expf(x) - 1);
}

void _cn_elu_backward(GradientStore *gs, VarNode *var) {
    float change = var->num > 0 ? 1 : var->num + CLEAR_NET_ACT_NEG_SCALE;
    GET_NODE(var->prev_left).grad += change * var->grad;
}

size_t cn_eluv(GradientStore *gs, size_t x) {
    float val = cn_elu(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_elu_backward);
    return out;
}

void cn_backward(GradientStore *gs, size_t y) {
    GET_NODE(y).grad = 1;
    VarNode *var;
    for (size_t i = gs->length - 1; i > 0; --i) {
        var = &GET_NODE(i);
        if (var->backward) {
            var->backward(gs, var);
        }
    }
}

/* Implement: Linear Algebra */
#define MAT_ID(mat, r, c) (mat).gs_id + ((r) * (mat).stride) + (c)
#define MAT_AT(mat, r, c) (mat).elements[(r) * (mat).stride + (c)]
#define VEC_ID(vec, i) (vec).gs_id + (i)
#define VEC_AT(vec, i) (vec).elements[i]
#define CN_PRINT_MATRIX(mat) cn_print_matrix((mat), #mat)
#define _CN_PRINT_VECTOR(vec) _cn_print_vector((vec), #vec)

struct Matrix {
    float *elements;
    float *grad_stores;
    size_t gs_id;
    size_t stride;
    size_t nrows;
    size_t ncols;
};

Matrix cn_alloc_matrix(size_t nrows, size_t ncols) {
    Matrix mat;
    mat.nrows = nrows;
    mat.ncols = ncols;
    mat.stride = ncols;
    mat.elements = CLEAR_NET_ALLOC(nrows * ncols * sizeof(*mat.elements));
    CLEAR_NET_ASSERT(mat.elements != NULL);
    mat.grad_stores = NULL;
    mat.gs_id = 0;
    return mat;
}

void cn_dealloc_matrix(Matrix *mat) {
    CLEAR_NET_DEALLOC(mat->elements);
    if (mat->grad_stores != NULL) {
        CLEAR_NET_DEALLOC(mat->grad_stores);
    }
    mat->nrows = 0;
    mat->ncols = 0;
    mat->stride = 0;
    mat->gs_id = 0;
    mat->elements = NULL;
}

Matrix cn_form_matrix(size_t nrows, size_t ncols, size_t stride,
                      float *elements) {
    return (Matrix){.gs_id = 0,
                    .nrows = nrows,
                    .ncols = ncols,
                    .stride = stride,
                    .elements = elements};
}

void cn_print_matrix(Matrix mat, char *name) {
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

void cn_shuffle_matrix_rows(Matrix mat) {
    for (size_t i = 0; i < mat.nrows; ++i) {
        size_t j = i + rand() % (mat.nrows - i);
        if (i != j) {
            for (size_t k = 0; k < mat.ncols; ++k) {
                float t = MAT_AT(mat, i, k);
                MAT_AT(mat, i, k) = MAT_AT(mat, j, k);
                MAT_AT(mat, j, k) = t;
            }
        }
    }
}

struct Vector {
    float *elements;
    float *grad_stores;
    size_t gs_id;
    size_t nelem;
};

Vector _cn_alloc_vector(size_t nelem) {
    Vector vec;
    vec.nelem = nelem;
    vec.elements = CLEAR_NET_ALLOC(nelem * sizeof(*vec.elements));
    CLEAR_NET_ASSERT(vec.elements != NULL);
    vec.grad_stores = NULL;
    return vec;
}

void _cn_dealloc_vector(Vector *vec) {
    CLEAR_NET_DEALLOC(vec->elements);
    if (vec->grad_stores != NULL) {
        CLEAR_NET_DEALLOC(vec->grad_stores);
    }
    vec->nelem = 0;
    vec->gs_id = 0;
    vec->elements = NULL;
}

Vector cn_form_vector(size_t nelem, float *elements) {
    return (Vector){
        .gs_id = 0,
        .nelem = nelem,
        .elements = elements,
    };
}

void _cn_print_vector(Vector vec, char *name) {
    printf("%s = [\n", name);
    printf("    ");
    for (size_t i = 0; i < vec.nelem; ++i) {
        printf("%f ", VEC_AT(vec, i));
    }
    printf("\n]\n");
}

void _cn_print_vector_res(Vector vec) {
    for (size_t j = 0; j < vec.nelem; ++j) {
        printf("%f ", VEC_AT(vec, j));
    }
    printf("| ");
}

/* Implement: Net */
struct NetConfig {
    size_t *shape;
    size_t shape_allocated;
    Activation *activations;
    size_t activations_allocated;
    size_t nparam;
    size_t nlayers;
    size_t with_momentum;
    float momentum_beta;
    float rate;
};

struct DenseLayer {
    Matrix weights;
    Vector biases;
    Activation act;
    size_t *output_gs_ids;
    Vector output;
};

struct Net {
    DenseLayer *layers;
    GradientStore computation_graph;
    NetConfig hparams;
};

NetConfig cn_init_net_conf(size_t *shape, size_t shape_allocated,
                           size_t nlayers, Activation *acts,
                           size_t activations_allocated, float rate) {
    return (NetConfig){.shape = shape,
                       .shape_allocated = shape_allocated,
                       .nlayers = nlayers,
                       .activations = acts,
                       .activations_allocated = activations_allocated,
                       .rate = rate};
}

NetConfig cn_alloc_default_conf(size_t *shape, size_t shape_allocated,
                                size_t nlayers) {
    Activation *acts =
        (Activation *)CLEAR_NET_ALLOC((nlayers - 1) * sizeof(Activation));
    for (size_t i = 0; i < nlayers - 1; i++) {
        acts[i] = i == nlayers - 2 ? Sigmoid : ReLU;
    }
    size_t activations_allocated = 1;
    float rate = 0.5;
    return cn_init_net_conf(shape, shape_allocated, nlayers, acts,
                            activations_allocated, rate);
}

void cn_with_momentum(NetConfig *nc, float momentum_beta) {
    nc->with_momentum = 1;
    nc->momentum_beta = momentum_beta;
}

Net cn_alloc_net(NetConfig net_conf) {
    CLEAR_NET_ASSERT(net_conf.nlayers != 0);

    Net net;
    net.layers = CLEAR_NET_ALLOC((net_conf.nlayers - 1) * sizeof(DenseLayer));
    // Length calculation
    // | number of weights | biases
    // (shape[0] * shape[1]) + shape[1]
    size_t offset = 1;
    for (size_t i = 0; i < net_conf.nlayers - 1; ++i) {
        DenseLayer layer;
        layer.act = net_conf.activations[i];
        Matrix mat = cn_alloc_matrix(net_conf.shape[i], net_conf.shape[i + 1]);
        mat.gs_id = offset;
        if (net_conf.with_momentum) {
            mat.grad_stores = CLEAR_NET_ALLOC(mat.nrows * mat.ncols *
                                              sizeof(*mat.grad_stores));
            _cn_fill_floats(mat.grad_stores, mat.nrows * mat.ncols, 0);
        }
        layer.weights = mat;
        offset += (layer.weights.nrows * layer.weights.ncols);

        Vector vec = _cn_alloc_vector(net_conf.shape[i + 1]);
        vec.gs_id = offset;
        if (net_conf.with_momentum) {
            vec.grad_stores =
                CLEAR_NET_ALLOC(vec.nelem * sizeof(*vec.grad_stores));
            _cn_fill_floats(vec.grad_stores, vec.nelem, 0);
        }
        layer.biases = vec;
        offset += layer.biases.nelem;

        layer.output_gs_ids =
            CLEAR_NET_ALLOC(layer.biases.nelem * sizeof(*layer.output_gs_ids));
        Vector out = _cn_alloc_vector(layer.biases.nelem);
        out.gs_id = 0;
        layer.output = out;
        net.layers[i] = layer;
    }
    net_conf.nparam = offset - 1;
    net.hparams = net_conf;
    net.computation_graph = cn_alloc_gradient_store(net.hparams.nparam);
    return net;
}

void cn_dealloc_net(Net *net) {
    for (size_t i = 0; i < net->hparams.nlayers - 1; ++i) {
        cn_dealloc_matrix(&net->layers[i].weights);
        _cn_dealloc_vector(&net->layers[i].biases);
        _cn_dealloc_vector(&net->layers[i].output);
        CLEAR_NET_DEALLOC(net->layers[i].output_gs_ids);
    }
    if (net->hparams.shape_allocated) {
        CLEAR_NET_DEALLOC(net->hparams.shape);
    }
    if (net->hparams.activations_allocated) {
        CLEAR_NET_DEALLOC(net->hparams.activations);
    }
    cn_dealloc_gradient_store(&net->computation_graph);
}

size_t _cn_activate(GradientStore *gs, size_t id, Activation act) {
    switch (act) {
    case ReLU:
        return cn_reluv(gs, id);
    case Sigmoid:
        return cn_sigmoidv(gs, id);
    case Tanh:
        return cn_hyper_tanv(gs, id);
    case LeakyReLU:
        return cn_leaky_reluv(gs, id);
    case ELU:
        return cn_eluv(gs, id);
    }
}

float activate(float x, Activation act) {
    switch (act) {
    case ReLU:
        return cn_relu(x);
    case Sigmoid:
        return cn_sigmoid(x);
    case Tanh:
        return cn_hyper_tan(x);
    case LeakyReLU:
        return cn_leaky_relu(x);
    case ELU:
        return cn_elu(x);
    }
}

void cn_randomize_net(Net net, float lower, float upper) {
    for (size_t i = 0; i < net.hparams.nlayers - 1; ++i) {
        DenseLayer layer = net.layers[i];
        for (size_t j = 0; j < layer.weights.nrows; ++j) {
            for (size_t k = 0; k < layer.weights.ncols; ++k) {
                MAT_AT(layer.weights, j, k) =
                    _cn_randf() * (upper - lower) + lower;
            }
        }
        for (size_t j = 0; j < layer.biases.nelem; ++j) {
            VEC_AT(layer.biases, j) = _cn_randf() * (upper - lower) + lower;
        }
    }
}

void cn_print_net(Net net, char *name) {
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < net.hparams.nlayers - 1; ++i) {
        DenseLayer layer = net.layers[i];
        snprintf(buf, sizeof(buf), "weight matrix: %zu", i);
        cn_print_matrix(layer.weights, buf);
        snprintf(buf, sizeof(buf), "bias vector: %zu", i);
        _cn_print_vector(layer.biases, buf);
    }
}

Vector _cn_predict_layer(DenseLayer layer, GradientStore *gs,
                         Vector prev_output) {
    for (size_t i = 0; i < layer.weights.ncols; ++i) {
        size_t res = cn_init_leaf_var(gs, 0);
        for (size_t j = 0; j < prev_output.nelem; ++j) {
            res = cn_add(gs, res,
                         cn_multiply(gs, MAT_ID(layer.weights, j, i),
                                     VEC_ID(prev_output, j)));
        }
        res = cn_add(gs, res, VEC_ID(layer.biases, i));
        res = _cn_activate(gs, res, layer.act);
        layer.output_gs_ids[i] = res;
    }

    Vector out = (Vector){
        .gs_id = gs->length,
        .nelem = layer.weights.ncols,
    };

    for (size_t i = 0; i < out.nelem; ++i) {
        VarNode to_copy = GET_NODE(layer.output_gs_ids[i]);
        _cn_init_var(gs, to_copy.num, to_copy.prev_left, to_copy.prev_right,
                     to_copy.backward);
    }
    return out;
}

Vector _cn_predict(Net *net, GradientStore *gs, Vector input) {
    CLEAR_NET_ASSERT(input.nelem == net->layers[0].weights.nrows);
    Vector guess = input;
    for (size_t i = 0; i < net->hparams.nlayers - 1; ++i) {
        guess = _cn_predict_layer(net->layers[i], gs, guess);
    }
    return guess;
}

float _cn_find_grad(Net *net, GradientStore *gs, Vector input, Vector target) {
    input.gs_id = gs->length;
    for (size_t i = 0; i < input.nelem; ++i) {
        cn_init_leaf_var(gs, VEC_AT(input, i));
    }
    Vector prediction = _cn_predict(net, gs, input);

    target.gs_id = gs->length;
    for (size_t i = 0; i < target.nelem; ++i) {
        cn_init_leaf_var(gs, VEC_AT(target, i));
    }
    size_t loss = cn_init_leaf_var(gs, 0);
    for (size_t i = 0; i < target.nelem; ++i) {
        loss = cn_add(
            gs, loss,
            cn_raise(gs,
                     cn_subtract(gs, VEC_ID(prediction, i), VEC_ID(target, i)),
                     cn_init_leaf_var(gs, 2)));
    }
    cn_backward(gs, loss);

    return GET_NODE(loss).num;
}

float cn_learn(Net *net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t train_size = input.nrows;
    net->computation_graph.length = 1;
    GradientStore *gs = &net->computation_graph;

    for (size_t i = 0; i < net->hparams.nlayers - 1; ++i) {
        for (size_t j = 0; j < net->layers[i].weights.nrows; ++j) {
            for (size_t k = 0; k < net->layers[i].weights.ncols; ++k) {
                cn_init_leaf_var(gs, MAT_AT(net->layers[i].weights, j, k));
            }
        }
        for (size_t j = 0; j < net->layers[i].biases.nelem; ++j) {
            cn_init_leaf_var(gs, VEC_AT(net->layers[i].biases, j));
        }
    }

    float total_loss = 0;
    Vector input_vec;
    Vector target_vec;
    for (size_t i = 0; i < train_size; ++i) {
        input_vec = cn_form_vector(input.ncols, &MAT_AT(input, i, 0));
        target_vec = cn_form_vector(target.ncols, &MAT_AT(target, i, 0));
        total_loss += _cn_find_grad(net, gs, input_vec, target_vec);
        gs->length = net->hparams.nparam + 1;
    }
    float coef = net->hparams.rate / train_size;

    for (size_t i = 0; i < net->hparams.nlayers - 1; ++i) {
        for (size_t j = 0; j < net->layers[i].weights.nrows; ++j) {
            for (size_t k = 0; k < net->layers[i].weights.ncols; ++k) {
                float change =
                    coef * GET_NODE(MAT_ID(net->layers[i].weights, j, k)).grad;
                if (net->hparams.with_momentum) {
                    net->layers[i].weights.grad_stores[j * k] =
                        net->hparams.momentum_beta *
                            net->layers[i].weights.grad_stores[j * k] +
                        ((1 - net->hparams.momentum_beta) * change);
                    change = net->layers[i].weights.grad_stores[j * k];
                }
                MAT_AT(net->layers[i].weights, j, k) -= change;
            }
        }
        for (size_t j = 0; j < net->layers[i].biases.nelem; ++j) {
            float change =
                coef * GET_NODE(VEC_ID(net->layers[i].biases, j)).grad;
            if (net->hparams.with_momentum) {
                net->layers[i].biases.grad_stores[j] =
                    net->hparams.momentum_beta *
                        net->layers[i].biases.grad_stores[j] +
                    ((1 - net->hparams.momentum_beta) * change);
                change = net->layers[i].biases.grad_stores[j];
            }
            VEC_AT(net->layers[i].biases, j) -= change;
        }
    }

    return total_loss / train_size;
}

Vector cn_predict_layer(DenseLayer layer, Vector prev_output) {
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

Vector cn_predict(Net net, Vector input) {
    Vector guess = input;

    for (size_t i = 0; i < net.hparams.nlayers - 1; ++i) {
        guess = cn_predict_layer(net.layers[i], guess);
    }

    return guess;
}

float cn_error(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t size = input.nrows;
    float loss = 0;
    for (size_t i = 0; i < size; ++i) {
        Vector in = cn_form_vector(input.ncols, &MAT_AT(input, i, 0));
        Vector tar = cn_form_vector(target.ncols, &MAT_AT(target, i, 0));
        Vector out = cn_predict(net, in);
        for (size_t j = 0; j < out.nelem; ++j) {
            loss += powf(VEC_AT(out, j) - VEC_AT(tar, j), 2);
        }
    }
    return loss / size;
}

void cn_get_batch(Matrix *batch_in, Matrix *batch_tar, Matrix all_input,
                  Matrix all_target, size_t batch_num, size_t batch_size) {
    *batch_in = cn_form_matrix(batch_size, all_input.ncols, all_input.stride,
                               &MAT_AT(all_input, batch_num * batch_size, 0));
    *batch_tar = cn_form_matrix(batch_size, all_target.ncols, all_target.stride,
                                &MAT_AT(all_target, batch_num * batch_size, 0));
}

void cn_print_net_results(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t size = input.nrows;
    printf("Input | Net Output | Target\n");
    float loss = 0;
    for (size_t i = 0; i < size; ++i) {
        Vector in = cn_form_vector(input.ncols, &MAT_AT(input, i, 0));
        Vector tar = cn_form_vector(target.ncols, &MAT_AT(target, i, 0));
        Vector out = cn_predict(net, in);
        for (size_t j = 0; j < out.nelem; ++j) {
            loss += powf(VEC_AT(out, j) - VEC_AT(tar, j), 2);
        }
        _cn_print_vector_res(in);
        _cn_print_vector_res(out);
        _cn_print_vector_res(tar);
        printf("\n");
    }
    printf("Average Loss: %f\n", loss / size);
}

void cn_save_net_to_file(Net net, char *file_name) {
    FILE *fp = fopen(file_name, "wb");
    fwrite(&net.hparams.nlayers, sizeof(net.hparams.nlayers), 1, fp);
    fwrite(net.hparams.shape, sizeof(*net.hparams.shape), net.hparams.nlayers,
           fp);
    fwrite(&net.hparams.rate, sizeof(net.hparams.rate), 1, fp);
    fwrite(net.hparams.activations, sizeof(*net.hparams.activations),
           net.hparams.nlayers, fp);
    Matrix weights;
    Vector biases;
    for (size_t i = 0; i < net.hparams.nlayers - 1; ++i) {
        weights = net.layers[i].weights;
        biases = net.layers[i].biases;
        fwrite(weights.elements, sizeof(*weights.elements),
               weights.nrows * weights.ncols, fp);
        fwrite(biases.elements, sizeof(*biases.elements), biases.nelem, fp);
    }
    fclose(fp);
}

Net cn_alloc_net_from_file(char *file_name) {
    FILE *fp = fopen(file_name, "rb");
    CLEAR_NET_ASSERT(fp != NULL);
    size_t nlayers;
    fread(&nlayers, sizeof(nlayers), 1, fp);
    size_t *shape = (size_t *)CLEAR_NET_ALLOC(nlayers * sizeof(nlayers));
    CLEAR_NET_ASSERT(shape != NULL);
    fread(shape, sizeof(*shape), nlayers, fp);
    float rate;
    fread(&rate, sizeof(rate), 1, fp);
    Activation *acts =
        (Activation *)CLEAR_NET_ALLOC(nlayers * sizeof(Activation));
    fread(acts, sizeof(*acts), nlayers, fp);
    NetConfig hparams = cn_init_net_conf(shape, 1, nlayers, acts, 1, rate);
    Net net = cn_alloc_net(hparams);
    Matrix weights;
    Vector biases;
    for (size_t i = 0; i < net.hparams.nlayers - 1; ++i) {
        weights = net.layers[i].weights;
        fread(weights.elements, sizeof(*weights.elements),
              weights.nrows * weights.ncols, fp);
        biases = net.layers[i].biases;
        fread(biases.elements, sizeof(*biases.elements), biases.nelem, fp);
    }
    fclose(fp);
    return net;
}

#endif // CLEAR_NET_IMPLEMENTATION
/* Ending */

/* License
   Creative Commons Legal Code

   CC0 1.0 Universal

   CREATIVE COMMONS CORPORATION IS NOT A LAW FIRM AND DOES NOT PROVIDE
   LEGAL SERVICES. DISTRIBUTION OF THIS DOCUMENT DOES NOT CREATE AN
   ATTORNEY-CLIENT RELATIONSHIP. CREATIVE COMMONS PROVIDES THIS
   INFORMATION ON AN "AS-IS" BASIS. CREATIVE COMMONS MAKES NO WARRANTIES
   REGARDING THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS
   PROVIDED HEREUNDER, AND DISCLAIMS LIABILITY FOR DAMAGES RESULTING FROM
   THE USE OF THIS DOCUMENT OR THE INFORMATION OR WORKS PROVIDED
   HEREUNDER.

   Statement of Purpose

   The laws of most jurisdictions throughout the world automatically confer
   exclusive Copyright and Related Rights (defined below) upon the creator
   and subsequent owner(s) (each and all, an "owner") of an original work of
   authorship and/or a database (each, a "Work").

   Certain owners wish to permanently relinquish those rights to a Work for
   the purpose of contributing to a commons of creative, cultural and
   scientific works ("Commons") that the public can reliably and without fear
   of later claims of infringement build upon, modify, incorporate in other
   works, reuse and redistribute as freely as possible in any form whatsoever
   and for any purposes, including without limitation commercial purposes.
   These owners may contribute to the Commons to promote the ideal of a free
   culture and the further production of creative, cultural and scientific
   works, or to gain reputation or greater distribution for their Work in
   part through the use and efforts of others.

   For these and/or other purposes and motivations, and without any
   expectation of additional consideration or compensation, the person
   associating CC0 with a Work (the "Affirmer"), to the extent that he or she
   is an owner of Copyright and Related Rights in the Work, voluntarily
   elects to apply CC0 to the Work and publicly distribute the Work under its
   terms, with knowledge of his or her Copyright and Related Rights in the
   Work and the meaning and intended legal effect of CC0 on those rights.

   1. Copyright and Related Rights. A Work made available under CC0 may be
   protected by copyright and related or neighboring rights ("Copyright and
   Related Rights"). Copyright and Related Rights include, but are not
   limited to, the following:

   i. the right to reproduce, adapt, distribute, perform, display,
   communicate, and translate a Work;
   ii. moral rights retained by the original author(s) and/or performer(s);
   iii. publicity and privacy rights pertaining to a person's image or
   likeness depicted in a Work;
   iv. rights protecting against unfair competition in regards to a Work,
   subject to the limitations in paragraph 4(a), below;
   v. rights protecting the extraction, dissemination, use and reuse of data
   in a Work;
   vi. database rights (such as those arising under Directive 96/9/EC of the
   European Parliament and of the Council of 11 March 1996 on the legal
   protection of databases, and under any national implementation
   thereof, including any amended or successor version of such
   directive); and
   vii. other similar, equivalent or corresponding rights throughout the
   world based on applicable law or treaty, and any national
   implementations thereof.

   2. Waiver. To the greatest extent permitted by, but not in contravention
   of, applicable law, Affirmer hereby overtly, fully, permanently,
   irrevocably and unconditionally waives, abandons, and surrenders all of
   Affirmer's Copyright and Related Rights and associated claims and causes
   of action, whether now known or unknown (including existing as well as
   future claims and causes of action), in the Work (i) in all territories
   worldwide, (ii) for the maximum duration provided by applicable law or
   treaty (including future time extensions), (iii) in any current or future
   medium and for any number of copies, and (iv) for any purpose whatsoever,
   including without limitation commercial, advertising or promotional
   purposes (the "Waiver"). Affirmer makes the Waiver for the benefit of each
   member of the public at large and to the detriment of Affirmer's heirs and
   successors, fully intending that such Waiver shall not be subject to
   revocation, rescission, cancellation, termination, or any other legal or
   equitable action to disrupt the quiet enjoyment of the Work by the public
   as contemplated by Affirmer's express Statement of Purpose.

   3. Public License Fallback. Should any part of the Waiver for any reason
   be judged legally invalid or ineffective under applicable law, then the
   Waiver shall be preserved to the maximum extent permitted taking into
   account Affirmer's express Statement of Purpose. In addition, to the
   extent the Waiver is so judged Affirmer hereby grants to each affected
   person a royalty-free, non transferable, non sublicensable, non exclusive,
   irrevocable and unconditional license to exercise Affirmer's Copyright and
   Related Rights in the Work (i) in all territories worldwide, (ii) for the
   maximum duration provided by applicable law or treaty (including future
   time extensions), (iii) in any current or future medium and for any number
   of copies, and (iv) for any purpose whatsoever, including without
   limitation commercial, advertising or promotional purposes (the
   "License"). The License shall be deemed effective as of the date CC0 was
   applied by Affirmer to the Work. Should any part of the License for any
   reason be judged legally invalid or ineffective under applicable law, such
   partial invalidity or ineffectiveness shall not invalidate the remainder
   of the License, and in such case Affirmer hereby affirms that he or she
   will not (i) exercise any of his or her remaining Copyright and Related
   Rights in the Work or (ii) assert any associated claims and causes of
   action with respect to the Work, in either case contrary to Affirmer's
   express Statement of Purpose.

   4. Limitations and Disclaimers.

   a. No trademark or patent rights held by Affirmer are waived, abandoned,
   surrendered, licensed or otherwise affected by this document.
   b. Affirmer offers the Work as-is and makes no representations or
   warranties of any kind concerning the Work, express, implied,
   statutory or otherwise, including without limitation warranties of
   title, merchantability, fitness for a particular purpose, non
   infringement, or the absence of latent or other defects, accuracy, or
   the present or absence of errors, whether or not discoverable, all to
   the greatest extent permissible under applicable law.
   c. Affirmer disclaims responsibility for clearing rights of other persons
   that may apply to the Work or any use thereof, including without
   limitation any person's Copyright and Related Rights in the Work.
   Further, Affirmer disclaims responsibility for obtaining any necessary
   consents, permissions or other rights required for any use of the
   Work.
   d. Affirmer understands and acknowledges that Creative Commons is not a
   party to this document and has no duty or obligation with respect to
   this CC0 or use of the Work.
*/
