/*
   Clear Net by Timothy Cronin

   To the extent possible under law, the person who associated CC0 with
   Clear Net has waived all copyright and related or neighboring rights
   to Clear Net.

   You should have received a copy of the CC0 legalcode along with this
   work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

   See end of file for full license.
*/

/***
    TODO think about MAT_AT being a mocro and padded_mat_at being a function, kind of ugly, not sure how to change, use this maybe pretty clean
    #define PADDED_MAT_AT(mat, row, col) \
    ((row < 0 || col < 0 || row >= (long)mat.nrows || col >= (long)mat.ncols) ? 0 : MAT_AT(mat, row, col))
    Implement the functions then call them from the testing file, then change to a macr
***/

/* Beginning */
#ifndef CLEAR_NET
#define CLEAR_NET

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

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
#ifndef CLEAR_NET_INITIAL_GRAPH_LENGTH
#define CLEAR_NET_INITIAL_GRAPH_LENGTH 10
#endif // CLEAR_NET_INITIAL_GRAPH_LENGTH

/* Declaration: Helpers */
float _cn_randf(void);
void _cn_fill_floats(float *ptr, size_t len, float val);

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
float cn_activate(float x, Activation act);

/* Declaration: Automatic Differentiation Engine */
typedef struct VarNode VarNode;
typedef struct GradientStore GradientStore;
typedef void BackWardFunction(GradientStore *nl, VarNode *var);

GradientStore cn_alloc_gradient_store(size_t length);
void cn_realloc_gradient_store(GradientStore *gs, size_t new_len);
void cn_dealloc_gradient_store(GradientStore *nl);
VarNode cn_create_var(float num, size_t prev_left, size_t prev_right, BackWardFunction *backward);
size_t _cn_init_var(GradientStore *nl, float num, size_t prev_left,
                    size_t prev_right, BackWardFunction *backward);
size_t cn_init_leaf_var(GradientStore *nl, float num);
void _cn_add_backward(GradientStore *nl, VarNode *var);
size_t cn_add(GradientStore *nl, size_t left, size_t right);
void _cn_subtract_backward(GradientStore *nl, VarNode *var);
size_t cn_subtract(GradientStore *nl, size_t left, size_t right);
void _cn_multiply_backward(GradientStore *nl, VarNode *var);
size_t cn_multiply(GradientStore *nl, size_t left, size_t right);
void _cn_raise_backward(GradientStore *gs, VarNode *var);
size_t cn_raise(GradientStore *gs, size_t to_raise, size_t pow);
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
size_t _cn_activate(GradientStore *nl, size_t id, Activation act);
void cn_backward(GradientStore *nl, size_t y);

/* Declaration: Linear Algebra */
typedef struct Matrix Matrix;

Matrix cn_alloc_matrix(size_t nrows, size_t ncols);
void cn_dealloc_matrix(Matrix *mat);
Matrix cn_form_matrix(size_t nrows, size_t ncols, size_t stride,
                      float *elements);
void _cn_randomize_matrix(Matrix mat, float lower, float upper);
void cn_shuffle_matrix_rows(Matrix mat);
void cn_print_matrix(Matrix mat, char *name);

typedef struct Vector Vector;
Vector _cn_alloc_vector(size_t nelem);
void _cn_dealloc_vector(Vector *vec);
Vector cn_form_vector(size_t nelem, float *elements);
void _cn_randomize_vector(Vector vec, float lower, float upper);
void _cn_print_vector(Vector vec, char *name);
void _cn_print_vector_res(Vector vec);

/* Declaration: Net Structs */
typedef struct NetConfig NetConfig;
typedef struct DenseLayer DenseLayer;
typedef struct ConvolutionalLayer ConvolutionalLayer;
typedef struct Filter Filter;
typedef enum {
  Same,
  Valid,
  Full,
} Padding;
typedef struct PoolingLayer PoolingLayer;
typedef struct GlobalPoolingLayer GlobalPoolingLayer;
typedef enum {
  Max,
  Average,
} Pooling;
typedef struct Net Net;

/* Declaration: Net Config */
NetConfig cn_init_net_conf(void);
void cn_with_momentum(NetConfig *nc, float momentum_beta);
void cn_set_neg_scale(float neg_scale);

/* Declaration: DenseLayer */
void cn_alloc_dense_layer(Net *net, size_t dim_input, size_t dim_output,
                          Activation act);
Vector cn_predict_layer(DenseLayer layer, Vector prev_output);
Vector _cn_predict_layer(DenseLayer layer, GradientStore *nl,
                         Vector prev_output);

/* Declaration: Convolutional Layer */
// TODO do the deallocations
ConvolutionalLayer cn_init_convolutional_layer(Padding padding, Activation act,
                                               size_t input_nrows,
                                               size_t input_ncols,
                                               size_t kernel_nrows,
                                               size_t kernel_ncols);
void _cn_randomize_convolutional_layer(ConvolutionalLayer *layer, float lower,
                                       float upper);
void cn_alloc_filter(ConvolutionalLayer *c_layer, size_t nkernels);
float cn_correlate(Matrix kern, Matrix input, long top_left_row,
                    long top_left_col);
void cn_correlate_layer(ConvolutionalLayer *layer, Matrix *input, size_t nimput);
void cn_apply_bias_and_act(ConvolutionalLayer *layer);
float padded_mat_at(Matrix mat, long row, long col);

/* Declaration: Pooling Layer */
GlobalPoolingLayer cn_alloc_global_pooling_layer(Pooling pooling, size_t noutput);
PoolingLayer cn_alloc_pooling_layer(Pooling pooling, size_t nimput,
                                 size_t input_nrows, size_t input_ncols,
                                 size_t kernel_nrows, size_t kernel_ncols);
void cn_global_pool_layer(GlobalPoolingLayer *pooler, Matrix *input,
                       size_t nimput);
void cn_pool_layer(PoolingLayer *pooler, Matrix *input, size_t nimput);

/* Declaration: Net */
Net cn_init_net(NetConfig net_conf);
void cn_dealloc_net(Net *net);
void cn_randomize_net(Net net, float lower, float upper);
float cn_learn(Net *net, Matrix input, Matrix target);
float _cn_find_grad(Net *net, GradientStore *gs, Vector input, Vector target);
Vector cn_predict(Net net, Vector input);
Vector _cn_predict(Net *net, GradientStore *nl, Vector input);
float cn_loss(Net net, Matrix input, Matrix target);
void cn_get_batch(Matrix *batch_in, Matrix *batch_tar, Matrix all_input,
                  Matrix all_target, size_t batch_num, size_t batch_size);
void cn_print_net(Net net, char *name);
void cn_print_net_results(Net net, Matrix input, Matrix target);
void cn_print_target_output_pairs(Net net, Matrix input, Matrix target);
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

#define RAND_RANGE(upper, lower) _cn_randf() * ((upper) - (lower)) + (lower);

/* Implement: Activation Functions */
float NEG_SCALE = 0.1;

float cn_sigmoid(float x) { return 1 / (1 + expf(-x)); }

float cn_relu(float x) { return x > 0 ? x : 0; }

float cn_hyper_tan(float x) { return tanhf(x); }

float cn_leaky_relu(float x) { return x >= 0 ? x : NEG_SCALE * x; }

float cn_elu(float x) { return x > 0 ? x : NEG_SCALE * (expf(x) - 1); }

float cn_activate(float x, Activation act) {
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

/* Implement: Automatic Differentiation Engine */
#define CLEAR_NET_EXTEND_LENGTH_FUNCTION(len)                                  \
    ((len) == 0 ? CLEAR_NET_INITIAL_GRAPH_LENGTH : ((len) * 1.5))
#define GET_NODE(id) (gs)->vars[(id)]

struct VarNode {
    float num;
    float grad;
    BackWardFunction *backward;
    size_t prev_left;
    size_t prev_right;
    size_t visited;
};

struct GradientStore {
    VarNode *vars;
    size_t length;
    size_t max_length;
};

GradientStore cn_alloc_gradient_store(size_t length) {
    return (GradientStore){
        .vars = CLEAR_NET_ALLOC(length * sizeof(VarNode)),
        .length = 1,
        .max_length = length,
    };
}

void cn_realloc_gradient_store(GradientStore *gs, size_t new_len) {
    gs->vars = CLEAR_NET_REALLOC(gs->vars, new_len * sizeof(*gs->vars));
    gs->max_length = new_len;
}

void cn_dealloc_gradient_store(GradientStore *gs) {
    CLEAR_NET_DEALLOC(gs->vars);
}

VarNode cn_create_var(float num, size_t prev_left, size_t prev_right,
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
    VarNode out = cn_create_var(num, prev_left, prev_right, backward);
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

void _cn_tanh_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += (1 - powf(var->num, 2)) * var->grad;
}

size_t cn_hyper_tanv(GradientStore *gs, size_t x) {
    float val = cn_hyper_tan(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_tanh_backward);
    return out;
}

void _cn_sigmoid_backward(GradientStore *gs, VarNode *var) {
    GET_NODE(var->prev_left).grad += var->num * (1 - var->num) * var->grad;
}

size_t cn_sigmoidv(GradientStore *gs, size_t x) {
    float val = cn_sigmoid(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_sigmoid_backward);
    return out;
}

void _cn_leaky_relu_backward(GradientStore *gs, VarNode *var) {
    float change = var->num >= 0 ? 1 : NEG_SCALE;
    GET_NODE(var->prev_left).grad += change * var->grad;
}

size_t cn_leaky_reluv(GradientStore *gs, size_t x) {
    float val = cn_leaky_relu(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_leaky_relu_backward);
    return out;
}

void _cn_elu_backward(GradientStore *gs, VarNode *var) {
    float change = var->num > 0 ? 1 : var->num + NEG_SCALE;
    GET_NODE(var->prev_left).grad += change * var->grad;
}

size_t cn_eluv(GradientStore *gs, size_t x) {
    float val = cn_elu(GET_NODE(x).num);
    size_t out = _cn_init_var(gs, val, x, 0, _cn_elu_backward);
    return out;
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

void _cn_randomize_matrix(Matrix mat, float lower, float upper) {
    for (size_t j = 0; j < mat.nrows; ++j) {
        for (size_t k = 0; k < mat.ncols; ++k) {
            MAT_AT(mat, j, k) = RAND_RANGE(lower, upper);
        }
    }
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

void _cn_randomize_vector(Vector vec, float lower, float upper) {
    for (size_t i = 0; i < vec.nelem; ++i) {
        VEC_AT(vec, i) = RAND_RANGE(lower, upper);
    }
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

/* Implement: Net Structs */
struct NetConfig {
    size_t nparams;
    size_t nlayers;
    size_t with_momentum;
    float momentum_beta;
    float neg_scale;
    float rate;
};

struct Net {
    DenseLayer *layers;
    GradientStore computation_graph;
    NetConfig hparams;
};

struct DenseLayer {
    Matrix weights;
    Vector biases;
    Activation act;
    size_t *output_gs_ids;
    Vector output;
};

struct ConvolutionalLayer {
    Filter *filters;
    Matrix *outputs;
    size_t nfilters;
    Padding padding;
    Activation act;
    size_t input_nrows;
    size_t input_ncols;
    size_t output_nrows;
    size_t output_ncols;
    size_t k_nrows;
    size_t k_ncols;
};

struct Filter {
    Matrix *kernels;
    size_t nkernels;
    Matrix biases;
};

struct PoolingLayer {
    Matrix *outputs;
    Pooling pooling;
    size_t noutput;
    size_t k_nrows;
    size_t k_ncols;
    size_t output_nrows;
    size_t output_ncols;
};

struct GlobalPoolingLayer {
    Vector output;
    Pooling pooling;
};

/* Implement: Net Config */
NetConfig cn_init_net_conf(void) {
    return (NetConfig){
        .nparams = 0,
        .rate = 0.5f,
        .nlayers = 0,
        .neg_scale = 0.1,
        .with_momentum = 0,
        .momentum_beta = 0,
    };
}

void cn_with_momentum(NetConfig *nc, float momentum_beta) {
    nc->with_momentum = 1;
    nc->momentum_beta = momentum_beta;
}

void cn_set_neg_scale(float neg_scale) { NEG_SCALE = neg_scale; }

/* Implement: DenseLayer */
void cn_alloc_dense_layer(Net *net, size_t dim_input, size_t dim_output, Activation act) {
    if (net->hparams.nlayers != 0) {
        CLEAR_NET_ASSERT(net->layers[net->hparams.nlayers - 1].output.nelem == dim_input);
    }
    net->layers = CLEAR_NET_REALLOC(net->layers, (net->hparams.nlayers + 1) * sizeof(*net->layers));
    DenseLayer layer;
    layer.act = act;

    size_t offset = net->hparams.nparams + 1;
    Matrix weights = cn_alloc_matrix(dim_input, dim_output);
    if (net->hparams.with_momentum) {
        weights.grad_stores = CLEAR_NET_ALLOC(weights.nrows * weights.ncols * sizeof(*weights.grad_stores));
        _cn_fill_floats(weights.grad_stores, weights.nrows * weights.ncols, 0);
    }
    weights.gs_id = offset;
    layer.weights = weights;
    offset += layer.weights.nrows * layer.weights.ncols;

    Vector biases = _cn_alloc_vector(dim_output);
    biases.gs_id = offset;
    if (net->hparams.with_momentum) {
        biases.grad_stores = CLEAR_NET_ALLOC(biases.nelem * sizeof(*biases.grad_stores));
        _cn_fill_floats(biases.grad_stores, biases.nelem, 0);
    }
    layer.biases = biases;
    offset += layer.biases.nelem;

    Vector out = _cn_alloc_vector(dim_output);
    out.gs_id = 0;
    layer.output = out;
    layer.output_gs_ids =
        CLEAR_NET_ALLOC(layer.biases.nelem * sizeof(*layer.output_gs_ids));

    net->hparams.nparams = offset - 1;
    net->layers[net->hparams.nlayers] = layer;
    net->hparams.nlayers++;
    cn_realloc_gradient_store(&net->computation_graph, net->hparams.nparams);
}

Vector cn_predict_layer(DenseLayer layer, Vector prev_output) {
    for (size_t i = 0; i < layer.weights.ncols; ++i) {
        float res = 0;
        for (size_t j = 0; j < prev_output.nelem; ++j) {
            res += MAT_AT(layer.weights, j, i) * VEC_AT(prev_output, j);
        }
        res += VEC_AT(layer.biases, i);
        res = cn_activate(res, layer.act);
        layer.output.elements[i] = res;
    }
    return layer.output;
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

/* Implement: Convolutional Layer */
ConvolutionalLayer cn_init_convolutional_layer(Padding padding, Activation act,
                                               size_t input_nrows,
                                               size_t input_ncols,
                                               size_t kernel_nrows,
                                               size_t kernel_ncols) {
    size_t output_nrows;
    size_t output_ncols;
    switch(padding) {
    case Same:
        output_nrows = input_nrows;
        output_ncols = input_ncols;
        break;
    case Full:
        output_nrows = input_nrows + kernel_nrows - 1;
        output_ncols = input_ncols + kernel_ncols - 1;
        break;
    case Valid:
        output_nrows = input_nrows - kernel_nrows + 1;
        output_ncols = input_ncols - kernel_ncols + 1;
        break;
    }
    return (ConvolutionalLayer){
         .nfilters = 0,
        .filters = NULL,
        .outputs = NULL,
        .padding = padding,
        .act = act,
        .input_nrows = input_nrows,
        .input_ncols = input_ncols,
        .output_nrows = output_nrows,
        .output_ncols = output_ncols,
        .k_nrows = kernel_nrows,
        .k_ncols = kernel_ncols,
    };
}

void _cn_randomize_convolutional_layer(ConvolutionalLayer *layer, float lower,
                                       float upper) {
    for (size_t i = 0; i < layer->nfilters; ++i) {
        for (size_t j = 0; j < layer->filters[i].nkernels; ++j) {
            _cn_randomize_matrix(layer->filters[i].kernels[j], lower, upper);
        }
        _cn_randomize_matrix(layer->filters[i].biases, lower, upper);
    }
}

void cn_alloc_filter(ConvolutionalLayer *c_layer, size_t nkernels) {
    Filter filter;
    filter.nkernels = nkernels;
    filter.kernels = CLEAR_NET_ALLOC(nkernels * sizeof(Matrix));
    for (size_t i = 0; i < filter.nkernels; ++i) {
        filter.kernels[i] = cn_alloc_matrix(c_layer->k_nrows, c_layer->k_ncols);
    }
    filter.biases = cn_alloc_matrix(c_layer->output_nrows, c_layer->output_ncols);
    c_layer->filters = CLEAR_NET_REALLOC(c_layer->filters, (c_layer->nfilters + 1) * sizeof(Filter));
    c_layer->filters[c_layer->nfilters] = filter;
    c_layer->outputs = CLEAR_NET_REALLOC(c_layer->outputs, (c_layer->nfilters + 1) * sizeof(Matrix));
    c_layer->outputs[c_layer->nfilters] = cn_alloc_matrix(c_layer->output_nrows, c_layer->output_ncols);
    c_layer->nfilters++;
}

float cn_correlate(Matrix kern, Matrix input, long top_left_row,
                    long top_left_col) {
    float res = 0;
    for (size_t i = 0; i < kern.nrows; ++i) {
        for (size_t j = 0; j < kern.ncols; ++j) {
            res += padded_mat_at(input, top_left_row + i, top_left_col + j) * MAT_AT(kern, i, j);
        }
    }
    return res;
}

void cn_correlate_layer(ConvolutionalLayer *layer, Matrix *input,
                         size_t nimput) {
    float res;
    size_t row_padding;
    size_t col_padding;

    // Traversing the output with k and l
    // Applying that to those indices to input causes segfault
    // Each filter goes through the whole input
    // so we need to go through the whole input for each filter, which creates its own output
    for (size_t i = 0; i < nimput; ++i) {
        for (size_t j = 0; j < layer->nfilters; ++j) {
            for (size_t k = 0; k < layer->outputs[j].nrows; ++k) { // this should traverse the input
                for (size_t l = 0; l < layer->outputs[j].ncols; ++l) {
                    for (size_t m = 0; m < layer->filters[j].nkernels; ++m) {
                        switch (layer->padding) {
                        case Same:
                            row_padding =
                                (layer->filters[j].kernels[m].nrows - 1) / 2;
                            col_padding =
                                (layer->filters[j].kernels[m].ncols - 1) / 2;
                            break;
                        case Full:
                            row_padding =
                                layer->filters[j].kernels[m].nrows - 1;
                            col_padding =
                                layer->filters[j].kernels[m].ncols - 1;
                            break;
                        case Valid:
                            row_padding = 0;
                            col_padding = 0;
                            break;
                        }
                        long top_left_row = (long)k - row_padding;
                        long top_left_col = (long)l - col_padding;

                        res = cn_correlate(layer->filters[j].kernels[m], input[i],
                                          top_left_row, top_left_col);

                        MAT_AT(layer->outputs[j], k, l) += res;
                    }
                }
            }
        }
    }
}

void cn_apply_bias_and_act(ConvolutionalLayer *layer) {
        for (size_t i = 0; i < layer->nfilters; ++i) {
        for (size_t j = 0; j < layer->outputs[i].nrows; ++j) {
            for (size_t k = 0; k < layer->outputs[i].ncols; ++k) {
                MAT_AT(layer->outputs[i], j, k) =
                    MAT_AT(layer->outputs[i], j, k) + MAT_AT(layer->filters[i].biases, j, k);
                MAT_AT(layer->outputs[i], j, k) = cn_activate(MAT_AT(layer->outputs[i], j, k), layer->act);
            }
        }
    }
}
float padded_mat_at(Matrix mat, long row, long col) {
    if (row < 0 || col < 0 || row >= (long)mat.nrows ||
        col >= (long)mat.ncols) {
        return 0;
    }
    return MAT_AT(mat, row, col);
}

/* Implement: Pooling Layer */
GlobalPoolingLayer cn_alloc_global_pooling_layer(Pooling pooling,
                                                  size_t noutput) {
    return (GlobalPoolingLayer) {
        .pooling = pooling,
        .output = _cn_alloc_vector(noutput),
    };
}

PoolingLayer cn_alloc_pooling_layer(Pooling pooling, size_t nimput,
                                     size_t input_nrows, size_t input_ncols,
                                     size_t kernel_nrows, size_t kernel_ncols) {
    PoolingLayer pooler;
    pooler.pooling = pooling;
    pooler.k_nrows = kernel_nrows;
    pooler.k_ncols = kernel_ncols;
    pooler.output_nrows = input_nrows / kernel_nrows;
    pooler.output_ncols = input_ncols / kernel_ncols;
    pooler.outputs = CLEAR_NET_ALLOC(nimput * sizeof(Matrix));
    for (size_t i = 0; i < nimput; ++i) {
        pooler.outputs[i] = cn_alloc_matrix(pooler.output_nrows, pooler.output_ncols);
    }
    pooler.noutput = nimput;

    return pooler;
}

void cn_global_pool_layer(GlobalPoolingLayer *pooler, Matrix *input,
                       size_t nimput) {
    for (size_t i = 0; i < nimput; ++i) {
        float max_store = -1 * FLT_MAX;
        float avg_res = 0;
        float cur;
        size_t nelements = input[i].nrows * input[i].ncols;
        for (size_t j = 0; j < input[i].nrows; ++j) {
            for (size_t k = 0; k < input[i].ncols; ++k) {
                cur = MAT_AT(input[i], j, k);
                switch (pooler->pooling) {
                case(Max):
                    if (cur > max_store) {
                        max_store = cur;
                    }
                    break;
                case(Average):
                    avg_res += cur;
                    break;
                }
            }
        }
        switch(pooler->pooling) {
        case(Max):
            VEC_AT(pooler->output, i) = max_store;
            break;
        case(Average):
            VEC_AT(pooler->output, i) = avg_res / nelements;
            break;
        }
    }
}

void cn_pool_layer(PoolingLayer *pooler, Matrix *input, size_t nimput) {
    for (size_t i = 0; i < nimput; ++i) {
        for (size_t j = 0; j < input[i].nrows; j += pooler->k_nrows) {
            for (size_t k = 0; k < input[i].ncols; k += pooler->k_ncols) {
                float max_store = -1 * FLT_MAX;
                float avg_store = 0;
                float cur;
                size_t nelements = pooler->k_nrows * pooler->k_ncols;
                for (size_t l = 0; l < pooler->k_nrows; ++l) {
                    for (size_t m = 0; m < pooler->k_ncols; ++m) {
                        cur = MAT_AT(input[i], j + l, k + m);
                        switch(pooler->pooling) {
                        case(Max):
                            if (cur > max_store) {
                                max_store = cur;
                            }
                            break;
                        case(Average):
                            avg_store += cur;
                            break;
                        }
                    }
                }
                switch(pooler->pooling) {
                case(Max):
                    MAT_AT(pooler->outputs[i], j / pooler->k_nrows, k / pooler->k_ncols) = max_store;
                    break;
                case(Average):
                    MAT_AT(pooler->outputs[i], j / pooler->k_nrows, k / pooler->k_ncols) = avg_store / nelements;
                    break;
                }
            }
        }
    }
}

/* Implement: Net */
Net cn_init_net(NetConfig net_conf) {
    return (Net) {
        .hparams = net_conf,
        .layers = NULL,
        .computation_graph = cn_alloc_gradient_store(0),
    };
}

void cn_dealloc_net(Net *net) {
    for (size_t i = 0; i < net->hparams.nlayers - 1; ++i) {
        cn_dealloc_matrix(&net->layers[i].weights);
        _cn_dealloc_vector(&net->layers[i].biases);
        _cn_dealloc_vector(&net->layers[i].output);
        CLEAR_NET_DEALLOC(net->layers[i].output_gs_ids);
    }
    cn_dealloc_gradient_store(&net->computation_graph);
}

void cn_randomize_net(Net net, float lower, float upper) {
    for (size_t i = 0; i < net.hparams.nlayers; ++i) {
        _cn_randomize_matrix(net.layers[i].weights, lower, upper);
        _cn_randomize_vector(net.layers[i].biases, lower, upper);
    }
}

float cn_learn(Net *net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    size_t train_size = input.nrows;
    net->computation_graph.length = 1;
    GradientStore *gs = &net->computation_graph;

    for (size_t i = 0; i < net->hparams.nlayers; ++i) {
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
        gs->length = net->hparams.nparams + 1;
    }
    float coef = net->hparams.rate / train_size;

    for (size_t i = 0; i < net->hparams.nlayers; ++i) {
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

Vector cn_predict(Net net, Vector input) {
    Vector guess = input;

    for (size_t i = 0; i < net.hparams.nlayers; ++i) {
        guess = cn_predict_layer(net.layers[i], guess);
    }

    return guess;
}

Vector _cn_predict(Net *net, GradientStore *gs, Vector input) {
    CLEAR_NET_ASSERT(input.nelem == net->layers[0].weights.nrows);
    Vector guess = input;
    for (size_t i = 0; i < net->hparams.nlayers; ++i) {
        guess = _cn_predict_layer(net->layers[i], gs, guess);
    }
    return guess;
}

float cn_loss(Net net, Matrix input, Matrix target) {
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

void cn_print_net(Net net, char *name) {
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < net.hparams.nlayers; ++i) {
        DenseLayer layer = net.layers[i];
        snprintf(buf, sizeof(buf), "weight matrix: %zu", i);
        cn_print_matrix(layer.weights, buf);
        snprintf(buf, sizeof(buf), "bias vector: %zu", i);
        _cn_print_vector(layer.biases, buf);
    }
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

void cn_print_target_output_pairs(Net net, Matrix input, Matrix target) {
    CLEAR_NET_ASSERT(input.nrows == target.nrows);
    Vector in;
    Vector tar;
    Vector out;
    for (size_t i = 0; i < input.nrows; ++i) {
        in = cn_form_vector(input.ncols, &MAT_AT(input, i, 0));
        tar = cn_form_vector(target.ncols, &MAT_AT(target, i, 0));
        out = cn_predict(net, in);
        printf("------------\n");
        printf("target: ");
        for (size_t j = 0; j < tar.nelem; ++j) {
            printf("%f ", VEC_AT(tar, j));
        }
        printf("\n");
        printf("output: ");
        for (size_t j = 0; j < out.nelem; ++j) {
            printf("%f ", VEC_AT(out, j));
        }
        printf("\n");
    }
}

void cn_save_net_to_file(Net net, char *file_name) {
    FILE *fp = fopen(file_name, "wb");
    Matrix weights;
    Vector biases;
    fwrite(&net.hparams.nlayers, sizeof(net.hparams.nlayers), 1, fp);
    fwrite(&net.hparams.rate, sizeof(net.hparams.rate), 1, fp);
    fwrite(&net.hparams.with_momentum, sizeof(net.hparams.with_momentum), 1, fp);
    fwrite(&net.hparams.momentum_beta, sizeof(net.hparams.momentum_beta), 1, fp);
    fwrite(&NEG_SCALE, sizeof(NEG_SCALE), 1, fp);
    for (size_t i = 0; i < net.hparams.nlayers; ++i) {
        fwrite(&net.layers[i].act, sizeof(net.layers[i].act), 1, fp);
        weights = net.layers[i].weights;
        biases = net.layers[i].biases;
        fwrite(&weights.nrows, sizeof(weights.nrows), 1, fp);
        fwrite(&weights.ncols, sizeof(weights.ncols), 1, fp);
        fwrite(weights.elements, sizeof(*weights.elements),
               weights.nrows * weights.ncols, fp);
        fwrite(biases.elements, sizeof(*biases.elements), biases.nelem, fp);
    }
    fclose(fp);
}

Net cn_alloc_net_from_file(char *file_name) {
    FILE *fp = fopen(file_name, "rb");
    CLEAR_NET_ASSERT(fp != NULL);
    NetConfig nc = cn_init_net_conf();
    size_t nlayers;
    fread(&nlayers, sizeof(nc.nlayers), 1, fp);
    fread(&nc.rate, sizeof(nc.rate), 1, fp);
    fread(&nc.with_momentum, sizeof(nc.with_momentum), 1, fp);
    fread(&nc.momentum_beta, sizeof(nc.momentum_beta), 1, fp);
    fread(&NEG_SCALE, sizeof(NEG_SCALE), 1, fp);
    Net net = cn_init_net(nc);
    Activation act;
    size_t input_dim;
    size_t output_dim;
    Matrix weights;
    Vector biases;
    for (size_t i = 0; i < nlayers; ++i) {
        fread(&act, sizeof(act), 1, fp);
        fread(&input_dim, sizeof(input_dim), 1, fp);
        fread(&output_dim, sizeof(output_dim), 1, fp);
        cn_alloc_dense_layer(&net, input_dim, output_dim, act);
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
