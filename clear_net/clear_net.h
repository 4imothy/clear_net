#ifndef CLEAR_NET
#define CLEAR_NET
#include "autodiff.h"
#include "defines.h"

// TODO when doing examples should create a public matrix type for scalar types
// which is then used by the people

typedef struct {
    struct {
        ulong (*add)(CompGraph *cg, ulong left, ulong right);
        ulong (*sub)(CompGraph *cg, ulong left, ulong right);
        ulong (*mul)(CompGraph *cg, ulong left, ulong right);
        ulong (*raise)(CompGraph *cg, ulong to_raise, ulong pow);
        ulong (*relu)(CompGraph *cg, ulong x);
        ulong (*leakyRelu)(CompGraph *cg, ulong x);
        ulong (*htan)(CompGraph *cg, ulong x);
        ulong (*sigmoid)(CompGraph *cg, ulong x);
        ulong (*elu)(CompGraph *cg, ulong x);
        void (*backprop)(CompGraph *cg, ulong last);
    } ad;
} _cn_names;

extern _cn_names const cn;
#endif // CLEAR_NET
