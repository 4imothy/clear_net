#include "scalar.h"

#define IS_FLOAT sizeof(scalar) == sizeof(float)
#define IS_DOUBLE sizeof(scalar) == sizeof(double)
#define IS_LONG_DOUBLE sizeof(scalar) == sizeof(long double)

// TODO this causes issues with printing with %f so create a new to_char
// function for easier printing

char *type_not_supported_message = "Type of Scalar Not Supported";

scalar pows(scalar to_raise, scalar raiser) {
    if (IS_FLOAT) {
        return pow(to_raise, raiser);
    } else if (IS_DOUBLE) {
        return powf(to_raise, raiser);
    } else if (IS_LONG_DOUBLE) {
        return powl(to_raise, raiser);
    }
    CLEAR_NET_ASSERT(0 && type_not_supported_message);
    return 0;
}

scalar tanhs(scalar x) {
    if (IS_DOUBLE) {
        return tanh(x);
    } else if (IS_FLOAT) {
        return tanhf(x);
    } else if (IS_LONG_DOUBLE) {
        return tanhl(x);
    }
    CLEAR_NET_ASSERT(0 && type_not_supported_message);
    return 0;
}

scalar exps(scalar x) {
    if (IS_FLOAT) {
        return expf(x);
    } else if (IS_DOUBLE) {
        return exp(x);
    } else if (IS_LONG_DOUBLE) {
        return expl(x);
    }
    CLEAR_NET_ASSERT(0 && type_not_supported_message);
    return 0;
}

scalar randRange(scalar lower, scalar upper) {
    return ((scalar)rand() / RAND_MAX) * (upper - lower) + lower;
}
