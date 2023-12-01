#ifndef CN_GRAPH_UTILS
#define CN_GRAPH_UTILS
#include "clear_net.h"

#define MAT_ID(mat, r, c) (mat).start_id + ((r) * (mat).ncols) + (c)
#define VEC_ID(vec, i) (vec).start_id + (i)

// for when elements are together in the computation
typedef struct {
    ulong start_id;
    ulong nrows;
    ulong ncols;
} Mat;

typedef struct {
    ulong start_id;
    ulong nelem;
} Vec;

typedef struct {
    ulong *elem;
    ulong ncols;
    ulong nrows;
    ulong stride;
} UMat;

typedef struct {
    ulong *elem;
    ulong nelem;
} UVec;

typedef union {
    UVec vec;
    UMat mat;
    UMat *mats;
} UStore;

typedef enum {
    UVEC,
    UMAT,
    UMATLIST,
} UType;

typedef struct {
    UStore in;
    UType type;
    ulong nchannels;
} UData;

Mat createMat(CompGraph *cg, ulong nrows, ulong ncols, ulong *offset);
void zeroMat(Mat *mat);
void printMat(CompGraph *cg, Mat *mat, char *name);
void randomizeMat(CompGraph *cg, Mat *mat, scalar lower, scalar upper);
void applyMatGrads(CompGraph *cg, Mat *mat, scalar rate, bool momentum, scalar beta);
Vec createVec(CompGraph *cg, ulong nelem, ulong *offset);
void zeroVec(Vec *vec);
void printVec(CompGraph *cg, Vec *vec, char *name);
void randomizeVec(CompGraph *cg, Vec *vec, scalar lower, scalar upper);
void applyVecGrads(CompGraph *cg, Vec *vec, scalar rate, bool momentum, scalar beta);
UMat allocUMat(ulong nrows, ulong ncols);
void deallocUMat(UMat *umat);
UVec allocUVec(ulong nelem);
void deallocUVec(UVec *uvec);
void printUVecInline(CompGraph *cg, UVec *uvec);
UMat* allocUMatList(ulong nrows, ulong ncols, ulong nchannels);
void deallocUMatList(UMat **list, ulong nelem);
void deallocUData(UData *data);

#endif // CN_GRAPH_UTILS
