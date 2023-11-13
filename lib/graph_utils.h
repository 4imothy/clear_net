#include "clear_net.h"
#define MAT_ID(mat, r, c) (mat).start_id + ((r) * (mat).ncols) + (c)
#define VEC_ID(vec, i) (vec).start_id + (i)


// for when elements are together in the computation
typedef struct {
    ulong start_id;
    ulong nrows;
    ulong ncols;
    ulong stride;
} Mat;

typedef struct {
    ulong start_id;
    ulong nelem;
} Vec;

// for when elements are not together in the computation graph
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


typedef struct {
    UMat *mats;
    ulong nelem;
} UMatList;

// Used when the input type for a net is unknown
typedef union {
    UVec vec;
    UMat mat;
    UMatList mat_list;
} UVecMatU;

typedef enum {
    UVector,
    UMatrix,
    UMatrixList,
} UType;

typedef struct {
    UVecMatU data;
    UType type;
} UData;

Mat createMat(CompGraph *cg, ulong nrows, ulong ncols, ulong *offset);
void zeroMat(Mat *mat);
void printMat(CompGraph *cg, Mat *mat, char *name);
void randomizeMat(CompGraph *cg, Mat *mat, scalar lower, scalar upper);
void applyMatGrads(CompGraph *cg, Mat *mat, scalar rate);
Vec createVec(CompGraph *cg, ulong nelem, ulong *offset);
void zeroVec(Vec *vec);
void printVec(CompGraph *cg, Vec *vec, char *name);
void randomizeVec(CompGraph *cg, Vec *vec, scalar lower, scalar upper);
void applyVecGrads(CompGraph *cg, Vec *vec, scalar rate);
UMat allocUMat(ulong nrows, ulong ncols);
void deallocUMat(UMat *umat);
UVec allocUVec(ulong nelem);
void deallocUVec(UVec *uvec);
UMatList allocUMatList(ulong nrows, ulong ncols, ulong nchannels);
void deallocUMatList(UMatList *list);
void deallocUData(UData *data);
