#ifndef QBH_FP_ISLAND_SWIGLU_H
#define QBH_FP_ISLAND_SWIGLU_H
#include <stdint.h>
#include <stddef.h>
/* params = gate scale, gate zero, up scale, up zero, middle scale. */
void qbh_fp_island_swiglu(const uint8_t *g,const uint8_t *u,uint8_t *lo,
    uint8_t *hi,size_t elements,const float params[5]);
/* scratch: two aligned FP32 arrays of elements each; phase 0 DQ, 1 FP, 2 Q.
 * FP output reuses the gate float array after both inputs have been read. */
void qbh_fp_island_swiglu_phase(uint32_t phase,const uint8_t *g,const uint8_t *u,
 uint8_t *lo,uint8_t *hi,size_t elements,const float params[5],float *scratch);
#endif
