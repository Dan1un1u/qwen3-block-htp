#ifndef QBH_FP_ISLAND_SWIGLU_H
#define QBH_FP_ISLAND_SWIGLU_H
#include <stdint.h>
#include <stddef.h>
/* params = gate scale, gate zero, up scale, up zero, middle scale. */
void qbh_fp_island_swiglu(const uint8_t *g,const uint8_t *u,uint8_t *lo,
    uint8_t *hi,size_t elements,const float params[5]);
#endif
