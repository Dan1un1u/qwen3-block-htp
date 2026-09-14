/* L32-0024 exact dyadic dense-dot helpers; no floating converter or FWHT. */
#ifndef QBH_LLAMA_R4_EXACT_H
#define QBH_LLAMA_R4_EXACT_H
#include <stdint.h>
/* Finite binary16 in units of 2^-24. Caller rejects exponent31. */
static inline int64_t qbh_r4_half_units(uint16_t h) {
    uint32_t e=(h>>10)&31U,m=h&1023U;
    int64_t v=e?((int64_t)(m+1024U)<<(e-1U)):(int64_t)m;
    return (h&32768U)?-v:v;
}
/* Exact signed integer times 2^base -> binary16 round-to-nearest-even.
 * R4 bounds: <=512*maxhalf*2^24*181 <2^57; no int64 overflow. */
static inline uint16_t qbh_r4_round_units(int64_t value,int base) {
    uint16_t sign=value<0?32768U:0U;
    uint64_t v=value<0?(uint64_t)(-value):(uint64_t)value;
    if(!v)return sign;
    int p=63-__builtin_clzll(v),e=p+base;
    int shift=e>=-14?p-10:-24-base;
    uint64_t q;
    if(shift>0) {
        q=v>>shift;
        uint64_t rem=v&(((uint64_t)1<<shift)-1U),half=(uint64_t)1<<(shift-1);
        q+=(rem>half || (rem==half && (q&1U)));
    } else q=v<<(-shift);
    if(e<-14)return sign|(uint16_t)q;
    if(q==2048U){q=1024U;++e;}
    if(e>15)return sign|31744U;
    return sign|(uint16_t)((e+15)*1024+(int)q-1024);
}
#endif
