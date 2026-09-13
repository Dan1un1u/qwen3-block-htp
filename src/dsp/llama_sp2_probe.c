/* L32-0008 isolated component, never selected by the model runtime. */
#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <hmx_hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <qurt.h>
#include <stdint.h>
#include <string.h>
#include "llama_sp2_probe.h"
#include "mlp_u8.h"
#include "hmx_u8s8_projection.h"
#include "qbh_user_dma.h"
#define RT 0x700U
static struct qbh_dma_aligned_desc_1d probe_dma; /* DMA descriptors reside in DDR, payload intermediates in VTCM. */
static uint32_t aligned(uint32_t x){return (x+2047U)&~2047U;}
static int valid(uint32_t off,uint32_t size,uint32_t total){return off>=128U && off<=total && size<=total-off;}
/* Non-saturating retain conversion exposes four radix256 digits without
 * modifying the accumulator. Identity mantissa; exponents24,16,8,0. */
static void read_i32(uint8_t *bytes,uint32_t *bias){
 for(uint32_t j=0;j<4;j++){
  for(uint32_t n=0;n<32;n++){bias[n]=(24U-8U*j)<<10;bias[32+n]=0;}
  Q6_bias_mxmem2_A(bias);
  Q6_mxmem_AR_after_retain_cm_ub(bytes+2048*j,RT);
  asm volatile("barrier":::"memory");
 }
}
static void unpack_digits(const uint8_t *bytes,int32_t *wide,uint32_t digits){
 for(uint32_t base=0;base<2048;base+=128){
  HVX_Vector value[4]={Q6_V_vzero(),Q6_V_vzero(),Q6_V_vzero(),Q6_V_vzero()};
  for(uint32_t digit=0;digit<digits;digit++){
   HVX_Vector x=*(const HVX_Vector*)(bytes+2048*digit+base);
   HVX_VectorPair h=Q6_Wuh_vunpack_Vub(x);
   HVX_VectorPair wl=Q6_Wuw_vunpack_Vuh(Q6_V_lo_W(h)),wh=Q6_Wuw_vunpack_Vuh(Q6_V_hi_W(h));
   HVX_Vector w[4]={Q6_V_lo_W(wl),Q6_V_hi_W(wl),Q6_V_lo_W(wh),Q6_V_hi_W(wh)};
   for(uint32_t j=0;j<4;j++)value[j]=Q6_V_vor_VV(value[j],Q6_Vw_vasl_VwR(w[j],8*digit));
  }
  for(uint32_t j=0;j<4;j++)*(HVX_Vector*)(wide+base+32*j)=value[j];
 }
}
/* Exhaustive gather scheduling audit, separate from the model runtime.
 * Input is one immutable LUT; outputs are old low/high and new low/high. */
static int lsp2_gather_audit(uint8_t *shared,uint32_t bytes,uint8_t *vtcm,uint32_t vbytes) {
 struct lsp2_header *h=(struct lsp2_header *)shared;
 if(h->magic!=LSP2_MAGIC || h->abi!=1U || h->bytes!=bytes || vbytes!=8388608U ||
    !valid(h->input_offset,131072U,bytes) || !valid(h->output_offset,262144U,bytes))return AEE_EBADPARM;
 uint8_t *base=(uint8_t *)(((uintptr_t)vtcm+65535U)&~(uintptr_t)65535U);
 uint16_t *lut=(uint16_t *)base;
 uint8_t *g=base+131072U,*u=g+65536U,*out=u+65536U,*scratch=out+262144U;
 if(scratch+256U>vtcm+vbytes)return AEE_ENOMEMORY;
 memcpy(lut,shared+h->input_offset,131072U);
 for(uint32_t i=0;i<65536U;i++){g[i]=(uint8_t)(i>>8);u[i]=(uint8_t)i;}
 qbh_mlp_gate_up_sp2_lut_hvx(g,u,out,out+65536U,65536U,lut,scratch);
 qbh_mlp_gate_up_sp2_lut_pipelined_hvx(g,u,out+131072U,out+196608U,65536U,lut,scratch);
 uint32_t mismatch=0;
 for(uint32_t i=0;i<65536U;i++) {
   uint16_t expected=lut[i];
   if(out[i]!=(uint8_t)expected || out[i+65536U]!=(uint8_t)(expected>>8) ||
      out[i+131072U]!=(uint8_t)expected || out[i+196608U]!=(uint8_t)(expected>>8))mismatch++;
 }
 memcpy(shared+h->output_offset,out,262144U);
 h->streams=65536U;h->conversions=mismatch;h->vtcm_bytes=vbytes;
 h->peak_bytes=(uint32_t)(scratch+256U-vtcm);h->status=mismatch?AEE_EFAILED:AEE_SUCCESS;
 return h->status;
}
int lsp2_run(int fd,uint32_t bytes,uint8_t *vtcm,uint32_t vbytes,uint32_t ctx){
 uint8_t *shared=0;int ret=HAP_mmap_get(fd,(void**)&shared,0);if(ret||!shared)return AEE_EFAILED;
 ret=qurt_mem_cache_clean((qurt_addr_t)shared,bytes,QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE);
 if(ret){HAP_mmap_put(fd);return AEE_EFAILED;}
 struct lsp2_header *h=(struct lsp2_header*)shared;
 if(bytes>=128U && h->mode==4U) {
   ret=lsp2_gather_audit(shared,bytes,vtcm,vbytes);
   int e=qurt_mem_cache_clean((qurt_addr_t)shared,bytes,QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE);
   HAP_mmap_put(fd);return ret?ret:(e?AEE_EFAILED:AEE_SUCCESS);
 }
 if(bytes<128 || h->magic!=LSP2_MAGIC || h->abi!=1 || h->bytes!=bytes || h->mode>3 || !h->rows || h->rows>64 || (h->mode==1 && h->rows>32) || !h->k || h->k>8192 || h->k%32 || !h->n || h->n>2048 || h->n%32 || vbytes!=8388608U || !valid(h->input_offset,h->rows*h->k*2,bytes) || !valid(h->weight_offset,h->n*h->k/2,bytes) || !valid(h->sum_offset,h->n*4,bytes) || !valid(h->output_offset,h->rows*h->n*4,bytes)) {HAP_mmap_put(fd);return AEE_EBADPARM;}
 struct lsp2_header c=*h;c.status=-1;c.vtcm_bytes=vbytes;c.streams=0;c.conversions=0;
 c.load_ticks=c.pack_ticks=c.dma_ticks=c.mac_ticks=c.convert_ticks=c.merge_ticks=c.publish_ticks=0;
 uint32_t cursor=aligned(c.rows*c.k*2);int16_t *input=(int16_t*)vtcm;
 uint8_t *a0=vtcm+cursor;cursor+=c.k*64;
 uint8_t *a1=vtcm+cursor;cursor+=c.k*64;
 uint8_t *weight=vtcm+cursor;cursor+=aligned(c.k*16);
 uint8_t *raw0=vtcm+cursor;cursor+=8192;
 uint8_t *raw1=vtcm+cursor;cursor+=8192;
 int32_t *wide0=(int32_t*)(vtcm+cursor);cursor+=8192;
 int32_t *wide1=(int32_t*)(vtcm+cursor);cursor+=8192;
 uint8_t *pack_scratch=vtcm+cursor;cursor+=2048;
 uint32_t *bias=(uint32_t*)(vtcm+cursor);cursor+=2048;
 int32_t *sums=(int32_t*)(vtcm+cursor);cursor+=aligned(c.n*4);
 int32_t *output=(int32_t*)(vtcm+cursor);cursor+=aligned(c.rows*c.n*4);
 struct qbh_dma_aligned_desc_1d *dma=&probe_dma;
 c.peak_bytes=cursor;if(cursor>vbytes){HAP_mmap_put(fd);return AEE_ENOMEMORY;}
 ret=HAP_compute_res_hmx_lock2(ctx,HAP_COMPUTE_RES_HMX_SHARED);if(ret){HAP_mmap_put(fd);return ret;}
 uint64_t start=HAP_perf_get_qtimer_count(),t=start;
 memcpy(input,shared+c.input_offset,c.rows*c.k*2);memcpy(sums,shared+c.sum_offset,c.n*4);
 c.load_ticks=HAP_perf_get_qtimer_count()-t;t=HAP_perf_get_qtimer_count();
 memset(a0,0,c.k*64);memset(a1,128,c.k*64);
 /* Fixture input is logical signed16; producer fusion is deliberately a
  * later experiment. Vector packing writes only the live native row spans. */
 for(uint32_t row=0;row<c.rows;row++)for(uint32_t k=0;k<c.k;k+=64){
  HVX_Vector v=*(const HVX_UVector*)(input+row*c.k+k);
  *(HVX_Vector*)pack_scratch=Q6_Vb_vpacke_VhVh(v,v);
  *(HVX_Vector*)(pack_scratch+128)=Q6_V_vxor_VV(Q6_Vb_vpacko_VhVh(v,v),Q6_V_vsplat_R(0x80808080));
  for(uint32_t j=0;j<2 && k+32*j<c.k;j++){
   uint32_t off=((k/32)+j)*2048+row*32;
   memcpy(a0+off,pack_scratch+j*32,32);
   if(c.mode==1)memcpy(a0+off+c.rows*32,pack_scratch+128+j*32,32);
   if(c.mode==2)memcpy(a1+off,pack_scratch+128+j*32,32);
  }
 }
 c.pack_ticks=HAP_perf_get_qtimer_count()-t;
 for(uint32_t nt=0;nt<c.n/32;nt++){
  t=HAP_perf_get_qtimer_count();memset(dma,0,sizeof(*dma));dma->descriptor.length=c.k*16;dma->descriptor.src_bypass=1;dma->descriptor.ordered=1;dma->descriptor.src=(uint32_t)(uintptr_t)(shared+c.weight_offset+nt*c.k*16);dma->descriptor.dst=(uint32_t)(uintptr_t)weight;
  if(qbh_dma_start(dma)||qbh_dma_wait_idle()){ret=AEE_EFAILED;goto done;}
  c.dma_ticks+=HAP_perf_get_qtimer_count()-t;
  uint32_t passes=c.mode==2?2:1;
  for(uint32_t p=0;p<passes;p++){
   t=HAP_perf_get_qtimer_count();for(uint32_t n=0;n<32;n++){bias[n]=24U<<10;bias[32+n]=0;}
   qbh_hmx_begin_u8s8_output(bias);c.streams+=qbh_hmx_accumulate_u8n4_projection(p?a1:a0,weight,c.k/32);
   c.mac_ticks+=HAP_perf_get_qtimer_count()-t;t=HAP_perf_get_qtimer_count();
   if(c.mode==3){qbh_hmx_store_u8_output(raw0);c.conversions++;}
   else {read_i32(p?raw1:raw0,bias);c.conversions+=4;}
   c.convert_ticks+=HAP_perf_get_qtimer_count()-t;
  }
  t=HAP_perf_get_qtimer_count();
  unpack_digits(raw0,wide0,c.mode==3?1:4);if(c.mode==2)unpack_digits(raw1,wide1,4);
  for(uint32_t row=0;row<c.rows;row++){
   {
    HVX_Vector v=*(const HVX_Vector*)(wide0+row*32);
    if(c.mode==1 || c.mode==2){
     const int32_t *hp=c.mode==1?wide0+(row+c.rows)*32:wide1+row*32;
     HVX_Vector high=*(const HVX_Vector*)hp,sum=*(const HVX_Vector*)(sums+nt*32);
     /* Wrapping intermediates are exact modulo2^32. Final signed result is
      * bounded by8192*32768*7 <2^31, including signed16 stress fixtures. */
     v=Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(v,Q6_Vw_vasl_VwR(high,8)),Q6_Vw_vasl_VwR(sum,15));
    }
    *(HVX_Vector*)(output+row*c.n+nt*32)=v;
   }
  }
  c.merge_ticks+=HAP_perf_get_qtimer_count()-t;
 }
 t=HAP_perf_get_qtimer_count();memcpy(shared+c.output_offset,output,c.rows*c.n*4);c.publish_ticks=HAP_perf_get_qtimer_count()-t;ret=AEE_SUCCESS;
 done:
 c.total_ticks=HAP_perf_get_qtimer_count()-start;
 {int u=HAP_compute_res_hmx_unlock2(ctx,HAP_COMPUTE_RES_HMX_SHARED);if(!ret)ret=u;}
 c.status=ret;*h=c;
 {int e=qurt_mem_cache_clean((qurt_addr_t)shared,bytes,QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE);if(!ret && e)ret=AEE_EFAILED;}
 HAP_mmap_put(fd);return ret;
}
