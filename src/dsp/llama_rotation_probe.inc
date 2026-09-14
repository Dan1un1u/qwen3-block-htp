/* L32-0015 isolated dense HMX rotation and post-R4 SP2 contract probe.
 * Explicit matrix products only. Diagnostic output copies are outside core_ticks;
 * neither core nor RPC times are a full-block speed result. */
#include "hmx_fp16.h"
static size_t lrpos(uint32_t r,uint32_t c,uint32_t nt) {
 return qbh_hmx_fp16_matrix_tile_offset(r/32,c/32,nt)+qbh_hmx_fp16_tile_offset(r%32,c%32);
}
static void lrscale(void *p,float s) {
 float v[32] __attribute__((aligned(128)));
 for(uint32_t i=0;i<32;i++)v[i]=s;
 qbh_hmx_fp16_init_channel_scales(p,v);
}
static int lrprobe(uint8_t *shared,uint32_t bytes,uint8_t *vtcm,uint32_t vbytes,uint32_t ctx) {
 struct lsp2_header *h=(void*)shared;
 const uint32_t isr4=h->mode==7U, rows=h->rows, n=isr4?8192U:64U;
 if(h->magic!=LSP2_MAGIC || h->abi!=1 || h->bytes!=bytes || h->k!=n || h->n!=n ||
    !rows || rows>(isr4?64U:2560U) || vbytes!=8388608U ||
    !valid(h->input_offset,rows*n*2U,bytes) ||
    !valid(h->output_offset,rows*n*(isr4?6U:2U),bytes) ||
    (isr4 && !valid(h->weight_offset,131072U,bytes)))return AEE_EBADPARM;
 uint8_t *base=(uint8_t*)(((uintptr_t)vtcm+2047U)&~(uintptr_t)2047U);
 __fp16 *input=(void*)base,*act=(void*)(base+1048576U),*out=(void*)(base+2097152U),*weight=(void*)(base+3145728U);
 __fp16 *tmp=(void*)(base+3670016U),*tmpout=(void*)(base+3932160U);
 uint16_t *lut=(void*)(base+4194304U);void *sc=base+4325376U;
 if(base+4327424U>vtcm+vbytes)return AEE_ENOMEMORY;
 uint64_t start=HAP_perf_get_qtimer_count(),t=start;
 memcpy(input,shared+h->input_offset,rows*n*2U);
 if(isr4)memcpy(lut,shared+h->weight_offset,131072U);
 h->load_ticks=HAP_perf_get_qtimer_count()-t;h->pack_ticks=h->mac_ticks=h->convert_ticks=h->merge_ticks=h->publish_ticks=h->dma_ticks=0;
 h->vtcm_bytes=vbytes;h->peak_bytes=(uint32_t)(base+4327424U-vtcm);h->streams=0;h->conversions=0;
 int ret=HAP_compute_res_hmx_lock2(ctx,HAP_COMPUTE_RES_HMX_SHARED);if(ret)return ret;
 t=HAP_perf_get_qtimer_count();
 const uint32_t dim=isr4?512U:64U, count=isr4?rows*16U:rows, padded=(count+31U)&~31U;
 memset(act,0,padded*dim*2U);
 for(uint32_t a=0;a<(isr4?16U:1U);a++)for(uint32_t rr=0;rr<rows;rr++)for(uint32_t c=0;c<dim;c++)
  act[lrpos(a*rows+rr,c,dim/32U)]=input[rr*n+a*dim+c];
 for(uint32_t j=0;j<dim;j++)for(uint32_t k=0;k<dim;k++)
  weight[((j/32U)*(dim/32U)+k/32U)*1024U+qbh_hmx_fp16_tile_offset(k%32U,j%32U)]=(__fp16)((__builtin_popcount(j&k)&1)?-1:1);
 lrscale(sc,isr4?(float)(__fp16)0.04419417382415922f:0.125f);
 h->pack_ticks+=HAP_perf_get_qtimer_count()-t;t=HAP_perf_get_qtimer_count();
 qbh_hmx_fp16_matmul_tiles(act,weight,sc,out,padded/32U,dim/32U,dim/32U);
 h->mac_ticks+=HAP_perf_get_qtimer_count()-t;h->streams++;
 t=HAP_perf_get_qtimer_count();
 __fp16 *audit=(void*)(shared+h->output_offset);
 for(uint32_t a=0;a<(isr4?16U:1U);a++)for(uint32_t rr=0;rr<rows;rr++)for(uint32_t c=0;c<dim;c++)
  audit[rr*n+a*dim+c]=out[lrpos(a*rows+rr,c,dim/32U)];
 h->publish_ticks+=HAP_perf_get_qtimer_count()-t;
 if(isr4) {
  t=HAP_perf_get_qtimer_count();memset(weight,0,2048U);
  for(uint32_t j=0;j<16;j++)for(uint32_t k=0;k<16;k++)weight[qbh_hmx_fp16_tile_offset(k,j)]=(__fp16)((__builtin_popcount(j&k)&1)?-1:1);
  lrscale(sc,0.25f);h->pack_ticks+=HAP_perf_get_qtimer_count()-t;
  for(uint32_t first=0;first<rows;first+=8U) {
   uint32_t nr=rows-first<8U?rows-first:8U;t=HAP_perf_get_qtimer_count();memset(tmp,0,nr*512U*32U*2U);
   for(uint32_t rr=0;rr<nr;rr++)for(uint32_t c=0;c<512;c++)for(uint32_t a=0;a<16;a++)
    tmp[lrpos(rr*512U+c,a,1U)]=out[lrpos(a*rows+first+rr,c,16U)];
   h->pack_ticks+=HAP_perf_get_qtimer_count()-t;t=HAP_perf_get_qtimer_count();
   qbh_hmx_fp16_matmul_tiles(tmp,weight,sc,tmpout,nr*16U,1U,1U);
   h->mac_ticks+=HAP_perf_get_qtimer_count()-t;h->streams++;t=HAP_perf_get_qtimer_count();
   for(uint32_t rr=0;rr<nr;rr++)for(uint32_t c=0;c<512;c++)for(uint32_t a=0;a<16;a++) {
    uint16_t bits;__fp16 val=tmpout[lrpos(rr*512U+c,a,1U)];memcpy(&bits,&val,2);
    uint32_t ix=(first+rr)*8192U+a*512U+c;
    input[ix]=val;((uint16_t*)act)[ix]=lut[bits];
   }
   h->convert_ticks+=HAP_perf_get_qtimer_count()-t;
  }
  t=HAP_perf_get_qtimer_count();memcpy(audit+rows*n,input,rows*n*2U);memcpy(audit+2U*rows*n,act,rows*n*2U);h->publish_ticks+=HAP_perf_get_qtimer_count()-t;h->conversions=rows*n;
 }
 h->total_ticks=HAP_perf_get_qtimer_count()-start;
 ret=HAP_compute_res_hmx_unlock2(ctx,HAP_COMPUTE_RES_HMX_SHARED);h->status=ret;return ret;
}
