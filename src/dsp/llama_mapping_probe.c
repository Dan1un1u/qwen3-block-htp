/* L32-0048: isolated delayed-map probe; no inference arithmetic changes. */
#include <AEEStdErr.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <stdint.h>
#include <string.h>
#include "llama_mapping_probe.h"
#include "qbh_user_dma.h"
static int lmp_dma(void *dst,const void *src,uint32_t bytes,int to_ddr) {
 struct qbh_dma_aligned_desc_1d d __attribute__((aligned(64)));
 memset(&d,0,sizeof(d));
 d.descriptor.src=(uint32_t)(uintptr_t)src;
 d.descriptor.dst=(uint32_t)(uintptr_t)dst;
 d.descriptor.length=bytes;
 d.descriptor.src_bypass=to_ddr?0:1;
 d.descriptor.dst_bypass=to_ddr?1:0;
 d.descriptor.ordered=1;
 asm volatile("syncht" ::: "memory");
 if(qbh_dma_start(&d)!=0 || qbh_dma_wait_idle()!=0)return -1;
 asm volatile("syncht" ::: "memory");
 return d.descriptor.dstate==QBH_DMA_DESC_COMPLETE?0:-1;
}
int lmp_run(int fd,uint32_t bytes,void *vtcm,uint32_t vtcm_bytes) {
 struct lmp_header *h=NULL;
 void *active[2]={NULL,NULL};
 uint32_t ev_ids[2]={0,0}, next=0;
 int ret=HAP_mmap_get(fd,(void **)&h,NULL);
 if(ret || !h)return AEE_EFAILED;
 if(bytes!=sizeof(*h) || qurt_mem_cache_clean((qurt_addr_t)h,bytes,QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE)){
  HAP_mmap_put(fd);return AEE_EBADPARM;
 }
 h->status=-1;h->vtcm_bytes=vtcm_bytes;
 if(h->magic!=LMP_MAGIC || h->version!=1 || !h->count || h->count>LMP_MAX_BUFFERS ||
    !h->window || h->window>2 || h->cycles<1 ||
    (uint64_t)h->cycles*h->count>LMP_MAX_EVENTS || vtcm_bytes!=8388608U)goto done;
 for(uint32_t b=0;b<h->count;b++)if(h->fds[b]<0 || h->sizes[b]<LMP_PAGE_BYTES*LMP_SAMPLES ||
    h->sizes[b]%LMP_PAGE_BYTES || h->sizes[b]>1073741824U)goto done;
 {
 uint64_t start=HAP_perf_get_qtimer_count();
 for(uint32_t cycle=0;cycle<h->cycles;cycle++) {
  for(uint32_t b=0;b<h->count;b++) {
   uint32_t id=(cycle&1U)?h->count-1U-b:b;
   uint32_t slot=next++%h->window;
   if(active[slot]) {
    struct lmp_event *old=&h->events[ev_ids[slot]];
    uint64_t t=HAP_perf_get_qtimer_count();
    old->unmap_result=HAP_munmap2(active[slot],old->bytes);
    old->unmap_ticks=HAP_perf_get_qtimer_count()-t;
    if(old->unmap_result){h->status=-3;goto finish;}
    active[slot]=NULL;
   }
   uint32_t ei=h->event_count++;
   struct lmp_event *e=&h->events[ei];
   e->cycle=cycle;e->buffer=id;e->bytes=h->sizes[id];e->unmap_result=-999;
   uint64_t t=HAP_perf_get_qtimer_count();
   void *ptr=HAP_mmap2(NULL,e->bytes,HAP_PROT_READ|HAP_PROT_WRITE,0,h->fds[id],0);
   e->map_ticks=HAP_perf_get_qtimer_count()-t;
   e->map_result=(ptr && ptr!=(void *)-1)?0:-1;
   if(e->map_result){h->status=-2;goto finish;}
   e->va=(uint32_t)(uintptr_t)ptr;
   active[slot]=ptr;ev_ids[slot]=ei;
   if((uint64_t)e->va+e->bytes>0x100000000ULL){h->status=-4;goto finish;}
   for(uint32_t sample=0;sample<LMP_SAMPLES;sample++) {
    uint8_t *src=(uint8_t *)ptr+lmp_offset(e->bytes,sample);
    t=HAP_perf_get_qtimer_count();
    if(lmp_dma(vtcm,src,LMP_PAGE_BYTES,0)){h->status=-5;goto finish;}
    e->dma_ticks+=HAP_perf_get_qtimer_count()-t;
    uint32_t *v=(uint32_t *)vtcm;
    for(uint32_t w=0;w<512;w++) {
     e->checks++;
     if(v[w]!=lmp_value(id,sample,w))e->mismatches++;
     if(cycle){e->checks++;if(v[w+512]!=(lmp_value(id,sample,w)^cycle))e->mismatches++;}
     v[w+512]=lmp_value(id,sample,w)^(cycle+1);
    }
    t=HAP_perf_get_qtimer_count();
    if(lmp_dma(src+2048,(uint8_t *)vtcm+2048,2048,1)){h->status=-5;goto finish;}
    e->dma_ticks+=HAP_perf_get_qtimer_count()-t;
   }
   h->mismatches+=e->mismatches;
   if(e->mismatches){h->status=-6;goto finish;}
   h->completed++;
  }
 }
 h->status=0;
 finish:
 for(uint32_t i=0;i<h->window;i++)if(active[i]){
  struct lmp_event *e=&h->events[ev_ids[i]];
  uint64_t t=HAP_perf_get_qtimer_count();
  int err=HAP_munmap2(active[i],e->bytes);
  if(e->unmap_result==-999){e->unmap_result=err;e->unmap_ticks=HAP_perf_get_qtimer_count()-t;}
  if(err)h->cleanup_errors++;
 }
 h->total_ticks=HAP_perf_get_qtimer_count()-start;
 }
 done:
 ret=h->status==0 && h->cleanup_errors==0?AEE_SUCCESS:AEE_EFAILED;
 if(qurt_mem_cache_clean((qurt_addr_t)h,bytes,QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE))ret=AEE_EFAILED;
 if(HAP_mmap_put(fd))ret=AEE_EFAILED;
 return ret;
}
