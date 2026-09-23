#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stdint.h>
#include <string.h>
#include "bandwidth_protocol.h"
#include "qbh_user_dma.h"
#include "hmx_fp16.h"
#define MAXW 6
static uint8_t stacks[MAXW][8192] __attribute__((aligned(128)));
struct job {uint8_t *p,*out;struct bw_header *h;uint32_t id;int status;qurt_sem_t ready,go,done;uint64_t t0,t1,c0,c1;};
static struct job jobs[MAXW];
static struct qbh_dma_aligned_desc_1d desc[256];
/* Eight independent volatile vector reads; no per-vector reduction on critical path. */
__attribute__((noinline)) static void hvx_loop(uint8_t *p,uint8_t *out,uint32_t bytes,uint32_t repeats,uint32_t mode) {
 HVX_Vector a=Q6_V_vsplat_R(0x5a5a5a5a),b=a,c=a,d=a,e=a,f=a,g=a,h=a;
 for(uint32_t r=0;r<repeats;r++)for(uint32_t i=0;i<bytes;i+=1024){
  if(mode!=1) {
   asm volatile("%0=vmem(%8+#0)\n%1=vmem(%8+#1)\n%2=vmem(%8+#2)\n%3=vmem(%8+#3)\n%4=vmem(%8+#4)\n%5=vmem(%8+#5)\n%6=vmem(%8+#6)\n%7=vmem(%8+#7)\n":"=v"(a),"=v"(b),"=v"(c),"=v"(d),"=v"(e),"=v"(f),"=v"(g),"=v"(h):"r"(p+i):"memory");
  }
  if(mode!=0){HVX_Vector *q=(HVX_Vector*)(out+i);q[0]=a;q[1]=b;q[2]=c;q[3]=d;q[4]=e;q[5]=f;q[6]=g;q[7]=h;asm volatile("":::"memory");}
 }
 if(mode==0){HVX_Vector *q=(HVX_Vector*)out;q[0]=a;q[1]=b;q[2]=c;q[3]=d;q[4]=e;q[5]=f;q[6]=g;q[7]=h;}
 asm volatile("barrier":::"memory");
}
static void worker(void *arg){struct job *j=arg;j->status=qurt_hvx_lock(QURT_HVX_MODE_128B);qurt_sem_up(&j->ready);
 for(uint32_t r=0;r<j->h->rounds;r++){qurt_sem_down(&j->go);j->t0=HAP_perf_get_qtimer_count();j->c0=HAP_perf_get_pcycles();if(!j->status)hvx_loop(j->p,j->out,j->h->bytes,j->h->repeats,j->h->mode);j->c1=HAP_perf_get_pcycles();j->t1=HAP_perf_get_qtimer_count();qurt_sem_up(&j->done);}
 if(!j->status)qurt_hvx_unlock();
}
static uint8_t pattern(uint32_t i){return (uint8_t)((i*13U+(i>>12)*7U)^0x5aU);}
static inline void hmx_issue(uint8_t *a,uint8_t *w,uint32_t stream,uint32_t mode){
 uint32_t ar=1792U+124U+(stream-1U)*2048U;
 uint32_t wr=stream*(mode==5?512U:1024U)-1U;
 if(mode==3){uint32_t n=stream*2048U-1U;asm volatile("{ activation.hf = mxmem(%0,%1):deep\n weight.hf = mxmem(%2,%1) }\n"::"r"(a),"r"(n),"r"(w):"memory");}
 else if(mode==4)asm volatile("{ activation.ub = mxmem(%0,%1):deep:cm\n weight.b = mxmem(%2,%3) }\n"::"r"(a),"r"(ar),"r"(w),"r"(wr):"memory");
 else asm volatile("{ activation.ub = mxmem(%0,%1):deep:cm\n weight.n = mxmem(%2,%3) }\n"::"r"(a),"r"(ar),"r"(w),"r"(wr):"memory");
}
static void hmx_run(uint8_t *v,struct bw_header *h){
 uint8_t *w=v+4U*1024U*1024U,*out=v+7U*1024U*1024U;uint32_t *bias=(void*)(out+4096);
 uint32_t step=h->stream*2048U,wb=h->stream*(h->mode==3?2048U:h->mode==4?1024U:512U);
 uint32_t chunks=h->bytes/step;
 if(h->mode==3){for(uint32_t i=0;i<h->bytes/2;i++){((uint16_t*)v)[i]=0x1c00;((uint16_t*)w)[i]=0x1c00;}qbh_hmx_fp16_init_unity_scale(bias);}
 else{memset(v,1,h->bytes);memset(w,h->mode==5?0x11:1,chunks*wb);for(unsigned n=0;n<32;n++){bias[n]=24U<<10;bias[n+32]=0;}}
 Q6_bias_mxmem2_A(bias);
 for(uint32_t r=0;r<h->rounds;r++){
 uint64_t t=HAP_perf_get_qtimer_count(),c=HAP_perf_get_pcycles();
 for(uint32_t rep=0;rep<h->repeats;rep++){
  if(h->mode==3)asm volatile("mxclracc.hf":::"memory");else Q6_mxclracc();
  for(uint32_t j=0;j<chunks;j++)hmx_issue(v+j*step,w+j*wb,h->stream,h->mode);
  if(h->mode==3)asm volatile("cvt.hf=acc(%0)\nmxmem(%1,%2)=cvt\n"::"r"(2),"r"(out),"r"(0):"memory");
  else Q6_mxmem_AR_after_cm_sat_ub(out,1792U);
 }
 asm volatile("barrier":::"memory");h->cycles[r]=HAP_perf_get_pcycles()-c;h->ticks[r]=HAP_perf_get_qtimer_count()-t;
 }
 h->payload_bytes=(uint64_t)h->repeats*chunks*(step+wb);
 if(h->mode==3){__fp16 expected=(__fp16)((float)(chunks*h->stream*32U)/65536.0f);uint16_t bits;memcpy(&bits,&expected,2);for(unsigned i=0;i<1024;i++)h->errors+=((uint16_t*)out)[i]!=bits;}
 else{uint32_t x=chunks*h->stream*32U;if(x>255)x=255;for(unsigned i=0;i<2048;i++)h->errors+=out[i]!=x;}
 /* Perturb activation and weight separately and verify zero sensitivity, untimed. */
 for(unsigned which=0;which<2;which++){
  uint8_t *q=which?w:v;uint32_t len=which?wb:step;memset(q,0,len);
  if(h->mode==3)asm volatile("mxclracc.hf":::"memory");else Q6_mxclracc();hmx_issue(v,w,h->stream,h->mode);
  if(h->mode==3)asm volatile("cvt.hf=acc(%0)\nmxmem(%1,%2)=cvt\n"::"r"(2),"r"(out),"r"(0):"memory");else Q6_mxmem_AR_after_cm_sat_ub(out,1792U);
  asm volatile("barrier":::"memory");for(unsigned i=0;i<2048;i++)h->errors+=out[i]!=0;
  if(h->mode==3)for(uint32_t i=0;i<len/2;i++)((uint16_t*)q)[i]=0x1c00;else memset(q,which&&h->mode==5?0x11:1,len);
 }
}
int qbh_bandwidth_run(void *vtcm,uint32_t vbytes,uint32_t hmx,int fd,uint32_t size){
 uint8_t *mem=0,*v=vtcm;int result=HAP_mmap_get(fd,(void**)&mem,0);if(result||!mem)return AEE_EFAILED;
 qurt_mem_cache_clean((qurt_addr_t)mem,sizeof(struct bw_header),QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE);
 struct bw_header *h=(void*)mem;h->status=1;h->vtcm_bytes=vbytes;h->errors=0;
 if(h->magic!=BW_MAGIC||size<BW_DRAM+BW_PAYLOAD||h->rounds!=BW_ROUNDS||h->mode>6||!h->bytes||h->bytes%1024||!h->repeats||h->repeats>100000||!h->workers||h->workers>MAXW||!h->stream||h->stream>32||!h->depth||h->depth>256||h->bypass>1)goto done;
 if(h->mode<3){
  if((uint64_t)h->workers*h->bytes*2U>vbytes)goto done;
  qurt_thread_t threads[MAXW];
  for(uint32_t i=0;i<h->workers;i++){struct job *j=&jobs[i];memset(j,0,sizeof(*j));j->p=v+i*h->bytes*2U;j->out=j->p+h->bytes;j->h=h;j->id=i;memset(j->p,0xa5,h->bytes);memset(j->out,0,h->bytes);qurt_sem_init_val(&j->ready,0);qurt_sem_init_val(&j->go,0);qurt_sem_init_val(&j->done,0);qurt_thread_attr_t a;qurt_thread_attr_init(&a);qurt_thread_attr_set_name(&a,"bw-hvx");qurt_thread_attr_set_stack_addr(&a,stacks[i]);qurt_thread_attr_set_stack_size(&a,sizeof(stacks[i]));qurt_thread_attr_set_priority(&a,qurt_thread_get_priority(qurt_thread_get_id()));if(qurt_thread_create(&threads[i],&a,worker,j))goto done;}
  for(uint32_t i=0;i<h->workers;i++)qurt_sem_down(&jobs[i].ready);
  for(uint32_t r=0;r<h->rounds;r++){
   for(uint32_t i=0;i<h->workers;i++)qurt_sem_up(&jobs[i].go);
   for(uint32_t i=0;i<h->workers;i++)qurt_sem_down(&jobs[i].done);
   uint64_t t0=UINT64_MAX,t1=0,c0=UINT64_MAX,c1=0;
   for(uint32_t i=0;i<h->workers;i++){struct job*j=&jobs[i];if(j->t0<t0)t0=j->t0;if(j->t1>t1)t1=j->t1;if(j->c0<c0)c0=j->c0;if(j->c1>c1)c1=j->c1;}
   h->ticks[r]=t1-t0;h->cycles[r]=c1-c0;
  }
  for(uint32_t i=0;i<h->workers;i++){int st;qurt_thread_join(threads[i],&st);h->errors+=jobs[i].status!=0;uint32_t len=h->mode==0?1024:h->bytes;for(uint32_t b=0;b<len;b++)h->errors+=jobs[i].out[b]!=(h->mode==1?0x5a:0xa5);qurt_sem_destroy(&jobs[i].ready);qurt_sem_destroy(&jobs[i].go);qurt_sem_destroy(&jobs[i].done);}
  h->payload_bytes=(uint64_t)h->workers*h->bytes*h->repeats*(h->mode==2?2U:1U);
 }else if(h->mode<6){
  if(h->bytes>3U*1024U*1024U||h->bytes%(h->stream*2048U))goto done;
  if(HAP_compute_res_hmx_lock2(hmx,HAP_COMPUTE_RES_HMX_SHARED))goto done;
  hmx_run(v,h);HAP_compute_res_hmx_unlock2(hmx,HAP_COMPUTE_RES_HMX_SHARED);
 }else{
  if((uint64_t)h->bytes*h->depth>vbytes||h->bytes*h->depth>BW_DRAM)goto done;
  uint32_t span=h->bytes*h->depth,groups=BW_DRAM/span,last=0;
  for(uint32_t r=0;r<h->rounds;r++){
   uint64_t t=HAP_perf_get_qtimer_count(),c=HAP_perf_get_pcycles();
   for(uint32_t rep=0;rep<h->repeats;rep++){
    uint32_t base=(rep%groups)*span;last=base;
    for(uint32_t d=0;d<h->depth;d++){struct qbh_dma_desc_1d *x=&desc[d].descriptor;x->next=d+1<h->depth?(uint32_t)(uintptr_t)&desc[d+1]:0;x->control=0;x->length=h->bytes;x->src_bypass=h->bypass;x->dst_bypass=1;x->src=(uint32_t)(uintptr_t)(mem+BW_PAYLOAD+base+d*h->bytes);x->dst=(uint32_t)(uintptr_t)(v+d*h->bytes);}
    if(qbh_dma_start(desc)||qbh_dma_wait_idle()){h->errors++;break;}
   }
   asm volatile("barrier":::"memory");h->ticks[r]=HAP_perf_get_qtimer_count()-t;h->cycles[r]=HAP_perf_get_pcycles()-c;
   for(uint32_t b=0;b<span;b++)h->errors+=v[b]!=pattern(last+b);
  }
  h->payload_bytes=(uint64_t)h->repeats*span;
 }
 h->status=0;
 done:qurt_mem_cache_clean((qurt_addr_t)mem,sizeof(*h),QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE);HAP_mmap_put(fd);return h->status?AEE_EBADPARM:AEE_SUCCESS;
}
