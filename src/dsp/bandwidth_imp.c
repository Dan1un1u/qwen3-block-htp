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
static uint8_t pattern(uint32_t i);
static uint8_t stacks[MAXW][8192] __attribute__((aligned(128)));
struct job {uint8_t *p,*out;struct bw_header *h;uint32_t id;int status;qurt_sem_t ready,go,done;uint64_t t0,t1,c0,c1;};
static struct job jobs[MAXW];
static struct qbh_dma_aligned_desc_1d desc[256];
/* Eight independent volatile vector reads; no per-vector reduction on critical path. */
__attribute__((noinline)) static void hvx_loop(uint8_t *p,uint8_t *out,uint32_t bytes,uint32_t repeats,uint32_t mode) {
 HVX_Vector a=Q6_V_vsplat_R(0x5a5a5a5a),b=a,c=a,d=a,e=a,f=a,g=a,h=a;
 if(mode==0 && bytes%4096U==0U){
  for(uint32_t rep=0;rep<repeats;rep++){
   uint8_t *ptr=p;
   asm volatile("loop0(1f,%9)\n"
    "1:\n"
    "%0=vmem(%8+#0)\n"
    "%1=vmem(%8+#1)\n"
    "%2=vmem(%8+#2)\n"
    "%3=vmem(%8+#3)\n"
    "%4=vmem(%8+#4)\n"
    "%5=vmem(%8+#5)\n"
    "%6=vmem(%8+#6)\n"
    "{%7=vmem(%8+#7)\n %8=add(%8,#1024)}\n"
    "%0=vmem(%8+#0)\n"
    "%1=vmem(%8+#1)\n"
    "%2=vmem(%8+#2)\n"
    "%3=vmem(%8+#3)\n"
    "%4=vmem(%8+#4)\n"
    "%5=vmem(%8+#5)\n"
    "%6=vmem(%8+#6)\n"
    "{%7=vmem(%8+#7)\n %8=add(%8,#1024)}\n"
    "%0=vmem(%8+#0)\n"
    "%1=vmem(%8+#1)\n"
    "%2=vmem(%8+#2)\n"
    "%3=vmem(%8+#3)\n"
    "%4=vmem(%8+#4)\n"
    "%5=vmem(%8+#5)\n"
    "%6=vmem(%8+#6)\n"
    "{%7=vmem(%8+#7)\n %8=add(%8,#1024)}\n"
    "%0=vmem(%8+#0)\n"
    "%1=vmem(%8+#1)\n"
    "%2=vmem(%8+#2)\n"
    "%3=vmem(%8+#3)\n"
    "%4=vmem(%8+#4)\n"
    "%5=vmem(%8+#5)\n"
    "%6=vmem(%8+#6)\n"
    "{%7=vmem(%8+#7)\n %8=add(%8,#1024)}:endloop0\n"
    :"=v"(a),"=v"(b),"=v"(c),"=v"(d),"=v"(e),"=v"(f),"=v"(g),"=v"(h),"+r"(ptr):"r"(bytes/4096U):"lc0","sa0","memory");
  }
  HVX_Vector *q=(HVX_Vector*)out;q[0]=a;q[1]=b;q[2]=c;q[3]=d;q[4]=e;q[5]=f;q[6]=g;q[7]=h;
  asm volatile("barrier":::"memory");return;
 }
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
static uint8_t pattern(uint32_t i){return (uint8_t)((i*13U+(i>>12)*7U+(i>>20)*29U)^0x5aU);}
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

/* EXP0311: common-window readers. All results count only batches completed
 * before the common deadline. Buffers and kernels are identical in solo/both. */
#define CW_MAX 7
static uint8_t cw_stacks[CW_MAX][8192] __attribute__((aligned(128)));
struct cw_job {
 uint8_t *a,*w,*out;uint32_t *bias;
 uint32_t role,fmt,bytes,stream,hmx,rounds;int status;
 qurt_sem_t ready,go,done;
 uint64_t begin,deadline,t0,t1,c0,c1,count;
};
static struct cw_job cw_jobs[CW_MAX];
__attribute__((noinline)) static void cw_hmx_batch(struct cw_job *j) {
 uint32_t stream=j->stream,fmt=j->fmt,step=stream*2048U;
 uint32_t wb=stream*(fmt==3?2048U:fmt==4?1024U:512U),chunks=j->bytes/step;
 uint8_t *a=j->a,*w=j->w,*out=j->out;
 for(uint32_t rep=0;rep<16;rep++) {
  if(fmt==3)asm volatile("mxclracc.hf":::"memory");else Q6_mxclracc();
  for(uint32_t k=0;k<chunks;k++)hmx_issue(a+k*step,w+k*wb,stream,fmt);
  if(fmt==3)asm volatile("cvt.hf=acc(%0)\nmxmem(%1,%2)=cvt\n"::"r"(2),"r"(out),"r"(0):"memory");
  else Q6_mxmem_AR_after_cm_sat_ub(out,1792U);
 }
 asm volatile("barrier":::"memory");
}
static void cw_worker(void *arg) {
 struct cw_job *j=arg;
 j->status=j->role==0?qurt_hvx_lock(QURT_HVX_MODE_128B):j->role==1?HAP_compute_res_hmx_lock2(j->hmx,HAP_COMPUTE_RES_HMX_SHARED):0;
 if(j->role==1&&!j->status)Q6_bias_mxmem2_A(j->bias);
 qurt_sem_up(&j->ready);
 for(uint32_t round=0;round<j->rounds;round++) {
  qurt_sem_down(&j->go);j->count=0;
  while(HAP_perf_get_qtimer_count()<j->begin)asm volatile("pause(#8)":::"memory");
  j->t0=HAP_perf_get_qtimer_count();j->c0=HAP_perf_get_pcycles();
  if(!j->status)while(HAP_perf_get_qtimer_count()<j->deadline) {
   if(j->role==0)hvx_loop(j->a,j->out,524288U,8U,0U);
   else if(j->role==1)cw_hmx_batch(j);
   else { for(uint32_t k=0;k<4096;k++)asm volatile("nop":::"memory"); }
   asm volatile("barrier":::"memory");
   uint64_t end=HAP_perf_get_qtimer_count();
   if(end<=j->deadline)j->count++;
  }
  j->c1=HAP_perf_get_pcycles();j->t1=HAP_perf_get_qtimer_count();qurt_sem_up(&j->done);
 }
 if(!j->status){if(j->role==0)qurt_hvx_unlock();else if(j->role==1)HAP_compute_res_hmx_unlock2(j->hmx,HAP_COMPUTE_RES_HMX_SHARED);}
}
static int concurrent_run(uint8_t *v,uint32_t hmx,struct bw_header *h){
 const uint32_t fmt=h->bypass;
 uint32_t nv=h->mode==8?0:h->workers,nh=h->mode==7?0:1,nt=nv+nh;
 qurt_thread_t threads[CW_MAX];
 if(fmt==3){for(uint32_t k=0;k<6U*1024U*1024U/2;k++)((uint16_t*)v)[k]=0x1c00;}
 else{memset(v,1,6U*1024U*1024U);if(fmt==5)memset(v+1048576U,0x11,h->bytes/4U);}
 uint32_t *bias=(void*)(v+7U*1024U*1024U+4096U);
 if(fmt==3)qbh_hmx_fp16_init_unity_scale(bias);
 else for(unsigned k=0;k<32;k++){bias[k]=24U<<10;bias[k+32]=0;}
 for(uint32_t i=0;i<nt;i++){
  struct cw_job *j=&cw_jobs[i];memset(j,0,sizeof(*j));
  j->role=i<nv?0:h->mode==10?2:1;j->fmt=fmt;j->bytes=h->bytes;j->stream=h->stream;j->hmx=hmx;j->rounds=h->rounds;j->bias=bias;
  j->a=j->role==0?(h->depth==2?v:v+2U*1024U*1024U+i*524288U+(h->depth==3?i*2048U:0U)):v;
  j->w=v+1048576U;j->out=j->role==0?v+6U*1024U*1024U+i*2048U:v+7U*1024U*1024U;
  memset(j->out,0,2048);qurt_sem_init_val(&j->ready,0);qurt_sem_init_val(&j->go,0);qurt_sem_init_val(&j->done,0);
  qurt_thread_attr_t a;qurt_thread_attr_init(&a);qurt_thread_attr_set_name(&a,"bw-concurrent");qurt_thread_attr_set_stack_addr(&a,cw_stacks[i]);qurt_thread_attr_set_stack_size(&a,sizeof(cw_stacks[i]));qurt_thread_attr_set_priority(&a,qurt_thread_get_priority(qurt_thread_get_id()));
  if(qurt_thread_create(&threads[i],&a,cw_worker,j))return -1;
 }
 for(uint32_t i=0;i<nt;i++)qurt_sem_down(&cw_jobs[i].ready);
 uint64_t ticks=(uint64_t)h->repeats*192U/10U;
 for(uint32_t round=0;round<h->rounds;round++){
  uint64_t begin=HAP_perf_get_qtimer_count()+38400U,end=begin+ticks;
  for(uint32_t i=0;i<nt;i++){cw_jobs[i].begin=begin;cw_jobs[i].deadline=end;}
  asm volatile("barrier":::"memory");
  for(uint32_t i=0;i<nt;i++)qurt_sem_up(&cw_jobs[i].go);
  for(uint32_t i=0;i<nt;i++)qurt_sem_down(&cw_jobs[i].done);
  uint64_t c0=UINT64_MAX,c1=0,t0=UINT64_MAX,t1=0;
  h->hvx_bytes[round]=h->hmx_bytes[round]=h->start_delay[round]=h->finish_delay[round]=0;
  for(uint32_t i=0;i<nt;i++){
   struct cw_job *j=&cw_jobs[i];
   if(j->role==0)h->hvx_bytes[round]+=j->count*524288ULL*8ULL;
   if(j->role==1)h->hmx_bytes[round]+=j->count*16ULL*h->bytes*(fmt==3?8U:fmt==4?6U:5U)/4U;
   uint64_t startdelay=j->t0-begin,finishdelay=j->t1-end;
   if(startdelay>h->start_delay[round])h->start_delay[round]=startdelay;
   if(finishdelay>h->finish_delay[round])h->finish_delay[round]=finishdelay;
   if(j->c0<c0)c0=j->c0;if(j->c1>c1)c1=j->c1;
   if(j->t0<t0)t0=j->t0;if(j->t1>t1)t1=j->t1;
   h->errors+=j->status!=0||j->count==0;
  }
  h->ticks[round]=ticks;h->cycles[round]=(uint64_t)((double)(c1-c0)*ticks/(t1-t0));
 }
 for(uint32_t i=0;i<nt;i++){
  int st;qurt_thread_join(threads[i],&st);struct cw_job *j=&cw_jobs[i];
  if(j->role==0){for(uint32_t k=0;k<1024;k++)h->errors+=j->out[k]!=(fmt==3?(k%2?0x1c:0):1);}
  else if(j->role==1){
   if(fmt==3){__fp16 val=(__fp16)((float)(h->bytes/64U)/65536.0f);uint16_t bits;memcpy(&bits,&val,2);for(uint32_t k=0;k<1024;k++)h->errors+=((uint16_t*)j->out)[k]!=bits;}
   else for(uint32_t k=0;k<2048;k++)h->errors+=j->out[k]!=255;
  }
  qurt_sem_destroy(&j->ready);qurt_sem_destroy(&j->go);qurt_sem_destroy(&j->done);
 }
 return 0;
}

int qbh_bandwidth_run(void *vtcm,uint32_t vbytes,uint32_t hmx,int fd,uint32_t size){
 uint8_t *mem=0,*v=vtcm;int result=HAP_mmap_get(fd,(void**)&mem,0);if(result||!mem)return AEE_EFAILED;
 qurt_mem_cache_clean((qurt_addr_t)mem,sizeof(struct bw_header),QURT_MEM_CACHE_INVALIDATE,QURT_MEM_DCACHE);
 struct bw_header *h=(void*)mem;h->status=1;h->vtcm_bytes=vbytes;h->errors=0;h->checks[0]=(uint32_t)qurt_hvx_get_units();
 if(h->magic!=BW_MAGIC||size<BW_DRAM+BW_PAYLOAD||h->rounds!=BW_ROUNDS||h->mode>10||!h->bytes||h->bytes%1024||!h->repeats||h->repeats>100000||!h->workers||h->workers>MAXW||!h->stream||h->stream>32||!h->depth||h->depth>256||(h->mode<7?h->bypass>1:(h->bypass<3||h->bypass>5)))goto done;
 if(h->mode>=7){
  if(h->bytes!=1048576U||h->depth>3||h->repeats<1000U)goto done;
  if(concurrent_run(v,hmx,h))goto done;
 }else if(h->mode<3){
  if((uint64_t)h->workers*(h->bytes*2U+(h->depth-1U)*128U)>vbytes)goto done;
  qurt_thread_t threads[MAXW];
  for(uint32_t i=0;i<h->workers;i++){struct job *j=&jobs[i];memset(j,0,sizeof(*j));j->p=v+i*(h->bytes*2U+(h->depth-1U)*128U);j->out=j->p+h->bytes;j->h=h;j->id=i;for(uint32_t k=0;k<h->bytes;k++)j->p[k]=pattern(k+i*97U);memset(j->out,0,h->bytes);qurt_sem_init_val(&j->ready,0);qurt_sem_init_val(&j->go,0);qurt_sem_init_val(&j->done,0);qurt_thread_attr_t a;qurt_thread_attr_init(&a);qurt_thread_attr_set_name(&a,"bw-hvx");qurt_thread_attr_set_stack_addr(&a,stacks[i]);qurt_thread_attr_set_stack_size(&a,sizeof(stacks[i]));qurt_thread_attr_set_priority(&a,qurt_thread_get_priority(qurt_thread_get_id()));if(qurt_thread_create(&threads[i],&a,worker,j))goto done;}
  for(uint32_t i=0;i<h->workers;i++)qurt_sem_down(&jobs[i].ready);
  for(uint32_t r=0;r<h->rounds;r++){
   for(uint32_t i=0;i<h->workers;i++)qurt_sem_up(&jobs[i].go);
   for(uint32_t i=0;i<h->workers;i++)qurt_sem_down(&jobs[i].done);
   uint64_t t0=UINT64_MAX,t1=0,c0=UINT64_MAX,c1=0;
   for(uint32_t i=0;i<h->workers;i++){struct job*j=&jobs[i];if(j->t0<t0)t0=j->t0;if(j->t1>t1)t1=j->t1;if(j->c0<c0)c0=j->c0;if(j->c1>c1)c1=j->c1;}
   h->ticks[r]=t1-t0;h->cycles[r]=c1-c0;
  }
  for(uint32_t i=0;i<h->workers;i++){int st;qurt_thread_join(threads[i],&st);h->errors+=jobs[i].status!=0;uint32_t len=h->mode==0?1024:h->bytes;for(uint32_t b=0;b<len;b++)h->errors+=jobs[i].out[b]!=(h->mode==1?0x5a:pattern(b+(h->mode==0?h->bytes-1024U:0U)+i*97U));qurt_sem_destroy(&jobs[i].ready);qurt_sem_destroy(&jobs[i].go);qurt_sem_destroy(&jobs[i].done);}
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
  /* Untimed coverage of every source region, not just the last timed batch. */
  for(uint32_t group=0;group<groups;group++){
   uint32_t base=group*span;
   for(uint32_t d=0;d<h->depth;d++){struct qbh_dma_desc_1d *x=&desc[d].descriptor;x->control=0;x->length=h->bytes;x->src_bypass=h->bypass;x->dst_bypass=1;x->src=(uint32_t)(uintptr_t)(mem+BW_PAYLOAD+base+d*h->bytes);}
   if(qbh_dma_start(desc)||qbh_dma_wait_idle()){h->errors++;break;}
   for(uint32_t b=0;b<span;b++)h->errors+=v[b]!=pattern(base+b);
  }
  h->checks[1]=groups*span;
  h->payload_bytes=(uint64_t)h->repeats*span;
 }
 h->status=0;
 done:qurt_mem_cache_clean((qurt_addr_t)mem,sizeof(*h),QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE);HAP_mmap_put(fd);return h->status?AEE_EBADPARM:AEE_SUCCESS;
}
