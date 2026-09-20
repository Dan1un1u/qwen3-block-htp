/* L32-0059: complete SP2 Down component, same native W4 payload.
 * 11=one HVX lane, 12=four HVX lanes, 13=current HMX radix256 mapping.
 * All tensor workspaces are VTCM. Thread stacks/descriptors carry control only.
 * Input logical SP2 signed16; sum_offset: sums[N], scales[N], residual[M,N].
 * Output: exact signed32 accumulator followed by FP32 residual update.
 */
static uint8_t sp59_stacks[3][16384] __attribute__((aligned(128)));
static struct qbh_dma_aligned_desc_1d sp59_dma[2];
struct sp59_pool;
struct sp59_job {struct sp59_pool *p;uint32_t id;};
struct sp59_pool {
 qurt_thread_t thread[3];qurt_sem_t go[3],done[3];
 struct sp59_job job[3];volatile uint32_t stop;int lock_status[3];
 const uint8_t *weight;const uint32_t *meta;int32_t *output;
 uint32_t rows,k,n,first,count,created;
};
static HVX_Vector sp59_meta(HVX_Vector v) {
 HVX_Vector zero=Q6_V_vzero(),one=Q6_V_vsplat_R(1),s=Q6_Vw_vasr_VwR(v,31);
 HVX_Vector mag=Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(v,s),s);
 HVX_Vector e=Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(31),Q6_Vuw_vcl0_Vuw(mag));
 HVX_Vector bit=Q6_Vw_vasl_VwVw(one,e);
 HVX_Vector rem=Q6_Vw_vsub_VwVw(mag,bit);
 HVX_Vector f=Q6_Vw_vsub_VwVw(Q6_V_vsplat_R(31),Q6_Vuw_vcl0_Vuw(rem));
 HVX_Vector flags=Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(mag,zero),Q6_V_vsplat_R(65536),zero);
 flags=Q6_V_vor_VV(flags,Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(rem,zero),Q6_V_vsplat_R(131072),zero));
 return Q6_V_vor_VV(Q6_V_vor_VV(Q6_V_vand_VV(e,Q6_V_vsplat_R(15)),Q6_Vw_vasl_VwR(Q6_V_vand_VV(f,Q6_V_vsplat_R(15)),8)),
                   Q6_V_vor_VV(flags,Q6_V_vand_VV(s,Q6_V_vsplat_R(0x80000000U))));
}
static void sp59_prepare(const int16_t *input,uint32_t *meta,uint32_t count) {
 for(uint32_t i=0;i<count;i+=64U) {
  HVX_VectorPair v=Q6_Ww_vunpack_Vh(*(const HVX_Vector *)(input+i));
  *(HVX_Vector *)(meta+i)=sp59_meta(Q6_V_lo_W(v));
  *(HVX_Vector *)(meta+i+32U)=sp59_meta(Q6_V_hi_W(v));
 }
}
/* Four explicit accumulators keep partial sums in vector registers.
 * Each native128-byte W4 load supplies eight K positions x32 outputs.
 * No weight tensor expansion, integer multiply or horizontal reduction. */
#define SP59_ACC(ACC,ROW) do { \
 uint32_t m=meta[(size_t)(ROW)*k+kk+j]; \
 if(m&65536U) { \
  HVX_Vector sw=(m&0x80000000U)?Q6_Vw_vsub_VwVw(Q6_V_vzero(),w):w; \
  HVX_Vector prod=Q6_Vw_vasl_VwR(sw,m&15U); \
  if(m&131072U)prod=Q6_Vw_vadd_VwVw(prod,Q6_Vw_vasl_VwR(sw,(m>>8)&15U)); \
  ACC=Q6_Vw_vadd_VwVw(ACC,prod); \
 } \
} while(0)
static __attribute__((noinline)) void sp59_kernel(const uint8_t *weight,
 const uint32_t *meta,int32_t *out,uint32_t rows,uint32_t k,uint32_t n) {
 HVX_Vector mask=Q6_V_vsplat_R(15),eight=Q6_V_vsplat_R(8);
 for(uint32_t r=0;r<rows;r+=4U) {
  HVX_Vector a0=Q6_V_vzero(),a1=Q6_V_vzero(),a2=Q6_V_vzero(),a3=Q6_V_vzero();
  for(uint32_t kk=0;kk<k;kk+=8U) {
   HVX_Vector packed=*(const HVX_Vector *)(weight+(size_t)kk*16U);
   for(uint32_t j=0;j<8U;j++) {
    uint32_t shift=8U*(j&3U)+(j>=4U?4U:0U);
    HVX_Vector nib=Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(packed,shift),mask);
    HVX_Vector w=Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(nib,eight),eight);
    SP59_ACC(a0,r);
    if(rows>1U){SP59_ACC(a1,r+1U);SP59_ACC(a2,r+2U);SP59_ACC(a3,r+3U);}
   }
  }
  *(HVX_Vector *)(out+(size_t)r*n)=a0;
  if(rows>1U){*(HVX_Vector *)(out+(size_t)(r+1U)*n)=a1;*(HVX_Vector *)(out+(size_t)(r+2U)*n)=a2;*(HVX_Vector *)(out+(size_t)(r+3U)*n)=a3;}
 }
}
#undef SP59_ACC
static void sp59_work(struct sp59_pool *p,uint32_t id) {
 if(id<p->count)sp59_kernel(p->weight+(size_t)id*p->k*16U,p->meta,
   p->output+p->first+id*32U,p->rows,p->k,p->n);
}
static void sp59_worker(void *arg) {
 struct sp59_job *j=arg;struct sp59_pool *p=j->p;uint32_t i=j->id-1U;
 p->lock_status[i]=qurt_hvx_lock(QURT_HVX_MODE_128B);
 qurt_sem_up(&p->done[i]);
 for(;;){qurt_sem_down(&p->go[i]);if(p->stop)break;if(!p->lock_status[i])sp59_work(p,j->id);qurt_sem_up(&p->done[i]);}
 if(!p->lock_status[i])qurt_hvx_unlock();qurt_thread_exit(0);
}
static void sp59_pool_close(struct sp59_pool *p);
static int sp59_pool_open(struct sp59_pool *p) {
 for(uint32_t i=0;i<3U;i++) {
  qurt_sem_init_val(&p->go[i],0);qurt_sem_init_val(&p->done[i],0);
  p->job[i].p=p;p->job[i].id=i+1U;
  qurt_thread_attr_t a;qurt_thread_attr_init(&a);
  qurt_thread_attr_set_name(&a,"sp2-shift-add");
  qurt_thread_attr_set_stack_addr(&a,sp59_stacks[i]);qurt_thread_attr_set_stack_size(&a,sizeof(sp59_stacks[i]));
  qurt_thread_attr_set_priority(&a,qurt_thread_get_priority(qurt_thread_get_id()));
  qurt_thread_attr_set_detachstate(&a,QURT_THREAD_ATTR_CREATE_JOINABLE);
  if(qurt_thread_create(&p->thread[i],&a,sp59_worker,&p->job[i])!=QURT_EOK){sp59_pool_close(p);return -1;}
  p->created++;
  qurt_sem_down(&p->done[i]);if(p->lock_status[i]){sp59_pool_close(p);return -1;}
 }
 return 0;
}
static void sp59_pool_close(struct sp59_pool *p) {
 p->stop=1;for(uint32_t i=0;i<p->created;i++)qurt_sem_up(&p->go[i]);
 for(uint32_t i=0;i<p->created;i++){int status=0;qurt_thread_join(p->thread[i],&status);qurt_sem_destroy(&p->go[i]);qurt_sem_destroy(&p->done[i]);}
}
static int sp59_prefetch(uint32_t slot,void *dst,const void *src,uint32_t bytes) {
 struct qbh_dma_aligned_desc_1d *d=&sp59_dma[slot];memset(d,0,sizeof(*d));
 d->descriptor.length=bytes;d->descriptor.src_bypass=1;d->descriptor.ordered=1;
 d->descriptor.src=(uint32_t)(uintptr_t)src;d->descriptor.dst=(uint32_t)(uintptr_t)dst;
 return qbh_dma_start(d);
}
static void sp59_pack(const int16_t *input,uint8_t *a0,uint8_t *a1,
 uint32_t rows,uint32_t k,uint8_t *scratch) {
 memset(a0,0,k*64U);memset(a1,128,k*64U);
 for(uint32_t r=0;r<rows;r++)for(uint32_t c=0;c<k;c+=64U) {
  HVX_Vector v=*(const HVX_Vector *)(input+(size_t)r*k+c);
  *(HVX_Vector *)scratch=Q6_Vb_vpacke_VhVh(v,v);
  *(HVX_Vector *)(scratch+128U)=Q6_V_vxor_VV(Q6_Vb_vpacko_VhVh(v,v),Q6_V_vsplat_R(0x80808080));
  for(uint32_t j=0;j<2U;j++){uint32_t off=((c/32U)+j)*2048U+r*32U;
   memcpy(a0+off,scratch+j*32U,32U);
   memcpy((rows==1U?a0+off+32U:a1+off),scratch+128U+j*32U,32U);
  }
 }
}
static void sp59_hmx(const uint8_t *w,const uint8_t *a0,const uint8_t *a1,
 int32_t *out,const int32_t *sums,uint32_t rows,uint32_t k,uint32_t n,
 uint8_t *raw0,uint8_t *raw1,int32_t *wide0,int32_t *wide1,uint32_t *bias,struct lsp2_header *h) {
 uint32_t passes=rows==1U?1U:2U;
 for(uint32_t p=0;p<passes;p++) {
  for(uint32_t i=0;i<32U;i++){bias[i]=24U<<10;bias[i+32U]=0;}
  qbh_hmx_begin_u8s8_output(bias);h->streams+=qbh_hmx_accumulate_u8n4_projection(p?a1:a0,w,k/32U);
  for(uint32_t digit=0;digit<3U;digit++) {
   for(uint32_t i=0;i<32U;i++){bias[i]=(24U-8U*digit)<<10;bias[i+32U]=0;}
   Q6_bias_mxmem2_A(bias);Q6_mxmem_AR_after_retain_cm_ub((p?raw1:raw0)+2048U*digit,RT);
   asm volatile("barrier":::"memory");h->conversions++;
  }
 }
 unpack_digits(raw0,wide0,3);if(passes==2U)unpack_digits(raw1,wide1,3);
 HVX_Vector sum=*(const HVX_Vector *)sums;
 for(uint32_t r=0;r<rows;r++) {
  HVX_Vector l=*(HVX_Vector *)(wide0+r*32U);
  HVX_Vector h=*(HVX_Vector *)((passes==1U?wide0+32U:wide1)+r*32U);
  l=Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(l,8),8);h=Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(h,8),8);
  *(HVX_Vector *)(out+(size_t)r*n)=Q6_Vw_vsub_VwVw(Q6_Vw_vadd_VwVw(l,Q6_Vw_vasl_VwR(h,8)),Q6_Vw_vasl_VwR(sum,15));
 }
}
static int sp59_run(uint8_t *shared,uint32_t bytes,uint8_t *vtcm,uint32_t vbytes,uint32_t ctx) {
 struct lsp2_header *h=(void *)shared,c=*h;uint32_t m=c.rows,k=c.k,n=c.n;
 if(c.magic!=LSP2_MAGIC||c.abi!=1||c.bytes!=bytes||vbytes!=8388608U||
    (m!=1U&&m!=64U)||!k||k>8192U||k%64U||!n||n>2048U||n%32U||
    !valid(c.input_offset,m*k*2U,bytes)||!valid(c.weight_offset,n*k/2U,bytes)||
    !valid(c.sum_offset,n*8U+m*n*4U,bytes)||!valid(c.output_offset,m*n*8U,bytes))return AEE_EBADPARM;
 uint32_t pos=0;
 #define SP59_ALLOC(TYPE,NAME,SIZE) TYPE *NAME=(TYPE *)(vtcm+pos);pos+=aligned(SIZE)
 SP59_ALLOC(int16_t,input,m*k*2U);SP59_ALLOC(uint32_t,meta,m*k*4U);
 SP59_ALLOC(uint8_t,a0,k*64U);SP59_ALLOC(uint8_t,a1,k*64U);
 SP59_ALLOC(uint8_t,w0,k*64U);SP59_ALLOC(uint8_t,w1,k*64U);
 SP59_ALLOC(int32_t,out,m*n*4U);SP59_ALLOC(float,res,m*n*4U);
 SP59_ALLOC(int32_t,sums,n*4U);SP59_ALLOC(float,scales,n*4U);
 SP59_ALLOC(uint8_t,raw0,8192U);SP59_ALLOC(uint8_t,raw1,8192U);
 SP59_ALLOC(int32_t,wide0,8192U);SP59_ALLOC(int32_t,wide1,8192U);
 SP59_ALLOC(uint32_t,bias,2048U);SP59_ALLOC(uint8_t,scratch,2048U);
 #undef SP59_ALLOC
 if(pos>vbytes)return AEE_ENOMEMORY;
 c.status=-1;c.vtcm_bytes=vbytes;c.peak_bytes=pos;c.streams=c.conversions=0;
 c.load_ticks=c.pack_ticks=c.dma_ticks=c.mac_ticks=c.convert_ticks=c.merge_ticks=c.publish_ticks=0;
 struct sp59_pool pool;memset(&pool,0,sizeof(pool));pool.meta=meta;pool.output=out;pool.rows=m;pool.k=k;pool.n=n;
 int ret=0,hmx_locked=0;
 if(c.mode==12U && sp59_pool_open(&pool))return AEE_EFAILED;
 if(c.mode==13U){ret=HAP_compute_res_hmx_lock2(ctx,HAP_COMPUTE_RES_HMX_SHARED);if(ret)return ret;hmx_locked=1;}
 uint64_t start=HAP_perf_get_qtimer_count(),t=start;
 memcpy(input,shared+c.input_offset,m*k*2U);memcpy(sums,shared+c.sum_offset,n*4U);
 memcpy(scales,shared+c.sum_offset+n*4U,n*4U);memcpy(res,shared+c.sum_offset+n*8U,m*n*4U);
 c.load_ticks=HAP_perf_get_qtimer_count()-t;t=HAP_perf_get_qtimer_count();
 if(c.mode==13U)sp59_pack(input,a0,a1,m,k,scratch);else sp59_prepare(input,meta,m*k);
 c.pack_ticks=HAP_perf_get_qtimer_count()-t;
 uint32_t groups=(n+127U)/128U;
 t=HAP_perf_get_qtimer_count();
 if(sp59_prefetch(0,w0,shared+c.weight_offset,(n<128U?n:128U)*k/2U)){ret=AEE_EFAILED;goto end;}
 c.dma_ticks+=HAP_perf_get_qtimer_count()-t;
 for(uint32_t g=0;g<groups;g++) {
  t=HAP_perf_get_qtimer_count();if(qbh_dma_wait_idle()){ret=AEE_EFAILED;goto end;}
  if(g+1U<groups) {uint32_t remain=n-(g+1U)*128U;
   if(sp59_prefetch((g+1U)&1U,(g&1U)?w0:w1,shared+c.weight_offset+(size_t)(g+1U)*64U*k,(remain<128U?remain:128U)*k/2U)){ret=AEE_EFAILED;goto end;}}
  c.dma_ticks+=HAP_perf_get_qtimer_count()-t;t=HAP_perf_get_qtimer_count();
  pool.weight=(g&1U)?w1:w0;pool.first=g*128U;pool.count=(n-pool.first<128U?n-pool.first:128U)/32U;
  if(c.mode==12U){for(uint32_t i=0;i<3U;i++)qurt_sem_up(&pool.go[i]);sp59_work(&pool,0);for(uint32_t i=0;i<3U;i++)qurt_sem_down(&pool.done[i]);}
  else for(uint32_t i=0;i<pool.count;i++) {
   if(c.mode==11U)sp59_work(&pool,i);
   else sp59_hmx(pool.weight+(size_t)i*k*16U,a0,a1,out+pool.first+i*32U,sums+pool.first+i*32U,m,k,n,raw0,raw1,wide0,wide1,bias,&c);
  }
  c.mac_ticks+=HAP_perf_get_qtimer_count()-t;
 }
 t=HAP_perf_get_qtimer_count();
 for(uint32_t r=0;r<m;r++)for(uint32_t j=0;j<n;j+=32U) {
  HVX_Vector v=Q6_Vsf_equals_Vw(*(HVX_Vector *)(out+(size_t)r*n+j));
  HVX_Vector prod=Q6_Vqf32_vmpy_VsfVsf(v,*(HVX_Vector *)(scales+j));
  HVX_Vector sum=Q6_Vqf32_vadd_Vqf32Vsf(prod,*(HVX_Vector *)(res+(size_t)r*n+j));
  *(HVX_Vector *)(res+(size_t)r*n+j)=Q6_Vsf_equals_Vqf32(sum);
 }
 c.merge_ticks=HAP_perf_get_qtimer_count()-t;t=HAP_perf_get_qtimer_count();
 memcpy(shared+c.output_offset,out,m*n*4U);memcpy(shared+c.output_offset+m*n*4U,res,m*n*4U);
 c.publish_ticks=HAP_perf_get_qtimer_count()-t;
 end:c.total_ticks=HAP_perf_get_qtimer_count()-start;c.status=ret;*h=c;
 if(hmx_locked)HAP_compute_res_hmx_unlock2(ctx,HAP_COMPUTE_RES_HMX_SHARED);
 if(c.mode==12U)sp59_pool_close(&pool);
 return ret;
}
