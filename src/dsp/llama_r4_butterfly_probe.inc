/* L32-0069: post-SwiGLU FP16 -> normalized H8192 -> ordinary A8.
 * Component diagnostic only. No SP2/INT16 carrier, no HMX execution.
 * FP32 butterflies in VTCM; intra-vector stages stay in registers.
 * Four workers partition independent token rows; decode uses one worker.
 */
static const uint32_t bf69_lanes[32] __attribute__((aligned(128)))={
 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
static HVX_Vector bf69_splat(float x){uint32_t u;memcpy(&u,&x,4);return Q6_V_vsplat_R(u);}
static HVX_Vector bf69_mul(HVX_Vector a,HVX_Vector b){return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a,b));}
static __attribute__((noinline)) void bf69_rows(float *x,uint32_t rows) {
 const HVX_Vector lanes=*(const HVX_Vector *)bf69_lanes;
 for(uint32_t row=0;row<rows;row++) {
  float *v=x+row*8192U;
  for(uint32_t j=0;j<8192U;j+=32U) {
   HVX_Vector a=*(HVX_Vector *)(v+j);
   for(uint32_t h=1;h<32;h*=2U) {
    HVX_Vector b=Q6_V_vdelta_VV(a,Q6_V_vsplat_R((h*4U)*0x01010101U));
    HVX_VectorPred upper=Q6_Q_vcmp_gt_VwVw(Q6_V_vand_VV(lanes,Q6_V_vsplat_R(h)),Q6_V_vzero());
    a=Q6_V_vmux_QVV(upper,Q6_Vsf_vsub_VsfVsf(b,a),Q6_Vsf_vadd_VsfVsf(a,b));
   }
   *(HVX_Vector *)(v+j)=a;
  }
  for(uint32_t h=32;h<8192U;h*=2U)
   for(uint32_t base=0;base<8192U;base+=2U*h)
    for(uint32_t j=0;j<h;j+=32U) {
     HVX_Vector a=*(HVX_Vector *)(v+base+j),b=*(HVX_Vector *)(v+base+h+j);
     *(HVX_Vector *)(v+base+j)=Q6_Vsf_vadd_VsfVsf(a,b);
     *(HVX_Vector *)(v+base+h+j)=Q6_Vsf_vsub_VsfVsf(a,b);
    }
  for(uint32_t j=0;j<8192U;j+=32U)
   *(HVX_Vector *)(v+j)=bf69_mul(*(HVX_Vector *)(v+j),bf69_splat(0.011048543456039806f));
 }
}
struct bf69_pool;
struct bf69_job {struct bf69_pool *p;uint32_t id;};
struct bf69_pool {qurt_thread_t thread[3];qurt_sem_t go[3],done[3];struct bf69_job job[3];uint32_t created,rows;volatile uint32_t stop;int status[3];float *x;};
static uint8_t bf69_stacks[3][8192] __attribute__((aligned(128)));
static void bf69_work(struct bf69_pool *p,uint32_t id){uint32_t nr=p->rows/4U;bf69_rows(p->x+id*nr*8192U,nr);}
static void bf69_worker(void *arg){struct bf69_job *j=arg;struct bf69_pool *p=j->p;uint32_t i=j->id-1;
 p->status[i]=qurt_hvx_lock(QURT_HVX_MODE_128B);qurt_sem_up(&p->done[i]);
 for(;;){qurt_sem_down(&p->go[i]);if(p->stop)break;if(!p->status[i])bf69_work(p,j->id);qurt_sem_up(&p->done[i]);}
 if(!p->status[i])qurt_hvx_unlock();qurt_thread_exit(0);
}
static void bf69_close(struct bf69_pool *p){p->stop=1;for(uint32_t i=0;i<p->created;i++)qurt_sem_up(&p->go[i]);for(uint32_t i=0;i<p->created;i++){int status;qurt_thread_join(p->thread[i],&status);qurt_sem_destroy(&p->go[i]);qurt_sem_destroy(&p->done[i]);}}
static int bf69_open(struct bf69_pool *p){for(uint32_t i=0;i<3;i++){
 qurt_sem_init_val(&p->go[i],0);qurt_sem_init_val(&p->done[i],0);p->job[i].p=p;p->job[i].id=i+1;
 qurt_thread_attr_t a;qurt_thread_attr_init(&a);qurt_thread_attr_set_name(&a,"r4-fwht");qurt_thread_attr_set_stack_addr(&a,bf69_stacks[i]);qurt_thread_attr_set_stack_size(&a,sizeof(bf69_stacks[i]));qurt_thread_attr_set_priority(&a,qurt_thread_get_priority(qurt_thread_get_id()));qurt_thread_attr_set_detachstate(&a,QURT_THREAD_ATTR_CREATE_JOINABLE);
 if(qurt_thread_create(&p->thread[i],&a,bf69_worker,&p->job[i])!=QURT_EOK){bf69_close(p);return -1;}p->created++;qurt_sem_down(&p->done[i]);if(p->status[i]){bf69_close(p);return -1;}
 }return 0;}
static void bf69_dispatch(struct bf69_pool *p,uint32_t parallel){if(!parallel){bf69_rows(p->x,p->rows);return;}for(uint32_t i=0;i<3;i++)qurt_sem_up(&p->go[i]);bf69_work(p,0);for(uint32_t i=0;i<3;i++)qurt_sem_down(&p->done[i]);}
static int bf69_run(uint8_t *shared,uint32_t bytes,uint8_t *vtcm,uint32_t vbytes){
 struct lsp2_header *h=(void*)shared,c=*h;uint32_t rows=c.rows,count=rows*8192U;
 if(c.magic!=LSP2_MAGIC||c.abi!=1||c.bytes!=bytes||vbytes!=8388608U||c.k!=8192U||(rows!=1U&&rows!=64U)||(c.n!=1U&&c.n!=16U)||!valid(c.input_offset,count*2U,bytes)||!valid(c.weight_offset,12U,bytes)||!valid(c.output_offset,count*4U+524288U,bytes))return AEE_EBADPARM;
 float params[3];memcpy(params,shared+c.weight_offset,12U);
 if(!(params[0]>0.0f)||params[1]<0.0f||params[1]>255.0f)return AEE_EBADPARM;
 __fp16 *src=(void*)vtcm;float *x=(void*)(vtcm+aligned(count*2U));uint8_t *q=(void*)(x+count);
 uint32_t peak=(uint32_t)(q+524288U-vtcm);if(peak>vbytes)return AEE_ENOMEMORY;
 struct bf69_pool pool;memset(&pool,0,sizeof(pool));pool.x=x;pool.rows=rows;
 uint32_t parallel=(c.mode==17U&&rows==64U),reps=(uint32_t)params[2];if(!reps||reps>100U)return AEE_EBADPARM;
 if(parallel&&bf69_open(&pool))return AEE_EFAILED;
 c.load_ticks=c.pack_ticks=c.dma_ticks=c.mac_ticks=c.convert_ticks=c.merge_ticks=c.publish_ticks=0;c.streams=0;c.conversions=reps*c.n;c.vtcm_bytes=vbytes;c.peak_bytes=peak;
 uint64_t start=HAP_perf_get_qtimer_count(),t=start;memcpy(src,shared+c.input_offset,count*2U);c.load_ticks=HAP_perf_get_qtimer_count()-t;
 memset(q,(int)params[1],524288U);
 for(uint32_t rep=0;rep<reps;rep++) {
  t=HAP_perf_get_qtimer_count();
  for(uint32_t j=0;j<count;j+=64U){HVX_VectorPair f=Q6_Wsf_vcvt_Vhf(*(const HVX_Vector *)(src+j));HVX_VectorPair o=Q6_W_vshuff_VVR(Q6_V_hi_W(f),Q6_V_lo_W(f),-4);*(HVX_Vector *)(x+j)=Q6_V_lo_W(o);*(HVX_Vector *)(x+j+32U)=Q6_V_hi_W(o);}
  c.pack_ticks+=HAP_perf_get_qtimer_count()-t;t=HAP_perf_get_qtimer_count();
  if(c.mode!=18U)for(uint32_t layer=0;layer<c.n;layer++)bf69_dispatch(&pool,parallel);
  c.mac_ticks+=HAP_perf_get_qtimer_count()-t;t=HAP_perf_get_qtimer_count();
  for(uint32_t r=0;r<rows;r++)for(uint32_t j=0;j<8192U;j+=32U){
   HVX_Vector a=Q6_Vsf_vadd_VsfVsf(bf69_mul(*(HVX_Vector *)(x+r*8192U+j),bf69_splat(params[0])),bf69_splat(params[1]));
   HVX_Vector w=Q6_Vw_equals_Vsf(Q6_Vsf_vadd_VsfVsf(a,bf69_splat(.5f)));
   HVX_Vector packed=Q6_Vub_vpack_VhVh_sat(Q6_V_vzero(),Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(),w));
   size_t off=(size_t)(j/32U)*2048U+r*32U;uint32_t shift=off&127U;
   HVX_VectorPred pred=Q6_Q_and_QQn(Q6_Q_vsetq2_R(shift+32U),Q6_Q_vsetq_R(shift));
   Q6_vmem_QRIV(pred,(HVX_Vector *)(q+off),Q6_V_vlalign_VVR(packed,packed,shift));
  }
  c.convert_ticks+=HAP_perf_get_qtimer_count()-t;
 }
 t=HAP_perf_get_qtimer_count();memcpy(shared+c.output_offset,x,count*4U);memcpy(shared+c.output_offset+count*4U,q,524288U);c.publish_ticks=HAP_perf_get_qtimer_count()-t;c.total_ticks=HAP_perf_get_qtimer_count()-start;c.status=0;*h=c;if(parallel)bf69_close(&pool);return AEE_SUCCESS;
}
