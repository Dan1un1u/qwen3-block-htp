static const int8_t qbh_r4_h12[12][12]={
{1,1,1,1,1,1,1,1,1,1,1,1},
{1,-1,-1,1,-1,-1,-1,1,1,1,-1,1},
{1,1,-1,-1,1,-1,-1,-1,1,1,1,-1},
{1,-1,1,-1,-1,1,-1,-1,-1,1,1,1},
{1,1,-1,1,-1,-1,1,-1,-1,-1,1,1},
{1,1,1,-1,1,-1,-1,1,-1,-1,-1,1},
{1,1,1,1,-1,1,-1,-1,1,-1,-1,-1},
{1,-1,1,1,1,-1,1,-1,-1,1,-1,-1},
{1,-1,-1,1,1,1,-1,1,-1,-1,1,-1},
{1,-1,-1,-1,1,1,1,-1,1,-1,-1,1},
{1,1,-1,-1,-1,1,1,1,-1,1,-1,-1},
{1,-1,1,-1,-1,-1,1,1,1,-1,1,-1}
};

/* EXP0261 full R4 = H12 x H512. Dense matrix factors, never FWHT.
 * Lifetime: gate/up projections fully joined. hmx_activation (768KiB),
 * expanded_weight and expanded_weight_alt (1MiB each) are phase-dead.
 * Stage1 output stays in alt while up to16 tokens of stage2 use the other
 * two arenas; stage2 sign matrix occupies the dead HMX output tile. */
static size_t qbh_r4_pos(uint32_t r,uint32_t c,uint32_t nt) {
    return qbh_hmx_fp16_matrix_tile_offset(r/32U,c/32U,nt)+
           qbh_hmx_fp16_tile_offset(r%32U,c%32U);
}
static void qbh_r4_zero(void *p,uint32_t bytes) {
    for(uint32_t i=0;i<bytes/128U;++i)((HVX_Vector *)p)[i]=Q6_V_vzero();
}
static int qbh_r4_gemm(struct qbh_block_header *h,
    struct qbh_block_hmx_worker *worker,const __fp16 *a,const __fp16 *w,
    void *sc,__fp16 *out,uint32_t mt,uint32_t kt,uint32_t nt) {
    uint64_t t=HAP_perf_get_qtimer_count();
    int rc=qbh_hmx_submit(worker,QBH_BLOCK_HMX_FP16,a,w,sc,out,mt,kt,nt);
    h->dense_r4_matmul_ticks+=HAP_perf_get_qtimer_count()-t;
    ++h->hmx_command_count;++h->dense_r4_hmx_calls;
    h->hmx_fp16_tile_pair_count+=mt*kt*nt;
    return rc;
}
static void qbh_r4_prepare_tile(const uint8_t *gate,const uint8_t *up,
    __fp16 *act,uint32_t rows,uint32_t tile,const uint16_t *lut,uint8_t *scratch) {
    __fp16 tmp[128] __attribute__((aligned(128)));
    uint32_t a=tile/16U,k=tile%16U;
    for(uint32_t r=0;r<rows;r+=4U) {
        qbh_mlp_gate_up_lut_f16_hvx(gate+tile*2048U+r*32U,
            up+tile*2048U+r*32U,(uint16_t *)tmp,128U,lut,scratch);
        if(rows==64U) {
            __fp16 *dst=act+qbh_r4_pos(a*rows+r,k*32U,16U);
            ((HVX_Vector *)dst)[0]=Q6_Vh_vshuff_Vh(((HVX_Vector *)tmp)[0]);
            ((HVX_Vector *)dst)[1]=Q6_Vh_vshuff_Vh(((HVX_Vector *)tmp)[1]);
        } else for(uint32_t c=0;c<32U;++c)act[qbh_r4_pos(a,k*32U+c,16U)]=tmp[c];
    }
}
/* EXP0262: rearrange complete halfword pairs. Offsets map consecutive pairs
 * of channels directly to two HMX row tiles, without touching arithmetic. */
static void qbh_r4_layout_vector(const __fp16 *stage1,__fp16 *act,
    uint32_t rows,uint32_t first,uint32_t count) {
    const HVX_Vector offsets=*(const HVX_Vector *)qbh_attention_vscatter_offsets;
    qbh_r4_zero(act,count*32768U);
    for(uint32_t a=0;a<12U;++a)for(uint32_t r=0;r<count;r+=2U)
        for(uint32_t c=0;c<512U;c+=64U) {
            uint32_t sr=a*rows+first+r;
            const __fp16 *src=stage1+qbh_r4_pos(sr&~1U,c,16U);
            HVX_VectorPair v=Q6_W_vdeal_VVR(
                *(const HVX_Vector *)(src+1024U),*(const HVX_Vector *)src,-2);
            __fp16 *dst=act+r*16384U+(c/32U)*1024U+a*2U;
            Q6_vscatter_RMVwV((uint32_t)(uintptr_t)dst,4095U,offsets,
                (sr&1U)?Q6_V_hi_W(v):Q6_V_lo_W(v));
            if(r+1U<count)Q6_vscatter_RMVwV((uint32_t)(uintptr_t)(dst+16384U),
                4095U,offsets,Q6_V_hi_W(v));
        }
    asm volatile("barrier" ::: "memory");
}
/* EXP0264: retain the generic SF32 multiply/add/round/pack instruction
 * order, but prepare invariant constants outside all channel/row loops. */
struct qbh_r4_quant_constants { HVX_Vector coefficient,offset,rounding; };
static HVX_Vector qbh_r4_splat_sf(float f) {
    return Q6_Vsf_vadd_VsfVsf(Q6_V_vsplat_R(*(const int32_t *)&f),Q6_V_vzero());
}
static struct qbh_r4_quant_constants qbh_r4_quant_prepare(
    const struct qbh_block_qparam *qp) {
    struct qbh_r4_quant_constants q={qbh_r4_splat_sf(1.0f/qp->scale),
        qbh_r4_splat_sf((float)qp->zero_point),qbh_r4_splat_sf(0.5f)};
    return q;
}
static inline __attribute__((always_inline)) HVX_Vector qbh_r4_quant_half(
    HVX_Vector input,const struct qbh_r4_quant_constants *q) {
    HVX_VectorPair f=Q6_Wsf_vcvt_Vhf(input);
    HVX_Vector lo=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(f),q->coefficient));
    HVX_Vector hi=Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(f),q->coefficient));
    lo=Q6_Vw_equals_Vsf(Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vadd_VsfVsf(lo,q->offset),q->rounding));
    hi=Q6_Vw_equals_Vsf(Q6_Vsf_vadd_VsfVsf(Q6_Vsf_vadd_VsfVsf(hi,q->offset),q->rounding));
    HVX_VectorPair ordered=Q6_W_vshuff_VVR(hi,lo,-4);
    return Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(ordered),Q6_V_lo_W(ordered));
}
static inline __attribute__((always_inline)) HVX_Vector qbh_r4_quant_pair(
    HVX_Vector a,HVX_Vector b,const struct qbh_r4_quant_constants *q) {
    HVX_Vector lo=qbh_r4_quant_half(a,q),hi=qbh_r4_quant_half(b,q);
    return Q6_Vub_vpack_VhVh_sat(hi,lo);
}
/* Deliberate memory boundary for constant-only candidate OPT5. OPT6 feeds
 * identical instructions directly from gather registers. */
static __attribute__((noinline)) void qbh_r4_quant_prepared_array(
    const __fp16 *input,uint8_t *out,const struct qbh_r4_quant_constants *q) {
    ((HVX_Vector *)out)[0]=qbh_r4_quant_pair(((const HVX_Vector *)input)[0],((const HVX_Vector *)input)[1],q);
    ((HVX_Vector *)out)[1]=qbh_r4_quant_pair(((const HVX_Vector *)input)[2],((const HVX_Vector *)input)[3],q);
    asm volatile("barrier" ::: "memory");
}
/* Untimed actual-device exhaustive equivalence, including both signed
 * zeros and every finite half. Nonfinite bit patterns are replaced by0. */
static int qbh_r4_quant_selftest(const struct qbh_block_qparam *qp) {
    uint16_t input[256] __attribute__((aligned(128)));
    uint8_t old[256] __attribute__((aligned(128))),out[256] __attribute__((aligned(128)));
    const struct qbh_r4_quant_constants q=qbh_r4_quant_prepare(qp);
    for(uint32_t first=0;first<65536U;first+=256U) {
        for(uint32_t i=0;i<256U;++i)input[i]=((first+i)&0x7c00U)==0x7c00U?0U:(uint16_t)(first+i);
        qbh_hvx_quantize_f16_to_u8((const __fp16 *)input,old,256U,qp);
        qbh_r4_quant_prepared_array((const __fp16 *)input,out,&q);
        if(memcmp(old,out,256U))return -1;
    }
    return 0;
}
static inline __attribute__((always_inline)) HVX_Vector qbh_r4_gather_finish(
    const __fp16 *weight,HVX_Vector *scratch,HVX_Vector offsets,__fp16 *audit,
    uint32_t first,uint32_t count,uint32_t r,uint32_t a,uint32_t c) {
    if(r>=count)return Q6_V_vzero();
    const __fp16 *src=weight+r*16384U+(c/32U)*1024U+a*2U;
    Q6_vgather_ARMVw(scratch,(int32_t)(uintptr_t)src,4095U,offsets);
    HVX_Vector v=*(volatile HVX_Vector *)scratch;
    if(audit)*(HVX_Vector *)(audit+2U*393216U+(first+r)*6144U+a*512U+c)=v;
    return v;
}
static __attribute__((noinline)) void qbh_r4_finish_fused(struct qbh_block_header *h,
    const __fp16 *weight,uint8_t *middle,uint32_t first,uint32_t count,
    HVX_Vector *scratch,__fp16 *audit,uint32_t a_first,uint32_t a_count) {
    const struct qbh_r4_quant_constants q=qbh_r4_quant_prepare(&h->qparams[QBH_BLOCK_QP_MIDDLE]);
    const HVX_Vector offsets=*(const HVX_Vector *)qbh_attention_vscatter_offsets;
    for(uint32_t a=a_first;a<a_first+a_count;++a)for(uint32_t c=0;c<512U;c+=64U)
        for(uint32_t r=0;r<count;r+=4U) {
            HVX_Vector v0=qbh_r4_gather_finish(weight,scratch,offsets,audit,first,count,r,a,c);
            HVX_Vector v1=qbh_r4_gather_finish(weight,scratch,offsets,audit,first,count,r+1U,a,c);
            HVX_Vector lo=qbh_r4_quant_pair(v0,v1,&q);
            HVX_Vector v2=qbh_r4_gather_finish(weight,scratch,offsets,audit,first,count,r+2U,a,c);
            HVX_Vector v3=qbh_r4_gather_finish(weight,scratch,offsets,audit,first,count,r+3U,a,c);
            HVX_Vector hi=qbh_r4_quant_pair(v2,v3,&q);
            HVX_VectorPair x=Q6_W_vdeal_VVR(hi,lo,-32);
            *(HVX_Vector *)(middle+(a*16U+c/32U)*2048U+(first+r)*32U)=Q6_V_lo_W(x);
            *(HVX_Vector *)(middle+(a*16U+c/32U+1U)*2048U+(first+r)*32U)=Q6_V_hi_W(x);
        }
    asm volatile("barrier" ::: "memory");
}
static void qbh_r4_finish_prepared(struct qbh_block_header *h,
    const __fp16 *weight,uint8_t *middle,uint32_t first,uint32_t count,
    HVX_Vector *scratch,__fp16 *audit,uint32_t a_first,uint32_t a_count) {
    const struct qbh_r4_quant_constants q=qbh_r4_quant_prepare(&h->qparams[QBH_BLOCK_QP_MIDDLE]);
    const HVX_Vector offsets=*(const HVX_Vector *)qbh_attention_vscatter_offsets;
    const uint32_t fused=h->dense_r4_optimization==6U;
    for(uint32_t a=a_first;a<a_first+a_count;++a)for(uint32_t c=0;c<512U;c+=64U)
        for(uint32_t r=0;r<count;r+=4U) {
            HVX_Vector v[4];
            for(uint32_t rr=0;rr<4U;++rr) {
                v[rr]=Q6_V_vzero();
                if(r+rr<count) {
                    const __fp16 *src=weight+(r+rr)*16384U+(c/32U)*1024U+a*2U;
                    Q6_vgather_ARMVw(scratch,(int32_t)(uintptr_t)src,4095U,offsets);
                    v[rr]=*(volatile HVX_Vector *)scratch;
                    if(audit)*(HVX_Vector *)(audit+2U*393216U+
                        (first+r+rr)*6144U+a*512U+c)=v[rr];
                }
            }
            HVX_Vector lo,hi;
            if(fused) {
                lo=qbh_r4_quant_pair(v[0],v[1],&q);
                hi=qbh_r4_quant_pair(v[2],v[3],&q);
            } else {
                uint8_t codes[256] __attribute__((aligned(128)));
                qbh_r4_quant_prepared_array((const __fp16 *)v,codes,&q);
                lo=((HVX_Vector *)codes)[0];hi=((HVX_Vector *)codes)[1];
            }
            HVX_VectorPair x=Q6_W_vdeal_VVR(hi,lo,-32);
            *(HVX_Vector *)(middle+(a*16U+c/32U)*2048U+(first+r)*32U)=Q6_V_lo_W(x);
            *(HVX_Vector *)(middle+(a*16U+c/32U+1U)*2048U+(first+r)*32U)=Q6_V_hi_W(x);
        }
    asm volatile("barrier" ::: "memory");
}
static void qbh_r4_finish_range(struct qbh_block_header *h,
    const __fp16 *weight,uint8_t *middle,uint32_t first,uint32_t count,
    HVX_Vector *scratch,__fp16 *audit,uint32_t a_first,uint32_t a_count) {
    if(h->dense_r4_optimization==6U) {
        qbh_r4_finish_fused(h,weight,middle,first,count,scratch,audit,a_first,a_count);return;
    }
    if(h->dense_r4_optimization>=5U) {
        qbh_r4_finish_prepared(h,weight,middle,first,count,scratch,audit,a_first,a_count);return;
    }
    const HVX_Vector offsets=*(const HVX_Vector *)qbh_attention_vscatter_offsets;
    __fp16 tmp[256] __attribute__((aligned(128)));
    uint8_t codes[256] __attribute__((aligned(128)));
    for(uint32_t a=a_first;a<a_first+a_count;++a)for(uint32_t c=0;c<512U;c+=64U)
        for(uint32_t r=0;r<count;r+=4U) {
            for(uint32_t rr=0;rr<4U;++rr) {
                HVX_Vector v=Q6_V_vzero();
                if(r+rr<count) {
                    const __fp16 *src=weight+(r+rr)*16384U+(c/32U)*1024U+a*2U;
                    Q6_vgather_ARMVw(scratch,(int32_t)(uintptr_t)src,4095U,offsets);
                    v=*(volatile HVX_Vector *)scratch;
                }
                ((HVX_Vector *)tmp)[rr]=v;
                if(audit && r+rr<count)memcpy(audit+2U*393216U+
                    (first+r+rr)*6144U+a*512U+c,tmp+rr*64U,128U);
            }
            qbh_hvx_quantize_f16_to_u8(tmp,codes,256U,&h->qparams[QBH_BLOCK_QP_MIDDLE]);
            /* [r0c0,r0c1,r1c0,r1c1] -> native two tiles of four rows. */
            HVX_VectorPair x=Q6_W_vdeal_VVR(((HVX_Vector *)codes)[1],
                ((HVX_Vector *)codes)[0],-32);
            *(HVX_Vector *)(middle+(a*16U+c/32U)*2048U+(first+r)*32U)=Q6_V_lo_W(x);
            *(HVX_Vector *)(middle+(a*16U+c/32U+1U)*2048U+(first+r)*32U)=Q6_V_hi_W(x);
        }
    asm volatile("barrier" ::: "memory");
}
/* Three contexts share only immutable inputs; each owns a 256B VTCM gather
 * destination and disjoint whole channel tiles/groups. No atomic hot loop. */
struct qbh_r4_convert_task {
    struct qbh_block_header *h;
    struct qbh_block_buffers *b;
    const __fp16 *weight;
    uint8_t *middle;
    __fp16 *audit;
    uint32_t finish,first,count,stream;
    volatile uint32_t ready[6],abort;
    uint32_t consumed[2];
    uint64_t work_ticks[3];
};
static void qbh_r4_convert_worker(void *context,uint32_t id) {
    struct qbh_r4_convert_task *task=context;
    uint64_t t=HAP_perf_get_qtimer_count();
    uint8_t *scratch=task->b->w4u8_gather_scratch+id*256U;
    if(task->stream) {
        task->work_ticks[id]=0;
        for(uint32_t g=0;g<6U;++g) {
            while(!task->ready[g]) {
                if(task->abort)return;
                asm volatile("pause(#8)" ::: "memory");
            }
            asm volatile("barrier" ::: "memory");
            uint64_t work=HAP_perf_get_qtimer_count();
            for(uint32_t tile=g*32U+id*16U;tile<g*32U+(id+1U)*16U;++tile)
                qbh_r4_prepare_tile(task->b->gate,task->b->up,(__fp16 *)task->b->hmx_activation,
                    64U,tile,(const uint16_t *)task->b->w4u8_silu_lut,scratch);
            task->work_ticks[id]+=HAP_perf_get_qtimer_count()-work;
            ++task->consumed[id];
        }
        return;
    }
    if(task->finish)qbh_r4_finish_range(task->h,task->weight,task->middle,
        task->first,task->count,(HVX_Vector *)scratch,task->audit,id*4U,4U);
    else for(uint32_t tile=id*64U;tile<(id+1U)*64U;++tile)
        qbh_r4_prepare_tile(task->b->gate,task->b->up,(__fp16 *)task->b->hmx_activation,
            64U,tile,(const uint16_t *)task->b->w4u8_silu_lut,scratch);
    task->work_ticks[id]=HAP_perf_get_qtimer_count()-t;
}
static int qbh_r4_convert_parallel(struct qbh_block_w4f16_pool *pool,
    struct qbh_r4_convert_task *task) {
    if(!pool || pool->worker_count<2U)return -1;
    for(uint32_t i=0;i<pool->worker_count;++i)
        if(pool->jobs[i].command_kind!=QBH_BLOCK_HVX_POOL_NONE)return -1;
    pool->active_worker_count=2U;pool->extra_expand_worker_index=UINT32_MAX;
    for(uint32_t i=0;i<2U;++i) {
        pool->jobs[i].r4_convert_context=task;
        pool->jobs[i].command_kind=QBH_BLOCK_HVX_POOL_R4_CONVERT;
    }
    asm volatile("barrier" ::: "memory");
    for(uint32_t i=0;i<2U;++i)(void)qurt_sem_up(&pool->command_ready[i]);
    qbh_r4_convert_worker(task,2U);
    uint64_t t=HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);pool->active_worker_count=0U;
    task->h->dense_r4_parallel_join_ticks+=HAP_perf_get_qtimer_count()-t;
    for(uint32_t i=0;i<3U;++i)task->h->dense_r4_parallel_work_ticks+=task->work_ticks[i];
    ++task->h->dense_r4_parallel_dispatches;
    if(task->finish)task->h->dense_r4_parallel_finish_groups+=12U;
    else task->h->dense_r4_parallel_prepare_tiles+=192U;
    return 0;
}
/* Up native outputs and raw H512 input occupy independent arenas. A
 * ready publication follows the HMX completion barrier for that Up group.
 * Only these two workers own LUT gather scratch until the final join. */
static int qbh_r4_prefill_start(struct qbh_block_w4f16_pool *pool,
    struct qbh_r4_convert_task *task) {
    if(!pool || pool->worker_count<2U)return -1;
    for(uint32_t i=0;i<pool->worker_count;++i)
        if(pool->jobs[i].command_kind!=QBH_BLOCK_HVX_POOL_NONE)return -1;
    pool->active_worker_count=2U;pool->extra_expand_worker_index=UINT32_MAX;
    task->b->r4_prefill_ready=task->ready;
    for(uint32_t i=0;i<2U;++i) {
        pool->jobs[i].r4_convert_context=task;
        pool->jobs[i].command_kind=QBH_BLOCK_HVX_POOL_R4_CONVERT;
    }
    asm volatile("barrier" ::: "memory");
    for(uint32_t i=0;i<2U;++i)(void)qurt_sem_up(&pool->command_ready[i]);
    return 0;
}
static int qbh_r4_prefill_finish(struct qbh_block_w4f16_pool *pool,
    struct qbh_r4_convert_task *task,int abort) {
    if(abort)task->abort=1U;
    asm volatile("barrier" ::: "memory");
    uint64_t t=HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);pool->active_worker_count=0U;
    task->h->dense_r4_prefill_join_ticks+=HAP_perf_get_qtimer_count()-t;
    task->b->r4_prefill_ready=NULL;
    for(uint32_t i=0;i<2U;++i) {
        task->h->dense_r4_prefill_consume_count+=task->consumed[i];
        task->h->dense_r4_prefill_worker_ticks+=task->work_ticks[i];
        task->h->dense_r4_parallel_work_ticks+=task->work_ticks[i];
    }
    ++task->h->dense_r4_parallel_dispatches;
    if(abort || task->consumed[0]!=6U || task->consumed[1]!=6U)return -1;
    task->h->dense_r4_parallel_prepare_tiles+=192U;
    task->b->r4_prefill_prepared=1U;
    return 0;
}
static int qbh_r4_finish_vector(struct qbh_block_header *h,
    struct qbh_block_buffers *b,struct qbh_block_w4f16_pool *pool,
    const __fp16 *weight,uint8_t *middle,uint32_t first,uint32_t count,
    HVX_Vector *scratch,__fp16 *audit) {
    if(h->dense_r4_optimization>=3U && h->logical_m==64U) {
        struct qbh_r4_convert_task task={.h=h,.b=b,.weight=weight,.middle=middle,
            .audit=audit,.finish=1U,.first=first,.count=count};
        return qbh_r4_convert_parallel(pool,&task);
    }
    qbh_r4_finish_range(h,weight,middle,first,count,scratch,audit,0U,12U);
    return 0;
}
static int qbh_run_dense_r4(struct qbh_block_header *h,uint8_t *shared,
    struct qbh_block_buffers *b,struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,uint8_t *middle) {
    const uint32_t rows=h->logical_m, p1=(12U*rows+31U)&~31U;
    __fp16 *act=(__fp16 *)b->hmx_activation;
    __fp16 *weight=(__fp16 *)b->expanded_weight;
    __fp16 *stage1=(__fp16 *)b->expanded_weight_alt;
    __fp16 *w12=(__fp16 *)b->hmx_output;
    __fp16 tmp[128] __attribute__((aligned(128)));
    uint8_t codes[128] __attribute__((aligned(128)));
    float scales[32] __attribute__((aligned(128)));
    __fp16 *audit=h->dense_r4_audit_offset?(__fp16 *)(shared+h->dense_r4_audit_offset):NULL;
    if(h->variant!=QBH_BLOCK_W4U8 || h->dense_r4_mode>2U ||
       h->dense_r4_optimization>6U ||
       (rows!=1U && rows!=64U) ||
       (h->w4u8_decode_direct_n_gate_up_swiglu_stream && h->dense_r4_optimization<2U) ||
       (h->dense_r4_mode==2U && !audit) ||
       (audit && !qbh_range_valid(h->dense_r4_audit_offset,3U*786432U,h->shared_bytes)))return -1;
    if(audit && h->dense_r4_optimization>=5U &&
       qbh_r4_quant_selftest(&h->qparams[QBH_BLOCK_QP_MIDDLE]))return -1;
    /* These bounds are the existing owned allocations, no arena growth. */
    if((uintptr_t)b->expanded_weight_alt-(uintptr_t)b->expanded_weight<1048576U ||
       (uintptr_t)b->hmx_output-(uintptr_t)b->expanded_weight_alt<1048576U ||
       (uintptr_t)b->compressed_weight-(uintptr_t)b->hmx_activation<786432U) return -1;
    uint64_t t=HAP_perf_get_qtimer_count();
    const uint32_t streamed=rows==1U && h->dense_r4_optimization>=2U &&
        h->w4u8_decode_direct_n_gate_up_swiglu_stream;
    const uint32_t prefill_streamed=rows==64U && h->dense_r4_optimization==4U && b->r4_prefill_prepared;
    if(!streamed && !prefill_streamed) {
        if(!h->dense_r4_optimization || rows==1U)qbh_r4_zero(act,p1*512U*2U);
        if(h->dense_r4_optimization>=3U && rows==64U) {
            struct qbh_r4_convert_task task={.h=h,.b=b};
            if(qbh_r4_convert_parallel(pool,&task))return -1;
        } else for(uint32_t tile=0;tile<192U;++tile)qbh_r4_prepare_tile(b->gate,b->up,act,
            rows,tile,(const uint16_t *)b->w4u8_silu_lut,b->w4u8_gather_scratch);
    }
    /* Materialize H512 from exact dense H128 blocks with parity signs.
     * This generates matrix entries; activations never traverse a butterfly. */
    for(uint32_t n=0;n<16U;++n)for(uint32_t k=0;k<16U;++k) {
        const HVX_Vector *src=(const HVX_Vector *)(qbh_r3_sign_matrix+((n%4U)*4U+k%4U)*1024U);
        HVX_Vector *dst=(HVX_Vector *)(weight+(n*16U+k)*1024U);
        HVX_Vector sign=Q6_Vh_vsplat_R((__builtin_popcount((n/4U)&(k/4U))&1)?0x8000:0);
        for(uint32_t v=0;v<16U;++v)dst[v]=Q6_V_vxor_VV(src[v],sign);
    }
    for(uint32_t c=0;c<32U;++c)scales[c]=(float)(__fp16)0.04419417382415922f;
    qbh_hmx_fp16_init_channel_scales(b->channel_scale,scales);
    h->dense_r4_prepare_ticks+=HAP_perf_get_qtimer_count()-t;
    if(audit)for(uint32_t a=0;a<12U;++a)for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<512U;++c)
        audit[(a*rows+r)*512U+c]=act[qbh_r4_pos(a*rows+r,c,16U)];
    if(qbh_r4_gemm(h,worker,act,weight,b->channel_scale,stage1,p1/32U,16U,16U))return -1;
    if(h->dense_r4_mode==2U) {
        /* Untimed oracle replaces the result only after the HMX state transition. */
        for(uint32_t r=0;r<12U*rows;++r)for(uint32_t n=0;n<512U;++n) {
            double sum=0;
            for(uint32_t k=0;k<512U;++k)sum+=(double)audit[r*512U+k]*((__builtin_popcount(n&k)&1)?-1.0:1.0);
            stage1[qbh_r4_pos(r,n,16U)]=(__fp16)(sum*(double)(__fp16)0.04419417382415922f);
        }
    }
    if(audit)for(uint32_t r=0;r<12U*rows;++r)for(uint32_t c=0;c<512U;++c)
        audit[393216U+r*512U+c]=stage1[qbh_r4_pos(r,c,16U)];
    t=HAP_perf_get_qtimer_count();
    qbh_r4_zero(w12,2048U);
    /* Input row is [group], output column is [new group]. */
    for(uint32_t n=0;n<12U;++n)for(uint32_t k=0;k<12U;++k)
        w12[qbh_hmx_fp16_tile_offset(k,n)]=(__fp16)qbh_r4_h12[k][n];
    for(uint32_t c=0;c<32U;++c)scales[c]=(float)(__fp16)0.2886751345948129f;
    qbh_hmx_fp16_init_channel_scales(b->channel_scale,scales);
    h->dense_r4_prepare_ticks+=HAP_perf_get_qtimer_count()-t;
    if(h->dense_r4_optimization>=2U && rows==64U && h->dense_r4_mode==1U) {
        /* Two 256KiB input/output slots live inside the original arenas.
         * The previous output and next input use the opposite slot from the
         * active HMX command. Stage1, H12 and scales remain immutable. */
        t=HAP_perf_get_qtimer_count();
        qbh_r4_layout_vector(stage1,act,rows,0U,8U);
        h->dense_r4_layout_ticks+=HAP_perf_get_qtimer_count()-t;
        for(uint32_t batch=0;batch<8U;++batch) {
            uint32_t slot=batch&1U,other=slot^1U;
            __fp16 *ai=act+slot*131072U,*oi=weight+slot*131072U;
            qbh_hmx_start(worker,QBH_BLOCK_HMX_FP16,ai,w12,b->channel_scale,oi,128U,1U,1U);
            ++h->hmx_command_count;++h->dense_r4_hmx_calls;
            h->hmx_fp16_tile_pair_count+=128U;
            ++h->dense_r4_pipeline_batches;
            if(batch) {
                t=HAP_perf_get_qtimer_count();
                if(qbh_r4_finish_vector(h,b,pool,weight+other*131072U,middle,(batch-1U)*8U,8U,
                    (HVX_Vector *)b->w4u8_gather_scratch,audit)) {
                    (void)qbh_hmx_wait(worker);return -1;
                }
                uint64_t dt=HAP_perf_get_qtimer_count()-t;
                h->dense_r4_finish_ticks+=dt;h->dense_r4_pipeline_hvx_ticks+=dt;
            }
            if(batch+1U<8U) {
                t=HAP_perf_get_qtimer_count();
                qbh_r4_layout_vector(stage1,act+other*131072U,rows,(batch+1U)*8U,8U);
                uint64_t dt=HAP_perf_get_qtimer_count()-t;
                h->dense_r4_layout_ticks+=dt;h->dense_r4_pipeline_hvx_ticks+=dt;
            }
            t=HAP_perf_get_qtimer_count();
            int rc=qbh_hmx_wait(worker);
            h->dense_r4_matmul_ticks+=HAP_perf_get_qtimer_count()-t;
            if(rc)return -1;
        }
        t=HAP_perf_get_qtimer_count();
        if(qbh_r4_finish_vector(h,b,pool,weight+131072U,middle,56U,8U,
            (HVX_Vector *)b->w4u8_gather_scratch,audit))return -1;
        h->dense_r4_finish_ticks+=HAP_perf_get_qtimer_count()-t;
    } else
    for(uint32_t first=0;first<rows;first+=16U) {
        uint32_t count=rows-first<16U?rows-first:16U;
        t=HAP_perf_get_qtimer_count();
        if(h->dense_r4_optimization)qbh_r4_layout_vector(stage1,act,rows,first,count);
        else {
        qbh_r4_zero(act,count*512U*32U*2U);
        for(uint32_t r=0;r<count;++r)for(uint32_t a=0;a<12U;++a)for(uint32_t c=0;c<512U;++c)
            act[qbh_r4_pos(r*512U+c,a,1U)]=stage1[qbh_r4_pos(a*rows+first+r,c,16U)];
        }
        h->dense_r4_layout_ticks+=HAP_perf_get_qtimer_count()-t;
        if(qbh_r4_gemm(h,worker,act,w12,b->channel_scale,weight,count*16U,1U,1U))return -1;
        if(h->dense_r4_mode==2U)for(uint32_t r=0;r<count*512U;++r)for(uint32_t n=0;n<12U;++n) {
            double sum=0;
            for(uint32_t k=0;k<12U;++k)sum+=(double)act[qbh_r4_pos(r,k,1U)]*(double)qbh_r4_h12[k][n];
            weight[qbh_r4_pos(r,n,1U)]=(__fp16)(sum*(double)(__fp16)0.2886751345948129f);
        }
        t=HAP_perf_get_qtimer_count();
        if(h->dense_r4_optimization) {
            if(qbh_r4_finish_vector(h,b,pool,weight,middle,first,count,
                (HVX_Vector *)b->w4u8_gather_scratch,audit))return -1;
        }
        else for(uint32_t tile=0;tile<192U;++tile)for(uint32_t r=0;r<count;r+=4U) {
            uint32_t a=tile/16U,c0=tile%16U*32U,live=count-r<4U?count-r:4U;
            for(uint32_t rr=0;rr<4U;++rr)for(uint32_t c=0;c<32U;++c) {
                __fp16 z=rr<live?weight[qbh_r4_pos((r+rr)*512U+c0+c,a,1U)]:(__fp16)0;
                tmp[rr*32U+c]=z;
                if(audit && rr<live)audit[2U*393216U+(first+r+rr)*6144U+tile*32U+c]=z;
            }
            qbh_hvx_quantize_f16_to_u8(tmp,codes,128U,&h->qparams[QBH_BLOCK_QP_MIDDLE]);
            *(HVX_Vector *)(middle+tile*2048U+(first+r)*32U)=*(HVX_Vector *)codes;
        }
        h->dense_r4_finish_ticks+=HAP_perf_get_qtimer_count()-t;
    }
    ++h->dense_r4_calls;h->dense_r4_rows+=rows;
    if(audit)h->dense_r4_audit_bytes+=3U*rows*6144U*2U;
    return 0;
}
