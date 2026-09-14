#include "llama_r4_exact.h"
/* L32-0015 Llama full8192 H16 tensor H512; reused dense Qwen scheduling. */
static int qbh_r4_h16(uint32_t k,uint32_t n){return (__builtin_popcount(k&n)&1)?-1:1;}
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
    if(h->dense_r4_audit_offset) FARF(ALWAYS,"L32_R4 a=%p w=%p out=%p sc=%p mt=%u kt=%u nt=%u",(void*)a,(void*)w,(void*)out,sc,mt,kt,nt);
    if(h->dense_r4_audit_offset) {
        uint32_t *diag=(uint32_t *)((uint8_t *)h+h->dense_r4_audit_offset+3U*1048576U-128U);
        diag[0]=(uint32_t)(uintptr_t)a;diag[1]=(uint32_t)(uintptr_t)w;diag[2]=(uint32_t)(uintptr_t)out;
        diag[3]=(uint32_t)(uintptr_t)sc;diag[4]=mt;diag[5]=kt;diag[6]=nt;
        qurt_mem_cache_clean((qurt_addr_t)diag,128U,QURT_MEM_CACHE_FLUSH,QURT_MEM_DCACHE);
    }
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
    for(uint32_t a=0;a<16U;++a)for(uint32_t r=0;r<count;r+=2U)
        for(uint32_t c=0;c<512U;c+=64U) {
            uint32_t sr=a*rows+first+r;
            const __fp16 *src=stage1+qbh_r4_pos(sr&~1U,c,16U);
            HVX_VectorPair v=Q6_W_vdeal_VVR(
                *(const HVX_Vector *)(src+1024U),*(const HVX_Vector *)src,-2);
            /* Keep the declared 4KiB scatter window page aligned. Put the
             * group displacement in lane offsets, not the base address. */
            __fp16 *dst=act+r*16384U+(c/32U)*1024U;
            HVX_Vector group_offsets=Q6_Vw_vadd_VwVw(offsets,Q6_V_vsplat_R(a*4U));
            Q6_vscatter_RMVwV((uint32_t)(uintptr_t)dst,4095U,group_offsets,
                (sr&1U)?Q6_V_hi_W(v):Q6_V_lo_W(v));
            if(r+1U<count)Q6_vscatter_RMVwV((uint32_t)(uintptr_t)(dst+16384U),
                4095U,group_offsets,Q6_V_hi_W(v));
        }
    asm volatile("barrier" ::: "memory");
}
/* L32-0021: compute stage2 as H16 * stage1_token instead of
 * transpose(stage1_token) * H16. Same reduction order and FP16 scale;
 * token/channel columns are now native contiguous output tiles.
 * This is a storage/operand orientation change, not a fast Hadamard. */
static void qbh_r4_layout_weight(const __fp16 *stage1,__fp16 *dst,
    uint32_t rows,uint32_t first,uint32_t count) {
    qbh_r4_zero(dst,count*32768U);
    /* Pipeline batches start on even tokens and contain8 rows. Reuse
     * both token lanes in every native load. */
    for(uint32_t r=0;r<count;r+=2U)for(uint32_t g=0;g<16U;g+=2U)
        for(uint32_t c=0;c<512U;c+=64U) {
            uint32_t sr=g*rows+first+r,sr1=(g+1U)*rows+first+r;
            const __fp16 *a=stage1+qbh_r4_pos(sr&~1U,c,16U);
            const __fp16 *b=stage1+qbh_r4_pos(sr1&~1U,c,16U);
            HVX_VectorPair aa=Q6_W_vdeal_VVR(*(const HVX_Vector *)(a+1024U),*(const HVX_Vector *)a,-2);
            HVX_VectorPair bb=Q6_W_vdeal_VVR(*(const HVX_Vector *)(b+1024U),*(const HVX_Vector *)b,-2);
            HVX_VectorPair packed=Q6_W_vshuff_VVR(Q6_V_lo_W(bb),Q6_V_lo_W(aa),-2);
            HVX_VectorPair packed_next=Q6_W_vshuff_VVR(Q6_V_hi_W(bb),Q6_V_hi_W(aa),-2);
            __fp16 *out=dst+(r*16U+c/32U)*1024U+g*32U;
            *(HVX_Vector *)out=Q6_V_lo_W(packed);
            *(HVX_Vector *)(out+1024U)=Q6_V_hi_W(packed);
            *(HVX_Vector *)(out+16384U)=Q6_V_lo_W(packed_next);
            *(HVX_Vector *)(out+16384U+1024U)=Q6_V_hi_W(packed_next);
        }
    asm volatile("barrier" ::: "memory");
}
/* Post-R4 table is indexed by the exact FP16 bit pattern. Native planes
 * are produced directly; no intervening uniform U8 quantizer. */
static inline HVX_Vector qbh_r4_sp2_half(HVX_Vector bits,const uint16_t *lut,HVX_Vector *scratch) {
 HVX_Vector off=Q6_Vh_vasl_VhR(Q6_V_vand_VV(bits,Q6_Vh_vsplat_R(32767)),1);
 HVX_VectorPred upper=Q6_Q_vcmp_gt_VuhVuh(bits,Q6_Vh_vsplat_R(32767));
 Q6_vgather_AQRMVh(scratch,Q6_Q_not_Q(upper),(int32_t)(uintptr_t)lut,65535U,off);
 Q6_vgather_AQRMVh(scratch,upper,(int32_t)(uintptr_t)(lut+32768U),65535U,off);
 return *(volatile HVX_Vector*)scratch;
}
static void qbh_r4_finish_range(struct qbh_block_header *h,
 const struct qbh_block_buffers *b,const __fp16 *weight,uint8_t *middle,
 uint32_t first,uint32_t count,HVX_Vector *scratch,__fp16 *audit,uint32_t a_first,uint32_t a_count) {
 const HVX_Vector offsets=*(const HVX_Vector*)qbh_attention_vscatter_offsets;
 const uint16_t *lut=(const uint16_t*)(b->w4u8_silu_lut+QBH_MLP_LUT_BYTES);
 uint8_t *high=h->logical_m==64U?b->sp2_high:middle+128U;
 const uint32_t batch_rows=count>=8U?8U:4U;
 for(uint32_t a=a_first;a<a_first+a_count;a++)for(uint32_t c=0;c<512U;c+=64U)
  for(uint32_t r=0;r<count;r+=batch_rows) {
   /* Up to eight independent gather destinations hide producer/lookup latency.
    * Down output is phase-dead here; each context owns2048B within it. */
   for(uint32_t rr=0;rr<batch_rows;rr++) {
    if(r+rr<count) {
     const __fp16 *src=weight+(r+rr)*16384U+(c/32U)*1024U;
     if(h->logical_m==64U && h->dense_r4_mode!=2U) {
      src+=(a/2U)*64U;
      HVX_VectorPair pair=Q6_W_vdeal_VVR(*(const HVX_Vector *)(src+1024U),*(const HVX_Vector *)src,-2);
      scratch[rr]=(a&1U)?Q6_V_hi_W(pair):Q6_V_lo_W(pair);
     } else {
      HVX_Vector group_offsets=Q6_Vw_vadd_VwVw(offsets,Q6_V_vsplat_R(a*4U));
      Q6_vgather_ARMVw(scratch+rr,(int32_t)(uintptr_t)src,4095U,group_offsets);
     }
    } else scratch[rr]=Q6_V_vzero();
   }
   for(uint32_t rr=0;rr<batch_rows;rr++) {
    HVX_Vector bits=*(volatile HVX_Vector *)(scratch+rr);
    if(audit && r+rr<count)*(HVX_Vector*)(audit+2U*524288U+(first+r+rr)*8192U+a*512U+c)=bits;
    HVX_Vector off=Q6_Vh_vasl_VhR(Q6_V_vand_VV(bits,Q6_Vh_vsplat_R(32767)),1);
    HVX_VectorPred upper=Q6_Q_vcmp_gt_VuhVuh(bits,Q6_Vh_vsplat_R(32767));
    Q6_vgather_AQRMVh(scratch+8U+rr,Q6_Q_not_Q(upper),(int32_t)(uintptr_t)lut,65535U,off);
    Q6_vgather_AQRMVh(scratch+8U+rr,upper,(int32_t)(uintptr_t)(lut+32768U),65535U,off);
   }
   for(uint32_t rr=0;rr<batch_rows;rr+=4U) {
   /* Consume VTCM results directly. A variable-index vector array here
    * lowers to a 1KiB stack copy; never stage activations on the stack. */
   HVX_Vector e0=*(volatile HVX_Vector *)(scratch+8U+rr);
   HVX_Vector e1=*(volatile HVX_Vector *)(scratch+9U+rr);
   HVX_Vector e2=*(volatile HVX_Vector *)(scratch+10U+rr);
   HVX_Vector e3=*(volatile HVX_Vector *)(scratch+11U+rr);
   HVX_Vector lo=Q6_Vb_vpacke_VhVh(e1,e0),hi=Q6_Vb_vpacke_VhVh(e3,e2);
   HVX_VectorPair x=Q6_W_vdeal_VVR(hi,lo,-32);
   size_t pos=(a*16U+c/32U)*2048U+(first+r+rr)*32U;
   *(HVX_Vector*)(middle+pos)=Q6_V_lo_W(x);*(HVX_Vector*)(middle+pos+2048U)=Q6_V_hi_W(x);
   lo=Q6_Vb_vpacko_VhVh(e1,e0);hi=Q6_Vb_vpacko_VhVh(e3,e2);x=Q6_W_vdeal_VVR(hi,lo,-32);
   *(HVX_Vector*)(high+pos)=Q6_V_lo_W(x);*(HVX_Vector*)(high+pos+2048U)=Q6_V_hi_W(x);
   }
  }
 asm volatile("barrier":::"memory");
}
/* Three contexts share only immutable inputs; each owns a 2048B phase-dead Down gather
 * destination and disjoint whole channel tiles/groups. No atomic hot loop. */
struct qbh_r4_convert_task {
    struct qbh_block_header *h;
    struct qbh_block_buffers *b;
    const __fp16 *weight;
    uint8_t *middle;
    __fp16 *audit;
    uint32_t finish,first,count,stream,finish_workers,exact_stage1;
    volatile uint32_t ready[8],abort;
    uint32_t consumed[2];
    uint64_t work_ticks[3];
};
/* The H512 matrix is dead after the first HMX join. Three contexts use
 * disjoint 4KiB integer rows there; source activation and output alt stay VTCM.
 * Exact dense sums deliberately establish a correctness control before cost
 * reduction. 0x29a8 is exactly181/4096, hence the final lattice is2^-36. */
static void qbh_r4_exact_stage1(struct qbh_r4_convert_task *task,uint32_t id) {
    const uint16_t *src=(const uint16_t *)task->b->hmx_activation;
    uint16_t *out=(uint16_t *)task->b->expanded_weight_alt;
    int64_t *row=(int64_t *)(task->b->expanded_weight+id*4096U);
    uint32_t nr=16U*task->h->logical_m;
    for(uint32_t r=id*nr/3U;r<(id+1U)*nr/3U;++r) {
        for(uint32_t k=0;k<512U;++k) {
            uint16_t v=src[qbh_r4_pos(r,k,16U)];
            if((v&31744U)==31744U){task->abort=1U;return;}
            row[k]=qbh_r4_half_units(v);
        }
        for(uint32_t n=0;n<512U;++n) {
            int64_t sum=0;
            for(uint32_t k=0;k<512U;++k)
                sum+=(__builtin_popcount(n&k)&1)?-row[k]:row[k];
            uint16_t result=qbh_r4_round_units(sum*181,-36);
            if((result&31744U)==31744U){task->abort=1U;return;}
            out[qbh_r4_pos(r,n,16U)]=result;
        }
    }
}
/* Correct the second dense factor using the retained, now exact first factor.
 * Input/output orientation follows the existing prefill transposed pipeline. */
static void qbh_r4_exact_stage2(const __fp16 *stage1,__fp16 *out,
    uint32_t rows,uint32_t first,uint32_t count,int transposed) {
    for(uint32_t r=0;r<count;++r)for(uint32_t c=0;c<512U;++c)
        for(uint32_t a=0;a<16U;++a) {
            int64_t sum=0;
            for(uint32_t g=0;g<16U;++g) {
                uint16_t h=((const uint16_t *)stage1)[qbh_r4_pos(g*rows+first+r,c,16U)];
                int64_t v=qbh_r4_half_units(h);
                sum+=(__builtin_popcount(a&g)&1)?-v:v;
            }
            size_t pos=transposed?qbh_r4_pos(a,r*512U+c,count*16U):qbh_r4_pos(r*512U+c,a,1U);
            ((uint16_t *)out)[pos]=qbh_r4_round_units(sum,-26);
        }
}
static void qbh_r4_convert_worker(void *context,uint32_t id) {
    struct qbh_r4_convert_task *task=context;
    uint64_t t=HAP_perf_get_qtimer_count();
    if(task->exact_stage1) {qbh_r4_exact_stage1(task,id);task->work_ticks[id]=HAP_perf_get_qtimer_count()-t;return;}
    uint8_t *scratch=task->b->w4u8_gather_scratch+id*256U;
    if(task->stream) {
        task->work_ticks[id]=0;
        for(uint32_t g=0;g<8U;++g) {
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
    const uint32_t parts=task->finish_workers?task->finish_workers:3U;
    if(task->finish)qbh_r4_finish_range(task->h,task->b,task->weight,task->middle,
        task->first,task->count,(HVX_Vector *)(task->b->down+id*2048U),task->audit,id*16U/parts,(id+1U)*16U/parts-id*16U/parts);
    else for(uint32_t tile=id*256U/3U;tile<(id+1U)*256U/3U;++tile)
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
    if(task->finish)task->h->dense_r4_parallel_finish_groups+=16U;
    else if(!task->exact_stage1)task->h->dense_r4_parallel_prepare_tiles+=256U;
    return 0;
}
/* L32-0021: two finish workers consume the previous output while the
 * caller lays out the next input. Both use the inactive slot in distinct
 * arenas; current HMX input/output, Stage1 and LUT remain immutable.
 * Join all converters before advancing either slot or publishing Down. */
static int qbh_r4_finish_layout(struct qbh_block_w4f16_pool *pool,
    struct qbh_r4_convert_task *task,const __fp16 *stage1,__fp16 *next_act,
    uint32_t next_first) {
    if(!pool || pool->worker_count<2U)return -1;
    for(uint32_t i=0;i<pool->worker_count;++i)
        if(pool->jobs[i].command_kind!=QBH_BLOCK_HVX_POOL_NONE)return -1;
    uint64_t start=HAP_perf_get_qtimer_count();
    task->finish_workers=2U;
    pool->active_worker_count=2U;pool->extra_expand_worker_index=UINT32_MAX;
    for(uint32_t i=0;i<2U;++i) {
        pool->jobs[i].r4_convert_context=task;
        pool->jobs[i].command_kind=QBH_BLOCK_HVX_POOL_R4_CONVERT;
    }
    asm volatile("barrier" ::: "memory");
    for(uint32_t i=0;i<2U;++i)(void)qurt_sem_up(&pool->command_ready[i]);
    uint64_t t=HAP_perf_get_qtimer_count();
    qbh_r4_layout_weight(stage1,next_act,64U,next_first,8U);
    uint64_t layout=HAP_perf_get_qtimer_count()-t;
    t=HAP_perf_get_qtimer_count();
    qbh_w4f16_pool_wait(pool);pool->active_worker_count=0U;
    task->h->dense_r4_parallel_join_ticks+=HAP_perf_get_qtimer_count()-t;
    uint64_t elapsed=HAP_perf_get_qtimer_count()-start;
    /* Nonoverlapping caller attribution; worker work is a separate
     * diagnostic sum, never added to the complete Host/DSP ledger. */
    task->h->dense_r4_layout_ticks+=layout;
    task->h->dense_r4_finish_ticks+=elapsed-layout;
    task->h->dense_r4_pipeline_hvx_ticks+=elapsed;
    for(uint32_t i=0;i<2U;++i)task->h->dense_r4_parallel_work_ticks+=task->work_ticks[i];
    ++task->h->dense_r4_parallel_dispatches;
    task->h->dense_r4_parallel_finish_groups+=16U;
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
    if(abort || task->consumed[0]!=8U || task->consumed[1]!=8U)return -1;
    task->h->dense_r4_parallel_prepare_tiles+=256U;
    task->b->r4_prefill_prepared=1U;
    return 0;
}
static int qbh_r4_finish_vector(struct qbh_block_header *h,
    struct qbh_block_buffers *b,struct qbh_block_w4f16_pool *pool,
    const __fp16 *weight,uint8_t *middle,uint32_t first,uint32_t count,
    HVX_Vector *scratch,__fp16 *audit) {
    (void)scratch; /* finish owns phase-dead Down scratch */
    if(h->dense_r4_optimization>=3U && h->logical_m==64U) {
        struct qbh_r4_convert_task task={.h=h,.b=b,.weight=weight,.middle=middle,
            .audit=audit,.finish=1U,.first=first,.count=count};
        return qbh_r4_convert_parallel(pool,&task);
    }
    qbh_r4_finish_range(h,b,weight,middle,first,count,(HVX_Vector *)b->down,audit,0U,16U);
    return 0;
}
static int qbh_run_dense_r4(struct qbh_block_header *h,uint8_t *shared,
    struct qbh_block_buffers *b,struct qbh_block_hmx_worker *worker,
    struct qbh_block_w4f16_pool *pool,uint8_t *middle) {
    const uint32_t rows=h->logical_m, p1=(16U*rows+31U)&~31U;
    __fp16 *act=(__fp16 *)b->hmx_activation;
    __fp16 *weight=(__fp16 *)b->expanded_weight;
    __fp16 *stage1=(__fp16 *)b->expanded_weight_alt;
    __fp16 *w12=(__fp16 *)b->hmx_output;
    float scales[32] __attribute__((aligned(128)));
    __fp16 *audit=h->dense_r4_audit_offset?(__fp16 *)(shared+h->dense_r4_audit_offset):NULL;
    if(h->variant!=QBH_BLOCK_W4U8 || QBH_LLAMA_SP2(h)!=8U || h->dense_r4_optimization!=6U || h->dense_r4_mode>3U ||
       h->dense_r4_optimization>6U ||
       (rows!=1U && rows!=64U) ||
       (h->w4u8_decode_direct_n_gate_up_swiglu_stream && h->dense_r4_optimization<2U) ||
       (h->dense_r4_mode==2U && !audit) ||
       (audit && !qbh_range_valid(h->dense_r4_audit_offset,3U*1048576U,h->shared_bytes)))return -1;

    /* These bounds are the existing owned allocations, no arena growth. */
    if((uintptr_t)b->expanded_weight_alt-(uintptr_t)b->expanded_weight<1048576U ||
       (uintptr_t)b->hmx_output-(uintptr_t)b->expanded_weight_alt<1048576U ||
       (uintptr_t)b->compressed_weight-(uintptr_t)b->hmx_activation<1048576U) return -1;
    uint64_t t=HAP_perf_get_qtimer_count();
    const uint32_t streamed=rows==1U && h->dense_r4_optimization>=2U &&
        h->w4u8_decode_direct_n_gate_up_swiglu_stream;
    const uint32_t prefill_streamed=rows==64U && b->r4_prefill_prepared;
    if(!streamed && !prefill_streamed) {
        if(!h->dense_r4_optimization || rows==1U)qbh_r4_zero(act,p1*512U*2U);
        if(h->dense_r4_optimization>=3U && rows==64U) {
            struct qbh_r4_convert_task task={.h=h,.b=b};
            if(qbh_r4_convert_parallel(pool,&task))return -1;
        } else for(uint32_t tile=0;tile<256U;++tile)qbh_r4_prepare_tile(b->gate,b->up,act,
            rows,tile,(const uint16_t *)b->w4u8_silu_lut,b->w4u8_gather_scratch);
    }
    /* Materialize H512 from exact dense H128 blocks with parity signs.
     * This generates matrix entries; activations never traverse a butterfly. */
    for(uint32_t n=0;n<16U;++n)for(uint32_t k=0;k<16U;++k) {
        const HVX_Vector *src=(const HVX_Vector *)(qbh_r3_sign_matrix+((n%4U)*4U+k%4U)*1024U);
        HVX_Vector *dst=(HVX_Vector *)(weight+(n*16U+k)*1024U);
        /* 0x3c00 (+1) xor 0x15a8 = 0x29a8 (FP16 1/sqrt512).
         * Preserve sign and generate normalized weights at original XOR cost. */
        HVX_Vector sign=Q6_Vh_vsplat_R(0x15a8 ^ ((__builtin_popcount((n/4U)&(k/4U))&1)?0x8000:0));
        for(uint32_t v=0;v<16U;++v)dst[v]=Q6_V_vxor_VV(src[v],sign);
    }
    /* Same exact dyadic product, normalize the dense matrix before accumulation. */
    for(uint32_t c=0;c<32U;++c)scales[c]=1.0f;
    qbh_hmx_fp16_init_channel_scales(b->channel_scale,scales);
    h->dense_r4_prepare_ticks+=HAP_perf_get_qtimer_count()-t;
    if(audit)for(uint32_t a=0;a<16U;++a)for(uint32_t r=0;r<rows;++r)for(uint32_t c=0;c<512U;++c)
        audit[(a*rows+r)*512U+c]=act[qbh_r4_pos(a*rows+r,c,16U)];
    if(qbh_r4_gemm(h,worker,act,weight,b->channel_scale,stage1,p1/32U,16U,16U))return -1;
    if(h->dense_r4_mode==3U) {
        uint64_t refine=HAP_perf_get_qtimer_count();
        struct qbh_r4_convert_task task={.h=h,.b=b,.exact_stage1=1U};
        if(qbh_r4_convert_parallel(pool,&task) || task.abort)return -1;
        h->dense_r4_finish_ticks+=HAP_perf_get_qtimer_count()-refine;
    }
    if(h->dense_r4_mode==2U) {
        /* Untimed oracle replaces the result only after the HMX state transition. */
        for(uint32_t r=0;r<16U*rows;++r)for(uint32_t n=0;n<512U;++n) {
            double sum=0;
            for(uint32_t k=0;k<512U;++k)sum+=(double)audit[r*512U+k]*((__builtin_popcount(n&k)&1)?-1.0:1.0);
            stage1[qbh_r4_pos(r,n,16U)]=(__fp16)(sum*(double)(__fp16)0.04419417382415922f);
        }
    }
    if(audit)for(uint32_t r=0;r<16U*rows;++r)for(uint32_t c=0;c<512U;++c)
        audit[524288U+r*512U+c]=stage1[qbh_r4_pos(r,c,16U)];
    t=HAP_perf_get_qtimer_count();
    /* L32-0025: H16 entries carry the exact dyadic normalization.
     * +/-1 xor0x0800 is +/-0.25; mask AFTER xor so padded zeros stay zero.
     * Prefill consumes H16 as the left operand, decode as the right operand.
     * Same dot product, layouts and calls; unity output conversion. */
    for(uint32_t v=0;v<16U;++v)
        ((HVX_Vector *)w12)[v]=v<8U?Q6_V_vmux_QVV(Q6_Q_vsetq_R(64),
            Q6_V_vxor_VV(((const HVX_Vector *)qbh_r3_sign_matrix)[v],
                Q6_Vh_vsplat_R(0x0800)),Q6_V_vzero()):Q6_V_vzero();
    for(uint32_t c=0;c<32U;++c)scales[c]=1.0f;
    qbh_hmx_fp16_init_channel_scales(b->channel_scale,scales);
    h->dense_r4_prepare_ticks+=HAP_perf_get_qtimer_count()-t;
    if(h->dense_r4_optimization>=2U && rows==64U && h->dense_r4_mode!=2U) {
        /* Two 256KiB input/output slots live inside the original arenas.
         * The previous output and next input use the opposite slot from the
         * active HMX command. Stage1, H12 and scales remain immutable. */
        t=HAP_perf_get_qtimer_count();
        qbh_r4_layout_weight(stage1,act,rows,0U,8U);
        h->dense_r4_layout_ticks+=HAP_perf_get_qtimer_count()-t;
        for(uint32_t batch=0;batch<8U;++batch) {
            uint32_t slot=batch&1U,other=slot^1U;
            __fp16 *ai=act+slot*131072U,*oi=weight+slot*131072U;
            qbh_hmx_start(worker,QBH_BLOCK_HMX_FP16,w12,ai,b->channel_scale,oi,1U,1U,128U);
            ++h->hmx_command_count;++h->dense_r4_hmx_calls;
            h->hmx_fp16_tile_pair_count+=128U;
            ++h->dense_r4_pipeline_batches;
            if(batch && batch+1U<8U) {
                struct qbh_r4_convert_task task={.h=h,.b=b,
                    .weight=weight+other*131072U,.middle=middle,.audit=audit,
                    .finish=1U,.first=(batch-1U)*8U,.count=8U};
                if(qbh_r4_finish_layout(pool,&task,stage1,act+other*131072U,(batch+1U)*8U)) {
                    (void)qbh_hmx_wait(worker);return -1;
                }
            } else {
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
                qbh_r4_layout_weight(stage1,act+other*131072U,rows,(batch+1U)*8U,8U);
                uint64_t dt=HAP_perf_get_qtimer_count()-t;
                h->dense_r4_layout_ticks+=dt;h->dense_r4_pipeline_hvx_ticks+=dt;
            }
            }
            t=HAP_perf_get_qtimer_count();
            int rc=qbh_hmx_wait(worker);
            h->dense_r4_matmul_ticks+=HAP_perf_get_qtimer_count()-t;
            if(rc)return -1;
            if(h->dense_r4_mode==3U) {
                uint64_t refine=HAP_perf_get_qtimer_count();
                qbh_r4_exact_stage2(stage1,oi,rows,batch*8U,8U,1);
                h->dense_r4_finish_ticks+=HAP_perf_get_qtimer_count()-refine;
            }
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
        for(uint32_t r=0;r<count;++r)for(uint32_t a=0;a<16U;++a)for(uint32_t c=0;c<512U;++c)
            act[qbh_r4_pos(r*512U+c,a,1U)]=stage1[qbh_r4_pos(a*rows+first+r,c,16U)];
        }
        h->dense_r4_layout_ticks+=HAP_perf_get_qtimer_count()-t;
        if(qbh_r4_gemm(h,worker,act,w12,b->channel_scale,weight,count*16U,1U,1U))return -1;
        if(h->dense_r4_mode==3U) {
            uint64_t refine=HAP_perf_get_qtimer_count();
            qbh_r4_exact_stage2(stage1,weight,rows,first,count,0);
            h->dense_r4_finish_ticks+=HAP_perf_get_qtimer_count()-refine;
        }
        if(h->dense_r4_mode==2U)for(uint32_t r=0;r<count*512U;++r)for(uint32_t n=0;n<16U;++n) {
            double sum=0;
            for(uint32_t k=0;k<16U;++k)sum+=(double)act[qbh_r4_pos(r,k,1U)]*(double)qbh_r4_h16(k,n);
            weight[qbh_r4_pos(r,n,1U)]=(__fp16)(sum*(double)(__fp16)0.25f);
        }
        t=HAP_perf_get_qtimer_count();
        if(h->dense_r4_optimization) {
            if(qbh_r4_finish_vector(h,b,pool,weight,middle,first,count,
                (HVX_Vector *)b->w4u8_gather_scratch,audit))return -1;
        }
        h->dense_r4_finish_ticks+=HAP_perf_get_qtimer_count()-t;
    }
    ++h->dense_r4_calls;h->dense_r4_rows+=rows;
    if(audit)h->dense_r4_audit_bytes+=3U*rows*8192U*2U;
    return 0;
}
