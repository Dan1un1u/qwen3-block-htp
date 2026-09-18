
/* Four independent GQA groups share one serialized HMX/DMA owner. Each
 * HVX client owns one disjoint slot in the existing attention overlay. */
static void qbh_long_attention_tasks(struct qbh_block_w4f16_pool *pool,
                                    struct qbh_block_w4f16_job *job) {
    struct qbh_block_header *h=pool->attention_header;
    unsigned rows=pool->long_rows,past=pool->long_past,padded=pool->long_padded;
    /* Each resource has one serialized owner; cache DMA may overlap HMX on
     * another client's disjoint operand slot. All waits precede slot reuse. */
    qurt_mutex_t *dma_mutex=(h->long_optimization&32768U)
        ? &pool->long_dma_mutex : &pool->attention_hmx_mutex;
    unsigned valid=past+rows,tiles=padded/32;
    unsigned width=QBH_ATTENTION_Q_HEADS_PER_GROUP*64U;
    if(width<QBH_BLOCK_HEAD_DIM)width=QBH_BLOCK_HEAD_DIM;
    unsigned plane=padded*width;
    uint8_t *a=pool->attention_buffers->scores+job->worker_index*pool->long_stride;
    int8_t *w=(int8_t *)(a+plane);
    uint8_t *c=a+2*plane;
    uint32_t *qb=(uint32_t *)(a+3*plane);
    uint32_t *vb=qb+tiles*64;
    uint8_t *lut=pool->long_common_lut ? pool->attention_buffers->scores+4U*pool->long_stride : (uint8_t *)(vb+QBH_ATTENTION_HEAD_DIM_TILES*64);
    job->u8_attention_k_pack_ticks=job->u8_attention_v_pack_ticks=0;
    job->u8_attention_softmax_ticks=job->u8_attention_qk_hmx_ticks=0;
    job->u8_attention_av_hmx_ticks=job->u8_attention_hmx_queue_wait_ticks=0;
    job->u8_attention_v_recenter_saturation_count=job->u8_attention_group_count=0;
    for(;;) {
        unsigned g=qbh_atomic_fetch_increment(&pool->next_attention_task);
        if(g>=QBH_BLOCK_KV_HEADS || pool->attention_gqa_abort)break;
        const struct qbh_attention_config *cfg=&pool->attention_buffers->attention_configs[g];
        uint8_t *q=pool->attention_buffers->q+(size_t)g*QBH_ATTENTION_Q_HEADS_PER_GROUP*QBH_ATTENTION_HEAD_DIM_TILES*2048;
        const uint8_t *ck=pool->long_shared+h->kv_cache_k_offset+(size_t)g*h->kv_cache_capacity*QBH_BLOCK_HEAD_DIM;
        const uint8_t *cv=pool->long_shared+h->kv_cache_v_offset+(size_t)g*h->kv_cache_capacity*QBH_BLOCK_HEAD_DIM;
        if(!(h->long_optimization&4096U)) memset(a,0,qbh_align_up(valid*QBH_BLOCK_HEAD_DIM,128U));
        qurt_mutex_lock(dma_mutex);
        int rc=qbh_scan_cache_dma(h,a,ck,valid*QBH_BLOCK_HEAD_DIM,1);
        qurt_mutex_unlock(dma_mutex);
        if(rc){pool->attention_gqa_abort=1;break;}
        uint64_t t=HAP_perf_get_qtimer_count();
        if(h->long_optimization&16384U) qbh_attention_u8_pack_k_long_register(a,valid,padded,cfg,w,qb);
        else if(h->long_optimization&256U) qbh_attention_u8_pack_k_long_transpose(a,valid,padded,cfg,w,qb);
        else qbh_attention_u8_pack_k_long(a,valid,padded,cfg,w,qb);
        job->u8_attention_k_pack_ticks+=HAP_perf_get_qtimer_count()-t;
        if(qbh_attention_u8_pool_submit(pool,job,q,w,qb,c,QBH_ATTENTION_Q_HEADS_PER_GROUP,
            QBH_ATTENTION_HEAD_DIM_TILES,tiles,&job->u8_attention_qk_hmx_ticks))break;
        t=HAP_perf_get_qtimer_count();
        if(h->long_optimization&512U) qbh_attention_u8_softmax_long_native4_fast(c,a,rows,past,padded,cfg);
        else qbh_attention_u8_softmax_long_native4(c,a,rows,past,padded,cfg);
        job->u8_attention_softmax_ticks+=HAP_perf_get_qtimer_count()-t;
        if(!(h->long_optimization&4096U)) memset(c,0,qbh_align_up(valid*QBH_BLOCK_HEAD_DIM,128U));
        qurt_mutex_lock(dma_mutex);
        rc=qbh_scan_cache_dma(h,c,cv,valid*QBH_BLOCK_HEAD_DIM,1);
        qurt_mutex_unlock(dma_mutex);
        if(rc){pool->attention_gqa_abort=1;break;}
        t=HAP_perf_get_qtimer_count();
        if(!pool->long_common_lut)qbh_attention_u8_prepare_v_row_major_hvx(cfg,lut);
        if(h->long_optimization&1024U) qbh_attention_u8_pack_v_long_register(c,valid,padded,cfg,w,vb,lut,&job->u8_attention_v_recenter_saturation_count);
        else qbh_attention_u8_pack_v_long(c,valid,padded,cfg,w,vb,lut,&job->u8_attention_v_recenter_saturation_count);
        job->u8_attention_v_pack_ticks+=HAP_perf_get_qtimer_count()-t;
        if(qbh_attention_u8_pool_submit(pool,job,a,w,vb,q,QBH_ATTENTION_Q_HEADS_PER_GROUP,
            tiles,QBH_ATTENTION_HEAD_DIM_TILES,&job->u8_attention_av_hmx_ticks))break;
        t=HAP_perf_get_qtimer_count();
        /* Decode consumes only the first four physical rows, as in the
         * historical short-cache path. Keep all live outputs bit-identical. */
        if((h->long_optimization&8192U) && rows==1U)
            qbh_attention_u8_requant_av_rows(q,cfg,4U);
        else qbh_attention_u8_requant_av(q,cfg);
        job->u8_attention_av_hmx_ticks+=HAP_perf_get_qtimer_count()-t;
        ++job->u8_attention_group_count;
    }
    asm volatile("barrier" ::: "memory");
}
static int qbh_long_attention_parallel(struct qbh_block_header *h,uint8_t *shared,
 struct qbh_block_buffers *b,struct qbh_block_hmx_worker *worker,
 struct qbh_block_w4f16_pool *pool,unsigned rows,unsigned past) {
    unsigned padded=qbh_align_up(past+rows,32),tiles=padded/32;
    unsigned width=QBH_ATTENTION_Q_HEADS_PER_GROUP*64U;
    if(width<QBH_BLOCK_HEAD_DIM)width=QBH_BLOCK_HEAD_DIM;
    unsigned stride=qbh_align_up(3*padded*width+tiles*256+QBH_ATTENTION_HEAD_DIM_TILES*256+20480,2048);
    unsigned capacity=(uint32_t)(b->compressed_weight-b->scores);
    if(!pool || pool->worker_count<3 || stride*4+20480>capacity)return -1;
    for(unsigned j=0;j<3;++j)
        if(pool->jobs[j].command_kind!=QBH_BLOCK_HVX_POOL_NONE)return -1;
    uint64_t start=HAP_perf_get_qtimer_count();
    h->scan_attention_overlay_capacity_bytes=capacity;
    h->scan_attention_overlay_required_bytes=stride*4+20480;
    h->scan_total_kv_length=past+rows;h->scan_padded_kv_length=padded;
    pool->attention_header=h;pool->attention_buffers=b;pool->attention_hmx_worker=worker;
    pool->long_shared=shared;pool->long_rows=rows;pool->long_past=past;
    pool->long_padded=padded;pool->long_stride=stride;
    pool->long_common_lut=1;
    for(unsigned g=1;g<QBH_BLOCK_KV_HEADS;++g) {
        const struct qbh_attention_config *a=&b->attention_configs[0],*c=&b->attention_configs[g];
        if(a->v_zero_point!=c->v_zero_point || a->v_recenter_numerator!=c->v_recenter_numerator ||
           a->v_recenter_denominator!=c->v_recenter_denominator)pool->long_common_lut=0;
    }
    if(pool->long_common_lut)qbh_attention_u8_prepare_v_row_major_hvx(&b->attention_configs[0],b->scores+4U*stride);
    pool->next_attention_task=0;pool->attention_gqa_abort=0;
    pool->active_worker_count=3;
    for(unsigned j=0;j<3;++j)pool->jobs[j].command_kind=QBH_BLOCK_HVX_POOL_LONG_ATTENTION;
    asm volatile("barrier" ::: "memory");
    for(unsigned j=0;j<3;++j)qurt_sem_up(&pool->command_ready[j]);
    struct qbh_block_w4f16_job main_job;
    memset(&main_job,0,sizeof(main_job));main_job.worker_index=3;
    qbh_long_attention_tasks(pool,&main_job);
    qbh_w4f16_pool_wait(pool);
    for(unsigned j=0;j<4;++j) {
        const struct qbh_block_w4f16_job *job=j==3?&main_job:&pool->jobs[j];
        h->u8_attention_k_pack_ticks+=job->u8_attention_k_pack_ticks;
        h->u8_attention_v_pack_ticks+=job->u8_attention_v_pack_ticks;
        h->u8_attention_softmax_ticks+=job->u8_attention_softmax_ticks;
        h->u8_attention_qk_hmx_ticks+=job->u8_attention_qk_hmx_ticks;
        h->u8_attention_av_hmx_ticks+=job->u8_attention_av_hmx_ticks;
        h->u8_attention_pipeline_wait_ticks+=job->u8_attention_hmx_queue_wait_ticks;
        h->u8_attention_v_recenter_saturation_count+=job->u8_attention_v_recenter_saturation_count;
        h->u8_attention_group_count+=job->u8_attention_group_count;
    }
    h->u8_cache_full_prefix_pack_count+=2*QBH_BLOCK_KV_HEADS;
    h->u8_attention_qk_execution_count+=QBH_BLOCK_KV_HEADS*QBH_ATTENTION_Q_HEADS_PER_GROUP*tiles;
    h->u8_attention_av_execution_count+=QBH_BLOCK_KV_HEADS*QBH_ATTENTION_Q_HEADS_PER_GROUP*QBH_ATTENTION_HEAD_DIM_TILES;
    h->hmx_command_count+=2*QBH_BLOCK_KV_HEADS;
    h->hmx_u8s8_tile_pair_count+=2*QBH_BLOCK_KV_HEADS*QBH_ATTENTION_Q_HEADS_PER_GROUP*QBH_ATTENTION_HEAD_DIM_TILES*tiles;
    h->u8_attention_direct_o_tile_count+=QBH_BLOCK_HEADS*QBH_ATTENTION_HEAD_DIM_TILES;
    h->scan_dynamic_attention_ticks+=HAP_perf_get_qtimer_count()-start;
    return pool->attention_gqa_abort?-1:0;
}
