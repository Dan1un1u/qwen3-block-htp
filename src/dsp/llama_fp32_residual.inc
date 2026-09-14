/* L32-0016 accuracy prototype. Native packed W4 products, never U8 output.
 * Raw stores are signed24, proven at host preparation for EVERY possible U8
 * input. SP2's reconstructed signed32 bound is proved separately. No DDR spill.
 * Scalar FP32 epilogue deliberately prioritizes an auditable numerical contract;
 * its add time belongs to the O/Down projection ledger (no second residual add).
 */
static int32_t qbh_llama_raw24(const uint8_t *raw,uint32_t i) {
    return (int32_t)raw[i]+256*(int32_t)raw[2048U+i]+
        65536*(int32_t)(int8_t)raw[4096U+i];
}
static int qbh_llama_w4_fp32_residual(struct qbh_block_hmx_worker *w) {
    uint8_t *lo=w->sp2_scratch,*hi=lo+8192U;
    uint32_t *bias=(uint32_t *)(lo+16384U);
    if(!lo || !w->fp32_residual || w->m_tiles!=1U ||
       (w->fp32_rows!=1U && w->fp32_rows!=64U)) return AEE_EBADPARM;
    for(uint32_t nt=0;nt<w->n_tiles;nt++) {
        const uint8_t *weight=(const uint8_t *)w->weight+(size_t)nt*w->k_tiles*512U;
        const float *scale=(const float *)w->scale_or_bias+nt*64U;
        const int32_t *sum=(const int32_t *)(scale+32U);
        uint32_t passes=w->fp32_sp2 && w->fp32_rows==64U ? 2U:1U;
        for(uint32_t pass=0;pass<passes;pass++) {
            for(uint32_t n=0;n<32;n++){bias[n]=24U<<10;bias[n+32U]=0U;}
            qbh_hmx_begin_u8s8_output(bias);
            qbh_hmx_accumulate_u8n4_projection(pass?w->sp2_high:w->activation,weight,w->k_tiles);
            read_i32(pass?hi:lo,bias,3U);
        }
        for(uint32_t row=0;row<w->fp32_rows;row++) for(uint32_t n=0;n<32U;n++) {
            uint32_t i=row*32U+n;
            int32_t acc=qbh_llama_raw24(lo,i);
            if(w->fp32_sp2) {
                int32_t high=qbh_llama_raw24(w->fp32_rows==64U?hi:lo,
                    i+(w->fp32_rows==64U?0U:128U));
                /* int64 avoids intermediate overflow even when cancellation fits. */
                acc=(int32_t)((int64_t)acc+256LL*high-32768LL*sum[n]);
            } else acc-=w->fp32_input_zero*sum[n];
            float addition=(float)acc*scale[n];
            float *dst=w->fp32_residual+(size_t)row*QBH_BLOCK_HIDDEN+w->fp32_first_channel+nt*32U+n;
            *dst=*dst+addition;
        }
    }
    asm volatile("barrier":::"memory");
    return AEE_SUCCESS;
}
/* FP32 sum-of-squares/add/norm, original FP16 gamma, fixed post-Norm A8.
 * Physical activation is [K/32,64,32]; pad rows are explicitly zero-centered.
 * roundf is the scalar contract (nearest, ties away from zero). */
static void qbh_llama_fp32_norm(const float *x,const __fp16 *gamma,
    uint8_t *out,const struct qbh_block_qparam *q,uint32_t rows,
    uint32_t width,uint32_t native) {
    memset(out,q->zero_point,(size_t)QBH_BLOCK_M*width);
    for(uint32_t row=0;row<rows;row++) {
        float sum=0.0f;
        for(uint32_t c=0;c<width;c++) {float v=x[(size_t)row*width+c];sum+=v*v;}
        float inv=1.0f/sqrtf(sum/(float)width+QBH_MODEL_RMS_EPS);
        for(uint32_t c=0;c<width;c++) {
            float v=x[(size_t)row*width+c]*inv;
            v=v*(float)gamma[c];
            float code=roundf(v/q->scale+(float)q->zero_point);
            code=fminf(255.0f,fmaxf(0.0f,code));
            size_t i=native?(size_t)(c/32U)*2048U+row*32U+c%32U:(size_t)row*width+c;
            out[i]=(uint8_t)code;
        }
    }
    asm volatile("barrier":::"memory");
}
