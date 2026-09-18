
/* Head64 long-cache packers: two K rows per vector and register V LUT. */
void qbh_attention_u8_pack_k_long(const uint8_t *rows,uint32_t valid,uint32_t padded,
 const struct qbh_attention_config *cfg,int8_t *weight,uint32_t *bias) {
    const uint32_t div=1U<<cfg->score_shift,round=cfg->score_shift?div/2:0;
    const uint16_t cv=qbh_attention_u8_float_to_half_bits(512.0f/(float)div);
    uint32_t offsets[32] __attribute__((aligned(128)));
    for(unsigned i=0;i<32;++i) offsets[i]=((i%16)/8)*1024+(i%8)*128+(i/16)*4;
    HVX_Vector base=*(HVX_Vector *)offsets;
    memset(weight,0,(size_t)padded*QBH_ATTENTION_HEAD_DIM);
    for(unsigned t=0;t<padded/32;++t) {
        uint32_t *b=bias+t*64;
        *(HVX_Vector *)b=Q6_V_vsplat_R(cv);
        *((HVX_Vector *)b+1)=Q6_V_vsplat_R(128*div+round);
        for(unsigned r=0;r<32 && t*32+r<valid;r+=2) {
            unsigned count=valid-(t*32+r);
            HVX_Vector v=qbh_attention_u8_center_u8_to_s8(*(const HVX_Vector *)(rows+(size_t)(t*32+r)*64),cfg->k_zero_point);
            HVX_VectorPred live=count>=2?Q6_Q_vcmp_eq_VbVb(v,v):Q6_Q_vsetq_R(64);
            v=Q6_V_vmux_QVV(live,v,Q6_V_vzero());
            HVX_Vector sums=Q6_Vw_vrmpy_VbVb(v,Q6_Vb_vsplat_R(1));
            for(unsigned sh=32;sh>=4;sh>>=1)
                sums=Q6_Vw_vadd_VwVw(sums,Q6_V_vlalign_VVR(sums,Q6_V_vzero(),sh));
            b[32+r]=(uint32_t)(-(int32_t)cfg->q_zero_point*(int32_t)Q6_R_vextract_VR(sums,60)+128*(int32_t)div+(int32_t)round);
            if(count>=2)b[33+r]=(uint32_t)(-(int32_t)cfg->q_zero_point*(int32_t)Q6_R_vextract_VR(sums,124)+128*(int32_t)div+(int32_t)round);
            Q6_vscatter_QRMVwV(live,(uint32_t)(uintptr_t)(weight+t*2048),2047,
                Q6_Vw_vadd_VwVw(base,Q6_V_vsplat_R(r*4)),v);
        }
    }
    asm volatile("barrier" ::: "memory");
}
static void qbh_attention_u8_patch_v_long(
    const uint8_t *rows, uint32_t row_count,
    const struct qbh_attention_config *config,
    int8_t *k_tile_weights, uint32_t k_tile_stride_bytes,
    uint8_t *scratch, uint32_t *saturation_count) {
    int16_t *recenter_lut = (int16_t *)(
        scratch + QBH_ATTN_U8_VGATHER_LUT_OFFSET);
    uint8_t row_group[QBH_ATTN_U8_HVX_BYTES]
        __attribute__((aligned(QBH_ATTN_U8_HVX_BYTES)));

    if (rows == NULL || config == NULL ||
        k_tile_weights == NULL || scratch == NULL ||
        row_count > QBH_HMX_INPUT_CHANNELS ||
        k_tile_stride_bytes < QBH_HMX_WEIGHT_BYTES) {
        return;
    }
    if (saturation_count != NULL) {
        for (uint32_t index = 0U;
             index < row_count * QBH_ATTENTION_HEAD_DIM; ++index) {
            const int32_t centered =
                (int32_t)rows[index] - config->v_zero_point;
            const int32_t requantized =
                qbh_attention_u8_round_div_signed(
                    centered * (int32_t)config->v_recenter_numerator,
                    (int32_t)config->v_recenter_denominator);
            *saturation_count +=
                requantized < INT8_MIN || requantized > INT8_MAX;
        }
    }

    /* Exact256-entry byte LUT in registers: avoid two dependent VTCM
     * gathers per32-channel/four-token output vector. Each32-entry bank
     * occupies low bytes of halfwords, as required by vlut32. */
    HVX_Vector vluts[8];
    for(uint32_t bank=0;bank<8U;bank++) {
        HVX_Vector t=((const HVX_Vector *)recenter_lut)[bank/2U];
        if(bank&1U)t=Q6_V_vror_VR(t,64U);
        vluts[bank]=Q6_V_lo_W(Q6_W_vshuff_VVR(t,t,-64));
    }
    for (uint32_t n_tile = 0U;
         n_tile < QBH_ATTENTION_HEAD_DIM_TILES; ++n_tile) {
        int8_t *destination = k_tile_weights +
            (size_t)n_tile * k_tile_stride_bytes;
        for (uint32_t input_group = 0U;
             input_group * 4U < row_count; ++input_group) {
            HVX_Vector lanes[4];
            for(uint32_t lane=0;lane<4U;lane++) {
                uint32_t row=input_group*4U+lane;
                lanes[lane]=row<row_count
                    ?Q6_V_vror_VR(*(const HVX_Vector *)(rows+(row&~1U)*64U),(row&1U)*64U+n_tile*32U)
                    :Q6_Vb_vsplat_R(config->v_zero_point);
            }
            HVX_Vector ab=Q6_V_lo_W(Q6_W_vshuff_VVR(lanes[1],lanes[0],-32));
            HVX_Vector cd=Q6_V_lo_W(Q6_W_vshuff_VVR(lanes[3],lanes[2],-32));
            *(HVX_Vector *)row_group=Q6_V_lo_W(Q6_W_vshuff_VVR(cd,ab,-64));
            {
                const HVX_Vector values = *(const HVX_Vector *)row_group;
                HVX_Vector index=Q6_V_vand_VV(values,Q6_Vb_vsplat_R(31));
                HVX_Vector bank=Q6_Vub_vlsr_VubR(values,5);
                HVX_Vector recentered=Q6_Vb_vlut32_VbVbR_nomatch(index,vluts[0],0);
                for(uint32_t j=1;j<8U;j++)
                    recentered=Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VbVb(bank,Q6_Vb_vsplat_R(j)),
                        Q6_Vb_vlut32_VbVbR_nomatch(index,vluts[j],0),recentered);
                recentered = Q6_Vb_vdeal_Vb(recentered);
                recentered = Q6_Vb_vdeal_Vb(recentered);
                recentered = Q6_Vb_vdeal_Vb(recentered);
                recentered = Q6_Vb_vdeal_Vb(recentered);
                recentered = Q6_Vb_vdeal_Vb(recentered);
                *(HVX_Vector *)(destination +
                    (size_t)input_group * sizeof(HVX_Vector)) =
                    recentered;
            }
        }
    }
    asm volatile("barrier" ::: "memory");
}

void qbh_attention_u8_pack_v_long(const uint8_t *rows,
    uint32_t valid_tokens,uint32_t padded_tokens,
    const struct qbh_attention_config *config,int8_t *weight,uint32_t *bias,
    uint8_t *scratch,uint32_t *saturation_count) {
    const uint32_t kt=padded_tokens/32U,div=1U<<config->av_shift;
    const uint32_t rounding=config->av_shift?div/2U:0U;
    const uint16_t conversion=qbh_attention_u8_float_to_half_bits(512.0f/(float)div);
    const int32_t zp=config->av_multiplier==1U?config->output_zero_point:QBH_ATTENTION_HMX_CENTER;
    for(uint32_t i=0;i<QBH_ATTENTION_HEAD_DIM_TILES*kt*QBH_HMX_WEIGHT_BYTES/128U;++i)
        ((HVX_Vector *)weight)[i]=Q6_V_vzero();
    const uint8_t *saturated=scratch+QBH_ATTN_U8_VGATHER_LUT_OFFSET+QBH_ATTN_U8_VGATHER_LUT_BYTES;
    if(saturation_count && saturated[256]) {
        /* Positive recenter slope makes the non-saturating codes one interval.
         * Count on the same head128 vectors used by the packer, retaining exact
         * telemetry without a byte-at-a-time pass over the KV prefix. */
        uint32_t low=0U, high=255U;
        while(low<256U && saturated[low]) ++low;
        while(high>low && saturated[high]) --high;
        if(low==256U) {
            *saturation_count+=valid_tokens*QBH_ATTENTION_HEAD_DIM;
        } else {
            HVX_Vector count=Q6_V_vzero();
            const HVX_Vector vlo=Q6_Vb_vsplat_R(low),vhi=Q6_Vb_vsplat_R(high);
            for(uint32_t i=0;i<(valid_tokens+1U)/2U;++i) {
                HVX_Vector v=*(const HVX_Vector *)(rows+i*128U);
                HVX_VectorPred clipped=Q6_Q_or_QQ(Q6_Q_vcmp_gt_VubVub(vlo,v),
                                                Q6_Q_vcmp_gt_VubVub(v,vhi));
                clipped=Q6_Q_and_QQ(clipped,valid_tokens-i*2U>=2U?Q6_Q_vcmp_eq_VbVb(v,v):Q6_Q_vsetq_R(64));
                HVX_VectorPair ones=Q6_Wuh_vunpack_Vub(
                    Q6_V_vmux_QVV(clipped,Q6_Vb_vsplat_R(1),Q6_V_vzero()));
                count=Q6_Vh_vadd_VhVh(count,Q6_Vh_vadd_VhVh(Q6_V_lo_W(ones),Q6_V_hi_W(ones)));
            }
            *saturation_count+=qbh_attention_u8_sum_probability_half(count);
        }
    }
    for(uint32_t first=0;first<valid_tokens;first+=32U) {
        uint32_t count=valid_tokens-first;if(count>32U)count=32U;
        qbh_attention_u8_patch_v_long(rows+(size_t)first*QBH_ATTENTION_HEAD_DIM,count,
            config,weight+(first/32U)*QBH_HMX_WEIGHT_BYTES,kt*QBH_HMX_WEIGHT_BYTES,scratch,NULL);
    }
    for(uint32_t tile=0;tile<QBH_ATTENTION_HEAD_DIM_TILES;++tile)for(uint32_t i=0;i<32;++i) {
        uint32_t *b=bias+tile*(QBH_HMX_BIAS_BYTES/4U);b[i]=conversion;b[32+i]=(uint32_t)(zp*(int32_t)div)+rounding;
    }
    asm volatile("barrier" ::: "memory");
}

