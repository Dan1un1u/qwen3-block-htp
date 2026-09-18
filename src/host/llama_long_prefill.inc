/* L32-0050: full-model chunks share persistent KV, never hidden DDR.
 * Input/RoPE file loading and per-sample resets precede warm timing. */
static int qbh_run_long_prefill(struct qbh_session *session, int shared_fd,
    uint8_t *shared, uint32_t total_bytes, const char *package_root,
    struct qbh_block_header *h, const struct qbh_file_slot *tokens,
    const struct qbh_file_slot ropes[2]) {
    struct qbh_decode_session_state *state=(void *)(shared+h->replay_session_offset);
    const struct qbh_decode_session_state initial=*state;
    uint32_t length=h->long_prompt_tokens, decode=3U, repetitions=1U;
    const char *v=getenv("QBH_LONG_DECODE_STEPS");
    const char *audit=getenv("QBH_GENERATION_AUDIT_DIR");
    const int poison=getenv("QBH_LONG_PADDING_POISON")!=NULL;
    if (v && qbh_parse_u32(v,&decode)) return -1;
    v=getenv("QBH_LONG_REPEATS");
    if (v && qbh_parse_u32(v,&repetitions)) return -1;
    if (!length || length>768U || decode>63U || !repetitions || repetitions>10U ||
        length+decode>h->kv_cache_capacity ||
        (h->variant==QBH_BLOCK_W4U8 && !QBH_FP32_RESIDUAL(h)) || h->kv_cache_k_format!=QBH_KV_CACHE_FORMAT_HEAD_MAJOR_ROW_V1)
        return -1;
    uint32_t ids[768], fixed[64];
    uint16_t *cos=malloc((size_t)h->kv_cache_capacity*QBH_BLOCK_HEAD_DIM*2U);
    uint16_t *sin=malloc((size_t)h->kv_cache_capacity*QBH_BLOCK_HEAD_DIM*2U);
    if (!cos || !sin) {free(cos);free(sin);return -1;}
    int result=-1;
    if(qbh_read_named_tensor(package_root,"long_prompt_u32.bin",(void *)ids,(h->variant==QBH_BLOCK_W4U8?length:768U)*4U) ||
       qbh_read_named_tensor(package_root,"long_fixed_u32.bin",(void *)fixed,64U*4U) ||
       qbh_read_named_tensor(package_root,"long_rope_cos_f16.bin",(void *)cos,h->kv_cache_capacity*QBH_BLOCK_HEAD_DIM*2U) ||
       qbh_read_named_tensor(package_root,"long_rope_sin_f16.bin",(void *)sin,h->kv_cache_capacity*QBH_BLOCK_HEAD_DIM*2U)) goto finish;
    for(uint32_t i=0;i<length;i++)if(ids[i]>=QBH_QWEN3_VOCAB_SIZE)goto finish;
    for(uint32_t i=0;i<64;i++)if(fixed[i]>=QBH_QWEN3_VOCAB_SIZE)goto finish;
    for(uint32_t rep=0;rep<repetitions;rep++) {
        if(rep && (qbh_session_release(session)!=AEE_SUCCESS || qbh_session_prepare(session)!=AEE_SUCCESS))goto finish;
        *state=initial;
        for(uint32_t i=0;i<QBH_VERTICAL_SLICE_LAYER_COUNT;i++) {
            struct qbh_decode_layer_state *l=&state->layers[QBH_VERTICAL_SLICE_FIRST_LAYER+i];
            memset(shared+l->k_offset,poison?0xa5:0,l->k_bytes);
            memset(shared+l->v_offset,poison?0x5a:0,l->v_bytes);
        }
        uint64_t prefill_ns=0,decode_ns=0,stage_ns=0,prefill_stage_ns=0,decode_stage_ns=0;
        uint32_t chunks=(length+63U)/64U, selected=0;
        for(uint32_t step=0;step<chunks+decode;step++) {
            const uint32_t before=state->next_position;
            const uint32_t is_prefill=step<chunks;
            const uint32_t rows=is_prefill ? (length-before<64U?length-before:64U):1U;
            const uint64_t stage_start=qbh_monotonic_ns();
            memset(shared+tokens->offset,0,tokens->expected_bytes);
            if(poison)for(uint32_t i=0;i<64U;i++)((uint32_t *)(shared+tokens->offset))[i]=100U+i;
            if(is_prefill)memcpy(shared+tokens->offset,ids+before,rows*4U);
            else *(uint32_t *)(shared+tokens->offset)=getenv("QBH_LONG_GREEDY")?selected:fixed[step-chunks];
            memset(shared+ropes[0].offset,poison?0x3c:0,ropes[0].expected_bytes);
            memset(shared+ropes[1].offset,poison?0xbc:0,ropes[1].expected_bytes);
            memcpy(shared+ropes[0].offset,cos+(size_t)before*QBH_BLOCK_HEAD_DIM,rows*QBH_BLOCK_HEAD_DIM*2U);
            memcpy(shared+ropes[1].offset,sin+(size_t)before*QBH_BLOCK_HEAD_DIM,rows*QBH_BLOCK_HEAD_DIM*2U);
            h->scan_mode=is_prefill?QBH_BLOCK_SCAN_PREFILL:QBH_BLOCK_SCAN_DECODE;
            h->logical_m=rows;h->generation_token_count=rows;
            h->long_skip_head=is_prefill && step+1U<chunks;
            h->long_output_index=is_prefill?0U:step-chunks+1U;
            qbh_bind_host_slice_layer(h,0U);
            h->initial_kv_length=before;h->replay_first_position=before;
            h->replay_expected_step=state->completed_step_count;h->repeat_count=1U;
            h->dsp_status=QBH_BLOCK_STATUS_HOST_READY;
            const uint64_t stage_elapsed=qbh_monotonic_ns()-stage_start;
            stage_ns+=stage_elapsed;
            if(is_prefill)prefill_stage_ns+=stage_elapsed;else decode_stage_ns+=stage_elapsed;
            const uint64_t start=qbh_monotonic_ns();
            const int rc=qwen3_probe_run_block(session->handle,shared_fd,total_bytes);
            const uint64_t wall=qbh_monotonic_ns()-start;
            if(is_prefill)prefill_ns+=wall;else decode_ns+=wall;
            if(rc!=AEE_SUCCESS || h->dsp_status!=QBH_BLOCK_STATUS_OK ||
                state->next_position!=before+rows || state->completed_step_count!=step+1U ||
                h->vtcm_requested_bytes!=8388608U || h->vtcm_acquired_bytes!=8388608U ||
                h->intermediate_ddr_read_bytes || h->intermediate_ddr_write_bytes || h->intermediate_spill_fill_count) {
                fprintf(stderr,"long rep%u step%u rc%d dsp%d failure%u next%u rows%u\n",
                        rep,step,rc,h->dsp_status,h->projection_failure_step,state->next_position,rows);
                if(h->long_optimization&64U)fprintf(stderr,"KDIAG countW=%u countB=%u W=%llx B=%llx\n",h->scan_total_kv_length,h->scan_padded_kv_length,(unsigned long long)h->u8_attention_k_pack_ticks,(unsigned long long)h->u8_attention_v_pack_ticks);
                goto finish;
            }
            for(uint32_t i=0;i<QBH_VERTICAL_SLICE_LAYER_COUNT;i++)
                if(state->layers[QBH_VERTICAL_SLICE_FIRST_LAYER+i].valid_length!=before+rows)goto finish;
            selected=h->generation_selected_token_id;
            struct qbh_replay_step_result pr;memset(&pr,0,sizeof(pr));
            pr.host_wall_ns=wall;pr.first_position=before;pr.valid_length=before+rows;pr.step_index=step;
            pr.dsp_status=h->dsp_status;pr.vtcm_requested_bytes=h->vtcm_requested_bytes;pr.vtcm_acquired_bytes=h->vtcm_acquired_bytes;
            qbh_print_replay_profile(QBH_BLOCK_EXPERIMENT,"long_profile","chunk_step",
                h->variant,step,h,&pr,(const uint8_t *)&selected,sizeof(selected));
            printf("{\"record\":\"long_step\",\"sample\":%u,\"step\":%u,\"prefill\":%u,\"position\":%u,\"rows\":%u,\"head\":%u,\"token\":%u,\"code\":%u,\"host_wall_ns\":%" PRIu64 ",\"layer_hashes\":[",
                rep,step,is_prefill,before,rows,!h->long_skip_head,selected,h->generation_selected_logit_half_bits,wall);
            for(uint32_t i=0;i<QBH_VERTICAL_SLICE_LAYER_COUNT;i++)
                printf("%s\"%016" PRIx64 "\"",i?",":"",h->slice_profiles[i].output_hash);
            printf("]}\n");fflush(stdout);
            if(audit && h->generation_boundary_audit_enabled) {
                char name[96];
                for(uint32_t i=0;i<QBH_VERTICAL_SLICE_LAYER_COUNT;i++) {
                    struct qbh_decode_layer_state *l=&state->layers[QBH_VERTICAL_SLICE_FIRST_LAYER+i];
                    snprintf(name,sizeof(name),"long_r%u_s%02u_l%02u_k.bin",rep,step,i);
                    if(qbh_write_named_tensor(audit,name,shared+l->k_offset,l->k_bytes))goto finish;
                    snprintf(name,sizeof(name),"long_r%u_s%02u_l%02u_v.bin",rep,step,i);
                    if(qbh_write_named_tensor(audit,name,shared+l->v_offset,l->v_bytes))goto finish;
                }
                snprintf(name,sizeof(name),"long_r%u_s%02u_layer_first_rows.bin",rep,step);
                if(qbh_write_named_tensor(audit,name,shared+h->output_offset+68U*QBH_BLOCK_HIDDEN,
                    QBH_VERTICAL_SLICE_LAYER_COUNT*(QBH_FP32_RESIDUAL(h)?4U:2U)*QBH_BLOCK_HIDDEN))goto finish;
                if(!h->long_skip_head) {
                    snprintf(name,sizeof(name),"long_r%u_s%02u_hidden_norm.bin",rep,step);
                    if(qbh_write_named_tensor(audit,name,shared+h->output_offset,QBH_BLOCK_HIDDEN*(QBH_FP32_RESIDUAL(h)?(h->variant==QBH_BLOCK_W4U8?68U:6U):4U)))goto finish;
                }
            }
        }
        printf("{\"record\":\"long_complete\",\"sample\":%u,\"prompt_tokens\":%u,\"decode_steps\":%u,\"prefill_ns\":%" PRIu64 ",\"decode_ns\":%" PRIu64 ",\"host_staging_ns\":%" PRIu64 ",\"prefill_host_staging_ns\":%" PRIu64 ",\"decode_host_staging_ns\":%" PRIu64 "}\n",
            rep,length,decode,prefill_ns,decode_ns,stage_ns,prefill_stage_ns,decode_stage_ns);fflush(stdout);
    }
    result=0;
finish:free(cos);free(sin);return result;
}
