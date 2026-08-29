#include <AEEStdErr.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <remote.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hmx_u8s8_projection.h"
#include "mlp_imp.h"
#include "mlp_protocol.h"
#include "mlp_u8.h"
#include "qbh_user_dma.h"
#include "w4_parallel_pipeline.h"

#define QBH_MLP_DMA_TIMEOUT_TICKS UINT64_C(1920000)
#define QBH_MLP_SELF_TEST_ELEMENTS UINT32_C(65536)
#define QBH_MLP_GATE_UP_PAIR_SLOTS UINT32_C(8)
/* Both projection phases reuse the low VTCM arena.  Down's compressed slots
 * begin immediately after its Middle activation and therefore intentionally
 * overlay Gate/Up storage.  Keep the persistent activation LUT in a disjoint
 * high arena so repeat-N execution cannot overwrite it. */
#define QBH_MLP_LUT_VTCM_OFFSET UINT32_C(3145728)
#define QBH_MLP_GATHER_SCRATCH_VTCM_OFFSET \
    (QBH_MLP_LUT_VTCM_OFFSET + QBH_MLP_LUT_BYTES)
#define QBH_MLP_ALL_GATHER_SCRATCH_BYTES \
    (QBH_MAX_HVX_WORKERS * QBH_MLP_GATHER_SCRATCH_BYTES)
#define QBH_MLP_GATE_UP_MULTIPLIER_VTCM_OFFSET \
    (((QBH_MLP_GATHER_SCRATCH_VTCM_OFFSET + \
       QBH_MLP_ALL_GATHER_SCRATCH_BYTES + 127U) / 128U) * 128U)
#define QBH_MLP_DOWN_MULTIPLIER_VTCM_OFFSET \
    (QBH_MLP_GATE_UP_MULTIPLIER_VTCM_OFFSET + \
     QBH_MLP_GATE_UP_MULTIPLIER_BYTES)

static struct qbh_dma_aligned_desc_2d
    qbh_mlp_output_descriptors[QBH_DOWN_N / QBH_HMX_OUTPUT_CHANNELS]
        __attribute__((aligned(64)));

static int qbh_range_valid(uint32_t offset, uint32_t bytes,
                           uint32_t shared_bytes) {
    return offset >= sizeof(struct qbh_mlp_header) &&
           offset <= shared_bytes && bytes <= shared_bytes - offset;
}

static int qbh_mlp_header_valid(const struct qbh_mlp_header *header,
                                uint32_t shared_bytes,
                                const struct qbh_projection_layout *gate,
                                const struct qbh_projection_layout *down) {
    return header != NULL && gate != NULL && down != NULL &&
           header->magic == QBH_MLP_MAGIC &&
           header->abi_version == QBH_MLP_ABI_VERSION &&
           header->experiment == QBH_MLP_EXPERIMENT &&
           header->header_bytes == sizeof(*header) &&
           header->shared_bytes == shared_bytes &&
           header->repeat_count > 0U &&
           header->repeat_count <= QBH_HMX_MAX_REPEATS &&
           (header->gate_up_worker_count == 2U ||
            header->gate_up_worker_count == 3U ||
            header->gate_up_worker_count == 4U ||
            header->gate_up_worker_count == 6U) &&
           header->input_bytes == gate->activation_bytes &&
           header->gate_up_weight_bytes == gate->stored_weight_bytes &&
           header->down_weight_bytes == down->stored_weight_bytes &&
           header->activation_lut_bytes == QBH_MLP_LUT_BYTES &&
           header->gate_up_multiplier_bytes ==
               QBH_MLP_GATE_UP_MULTIPLIER_BYTES &&
           header->down_multiplier_bytes ==
               QBH_MLP_DOWN_MULTIPLIER_BYTES &&
           header->output_bytes == down->output_bytes &&
           qbh_range_valid(header->input_offset, header->input_bytes,
                           shared_bytes) &&
           qbh_range_valid(header->gate_up_weight_offset,
                           header->gate_up_weight_bytes, shared_bytes) &&
           qbh_range_valid(header->down_weight_offset,
                           header->down_weight_bytes, shared_bytes) &&
           qbh_range_valid(header->activation_lut_offset,
                           header->activation_lut_bytes, shared_bytes) &&
           qbh_range_valid(header->gate_up_multiplier_offset,
                           header->gate_up_multiplier_bytes,
                           shared_bytes) &&
           qbh_range_valid(header->down_multiplier_offset,
                           header->down_multiplier_bytes, shared_bytes) &&
           qbh_range_valid(header->output_offset, header->output_bytes,
                           shared_bytes);
}

static int qbh_configure_gate_up_handoff_layout(
    struct qbh_projection_layout *layout) {
    uint32_t activation_offset = qbh_align_up_u32(
        QBH_MLP_INTERMEDIATE_BYTES,
        QBH_HMX_ACTIVATION_BYTES);
    uint32_t compressed_offset = qbh_align_up_u32(
        activation_offset + layout->activation_bytes,
        QBH_W4_METADATA_ALIGNMENT);
    uint32_t expanded_offset = qbh_align_up_u32(
        compressed_offset + layout->compressed_slot_count *
                                layout->stored_weight_bundle_bytes,
        QBH_W4_METADATA_ALIGNMENT);
    uint32_t output_offset = qbh_align_up_u32(
        expanded_offset + layout->expanded_slot_count *
                              layout->expanded_chunk_slot_bytes,
        QBH_HMX_OUTPUT_BYTES);
    uint32_t plan_bytes = output_offset +
        QBH_MLP_GATE_UP_PAIR_SLOTS * 2U * QBH_HMX_OUTPUT_BYTES;

    layout->vtcm_activation_offset = activation_offset;
    layout->vtcm_compressed_slot0_offset = compressed_offset;
    layout->vtcm_compressed_slot1_offset =
        compressed_offset + layout->w4_bundle_bytes;
    layout->vtcm_chunked_expanded_slots_offset = expanded_offset;
    layout->vtcm_chunked_output_offset = output_offset;
    layout->vtcm_output_offset = output_offset;
    layout->vtcm_chunked_plan_bytes = plan_bytes;
    layout->vtcm_plan_bytes = plan_bytes;
    return plan_bytes <= QBH_W4U8_VTCM_BYTES ? 0 : -1;
}

static void qbh_reset_phase_header(struct qbh_probe_header *phase,
                                   uint32_t workers) {
    memset(phase, 0, sizeof(*phase));
    phase->magic = QBH_PROBE_MAGIC;
    phase->abi_version = QBH_PROBE_ABI_VERSION;
    phase->header_bytes = sizeof(*phase);
    phase->repeat_count = 1U;
    phase->requested_hvx_workers = workers;
    phase->dsp_status = QBH_PROBE_STATUS_DSP_RUNNING;
}

static int qbh_stage_activation(
    struct qbh_mlp_header *header,
    const struct qbh_projection_layout *layout, const uint8_t *source,
    uint8_t *destination) {
    struct qbh_dma_desc_2d descriptor __attribute__((aligned(64)));
    uint64_t start = HAP_perf_get_qtimer_count();

    for (uint32_t input_tile = 0; input_tile < layout->k_tiles;
         ++input_tile) {
        memset(&descriptor, 0, sizeof(descriptor));
        if (qbh_dma_wait_idle() != 0) {
            return -1;
        }
        descriptor.type = QBH_DMA_TYPE_2D;
        descriptor.src_bypass = 1;
        descriptor.dst_bypass = 0;
        descriptor.ordered = 1;
        descriptor.dstate = QBH_DMA_DESC_PENDING;
        descriptor.src = (uint32_t)(uintptr_t)(
            source + input_tile * QBH_HMX_INPUT_CHANNELS);
        descriptor.dst = (uint32_t)(uintptr_t)(
            destination + input_tile * QBH_HMX_ACTIVATION_BYTES);
        descriptor.roi_width = (uint16_t)QBH_HMX_INPUT_CHANNELS;
        descriptor.roi_height = (uint16_t)layout->m;
        descriptor.src_stride = (uint16_t)layout->k;
        descriptor.dst_stride = (uint16_t)QBH_HMX_INPUT_CHANNELS;
        if (qbh_dma_start(&descriptor) != 0 ||
            qbh_dma_wait_idle() != 0) {
            return -1;
        }
        asm volatile("barrier" : : : "memory");
    }
    header->input_stage_ticks += HAP_perf_get_qtimer_count() - start;
    return 0;
}

static int qbh_stage_bytes(const uint8_t *source, uint8_t *destination,
                           uint32_t bytes, uint64_t *ticks) {
    struct qbh_dma_desc_1d descriptor __attribute__((aligned(64)));
    uint64_t start = HAP_perf_get_qtimer_count();

    memset(&descriptor, 0, sizeof(descriptor));
    if (qbh_dma_wait_idle() != 0) {
        return -1;
    }
    descriptor.length = bytes;
    descriptor.type = QBH_DMA_TYPE_1D;
    descriptor.src_bypass = 1;
    descriptor.dst_bypass = 0;
    descriptor.ordered = 1;
    descriptor.dstate = QBH_DMA_DESC_PENDING;
    descriptor.src = (uint32_t)(uintptr_t)source;
    descriptor.dst = (uint32_t)(uintptr_t)destination;
    if (qbh_dma_start(&descriptor) != 0 || qbh_dma_wait_idle() != 0) {
        return -1;
    }
    asm volatile("barrier" : : : "memory");
    *ticks += HAP_perf_get_qtimer_count() - start;
    return 0;
}

static int qbh_stage_activation_lut(struct qbh_mlp_header *header,
                                    const uint8_t *source,
                                    uint8_t *destination) {
    return qbh_stage_bytes(source, destination, QBH_MLP_LUT_BYTES,
                           &header->activation_lut_stage_ticks);
}

static int qbh_run_activation_self_test(struct qbh_mlp_header *header,
                                        uint8_t *vtcm) {
    uint8_t *gate = vtcm;
    uint8_t *up = gate + QBH_MLP_SELF_TEST_ELEMENTS;
    uint8_t *actual = up + QBH_MLP_SELF_TEST_ELEMENTS;
    uint32_t mismatches = 0U;
    int lock_status;
    int unlock_status;

    for (uint32_t index = 0; index < QBH_MLP_SELF_TEST_ELEMENTS; ++index) {
        gate[index] = (uint8_t)(index >> 8U);
        up[index] = (uint8_t)index;
    }
    lock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (lock_status != AEE_SUCCESS) {
        return -1;
    }
    qbh_mlp_gate_up_lut_hvx(
        gate, up, actual, QBH_MLP_SELF_TEST_ELEMENTS,
        (const uint16_t *)(vtcm + QBH_MLP_LUT_VTCM_OFFSET),
        vtcm + QBH_MLP_GATHER_SCRATCH_VTCM_OFFSET);
    unlock_status = qurt_hvx_unlock();
    if (unlock_status != AEE_SUCCESS) {
        return -1;
    }
    for (uint32_t index = 0; index < QBH_MLP_SELF_TEST_ELEMENTS; ++index) {
        const uint16_t *lut = (const uint16_t *)(
            vtcm + QBH_MLP_LUT_VTCM_OFFSET);
        if (actual[index] !=
            (uint8_t)lut[((uint32_t)gate[index] << 8U) | up[index]]) {
            ++mismatches;
        }
    }
    header->activation_self_test_cases = QBH_MLP_SELF_TEST_ELEMENTS;
    header->activation_self_test_mismatches = mismatches;
    return mismatches == 0U ? 0 : -1;
}

static int qbh_requant_down_output(
    struct qbh_mlp_header *header,
    const struct qbh_projection_layout *layout, uint8_t *output_tiles,
    const uint8_t *multipliers) {
    uint64_t start = HAP_perf_get_qtimer_count();
    int lock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    int unlock_status;
    if (lock_status != AEE_SUCCESS) {
        return -1;
    }
    for (uint32_t tile = 0; tile < layout->n_tiles; ++tile) {
        qbh_mlp_requant_u8_hvx(
            output_tiles + (size_t)tile * QBH_HMX_OUTPUT_BYTES,
            QBH_HMX_OUTPUT_BYTES,
            multipliers + tile * QBH_HMX_OUTPUT_CHANNELS,
            QBH_MLP_DOWN_ZERO_POINT);
    }
    unlock_status = qurt_hvx_unlock();
    if (unlock_status != AEE_SUCCESS) {
        return -1;
    }
    header->down_requant_ticks +=
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

static int qbh_assemble_final_output(
    struct qbh_mlp_header *header,
    const struct qbh_projection_layout *layout, uint8_t *destination,
    const uint8_t *source_tiles) {
    uint64_t start = HAP_perf_get_qtimer_count();
    uint32_t spins = 0U;

    if (qbh_dma_wait_idle() != 0) {
        return -1;
    }
    memset(qbh_mlp_output_descriptors, 0,
           sizeof(*qbh_mlp_output_descriptors) * layout->n_tiles);
    for (uint32_t output_tile = 0; output_tile < layout->n_tiles;
         ++output_tile) {
        struct qbh_dma_desc_2d *descriptor =
            &qbh_mlp_output_descriptors[output_tile].descriptor;
        descriptor->next = output_tile + 1U < layout->n_tiles
                               ? (uint32_t)(uintptr_t)(
                                     &qbh_mlp_output_descriptors[
                                          output_tile + 1U]
                                          .descriptor)
                               : 0U;
        descriptor->type = QBH_DMA_TYPE_2D;
        descriptor->src_bypass = 0;
        descriptor->dst_bypass = 1;
        descriptor->ordered = 1;
        descriptor->dstate = QBH_DMA_DESC_PENDING;
        descriptor->src = (uint32_t)(uintptr_t)(
            source_tiles + (size_t)output_tile * QBH_HMX_OUTPUT_BYTES);
        descriptor->dst = (uint32_t)(uintptr_t)(
            destination + output_tile * QBH_HMX_OUTPUT_CHANNELS);
        descriptor->roi_width = (uint16_t)QBH_HMX_OUTPUT_CHANNELS;
        descriptor->roi_height = (uint16_t)layout->m;
        descriptor->src_stride = (uint16_t)QBH_HMX_OUTPUT_CHANNELS;
        descriptor->dst_stride = (uint16_t)layout->n;
        asm volatile("release(%0):at"
                     :
                     : "r"(descriptor)
                     : "memory");
    }
    if (qbh_dma_start(&qbh_mlp_output_descriptors[0].descriptor) != 0) {
        return -1;
    }
    header->final_output_dma_descriptor_count += layout->n_tiles;
    for (;;) {
        uint32_t status = Q6_R_dmpoll() & QBH_DMA_STATUS_MASK;
        volatile struct qbh_dma_desc_2d *last =
            &qbh_mlp_output_descriptors[layout->n_tiles - 1U].descriptor;
        if (last->dstate == QBH_DMA_DESC_COMPLETE) {
            break;
        }
        if (status == QBH_DMA_STATUS_ERROR) {
            header->final_output_dma_status = (int32_t)status;
            return -1;
        }
        ++spins;
        if ((spins & UINT32_C(255)) == 0U &&
            HAP_perf_get_qtimer_count() - start >
                QBH_MLP_DMA_TIMEOUT_TICKS) {
            ++header->final_output_dma_timeout_count;
            return -1;
        }
    }
    if (qbh_dma_wait_idle() != 0) {
        return -1;
    }
    header->final_output_ticks += HAP_perf_get_qtimer_count() - start;
    return 0;
}

AEEResult qbh_run_mlp_rpc(int32_t shared_fd, uint32_t shared_bytes,
                          uint8_t *vtcm, uint32_t hmx_context_id,
                          uint32_t prepared_session_run_index) {
    struct qbh_mlp_header *header = NULL;
    struct qbh_projection_layout gate_layout;
    struct qbh_projection_layout down_layout;
    uint8_t *shared = NULL;
    int cache_status;
    int result;

    if (vtcm == NULL || hmx_context_id == 0U ||
        shared_bytes < sizeof(struct qbh_mlp_header)) {
        return AEE_EBADPARM;
    }
    result = HAP_mmap_get(shared_fd, (void **)&shared, NULL);
    if (result != AEE_SUCCESS || shared == NULL) {
        return result != AEE_SUCCESS ? result : AEE_EFAILED;
    }
    header = (struct qbh_mlp_header *)shared;
    cache_status = qurt_mem_cache_clean(
        (qurt_addr_t)header, (qurt_size_t)sizeof(*header),
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (cache_status != 0) {
        result = AEE_EFAILED;
        goto cleanup;
    }
    if (qbh_projection_layout_init(
            QBH_PROJECTION_GATE_UP_PAIR,
            header->weight_storage_variant,
            QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4, 8U,
            QBH_W4_COARSE_CHUNK_TILES, &gate_layout) != 0 ||
        qbh_projection_layout_init(
            QBH_PROJECTION_DOWN, header->weight_storage_variant,
            QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2, 4U,
            QBH_W4_WIDE_CHUNK_TILES, &down_layout) != 0 ||
        qbh_configure_gate_up_handoff_layout(&gate_layout) != 0 ||
        !qbh_mlp_header_valid(header, shared_bytes, &gate_layout,
                              &down_layout)) {
        header->dsp_status = QBH_MLP_STATUS_BAD_HEADER;
        result = AEE_EBADPARM;
        goto publish_header;
    }
    cache_status = qurt_mem_cache_clean(
        (qurt_addr_t)(shared + header->input_offset),
        (qurt_size_t)header->input_bytes, QURT_MEM_CACHE_INVALIDATE,
        QURT_MEM_DCACHE);
    if (cache_status == 0) {
        cache_status = qurt_mem_cache_clean(
            (qurt_addr_t)(shared + header->gate_up_weight_offset),
            (qurt_size_t)header->gate_up_weight_bytes,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    if (cache_status == 0) {
        cache_status = qurt_mem_cache_clean(
            (qurt_addr_t)(shared + header->down_weight_offset),
            (qurt_size_t)header->down_weight_bytes,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    if (cache_status == 0) {
        cache_status = qurt_mem_cache_clean(
            (qurt_addr_t)(shared + header->activation_lut_offset),
            (qurt_size_t)header->activation_lut_bytes,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    if (cache_status == 0) {
        cache_status = qurt_mem_cache_clean(
            (qurt_addr_t)(shared + header->gate_up_multiplier_offset),
            (qurt_size_t)header->gate_up_multiplier_bytes,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    if (cache_status == 0) {
        cache_status = qurt_mem_cache_clean(
            (qurt_addr_t)(shared + header->down_multiplier_offset),
            (qurt_size_t)header->down_multiplier_bytes,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    header->cache_status = cache_status;
    if (cache_status != 0) {
        header->dsp_status = QBH_MLP_STATUS_CACHE_FAILED;
        result = AEE_EFAILED;
        goto publish_header;
    }

    header->dsp_status = QBH_MLP_STATUS_DSP_RUNNING;
    header->prepared_session_run_index = prepared_session_run_index;
    header->resource_vtcm_address = (uint32_t)(uintptr_t)vtcm;
    header->resource_hmx_context_id = hmx_context_id;
    header->vtcm_requested_bytes = QBH_W4U8_VTCM_BYTES;
    header->vtcm_acquired_bytes = QBH_W4U8_VTCM_BYTES;
    header->gate_up_output_vtcm_bytes =
        QBH_MLP_GATE_UP_PAIR_SLOTS * 2U * QBH_HMX_OUTPUT_BYTES;
    header->middle_vtcm_bytes = QBH_MLP_INTERMEDIATE_BYTES;
    header->activation_lut_vtcm_bytes = QBH_MLP_LUT_BYTES;
    header->requant_metadata_vtcm_bytes =
        QBH_MLP_GATE_UP_MULTIPLIER_BYTES +
        QBH_MLP_DOWN_MULTIPLIER_BYTES;
    header->activation_gather_scratch_vtcm_bytes =
        QBH_MLP_ALL_GATHER_SCRATCH_BYTES;
    header->final_output_vtcm_bytes = down_layout.output_tiles_bytes;
    header->gate_up_pair_slot_count = QBH_MLP_GATE_UP_PAIR_SLOTS;
    header->gate_up_pair_publish_count = 0U;
    header->gate_up_pair_consume_count = 0U;
    header->gate_up_full_tensor_materialized = 0U;
    header->vtcm_peak_plan_bytes =
        QBH_MLP_DOWN_MULTIPLIER_VTCM_OFFSET +
        QBH_MLP_DOWN_MULTIPLIER_BYTES;
    if (header->vtcm_peak_plan_bytes < gate_layout.vtcm_plan_bytes) {
        header->vtcm_peak_plan_bytes = gate_layout.vtcm_plan_bytes;
    }
    if (header->vtcm_peak_plan_bytes < down_layout.vtcm_plan_bytes) {
        header->vtcm_peak_plan_bytes = down_layout.vtcm_plan_bytes;
    }
    header->intermediate_ddr_read_bytes = 0U;
    header->intermediate_ddr_write_bytes = 0U;
    header->intermediate_dma_descriptor_count = 0U;
    header->intermediate_spill_fill_count = 0U;
    header->gate_up_output_dma_descriptor_count = 0U;
    header->middle_dma_descriptor_count = 0U;
    header->final_output_dma_descriptor_count = 0U;
    header->final_output_dma_timeout_count = 0U;
    header->final_output_dma_status = 0;
    header->input_stage_ticks = 0U;
    header->activation_lut_stage_ticks = 0U;
    header->requant_metadata_stage_ticks = 0U;
    header->gate_up_ticks = 0U;
    header->activation_ticks = 0U;
    header->down_ticks = 0U;
    header->down_requant_ticks = 0U;
    header->final_output_ticks = 0U;

    if (qbh_stage_activation_lut(
            header, shared + header->activation_lut_offset,
            vtcm + QBH_MLP_LUT_VTCM_OFFSET) != 0) {
        header->dsp_status = QBH_MLP_STATUS_INPUT_DMA_FAILED;
        result = AEE_EFAILED;
        goto publish_header;
    }
    if (qbh_stage_bytes(
            shared + header->gate_up_multiplier_offset,
            vtcm + QBH_MLP_GATE_UP_MULTIPLIER_VTCM_OFFSET,
            QBH_MLP_GATE_UP_MULTIPLIER_BYTES,
            &header->requant_metadata_stage_ticks) != 0 ||
        qbh_stage_bytes(
            shared + header->down_multiplier_offset,
            vtcm + QBH_MLP_DOWN_MULTIPLIER_VTCM_OFFSET,
            QBH_MLP_DOWN_MULTIPLIER_BYTES,
            &header->requant_metadata_stage_ticks) != 0) {
        header->dsp_status = QBH_MLP_STATUS_INPUT_DMA_FAILED;
        result = AEE_EFAILED;
        goto publish_header;
    }

    if (header->run_activation_self_test != 0U &&
        qbh_run_activation_self_test(header, vtcm) != 0) {
        header->dsp_status = QBH_MLP_STATUS_SELF_TEST_FAILED;
        result = AEE_EFAILED;
        goto publish_header;
    }

    header->qtimer_start = HAP_perf_get_qtimer_count();
    for (uint32_t repeat = 0; repeat < header->repeat_count; ++repeat) {
        uint64_t start;
        uint32_t pair_publish_count = 0U;
        uint32_t pair_consume_count = 0U;
        uint64_t activation_ticks = 0U;
        struct qbh_mlp_gate_up_handoff handoff = {
            .middle_activation = vtcm,
            .activation_lut = (const uint16_t *)(
                vtcm + QBH_MLP_LUT_VTCM_OFFSET),
            .output_multipliers =
                vtcm + QBH_MLP_GATE_UP_MULTIPLIER_VTCM_OFFSET,
            .activation_gather_scratch =
                vtcm + QBH_MLP_GATHER_SCRATCH_VTCM_OFFSET,
            .pair_slot_count = QBH_MLP_GATE_UP_PAIR_SLOTS,
            .pair_publish_count = &pair_publish_count,
            .pair_consume_count = &pair_consume_count,
            .activation_ticks = &activation_ticks,
        };

        if (qbh_stage_activation(
                header, &gate_layout, shared + header->input_offset,
                vtcm + gate_layout.vtcm_activation_offset) != 0) {
            header->dsp_status = QBH_MLP_STATUS_INPUT_DMA_FAILED;
            result = AEE_EFAILED;
            goto publish_header;
        }

        qbh_reset_phase_header(&header->gate_up_phase,
                               header->gate_up_worker_count);
        start = HAP_perf_get_qtimer_count();
        result = qbh_run_chunked_w4_pipeline_mlp(
            &header->gate_up_phase, &gate_layout,
            shared + header->gate_up_weight_offset,
            vtcm + gate_layout.vtcm_activation_offset, vtcm,
            hmx_context_id, &handoff);
        header->gate_up_ticks += HAP_perf_get_qtimer_count() - start;
        if (result != AEE_SUCCESS) {
            header->dsp_status = QBH_MLP_STATUS_GATE_UP_FAILED;
            goto publish_header;
        }

        header->gate_up_pair_publish_count += pair_publish_count;
        header->gate_up_pair_consume_count += pair_consume_count;
        header->activation_ticks += activation_ticks;
        if (pair_publish_count != QBH_GATE_UP_N /
                                      QBH_HMX_OUTPUT_CHANNELS ||
            pair_consume_count != pair_publish_count) {
            header->dsp_status = QBH_MLP_STATUS_ACTIVATION_FAILED;
            result = AEE_EFAILED;
            goto publish_header;
        }

        qbh_reset_phase_header(&header->down_phase, 6U);
        start = HAP_perf_get_qtimer_count();
        result = qbh_run_chunked_w4_pipeline(
            &header->down_phase, &down_layout,
            shared + header->down_weight_offset, vtcm, vtcm,
            hmx_context_id);
        header->down_ticks += HAP_perf_get_qtimer_count() - start;
        if (result != AEE_SUCCESS) {
            header->dsp_status = QBH_MLP_STATUS_DOWN_FAILED;
            goto publish_header;
        }
        if (qbh_requant_down_output(
                header, &down_layout,
                vtcm + down_layout.vtcm_output_offset,
                vtcm + QBH_MLP_DOWN_MULTIPLIER_VTCM_OFFSET) != 0) {
            header->dsp_status = QBH_MLP_STATUS_ACTIVATION_FAILED;
            result = AEE_EFAILED;
            goto publish_header;
        }
    }

    if (qbh_assemble_final_output(
            header, &down_layout, shared + header->output_offset,
            vtcm + down_layout.vtcm_output_offset) != 0) {
        header->dsp_status = QBH_MLP_STATUS_OUTPUT_DMA_FAILED;
        result = AEE_EFAILED;
        goto publish_header;
    }
    header->qtimer_end = HAP_perf_get_qtimer_count();
    header->total_ticks = header->qtimer_end - header->qtimer_start;
    header->dsp_status = QBH_MLP_STATUS_OK;
    result = AEE_SUCCESS;

publish_header:
    if (header != NULL) {
        int output_cache_status = 0;
        if (header->dsp_status == QBH_MLP_STATUS_OK) {
            output_cache_status = qurt_mem_cache_clean(
                (qurt_addr_t)(shared + header->output_offset),
                (qurt_size_t)header->output_bytes,
                QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        }
        if (output_cache_status == 0) {
            output_cache_status = qurt_mem_cache_clean(
                (qurt_addr_t)header, (qurt_size_t)sizeof(*header),
                QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        }
        if (output_cache_status != 0 && result == AEE_SUCCESS) {
            result = AEE_EFAILED;
        }
    }
cleanup:
    (void)HAP_mmap_put(shared_fd);
    return result;
}
