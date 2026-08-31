#include <AEEStdErr.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <HAP_power.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <remote.h>
#include <stdint.h>
#include <string.h>

#include <hexagon_types.h>

#include "hmx_u8s8_projection.h"
#include "hmx_convert_protocol.h"
#include "attention_imp.h"
#include "block_imp.h"
#include "mlp_imp.h"
#include "probe_protocol.h"
#include "qbh_user_dma.h"
#include "qwen3_probe.h"
#include "resource_protocol.h"
#include "w4_parallel_pipeline.h"
#include "w4_u8_expand.h"

#define QBH_HMX_WORKER_STACK_BYTES UINT32_C(16384)
#define QBH_DMA_DESCRIPTOR_TIMEOUT_TICKS UINT64_C(1920000)

struct qbh_probe_session {
    uint32_t open;
    uint32_t prepared;
    uint32_t prepared_run_count;
    uint32_t vtcm_context_id;
    uint32_t hmx_context_id;
    uint8_t *vtcm;
    int32_t prepare_result;
    int32_t vtcm_query_status;
    int32_t vtcm_configure_status;
    int32_t vtcm_acquire_status;
    int32_t vtcm_get_pointer_status;
    uint32_t vtcm_total_bytes;
    uint32_t vtcm_available_before_bytes;
    uint32_t vtcm_requested_bytes;
    uint32_t vtcm_minimum_page_bytes;
    uint32_t vtcm_minimum_required_bytes;
    uint32_t vtcm_granted_bytes;
    uint64_t vtcm_acquire_ticks;
    int power_context;
    int dcvs_powered;
    int hmx_powered;
    struct qbh_block_persistent_state block_persistent;
};

static struct qbh_probe_session probe_session;
static uint8_t hmx_worker_stack[QBH_HMX_WORKER_STACK_BYTES]
    __attribute__((aligned(128)));
static struct qbh_dma_aligned_desc_2d
    output_dma_descriptors[QBH_MAX_PROJECTION_N /
                           QBH_HMX_OUTPUT_CHANNELS]
        __attribute__((aligned(64)));

struct qbh_projection_worker_job {
    uint32_t hmx_context_id;
    const uint8_t *activation_tiles;
    const uint8_t *weight_slots[2];
    const uint32_t *bias_slots[2];
    uint8_t *output_tiles;
    uint32_t repeat_count;
    uint32_t k_tiles;
    uint32_t n_tiles;
    qurt_sem_t *ready[2];
    qurt_sem_t *free_slot[2];
    qurt_sem_t *started;
    volatile int32_t abort_status;
    int32_t lock_status;
    int32_t unlock_status;
    int32_t sync_status;
    uint32_t hmx_execution_count;
    uint32_t hmx_stream_count;
    uint32_t output_tile_count;
    uint64_t hmx_compute_ticks;
    uint64_t ready_wait_ticks;
};

static struct qbh_probe_session *qbh_session_from_handle(
    remote_handle64 handle) {
    return handle == (remote_handle64)&probe_session &&
                   probe_session.open != 0U
               ? &probe_session
               : NULL;
}

static AEEResult qbh_release_prepared_resources(
    struct qbh_probe_session *session) {
    HAP_power_request_t request;
    AEEResult result = AEE_SUCCESS;
    AEEResult current;

    if (session == NULL) {
        return AEE_EBADPARM;
    }
    if (session->hmx_context_id != 0U) {
        current = HAP_compute_res_release(session->hmx_context_id);
        if (result == AEE_SUCCESS && current != AEE_SUCCESS) {
            result = current;
        }
        session->hmx_context_id = 0U;
    }
    if (session->vtcm_context_id != 0U) {
        current = HAP_compute_res_release(session->vtcm_context_id);
        if (result == AEE_SUCCESS && current != AEE_SUCCESS) {
            result = current;
        }
        session->vtcm_context_id = 0U;
        session->vtcm = NULL;
    }
    if (session->hmx_powered) {
        memset(&request, 0, sizeof(request));
        request.type = HAP_power_set_HMX;
        request.hmx.power_up = 0;
        current = HAP_power_set(&session->power_context, &request);
        if (result == AEE_SUCCESS && current != AEE_SUCCESS) {
            result = current;
        }
        session->hmx_powered = 0;
    }
    if (session->dcvs_powered) {
        HAP_power_set_dcvs_v3_init(&request);
        current = HAP_power_set(&session->power_context, &request);
        if (result == AEE_SUCCESS && current != AEE_SUCCESS) {
            result = current;
        }
        session->dcvs_powered = 0;
    }
    session->power_context = 0;
    session->prepared = 0U;
    session->prepared_run_count = 0U;
    memset(&session->block_persistent, 0,
           sizeof(session->block_persistent));
    return result;
}

AEEResult qwen3_probe_open(const char *uri, remote_handle64 *handle) {
    (void)uri;
    if (handle == NULL) {
        return AEE_EBADPARM;
    }
    if (probe_session.open != 0U) {
        return AEE_EFAILED;
    }
    memset(&probe_session, 0, sizeof(probe_session));
    probe_session.open = 1U;
    *handle = (remote_handle64)&probe_session;
    return AEE_SUCCESS;
}

AEEResult qwen3_probe_close(remote_handle64 handle) {
    struct qbh_probe_session *session = qbh_session_from_handle(handle);
    AEEResult result;
    if (session == NULL) {
        return AEE_EBADPARM;
    }
    result = qbh_release_prepared_resources(session);
    memset(session, 0, sizeof(*session));
    return result;
}

AEEResult qwen3_probe_prepare(remote_handle64 handle) {
    struct qbh_probe_session *session = qbh_session_from_handle(handle);
    compute_res_attr_t vtcm_attributes;
    compute_res_attr_t hmx_attributes;
    HAP_power_request_t request;
    void *vtcm_pointer = NULL;
    unsigned int vtcm_total_bytes = 0U;
    unsigned int vtcm_available_before_bytes = 0U;
    unsigned int vtcm_granted_bytes = 0U;
    uint64_t acquire_start;
    AEEResult result;

    if (session == NULL) {
        return AEE_EBADPARM;
    }
    if (session->prepared != 0U) {
        return AEE_SUCCESS;
    }

    session->prepare_result = AEE_EFAILED;
    session->vtcm_query_status = AEE_EFAILED;
    session->vtcm_configure_status = AEE_EFAILED;
    session->vtcm_acquire_status = AEE_EFAILED;
    session->vtcm_get_pointer_status = AEE_EFAILED;

    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.dcvs_enable = 1;
    request.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    request.dcvs_v3.set_latency = 1;
    request.dcvs_v3.latency = 100;
    request.dcvs_v3.set_core_params = 1;
    request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_NOM;
    request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_TURBO_L3;
    request.dcvs_v3.core_params.target_corner =
        HAP_DCVS_VCORNER_TURBO_L3;
    request.dcvs_v3.set_bus_params = 1;
    request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_NOM;
    request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_TURBO_L3;
    request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_TURBO_L3;
    result = HAP_power_set(&session->power_context, &request);
    if (result != AEE_SUCCESS) {
        goto failure;
    }
    session->dcvs_powered = 1;

    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HMX;
    request.hmx.power_up = 1;
    result = HAP_power_set(&session->power_context, &request);
    if (result != AEE_SUCCESS) {
        goto failure;
    }
    session->hmx_powered = 1;

    session->vtcm_query_status = HAP_compute_res_query_VTCM(
        0U, &vtcm_total_bytes, NULL, &vtcm_available_before_bytes, NULL);
    session->vtcm_total_bytes = (uint32_t)vtcm_total_bytes;
    session->vtcm_available_before_bytes =
        (uint32_t)vtcm_available_before_bytes;
    session->vtcm_requested_bytes = session->vtcm_total_bytes;
    session->vtcm_minimum_page_bytes = QBH_FULL_VTCM_MIN_PAGE_BYTES;
    session->vtcm_minimum_required_bytes = 0U;
    if (session->vtcm_query_status != AEE_SUCCESS ||
        session->vtcm_total_bytes != QBH_EXPECTED_FULL_VTCM_BYTES) {
        result = session->vtcm_query_status != AEE_SUCCESS
                     ? session->vtcm_query_status
                     : AEE_EFAILED;
        goto failure;
    }

    session->vtcm_configure_status =
        HAP_compute_res_attr_init(&vtcm_attributes);
    if (session->vtcm_configure_status == AEE_SUCCESS) {
        session->vtcm_configure_status =
            HAP_compute_res_attr_set_serialize(&vtcm_attributes, 1);
    }
    if (session->vtcm_configure_status == AEE_SUCCESS) {
        session->vtcm_configure_status =
            HAP_compute_res_attr_set_vtcm_param_v2(
                &vtcm_attributes, session->vtcm_requested_bytes,
                session->vtcm_minimum_page_bytes,
                session->vtcm_minimum_required_bytes);
    }
    if (session->vtcm_configure_status != AEE_SUCCESS) {
        result = session->vtcm_configure_status;
        goto failure;
    }
    acquire_start = HAP_perf_get_qtimer_count();
    session->vtcm_context_id =
        HAP_compute_res_acquire(&vtcm_attributes, 1000000);
    session->vtcm_acquire_ticks =
        HAP_perf_get_qtimer_count() - acquire_start;
    if (session->vtcm_context_id == 0U) {
        session->vtcm_acquire_status = AEE_ERESOURCENOTFOUND;
        result = session->vtcm_acquire_status;
        goto failure;
    }
    session->vtcm_acquire_status = AEE_SUCCESS;
    session->vtcm_get_pointer_status =
        HAP_compute_res_attr_get_vtcm_ptr_v2(
            &vtcm_attributes, &vtcm_pointer, &vtcm_granted_bytes);
    session->vtcm = (uint8_t *)vtcm_pointer;
    session->vtcm_granted_bytes = vtcm_granted_bytes;
    if (session->vtcm_get_pointer_status != AEE_SUCCESS ||
        session->vtcm == NULL ||
        session->vtcm_granted_bytes != session->vtcm_requested_bytes) {
        result = session->vtcm_get_pointer_status != AEE_SUCCESS
                     ? session->vtcm_get_pointer_status
                     : AEE_EFAILED;
        goto failure;
    }

    result = HAP_compute_res_attr_init(&hmx_attributes);
    if (result != AEE_SUCCESS ||
        HAP_compute_res_attr_set_hmx_param(&hmx_attributes, 1) !=
            AEE_SUCCESS) {
        result = AEE_EFAILED;
        goto failure;
    }
    session->hmx_context_id =
        HAP_compute_res_acquire(&hmx_attributes, 100000);
    if (session->hmx_context_id == 0U) {
        result = AEE_ERESOURCENOTFOUND;
        goto failure;
    }
    session->prepared = 1U;
    session->prepared_run_count = 0U;
    memset(&session->block_persistent, 0,
           sizeof(session->block_persistent));
    session->prepare_result = AEE_SUCCESS;
    return AEE_SUCCESS;

failure:
    session->prepare_result = result;
    (void)qbh_release_prepared_resources(session);
    return result;
}

AEEResult qwen3_probe_release(remote_handle64 handle) {
    struct qbh_probe_session *session = qbh_session_from_handle(handle);
    return session != NULL ? qbh_release_prepared_resources(session)
                           : AEE_EBADPARM;
}

AEEResult qwen3_probe_resource_info(remote_handle64 handle, int32 shared_fd,
                                    uint32 shared_bytes) {
    struct qbh_probe_session *session = qbh_session_from_handle(handle);
    struct qbh_resource_header *header = NULL;
    uint8_t *shared = NULL;
    int cache_status;
    AEEResult result;

    if (session == NULL || shared_bytes < sizeof(*header)) {
        return AEE_EBADPARM;
    }
    result = HAP_mmap_get(shared_fd, (void **)&shared, NULL);
    if (result != AEE_SUCCESS || shared == NULL) {
        return result != AEE_SUCCESS ? result : AEE_EFAILED;
    }
    header = (struct qbh_resource_header *)shared;
    cache_status = qurt_mem_cache_clean(
        (qurt_addr_t)header, (qurt_size_t)sizeof(*header),
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (cache_status != 0 || header->magic != QBH_RESOURCE_MAGIC ||
        header->abi_version != QBH_RESOURCE_ABI_VERSION ||
        header->experiment != QBH_RESOURCE_EXPERIMENT ||
        header->header_bytes != sizeof(*header)) {
        if (cache_status == 0) {
            header->dsp_status = QBH_RESOURCE_STATUS_BAD_HEADER;
        }
        result = cache_status == 0 ? AEE_EBADPARM : AEE_EFAILED;
        goto publish;
    }

    header->dsp_status = QBH_RESOURCE_STATUS_OK;
    header->prepare_result = session->prepare_result;
    header->query_status = session->vtcm_query_status;
    header->configure_status = session->vtcm_configure_status;
    header->acquire_status = session->vtcm_acquire_status;
    header->get_pointer_status = session->vtcm_get_pointer_status;
    header->expected_total_bytes = QBH_EXPECTED_FULL_VTCM_BYTES;
    header->queried_total_bytes = session->vtcm_total_bytes;
    header->available_before_bytes = session->vtcm_available_before_bytes;
    header->requested_bytes = session->vtcm_requested_bytes;
    header->minimum_page_bytes = session->vtcm_minimum_page_bytes;
    header->minimum_required_bytes = session->vtcm_minimum_required_bytes;
    header->granted_bytes = session->vtcm_granted_bytes;
    header->context_id = session->vtcm_context_id;
    header->vtcm_address = (uint32_t)(uintptr_t)session->vtcm;
    header->exact_full_grant =
        session->prepared != 0U &&
        session->vtcm_total_bytes == QBH_EXPECTED_FULL_VTCM_BYTES &&
        session->vtcm_granted_bytes == session->vtcm_total_bytes;
    header->acquire_ticks = session->vtcm_acquire_ticks;
    result = AEE_SUCCESS;

publish:
    if (header != NULL) {
        int flush_status = qurt_mem_cache_clean(
            (qurt_addr_t)header, (qurt_size_t)sizeof(*header),
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        if (flush_status != 0 && result == AEE_SUCCESS) {
            result = AEE_EFAILED;
        }
    }
    (void)HAP_mmap_put(shared_fd);
    return result;
}

static int range_is_valid(uint32_t offset, uint32_t bytes,
                          uint32_t total_bytes) {
    return offset <= total_bytes && bytes <= total_bytes - offset;
}

AEEResult qwen3_probe_run_hmx_convert(remote_handle64 handle,
                                      int32 shared_fd,
                                      uint32 shared_bytes) {
    struct qbh_probe_session *session = qbh_session_from_handle(handle);
    struct qbh_hmx_convert_header *header = NULL;
    uint8_t *shared = NULL;
    uint8_t *activation;
    int8_t *weight;
    uint32_t *bias;
    uint8_t *output;
    int cache_status;
    AEEResult result;

    if (session == NULL || session->prepared == 0U ||
        session->vtcm == NULL || shared_bytes < sizeof(*header)) {
        return AEE_EBADSTATE;
    }
    result = HAP_mmap_get(shared_fd, (void **)&shared, NULL);
    if (result != AEE_SUCCESS || shared == NULL) {
        return result != AEE_SUCCESS ? result : AEE_EFAILED;
    }
    header = (struct qbh_hmx_convert_header *)shared;
    cache_status = qurt_mem_cache_clean(
        (qurt_addr_t)shared, (qurt_size_t)shared_bytes,
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (cache_status != 0) {
        header->dsp_status = QBH_HMX_CONVERT_STATUS_CACHE_FAILED;
        header->cache_status = cache_status;
        result = AEE_EFAILED;
        goto publish;
    }
    if (header->magic != QBH_HMX_CONVERT_MAGIC ||
        header->abi_version != QBH_HMX_CONVERT_ABI_VERSION ||
        header->header_bytes != sizeof(*header) ||
        header->total_bytes > shared_bytes ||
        !range_is_valid(header->activation_offset,
                        QBH_HMX_ACTIVATION_BYTES,
                        header->total_bytes) ||
        !range_is_valid(header->weight_offset, QBH_HMX_WEIGHT_BYTES,
                        header->total_bytes) ||
        !range_is_valid(header->bias_offset, QBH_HMX_BIAS_BYTES,
                        header->total_bytes) ||
        !range_is_valid(header->output_offset, QBH_HMX_OUTPUT_BYTES,
                        header->total_bytes)) {
        header->dsp_status = QBH_HMX_CONVERT_STATUS_BAD_HEADER;
        result = AEE_EBADPARM;
        goto publish;
    }

    activation = session->vtcm;
    weight = (int8_t *)(session->vtcm + QBH_HMX_ACTIVATION_BYTES);
    bias = (uint32_t *)(session->vtcm +
                        QBH_HMX_ACTIVATION_BYTES +
                        QBH_HMX_WEIGHT_BYTES);
    output = session->vtcm + QBH_HMX_OUTPUT_BYTES * 2U;
    memcpy(activation, shared + header->activation_offset,
           QBH_HMX_ACTIVATION_BYTES);
    memcpy(weight, shared + header->weight_offset,
           QBH_HMX_WEIGHT_BYTES);
    memcpy(bias, shared + header->bias_offset, QBH_HMX_BIAS_BYTES);

    header->hmx_context_id = session->hmx_context_id;
    header->vtcm_address = (uint32_t)(uintptr_t)session->vtcm;
    header->hmx_lock_status = HAP_compute_res_hmx_lock2(
        session->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    if (header->hmx_lock_status != AEE_SUCCESS) {
        header->dsp_status = QBH_HMX_CONVERT_STATUS_HMX_LOCK_FAILED;
        result = AEE_EFAILED;
        goto publish;
    }
    header->hmx_ticks = HAP_perf_get_qtimer_count();
    qbh_execute_u8s8_tile(activation, weight, bias, output);
    header->hmx_ticks = HAP_perf_get_qtimer_count() - header->hmx_ticks;
    header->hmx_unlock_status = HAP_compute_res_hmx_unlock2(
        session->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    if (header->hmx_unlock_status != AEE_SUCCESS) {
        header->dsp_status = QBH_HMX_CONVERT_STATUS_HMX_UNLOCK_FAILED;
        result = AEE_EFAILED;
        goto publish;
    }
    memcpy(shared + header->output_offset, output,
           QBH_HMX_OUTPUT_BYTES);
    header->dsp_status = QBH_HMX_CONVERT_STATUS_OK;
    result = AEE_SUCCESS;

publish:
    if (header != NULL) {
        cache_status = qurt_mem_cache_clean(
            (qurt_addr_t)shared, (qurt_size_t)shared_bytes,
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        if (cache_status != 0 && result == AEE_SUCCESS) {
            header->dsp_status = QBH_HMX_CONVERT_STATUS_CACHE_FAILED;
            header->cache_status = cache_status;
            result = AEE_EFAILED;
        }
    }
    (void)HAP_mmap_put(shared_fd);
    return result;
}

static int header_is_valid(const struct qbh_probe_header *header,
                           uint32_t shared_bytes,
                           struct qbh_projection_layout *layout) {
    if (header->magic != QBH_PROBE_MAGIC ||
        header->abi_version != QBH_PROBE_ABI_VERSION ||
        header->header_bytes != sizeof(*header) ||
        header->total_bytes > shared_bytes ||
        header->pattern < QBH_PATTERN_IDENTITY ||
        header->pattern > QBH_PATTERN_BOUNDARY ||
        header->input_zero_point > UINT8_MAX ||
        header->repeat_count == 0 ||
        header->repeat_count > QBH_HMX_MAX_REPEATS ||
        (header->output_assembly_mode != QBH_OUTPUT_ASSEMBLY_SCALAR &&
         header->output_assembly_mode !=
             QBH_OUTPUT_ASSEMBLY_LINKED_2D_DMA) ||
        (header->resource_lifetime_mode !=
             QBH_RESOURCE_LIFETIME_TRANSIENT &&
         header->resource_lifetime_mode !=
             QBH_RESOURCE_LIFETIME_PREPARED_SESSION)) {
        return 0;
    }
    if (qbh_projection_layout_init(header->projection_variant,
                                   header->weight_storage_variant,
                                   header->physical_plan,
                                   header->compressed_slot_count,
                                   header->chunk_tiles,
                                   layout) != 0) {
        return 0;
    }
    if (header->expanded_chunk_slot_count !=
        layout->expanded_slot_count) {
        return 0;
    }
    if (header->requested_hvx_workers == 0U ||
        header->requested_hvx_workers > QBH_MAX_HVX_WORKERS ||
        (qbh_physical_plan_is_full_bundle(header->physical_plan) &&
         header->requested_hvx_workers != 1U)) {
        return 0;
    }
    return range_is_valid(header->activation_offset,
                          layout->activation_bytes,
                          header->total_bytes) &&
           range_is_valid(header->weight_offset,
                          layout->stored_weight_bytes,
                          header->total_bytes) &&
           range_is_valid(header->output_offset, layout->output_bytes,
                          header->total_bytes);
}

static int vtcm_layout_is_aligned(
    const uint8_t *vtcm, const struct qbh_projection_layout *layout) {
    uintptr_t base = (uintptr_t)vtcm;
    return (base & UINT32_C(2047)) == 0 &&
           ((base + layout->vtcm_compressed_slot0_offset) &
            UINT32_C(255)) ==
               0 &&
           ((base + layout->vtcm_compressed_slot1_offset) &
            UINT32_C(255)) ==
               0 &&
           ((base + qbh_projection_compressed_slot_offset(
                        layout, layout->compressed_slot_count - 1U)) &
            UINT32_C(255)) == 0 &&
           ((base + layout->vtcm_expanded_slot0_offset) &
            UINT32_C(255)) == 0 &&
           ((base + layout->vtcm_expanded_slot1_offset) &
            UINT32_C(255)) == 0 &&
           ((base + layout->vtcm_chunked_expanded_slots_offset) &
            UINT32_C(255)) == 0 &&
           ((base + qbh_projection_expanded_chunk_offset(
                        layout,
                        layout->expanded_slot_count - 1U)) &
            UINT32_C(255)) == 0 &&
           ((base + layout->vtcm_output_offset) & UINT32_C(2047)) == 0 &&
           layout->vtcm_plan_bytes <= QBH_W4U8_VTCM_BYTES;
}

static int record_dma_wait(struct qbh_probe_header *header) {
    int status;
    ++header->dma_wait_count;
    status = qbh_dma_wait_idle();
    if (status != 0 && header->dma_status == 0) {
        header->dma_status = status;
    }
    return status;
}

static int stage_activation_tiles(struct qbh_probe_header *header,
                                  const struct qbh_projection_layout *layout,
                                  const uint8_t *source,
                                  uint8_t *destination) {
    struct qbh_dma_desc_2d descriptor __attribute__((aligned(64)));
    uint64_t start = HAP_perf_get_qtimer_count();

    for (uint32_t input_tile = 0; input_tile < layout->k_tiles;
         ++input_tile) {
        int status;
        memset(&descriptor, 0, sizeof(descriptor));
        if (record_dma_wait(header) != 0) {
            return -1;
        }
        descriptor.next = 0;
        descriptor.length = 0;
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
        descriptor.src_width_offset = 0;
        descriptor.dst_width_offset = 0;

        status = qbh_dma_start(&descriptor);
        ++header->dma_submit_count;
        ++header->dma_descriptor_count;
        if (status != 0) {
            header->dma_status = status;
            return -1;
        }
        if (record_dma_wait(header) != 0) {
            return -1;
        }
        ++header->dma_descriptor_completion_count;
        asm volatile("barrier" : : : "memory");
        ++header->activation_stage_count;
    }
    header->activation_stage_ticks =
        HAP_perf_get_qtimer_count() - start;
    return 0;
}

static int stage_weight_bundles(struct qbh_probe_header *header,
                                const uint8_t *source,
                                uint8_t *destination,
                                uint32_t bundle_bytes,
                                uint32_t bundle_count) {
    struct qbh_dma_desc_1d descriptor __attribute__((aligned(64)));
    int status;

    memset(&descriptor, 0, sizeof(descriptor));
    if (record_dma_wait(header) != 0) {
        return -1;
    }
    descriptor.next = 0;
    descriptor.length = bundle_bytes * bundle_count;
    descriptor.type = QBH_DMA_TYPE_1D;
    descriptor.src_bypass = 1;
    descriptor.dst_bypass = 0;
    descriptor.ordered = 1;
    descriptor.dstate = QBH_DMA_DESC_PENDING;
    descriptor.src = (uint32_t)(uintptr_t)source;
    descriptor.dst = (uint32_t)(uintptr_t)destination;

    status = qbh_dma_start(&descriptor);
    ++header->dma_submit_count;
    ++header->dma_descriptor_count;
    if (status != 0) {
        header->dma_status = status;
        return -1;
    }
    if (record_dma_wait(header) != 0) {
        return -1;
    }
    ++header->dma_descriptor_completion_count;
    asm volatile("barrier" : : : "memory");
    header->weight_bundle_stage_count += bundle_count;
    return 0;
}

static int start_linked_weight_bundles(
    struct qbh_probe_header *header,
    struct qbh_dma_aligned_desc_1d *descriptors,
    const uint8_t *source, uint8_t *destination,
    uint32_t bundle_bytes, uint32_t bundle_count) {
    if (bundle_count < 2U || bundle_count > 4U ||
        record_dma_wait(header) != 0) {
        return -1;
    }
    memset(descriptors, 0,
           sizeof(*descriptors) * bundle_count);
    for (uint32_t index = 0; index < bundle_count; ++index) {
        struct qbh_dma_desc_1d *descriptor =
            &descriptors[index].descriptor;
        descriptor->next = index + 1U < bundle_count
                               ? (uint32_t)(uintptr_t)(
                                     &descriptors[index + 1U].descriptor)
                               : 0U;
        descriptor->length = bundle_bytes;
        descriptor->type = QBH_DMA_TYPE_1D;
        descriptor->src_bypass = 1;
        descriptor->dst_bypass = 0;
        descriptor->ordered = 1;
        descriptor->dstate = QBH_DMA_DESC_PENDING;
        descriptor->src = (uint32_t)(uintptr_t)(
            source + (size_t)index * bundle_bytes);
        descriptor->dst = (uint32_t)(uintptr_t)(
            destination + (size_t)index * bundle_bytes);
        asm volatile("release(%0):at"
                     :
                     : "r"(descriptor)
                     : "memory");
    }
    if (qbh_dma_start(&descriptors[0].descriptor) != 0) {
        return -1;
    }
    ++header->dma_submit_count;
    header->dma_descriptor_count += bundle_count;
    ++header->dma_chain_count;
    return 0;
}

static int wait_linked_descriptor(
    struct qbh_probe_header *header,
    struct qbh_dma_desc_1d *descriptor) {
    uint64_t start = HAP_perf_get_qtimer_count();
    uint32_t spins = 0;

    ++header->dma_wait_count;
    for (;;) {
        uint32_t status = Q6_R_dmpoll() & QBH_DMA_STATUS_MASK;
        if (((volatile struct qbh_dma_desc_1d *)descriptor)->dstate ==
            QBH_DMA_DESC_COMPLETE) {
            asm volatile("barrier" : : : "memory");
            ++header->dma_descriptor_completion_count;
            return 0;
        }
        if (status == QBH_DMA_STATUS_ERROR) {
            header->dma_status = (int32_t)status;
            return -1;
        }
        ++spins;
        if ((spins & UINT32_C(255)) == 0U &&
            HAP_perf_get_qtimer_count() - start >
                QBH_DMA_DESCRIPTOR_TIMEOUT_TICKS) {
            ++header->dma_descriptor_timeout_count;
            header->dma_status = -1;
            return -1;
        }
    }
}

__attribute__((noinline)) static void assemble_row_major_output(
    const struct qbh_projection_layout *layout, uint8_t *destination,
    const uint8_t *source_tiles) {
    for (uint32_t row = 0; row < layout->m; ++row) {
        for (uint32_t output_tile = 0;
             output_tile < layout->n_tiles; ++output_tile) {
            memcpy(destination + qbh_projection_output_offset(
                                     layout, row,
                                     output_tile *
                                         QBH_HMX_OUTPUT_CHANNELS),
                   source_tiles +
                       (size_t)output_tile * QBH_HMX_OUTPUT_BYTES +
                       (size_t)row * QBH_HMX_OUTPUT_CHANNELS,
                   QBH_HMX_OUTPUT_CHANNELS);
        }
    }
    asm volatile("barrier" : : : "memory");
}

static int assemble_row_major_output_dma(
    struct qbh_probe_header *header,
    const struct qbh_projection_layout *layout, uint8_t *destination,
    const uint8_t *source_tiles) {
    uint64_t start = HAP_perf_get_qtimer_count();
    uint32_t spins = 0;

    if (layout->n_tiles >
            QBH_MAX_PROJECTION_N / QBH_HMX_OUTPUT_CHANNELS ||
        qbh_dma_wait_idle() != 0) {
        header->output_dma_status = -1;
        return -1;
    }
    ++header->output_dma_wait_count;
    memset(output_dma_descriptors, 0,
           sizeof(*output_dma_descriptors) * layout->n_tiles);
    for (uint32_t output_tile = 0; output_tile < layout->n_tiles;
         ++output_tile) {
        struct qbh_dma_desc_2d *descriptor =
            &output_dma_descriptors[output_tile].descriptor;
        descriptor->next = output_tile + 1U < layout->n_tiles
                               ? (uint32_t)(uintptr_t)(
                                     &output_dma_descriptors[
                                          output_tile + 1U].descriptor)
                               : 0U;
        descriptor->length = 0;
        descriptor->type = QBH_DMA_TYPE_2D;
        descriptor->src_bypass = 0;
        descriptor->dst_bypass = 1;
        descriptor->ordered = 1;
        descriptor->dstate = QBH_DMA_DESC_PENDING;
        descriptor->src = (uint32_t)(uintptr_t)(
            source_tiles + (size_t)output_tile * QBH_HMX_OUTPUT_BYTES);
        descriptor->dst = (uint32_t)(uintptr_t)(
            destination +
            (size_t)output_tile * QBH_HMX_OUTPUT_CHANNELS);
        descriptor->roi_width = (uint16_t)QBH_HMX_OUTPUT_CHANNELS;
        descriptor->roi_height = (uint16_t)layout->m;
        descriptor->src_stride = (uint16_t)QBH_HMX_OUTPUT_CHANNELS;
        descriptor->dst_stride = (uint16_t)layout->n;
        descriptor->src_width_offset = 0;
        descriptor->dst_width_offset = 0;
        asm volatile("release(%0):at"
                     :
                     : "r"(descriptor)
                     : "memory");
    }

    if (qbh_dma_start(&output_dma_descriptors[0].descriptor) != 0) {
        header->output_dma_status = -1;
        return -1;
    }
    ++header->output_dma_submit_count;
    header->output_dma_descriptor_count += layout->n_tiles;
    ++header->output_dma_chain_count;

    for (;;) {
        struct qbh_dma_desc_2d *last =
            &output_dma_descriptors[layout->n_tiles - 1U].descriptor;
        uint32_t status = Q6_R_dmpoll() & QBH_DMA_STATUS_MASK;
        if (((volatile struct qbh_dma_desc_2d *)last)->dstate ==
            QBH_DMA_DESC_COMPLETE) {
            asm volatile("barrier" : : : "memory");
            header->output_dma_descriptor_completion_count +=
                layout->n_tiles;
            break;
        }
        if (status == QBH_DMA_STATUS_ERROR) {
            header->output_dma_status = (int32_t)status;
            return -1;
        }
        ++spins;
        if ((spins & UINT32_C(255)) == 0U &&
            HAP_perf_get_qtimer_count() - start >
                QBH_DMA_DESCRIPTOR_TIMEOUT_TICKS) {
            ++header->output_dma_descriptor_timeout_count;
            header->output_dma_status = -1;
            return -1;
        }
    }
    if (qbh_dma_wait_idle() != 0) {
        header->output_dma_status = -1;
        return -1;
    }
    ++header->output_dma_wait_count;
    return 0;
}

static void hmx_worker_main(void *opaque) {
    struct qbh_projection_worker_job *job =
        (struct qbh_projection_worker_job *)opaque;
    int exit_status = AEE_SUCCESS;

    job->lock_status = HAP_compute_res_hmx_lock2(
        job->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    (void)qurt_sem_up(job->started);
    if (job->lock_status != AEE_SUCCESS) {
        qurt_thread_exit(job->lock_status);
    }

    for (uint32_t repeat = 0; repeat < job->repeat_count; ++repeat) {
        for (uint32_t output_tile = 0;
             output_tile < job->n_tiles; ++output_tile) {
            uint32_t linear_tile = repeat * job->n_tiles + output_tile;
            uint32_t slot = linear_tile & 1U;
            const uint32_t *bias_words;
            uint64_t wait_start = HAP_perf_get_qtimer_count();
            qurt_sem_down(job->ready[slot]);
            job->ready_wait_ticks +=
                HAP_perf_get_qtimer_count() - wait_start;
            if (job->abort_status != 0) {
                exit_status = job->abort_status;
                goto unlock;
            }

            bias_words = job->bias_slots[slot];
            uint64_t core_start = HAP_perf_get_qtimer_count();
            qbh_hmx_begin_u8s8_output(bias_words);
            job->hmx_stream_count += qbh_hmx_accumulate_u8s8_projection(
                job->activation_tiles,
                (const int8_t *)job->weight_slots[slot],
                job->k_tiles);
            job->hmx_execution_count += job->k_tiles;
            qbh_hmx_store_u8_output(
                job->output_tiles +
                (size_t)output_tile * QBH_HMX_OUTPUT_BYTES);
            job->hmx_compute_ticks +=
                HAP_perf_get_qtimer_count() - core_start;
            ++job->output_tile_count;

            (void)qurt_sem_up(job->free_slot[slot]);
        }
    }

unlock:
    job->unlock_status = HAP_compute_res_hmx_unlock2(
        job->hmx_context_id, HAP_COMPUTE_RES_HMX_SHARED);
    if (exit_status == AEE_SUCCESS && job->unlock_status != AEE_SUCCESS) {
        exit_status = job->unlock_status;
    }
    qurt_thread_exit(exit_status);
}

AEEResult qwen3_probe_run(remote_handle64 handle, int32 shared_fd,
                          uint32 shared_bytes) {
    struct qbh_probe_session *session = qbh_session_from_handle(handle);
    struct qbh_probe_header *header = NULL;
    struct qbh_projection_layout layout;
    compute_res_attr_t vtcm_attributes;
    compute_res_attr_t hmx_attributes;
    uint8_t *shared = NULL;
    uint8_t *vtcm = NULL;
    uint8_t *activation_tiles;
    uint8_t *compressed_slots[2];
    uint8_t *expanded_slots[2];
    uint8_t *output_tiles;
    uint32_t vtcm_context_id = 0;
    uint32_t hmx_context_id = 0;
    struct qbh_projection_worker_job hmx_job;
    qurt_thread_attr_t hmx_thread_attributes;
    qurt_thread_t hmx_thread;
    qurt_sem_t ready[2];
    qurt_sem_t free_slot[2];
    qurt_sem_t worker_started;
    int semaphores_initialized = 0;
    int hmx_thread_created = 0;
    int hmx_thread_joined = 0;
    int hmx_thread_exit_status = 0;
    HAP_power_request_t hmx_power_request;
    HAP_power_request_t dcvs_power_request;
    int hmx_power_context = 0;
    int hmx_powered = 0;
    int dcvs_powered = 0;
    int resources_from_session = 0;
    int hvx_locked = 0;
    int cache_result;
    int result = AEE_SUCCESS;
    int layout_valid = 0;
    uint64_t dsp_total_start = 0;
    uint64_t input_cache_start = 0;
    uint64_t input_cache_ticks = 0;

    if (session == NULL) {
        return AEE_EBADPARM;
    }

    result = HAP_mmap_get(shared_fd, (void **)&shared, NULL);
    if (result != AEE_SUCCESS || shared == NULL) {
        FARF(ERROR, "HAP_mmap_get failed: %d", result);
        return result != AEE_SUCCESS ? result : AEE_EFAILED;
    }
    if (shared_bytes < sizeof(struct qbh_probe_header)) {
        (void)HAP_mmap_put(shared_fd);
        return AEE_EBADSIZE;
    }

    header = (struct qbh_probe_header *)shared;
    input_cache_start = HAP_perf_get_qtimer_count();
    cache_result = qurt_mem_cache_clean((qurt_addr_t)header,
                                        (qurt_size_t)sizeof(*header),
                                        QURT_MEM_CACHE_INVALIDATE,
                                        QURT_MEM_DCACHE);
    if (cache_result != 0) {
        header->dsp_status = QBH_PROBE_STATUS_CACHE_INVALIDATE_FAILED;
        header->cache_status = cache_result;
        result = AEE_EFAILED;
        goto cleanup;
    }
    if (!header_is_valid(header, shared_bytes, &layout)) {
        header->dsp_status = QBH_PROBE_STATUS_BAD_HEADER;
        result = AEE_EBADPARM;
        goto cleanup;
    }
    layout_valid = 1;
    cache_result = qurt_mem_cache_clean(
        (qurt_addr_t)(shared + header->activation_offset),
        (qurt_size_t)layout.activation_bytes,
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (cache_result == 0) {
        cache_result = qurt_mem_cache_clean(
            (qurt_addr_t)(shared + header->weight_offset),
            (qurt_size_t)layout.stored_weight_bytes,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    input_cache_ticks =
        HAP_perf_get_qtimer_count() - input_cache_start;
    if (cache_result != 0) {
        header->dsp_status = QBH_PROBE_STATUS_CACHE_INVALIDATE_FAILED;
        header->cache_status = cache_result;
        result = AEE_EFAILED;
        goto cleanup;
    }
    resources_from_session =
        header->resource_lifetime_mode ==
        QBH_RESOURCE_LIFETIME_PREPARED_SESSION;
    if ((resources_from_session && session->prepared == 0U) ||
        (!resources_from_session && session->prepared != 0U)) {
        header->dsp_status = QBH_PROBE_STATUS_HMX_CONFIG_FAILED;
        result = AEE_EBADSTATE;
        goto cleanup;
    }

    header->dsp_status = QBH_PROBE_STATUS_DSP_RUNNING;
    header->cache_status = 0;
    header->vtcm_requested_bytes = QBH_W4U8_VTCM_BYTES;
    header->vtcm_acquired_bytes = 0;
    header->hmx_resource_status = 0;
    header->hmx_lock_status = 0;
    header->hmx_unlock_status = 0;
    header->hmx_release_status = 0;
    header->hmx_thread_create_status = 0;
    header->hmx_thread_join_status = 0;
    header->hmx_power_up_status = 0;
    header->hmx_power_down_status = 0;
    header->dcvs_power_setup_status = 0;
    header->dcvs_power_reset_status = 0;
    header->hmx_execution_count = 0;
    header->hmx_stream_count = 0;
    header->hvx_lock_status = 0;
    header->hvx_unlock_status = 0;
    header->projection_m = layout.m;
    header->projection_k = layout.k;
    header->projection_n = layout.n;
    header->k_tile_count = layout.k_tiles;
    header->n_tile_count = layout.n_tiles;
    header->expanded_chunk_slot_count = layout.expanded_slot_count;
    header->stored_weight_bundle_bytes =
        layout.stored_weight_bundle_bytes;
    header->expanded_weight_bundle_bytes =
        layout.expanded_weight_bundle_bytes;
    header->vtcm_plan_bytes = layout.vtcm_plan_bytes;
    header->k_streams_per_output = layout.k_streams_per_output;
    header->stored_weight_bytes_per_repeat = layout.stored_weight_bytes;
    header->expanded_weight_bytes_per_repeat =
        layout.expanded_weight_bytes;
    header->weight_expand_count = 0;
    header->activation_stage_count = 0;
    header->weight_bundle_stage_count = 0;
    header->output_tile_count = 0;
    header->dma_submit_count = 0;
    header->dma_wait_count = 0;
    header->dma_descriptor_count = 0;
    header->dma_chain_count = 0;
    header->dma_descriptor_completion_count = 0;
    header->dma_descriptor_timeout_count = 0;
    header->output_dma_submit_count = 0;
    header->output_dma_wait_count = 0;
    header->output_dma_descriptor_count = 0;
    header->output_dma_chain_count = 0;
    header->output_dma_descriptor_completion_count = 0;
    header->output_dma_descriptor_timeout_count = 0;
    header->output_dma_status = 0;
    header->resource_setup_in_run = 0;
    header->resource_release_in_run = 0;
    header->prepared_session_run_index = 0;
    header->resource_vtcm_address = 0;
    header->resource_hmx_context_id = 0;
    header->weight_slot_reuse_count = 0;
    header->expanded_chunk_slot_reuse_count = 0;
    header->chunks_per_output = layout.chunks_per_output;
    header->chunk_expand_count = 0;
    header->dma_status = 0;
    header->sync_status = 0;
    header->hvx_units_128b =
        ((uint32_t)qurt_hvx_get_units() >> 8U) & UINT32_C(0xff);
    header->hvx_workers_created = 0;
    header->hvx_workers_locked = 0;
    header->hvx_max_active_workers = 0;
    header->hvx_hmx_overlap_observed = 0;
    header->hvx_parallel_overlap_observed = 0;
    header->hvx_thread_create_status = 0;
    header->hvx_thread_join_status = 0;
    for (uint32_t worker = 0; worker < QBH_MAX_HVX_WORKERS; ++worker) {
        header->hvx_worker_lock_status[worker] = 0;
        header->hvx_worker_unlock_status[worker] = 0;
        header->hvx_worker_ticks[worker] = 0;
    }
    header->qtimer_start = 0;
    header->qtimer_end = 0;
    header->qtimer_elapsed = 0;
    header->pcycles_start = 0;
    header->pcycles_end = 0;
    header->activation_stage_ticks = 0;
    header->weight_stage_ticks = 0;
    header->weight_expand_ticks = 0;
    header->hmx_compute_ticks = 0;
    header->hmx_ready_wait_ticks = 0;
    header->producer_slot_wait_ticks = 0;
    header->expanded_slot_wait_ticks = 0;
    header->pipeline_ticks = 0;
    header->output_assembly_ticks = 0;
    header->dsp_total_ticks = 0;
    header->input_cache_ticks = input_cache_ticks;
    header->output_cache_ticks = 0;
    header->streaming_region_publish_count = 0U;
    header->streaming_ready_timeout_count = 0U;
    header->expand_window_start = 0;
    header->expand_window_end = 0;
    header->hmx_window_start = 0;
    header->hmx_window_end = 0;
    dsp_total_start = HAP_perf_get_qtimer_count();
    FARF(ALWAYS,
         "EXP0019 stage=header_valid projection=%u storage=%u plan=%u "
         "workers=%u compressed_slots=%u expanded_slots=%u chunk_tiles=%u "
         "output_mode=%u resource_mode=%u "
         "M=%u K=%u N=%u vtcm_plan=%u",
         layout.variant, layout.weight_storage_variant,
         layout.physical_plan, header->requested_hvx_workers,
         layout.compressed_slot_count, layout.expanded_slot_count,
         layout.chunk_tiles, header->output_assembly_mode,
         header->resource_lifetime_mode, layout.m, layout.k, layout.n,
         layout.vtcm_plan_bytes);

    if (resources_from_session) {
        vtcm_context_id = session->vtcm_context_id;
        hmx_context_id = session->hmx_context_id;
        vtcm = session->vtcm;
        header->vtcm_acquired_bytes = QBH_W4U8_VTCM_BYTES;
        header->resource_vtcm_address = (uint32_t)(uintptr_t)vtcm;
        header->resource_hmx_context_id = hmx_context_id;
        header->prepared_session_run_index =
            ++session->prepared_run_count;
    } else {
        header->resource_setup_in_run = 1U;
        memset(&dcvs_power_request, 0, sizeof(dcvs_power_request));
        dcvs_power_request.type = HAP_power_set_DCVS_v3;
        dcvs_power_request.dcvs_v3.dcvs_enable = 1;
        dcvs_power_request.dcvs_v3.dcvs_option =
            HAP_DCVS_V2_PERFORMANCE_MODE;
        dcvs_power_request.dcvs_v3.set_latency = 1;
        dcvs_power_request.dcvs_v3.latency = 100;
        dcvs_power_request.dcvs_v3.set_core_params = 1;
        dcvs_power_request.dcvs_v3.core_params.min_corner =
            HAP_DCVS_VCORNER_NOM;
        dcvs_power_request.dcvs_v3.core_params.max_corner =
            HAP_DCVS_VCORNER_TURBO_L3;
        dcvs_power_request.dcvs_v3.core_params.target_corner =
            HAP_DCVS_VCORNER_TURBO_L3;
        dcvs_power_request.dcvs_v3.set_bus_params = 1;
        dcvs_power_request.dcvs_v3.bus_params.min_corner =
            HAP_DCVS_VCORNER_NOM;
        dcvs_power_request.dcvs_v3.bus_params.max_corner =
            HAP_DCVS_VCORNER_TURBO_L3;
        dcvs_power_request.dcvs_v3.bus_params.target_corner =
            HAP_DCVS_VCORNER_TURBO_L3;
        header->dcvs_power_setup_status = HAP_power_set(
            &hmx_power_context, &dcvs_power_request);
        if (header->dcvs_power_setup_status != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_DCVS_POWER_FAILED;
            result = AEE_EFAILED;
            goto cleanup;
        }
        dcvs_powered = 1;

        memset(&hmx_power_request, 0, sizeof(hmx_power_request));
        hmx_power_request.type = HAP_power_set_HMX;
        hmx_power_request.hmx.power_up = 1;
        header->hmx_power_up_status = HAP_power_set(
            &hmx_power_context, &hmx_power_request);
        if (header->hmx_power_up_status != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_HMX_POWER_FAILED;
            result = AEE_EFAILED;
            goto cleanup;
        }
        hmx_powered = 1;

        result = HAP_compute_res_attr_init(&vtcm_attributes);
        if (result != AEE_SUCCESS ||
            HAP_compute_res_attr_set_serialize(&vtcm_attributes, 1) !=
                AEE_SUCCESS ||
            HAP_compute_res_attr_set_vtcm_param(
                &vtcm_attributes, QBH_W4U8_VTCM_BYTES, 1) !=
                AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_VTCM_CONFIG_FAILED;
            result = AEE_EFAILED;
            goto cleanup;
        }
        vtcm_context_id = HAP_compute_res_acquire(
            &vtcm_attributes, 100000);
        if (vtcm_context_id == 0) {
            header->dsp_status = QBH_PROBE_STATUS_VTCM_ACQUIRE_FAILED;
            result = AEE_ERESOURCENOTFOUND;
            goto cleanup;
        }
        vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(
            &vtcm_attributes);
        if (vtcm == NULL) {
            header->dsp_status = QBH_PROBE_STATUS_VTCM_POINTER_FAILED;
            result = AEE_EFAILED;
            goto cleanup;
        }
        header->vtcm_acquired_bytes = QBH_W4U8_VTCM_BYTES;
        header->resource_vtcm_address = (uint32_t)(uintptr_t)vtcm;

        result = HAP_compute_res_attr_init(&hmx_attributes);
        if (result != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_HMX_CONFIG_FAILED;
            header->hmx_resource_status = result;
            goto cleanup;
        }
        result = HAP_compute_res_attr_set_hmx_param(&hmx_attributes, 1);
        header->hmx_resource_status = result;
        if (result != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_HMX_CONFIG_FAILED;
            goto cleanup;
        }
        hmx_context_id = HAP_compute_res_acquire(
            &hmx_attributes, 100000);
        if (hmx_context_id == 0) {
            header->dsp_status = QBH_PROBE_STATUS_HMX_CONFIG_FAILED;
            header->hmx_resource_status = AEE_ERESOURCENOTFOUND;
            result = AEE_ERESOURCENOTFOUND;
            goto cleanup;
        }
        header->resource_hmx_context_id = hmx_context_id;
    }

    if (!vtcm_layout_is_aligned(vtcm, &layout)) {
        header->dsp_status = QBH_PROBE_STATUS_VTCM_ALIGNMENT_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }

    activation_tiles = vtcm + layout.vtcm_activation_offset;
    compressed_slots[0] =
        vtcm + layout.vtcm_compressed_slot0_offset;
    compressed_slots[1] =
        vtcm + layout.vtcm_compressed_slot1_offset;
    expanded_slots[0] = vtcm + layout.vtcm_expanded_slot0_offset;
    expanded_slots[1] = vtcm + layout.vtcm_expanded_slot1_offset;
    output_tiles = vtcm + layout.vtcm_output_offset;

    if (stage_activation_tiles(header, &layout,
                               shared + header->activation_offset,
                               activation_tiles) != 0) {
        header->dsp_status = QBH_PROBE_STATUS_DMA_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }

    if (qbh_physical_plan_is_chunked(layout.physical_plan)) {
        result = qbh_run_chunked_w4_pipeline(
            header, &layout, shared + header->weight_offset,
            activation_tiles, vtcm, hmx_context_id);
        if (result != AEE_SUCCESS) {
            if (header->dsp_status == QBH_PROBE_STATUS_DSP_RUNNING) {
                header->dsp_status =
                    header->dma_status != 0
                        ? QBH_PROBE_STATUS_DMA_FAILED
                        : (header->hmx_lock_status != AEE_SUCCESS
                               ? QBH_PROBE_STATUS_HMX_LOCK_FAILED
                               : (header->hvx_lock_status != AEE_SUCCESS
                                      ? QBH_PROBE_STATUS_HVX_LOCK_FAILED
                                      : QBH_PROBE_STATUS_SYNC_FAILED));
            }
            goto cleanup;
        }
        goto pipeline_complete;
    }

    header->hvx_lock_status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (header->hvx_lock_status != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_HVX_LOCK_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }
    hvx_locked = 1;

    qurt_sem_init_val(&ready[0], 0);
    qurt_sem_init_val(&ready[1], 0);
    qurt_sem_init_val(&free_slot[0], 1);
    qurt_sem_init_val(&free_slot[1], 1);
    qurt_sem_init_val(&worker_started, 0);
    semaphores_initialized = 1;

    memset(&hmx_job, 0, sizeof(hmx_job));
    hmx_job.hmx_context_id = hmx_context_id;
    hmx_job.activation_tiles = activation_tiles;
    hmx_job.weight_slots[0] = expanded_slots[0];
    hmx_job.weight_slots[1] = expanded_slots[1];
    if (qbh_weight_storage_is_packed_w4(
            layout.weight_storage_variant)) {
        hmx_job.bias_slots[0] = (const uint32_t *)(
            compressed_slots[0] + layout.w4_bias_offset);
        hmx_job.bias_slots[1] = (const uint32_t *)(
            compressed_slots[1] + layout.w4_bias_offset);
    } else {
        hmx_job.bias_slots[0] = (const uint32_t *)(
            expanded_slots[0] + layout.expanded_weight_chunk_bytes);
        hmx_job.bias_slots[1] = (const uint32_t *)(
            expanded_slots[1] + layout.expanded_weight_chunk_bytes);
    }
    hmx_job.output_tiles = output_tiles;
    hmx_job.repeat_count = header->repeat_count;
    hmx_job.k_tiles = layout.k_tiles;
    hmx_job.n_tiles = layout.n_tiles;
    hmx_job.ready[0] = &ready[0];
    hmx_job.ready[1] = &ready[1];
    hmx_job.free_slot[0] = &free_slot[0];
    hmx_job.free_slot[1] = &free_slot[1];
    hmx_job.started = &worker_started;

    qurt_thread_attr_init(&hmx_thread_attributes);
    qurt_thread_attr_set_name(&hmx_thread_attributes, "qbh-hmx-proj");
    qurt_thread_attr_set_stack_addr(&hmx_thread_attributes,
                                    hmx_worker_stack);
    qurt_thread_attr_set_stack_size(&hmx_thread_attributes,
                                    QBH_HMX_WORKER_STACK_BYTES);
    qurt_thread_attr_set_priority(
        &hmx_thread_attributes,
        qurt_thread_get_priority(qurt_thread_get_id()));
    qurt_thread_attr_set_detachstate(&hmx_thread_attributes,
                                     QURT_THREAD_ATTR_CREATE_JOINABLE);

    header->hmx_thread_create_status = qurt_thread_create(
        &hmx_thread, &hmx_thread_attributes, hmx_worker_main, &hmx_job);
    if (header->hmx_thread_create_status != QURT_EOK) {
        header->dsp_status = QBH_PROBE_STATUS_HMX_THREAD_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }
    hmx_thread_created = 1;

    qurt_sem_down(&worker_started);
    if (hmx_job.lock_status != AEE_SUCCESS) {
        header->hmx_lock_status = hmx_job.lock_status;
        header->dsp_status = QBH_PROBE_STATUS_HMX_LOCK_FAILED;
        result = AEE_EFAILED;
        goto join_worker;
    }

    header->qtimer_start = HAP_perf_get_qtimer_count();
    header->pcycles_start = HAP_perf_get_pcycles();
    for (uint32_t repeat = 0; repeat < header->repeat_count; ++repeat) {
        uint32_t dma_bundle_batch =
            qbh_physical_plan_dma_bundle_batch(layout.physical_plan);
        for (uint32_t output_base = 0;
             output_base < layout.n_tiles;
             output_base += dma_bundle_batch) {
            struct qbh_dma_aligned_desc_1d linked_descriptors[4];
            uint32_t linear_base =
                repeat * layout.n_tiles + output_base;
            uint32_t first_slot = linear_base & 1U;

            for (uint32_t batch_index = 0;
                 batch_index < dma_bundle_batch; ++batch_index) {
                uint32_t linear_tile = linear_base + batch_index;
                uint32_t slot = first_slot + batch_index;
                uint64_t wait_start = HAP_perf_get_qtimer_count();
                qurt_sem_down(&free_slot[slot]);
                header->producer_slot_wait_ticks +=
                    HAP_perf_get_qtimer_count() - wait_start;
                if (linear_tile >= 2U) {
                    ++header->weight_slot_reuse_count;
                }
            }

            uint8_t *stage_destination =
                qbh_weight_storage_is_packed_w4(
                    layout.weight_storage_variant)
                    ? compressed_slots[first_slot]
                    : expanded_slots[first_slot];
            if (qbh_physical_plan_uses_linked_dma(
                    layout.physical_plan)) {
                if (start_linked_weight_bundles(
                        header, linked_descriptors,
                        shared + header->weight_offset +
                            (size_t)output_base *
                                layout.stored_weight_bundle_bytes,
                        stage_destination,
                        layout.stored_weight_bundle_bytes,
                        dma_bundle_batch) != 0) {
                    header->dsp_status = QBH_PROBE_STATUS_DMA_FAILED;
                    result = AEE_EFAILED;
                    goto abort_worker;
                }
                for (uint32_t batch_index = 0;
                     batch_index < dma_bundle_batch; ++batch_index) {
                    uint64_t wait_start = HAP_perf_get_qtimer_count();
                    if (wait_linked_descriptor(
                            header,
                            &linked_descriptors[batch_index].descriptor) !=
                        0) {
                        header->dsp_status = QBH_PROBE_STATUS_DMA_FAILED;
                        result = AEE_EFAILED;
                        goto abort_worker;
                    }
                    header->weight_stage_ticks +=
                        HAP_perf_get_qtimer_count() - wait_start;
                    ++header->weight_bundle_stage_count;
                    asm volatile("barrier" : : : "memory");
                    (void)qurt_sem_up(
                        &ready[first_slot + batch_index]);
                }
                if (record_dma_wait(header) != 0) {
                    header->dsp_status = QBH_PROBE_STATUS_DMA_FAILED;
                    result = AEE_EFAILED;
                    goto abort_worker;
                }
            } else {
                uint64_t stage_start = HAP_perf_get_qtimer_count();
                if (stage_weight_bundles(
                        header,
                        shared + header->weight_offset +
                            (size_t)output_base *
                                layout.stored_weight_bundle_bytes,
                        stage_destination,
                        layout.stored_weight_bundle_bytes,
                        dma_bundle_batch) != 0) {
                    header->dsp_status = QBH_PROBE_STATUS_DMA_FAILED;
                    result = AEE_EFAILED;
                    goto abort_worker;
                }
                header->weight_stage_ticks +=
                    HAP_perf_get_qtimer_count() - stage_start;

                for (uint32_t batch_index = 0;
                     batch_index < dma_bundle_batch; ++batch_index) {
                    uint32_t slot = first_slot + batch_index;
                    if (qbh_weight_storage_is_packed_w4(
                            layout.weight_storage_variant)) {
                        uint64_t expand_start =
                            HAP_perf_get_qtimer_count();
                        if (layout.weight_storage_variant ==
                            QBH_WEIGHT_PACKED_W4_HMX_SCALE) {
                            qbh_unpack_w4_to_s8_hvx(
                                compressed_slots[slot],
                                (int8_t *)expanded_slots[slot],
                                layout.k_tiles);
                        } else {
                            qbh_expand_w4_to_s8_hvx(
                                compressed_slots[slot],
                                compressed_slots[slot] +
                                    layout.w4_scale_offset,
                                (int8_t *)expanded_slots[slot],
                                layout.k_tiles);
                        }
                        header->weight_expand_ticks +=
                            HAP_perf_get_qtimer_count() - expand_start;
                        ++header->weight_expand_count;
                    }
                    asm volatile("barrier" : : : "memory");
                    (void)qurt_sem_up(&ready[slot]);
                }
            }
        }
    }
    goto join_worker;

abort_worker:
    hmx_job.abort_status = result != AEE_SUCCESS ? result : AEE_EFAILED;
    (void)qurt_sem_up(&ready[0]);
    (void)qurt_sem_up(&ready[1]);

join_worker:
    if (hmx_thread_created && !hmx_thread_joined) {
        header->hmx_thread_join_status = qurt_thread_join(
            hmx_thread, &hmx_thread_exit_status);
        hmx_thread_joined = 1;
    }
    header->pcycles_end = HAP_perf_get_pcycles();
    header->qtimer_end = HAP_perf_get_qtimer_count();
    header->pipeline_ticks = header->qtimer_end - header->qtimer_start;
    header->qtimer_elapsed = header->pipeline_ticks;
    header->hmx_lock_status = hmx_job.lock_status;
    header->hmx_unlock_status = hmx_job.unlock_status;
    header->hmx_execution_count = hmx_job.hmx_execution_count;
    header->hmx_stream_count = hmx_job.hmx_stream_count;
    header->output_tile_count = hmx_job.output_tile_count;
    header->hmx_compute_ticks = hmx_job.hmx_compute_ticks;
    header->hmx_ready_wait_ticks = hmx_job.ready_wait_ticks;
    if (header->sync_status == 0 && hmx_job.sync_status != 0) {
        header->sync_status = hmx_job.sync_status;
    }
    if (result != AEE_SUCCESS ||
        header->hmx_thread_join_status != QURT_EOK ||
        hmx_thread_exit_status != AEE_SUCCESS ||
        header->hmx_lock_status != AEE_SUCCESS ||
        header->hmx_unlock_status != AEE_SUCCESS ||
        header->sync_status != 0) {
        if (header->dsp_status == QBH_PROBE_STATUS_DSP_RUNNING) {
            header->dsp_status =
                header->hmx_lock_status != AEE_SUCCESS
                    ? QBH_PROBE_STATUS_HMX_LOCK_FAILED
                    : (header->sync_status != 0
                           ? QBH_PROBE_STATUS_SYNC_FAILED
                           : QBH_PROBE_STATUS_HMX_THREAD_FAILED);
        }
        result = AEE_EFAILED;
        goto cleanup;
    }

pipeline_complete:
    ;
    uint64_t output_start = HAP_perf_get_qtimer_count();
    if (header->output_assembly_mode ==
        QBH_OUTPUT_ASSEMBLY_LINKED_2D_DMA) {
        if (assemble_row_major_output_dma(
                header, &layout, shared + header->output_offset,
                output_tiles) != 0) {
            header->dsp_status = QBH_PROBE_STATUS_OUTPUT_DMA_FAILED;
            result = AEE_EFAILED;
            goto cleanup;
        }
    } else {
        assemble_row_major_output(&layout,
                                  shared + header->output_offset,
                                  output_tiles);
    }
    header->output_assembly_ticks =
        HAP_perf_get_qtimer_count() - output_start;
    header->hvx_unlock_status = qurt_hvx_unlock();
    hvx_locked = 0;
    if (header->hvx_unlock_status != AEE_SUCCESS) {
        header->dsp_status = QBH_PROBE_STATUS_HVX_LOCK_FAILED;
        result = AEE_EFAILED;
        goto cleanup;
    }

    header->dsp_status = QBH_PROBE_STATUS_OK;
    result = AEE_SUCCESS;

cleanup:
    if (header != NULL && !resources_from_session &&
        header->resource_setup_in_run != 0U) {
        header->resource_release_in_run = 1U;
    }
    if (hmx_thread_created && !hmx_thread_joined) {
        hmx_job.abort_status = AEE_EFAILED;
        if (semaphores_initialized) {
            (void)qurt_sem_up(&ready[0]);
            (void)qurt_sem_up(&ready[1]);
        }
        header->hmx_thread_join_status = qurt_thread_join(
            hmx_thread, &hmx_thread_exit_status);
        hmx_thread_joined = 1;
    }
    if (semaphores_initialized) {
        qurt_sem_destroy(&worker_started);
        qurt_sem_destroy(&ready[0]);
        qurt_sem_destroy(&ready[1]);
        qurt_sem_destroy(&free_slot[0]);
        qurt_sem_destroy(&free_slot[1]);
    }
    if (hvx_locked) {
        header->hvx_unlock_status = qurt_hvx_unlock();
        hvx_locked = 0;
        if (header->hvx_unlock_status != AEE_SUCCESS &&
            result == AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_HVX_LOCK_FAILED;
            result = AEE_EFAILED;
        }
    }
    if (!resources_from_session && hmx_context_id != 0) {
        header->hmx_release_status = HAP_compute_res_release(hmx_context_id);
        if (header->hmx_release_status != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_HMX_RELEASE_FAILED;
            if (result == AEE_SUCCESS) {
                result = header->hmx_release_status;
            }
        }
    }
    if (!resources_from_session && vtcm_context_id != 0) {
        int release_result = HAP_compute_res_release(vtcm_context_id);
        if (release_result != AEE_SUCCESS && result == AEE_SUCCESS) {
            result = release_result;
        }
    }
    if (!resources_from_session && hmx_powered) {
        memset(&hmx_power_request, 0, sizeof(hmx_power_request));
        hmx_power_request.type = HAP_power_set_HMX;
        hmx_power_request.hmx.power_up = 0;
        header->hmx_power_down_status = HAP_power_set(
            &hmx_power_context, &hmx_power_request);
        if (header->hmx_power_down_status != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_HMX_POWER_FAILED;
            if (result == AEE_SUCCESS) {
                result = AEE_EFAILED;
            }
        }
    }
    if (!resources_from_session && dcvs_powered) {
        HAP_power_set_dcvs_v3_init(&dcvs_power_request);
        header->dcvs_power_reset_status = HAP_power_set(
            &hmx_power_context, &dcvs_power_request);
        if (header->dcvs_power_reset_status != AEE_SUCCESS) {
            header->dsp_status = QBH_PROBE_STATUS_DCVS_POWER_FAILED;
            if (result == AEE_SUCCESS) {
                result = AEE_EFAILED;
            }
        }
    }

    if (dsp_total_start != 0) {
        header->dsp_total_ticks =
            HAP_perf_get_qtimer_count() - dsp_total_start;
    }
    uint64_t output_cache_start = HAP_perf_get_qtimer_count();
    cache_result = 0;
    if (layout_valid) {
        cache_result = qurt_mem_cache_clean(
            (qurt_addr_t)(shared + header->output_offset),
            (qurt_size_t)layout.output_bytes,
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
    if (header != NULL) {
        header->output_cache_ticks =
            HAP_perf_get_qtimer_count() - output_cache_start;
        int header_cache_result = qurt_mem_cache_clean(
            (qurt_addr_t)header, (qurt_size_t)sizeof(*header),
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
        if (cache_result == 0) {
            cache_result = header_cache_result;
        }
    }
    if (cache_result != 0 && header != NULL) {
        header->dsp_status = QBH_PROBE_STATUS_CACHE_FLUSH_FAILED;
        header->cache_status = cache_result;
        if (result == AEE_SUCCESS) {
            result = AEE_EFAILED;
        }
    }

    (void)HAP_mmap_put(shared_fd);
    return result;
}

AEEResult qwen3_probe_run_mlp(remote_handle64 handle, int32 shared_fd,
                              uint32 shared_bytes) {
    struct qbh_probe_session *session = qbh_session_from_handle(handle);
    if (session == NULL || session->prepared == 0U ||
        session->vtcm == NULL || session->hmx_context_id == 0U) {
        return AEE_EBADSTATE;
    }
    return qbh_run_mlp_rpc(shared_fd, shared_bytes, session->vtcm,
                           session->hmx_context_id,
                           ++session->prepared_run_count);
}

AEEResult qwen3_probe_run_attention(remote_handle64 handle,
                                    int32 shared_fd,
                                    uint32 shared_bytes) {
    struct qbh_probe_session *session = qbh_session_from_handle(handle);
    uint32_t run_index;
    if (session == NULL || session->prepared == 0U ||
        session->vtcm == NULL || session->hmx_context_id == 0U) {
        return AEE_EBADSTATE;
    }
    run_index = ++session->prepared_run_count;
    return qbh_run_attention_rpc(
        shared_fd, shared_bytes, session->vtcm,
        session->vtcm_granted_bytes, session->hmx_context_id,
        run_index);
}

AEEResult qwen3_probe_run_block(remote_handle64 handle, int32 shared_fd,
                                uint32 shared_bytes) {
    struct qbh_probe_session *session = qbh_session_from_handle(handle);
    uint32_t run_index;
    if (session == NULL || session->prepared == 0U ||
        session->vtcm == NULL || session->hmx_context_id == 0U) {
        return AEE_EBADPARM;
    }
    run_index = ++session->prepared_run_count;
    return qbh_run_block_rpc(shared_fd, shared_bytes, session->vtcm,
                             session->vtcm_granted_bytes,
                             session->hmx_context_id, run_index,
                             &session->block_persistent);
}
