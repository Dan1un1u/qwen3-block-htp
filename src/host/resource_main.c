#include <AEEStdErr.h>
#include <inttypes.h>
#include <limits.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "host/session.h"
#include "qwen3_probe.h"
#include "resource_protocol.h"

int main(void) {
    struct qbh_session session = {(remote_handle64)-1, 0};
    struct qbh_resource_header *header = NULL;
    int shared_fd = -1;
    int mapped = 0;
    int open_result;
    int prepare_result = AEE_EFAILED;
    int info_result = AEE_EFAILED;
    int release_result = AEE_EFAILED;
    int close_result = AEE_EFAILED;
    int exit_code = 1;

    if (sizeof(*header) > INT_MAX) {
        return 2;
    }
    header = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED,
                          (int)sizeof(*header));
    if (header == NULL) {
        fprintf(stderr, "rpcmem_alloc failed\n");
        return 2;
    }
    shared_fd = rpcmem_to_fd(header);
    if (shared_fd < 0) {
        fprintf(stderr, "rpcmem_to_fd failed\n");
        goto cleanup;
    }
    memset(header, 0, sizeof(*header));
    header->magic = QBH_RESOURCE_MAGIC;
    header->abi_version = QBH_RESOURCE_ABI_VERSION;
    header->experiment = QBH_RESOURCE_EXPERIMENT;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->dsp_status = QBH_RESOURCE_STATUS_HOST_READY;

    open_result = qbh_session_open(&session);
    if (open_result != AEE_SUCCESS) {
        goto cleanup;
    }
    open_result = fastrpc_mmap(CDSP_DOMAIN_ID, shared_fd, header, 0,
                               sizeof(*header), FASTRPC_MAP_FD);
    if (open_result != AEE_SUCCESS) {
        fprintf(stderr, "fastrpc_mmap failed: 0x%08x\n",
                (unsigned int)open_result);
        goto cleanup;
    }
    mapped = 1;

    prepare_result = qbh_session_prepare(&session);
    info_result = qwen3_probe_resource_info(
        session.handle, shared_fd, (uint32_t)sizeof(*header));
    if (info_result != AEE_SUCCESS) {
        fprintf(stderr, "resource_info failed: 0x%08x\n",
                (unsigned int)info_result);
        goto cleanup;
    }
    release_result = qbh_session_release(&session);
    close_result = qbh_session_close(&session);

    printf("{\"experiment\":\"EXP-0022\","
           "\"gate\":\"fixed_full_vtcm_session\","
           "\"prepare_result\":%d,\"info_result\":%d,"
           "\"release_result\":%d,\"close_result\":%d,"
           "\"dsp_status\":%d,\"query_status\":%d,"
           "\"configure_status\":%d,\"acquire_status\":%d,"
           "\"get_pointer_status\":%d,"
           "\"expected_total_bytes\":%" PRIu32 ","
           "\"queried_total_bytes\":%" PRIu32 ","
           "\"available_before_bytes\":%" PRIu32 ","
           "\"requested_bytes\":%" PRIu32 ","
           "\"minimum_page_bytes\":%" PRIu32 ","
           "\"minimum_required_bytes\":%" PRIu32 ","
           "\"granted_bytes\":%" PRIu32 ","
           "\"context_id\":%" PRIu32 ","
           "\"vtcm_address\":%" PRIu32 ","
           "\"exact_full_grant\":%" PRIu32 ","
           "\"acquire_ticks\":%" PRIu64 "}\n",
           prepare_result, info_result, release_result, close_result,
           header->dsp_status, header->query_status,
           header->configure_status, header->acquire_status,
           header->get_pointer_status, header->expected_total_bytes,
           header->queried_total_bytes, header->available_before_bytes,
           header->requested_bytes, header->minimum_page_bytes,
           header->minimum_required_bytes, header->granted_bytes,
           header->context_id, header->vtcm_address,
           header->exact_full_grant, header->acquire_ticks);

    exit_code = prepare_result == AEE_SUCCESS &&
                        info_result == AEE_SUCCESS &&
                        release_result == AEE_SUCCESS &&
                        close_result == AEE_SUCCESS &&
                        header->dsp_status == QBH_RESOURCE_STATUS_OK &&
                        header->query_status == AEE_SUCCESS &&
                        header->configure_status == AEE_SUCCESS &&
                        header->acquire_status == AEE_SUCCESS &&
                        header->get_pointer_status == AEE_SUCCESS &&
                        header->expected_total_bytes ==
                            QBH_EXPECTED_FULL_VTCM_BYTES &&
                        header->queried_total_bytes ==
                            QBH_EXPECTED_FULL_VTCM_BYTES &&
                        header->requested_bytes ==
                            QBH_EXPECTED_FULL_VTCM_BYTES &&
                        header->minimum_page_bytes ==
                            QBH_FULL_VTCM_MIN_PAGE_BYTES &&
                        header->minimum_required_bytes == 0U &&
                        header->granted_bytes ==
                            QBH_EXPECTED_FULL_VTCM_BYTES &&
                        header->exact_full_grant == 1U
                    ? 0
                    : 1;

cleanup:
    if (session.handle != (remote_handle64)-1) {
        (void)qbh_session_close(&session);
    }
    if (mapped) {
        (void)fastrpc_munmap(CDSP_DOMAIN_ID, shared_fd, header,
                             sizeof(*header));
    }
    if (header != NULL) {
        rpcmem_free(header);
    }
    return exit_code;
}
