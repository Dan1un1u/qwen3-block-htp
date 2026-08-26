#include "host/session.h"

#include <AEEStdErr.h>
#include <stdio.h>
#include <string.h>

#include "qwen3_probe.h"

int qbh_session_open(struct qbh_session *session) {
    struct remote_rpc_control_unsigned_module unsigned_control;
    struct remote_rpc_control_latency latency_control;
    char uri[512];
    int result;

    if (session == NULL) {
        return AEE_EBADPARM;
    }
    session->handle = (remote_handle64)-1;

    unsigned_control.domain = CDSP_DOMAIN_ID;
    unsigned_control.enable = 1;
    result = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                    &unsigned_control,
                                    sizeof(unsigned_control));
    if (result != AEE_SUCCESS) {
        fprintf(stderr,
                "remote_session_control(unsigned PD) failed: 0x%08x\n",
                (unsigned int)result);
        return result;
    }

    result = snprintf(uri, sizeof(uri), "%s&_dom=cdsp", qwen3_probe_URI);
    if (result < 0 || (size_t)result >= sizeof(uri)) {
        return AEE_EBADSIZE;
    }

    result = qwen3_probe_open(uri, &session->handle);
    if (result != AEE_SUCCESS) {
        fprintf(stderr, "qwen3_probe_open failed: 0x%08x\n",
                (unsigned int)result);
        session->handle = (remote_handle64)-1;
        return result;
    }

    latency_control.enable = RPC_PM_QOS;
    latency_control.latency = 50;
    result = remote_handle64_control(session->handle,
                                     DSPRPC_CONTROL_LATENCY,
                                     &latency_control,
                                     sizeof(latency_control));
    if (result != AEE_SUCCESS) {
        fprintf(stderr,
                "warning: FastRPC QoS request failed: 0x%08x\n",
                (unsigned int)result);
    }
    return AEE_SUCCESS;
}

void qbh_session_close(struct qbh_session *session) {
    if (session != NULL && session->handle != (remote_handle64)-1) {
        (void)qwen3_probe_close(session->handle);
        session->handle = (remote_handle64)-1;
    }
}
