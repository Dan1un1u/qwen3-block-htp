#ifndef QWEN3_BLOCK_HTP_BLOCK_IMP_H
#define QWEN3_BLOCK_HTP_BLOCK_IMP_H

#include <AEEStdErr.h>
#include <stdint.h>

AEEResult qbh_run_block_rpc(int32_t shared_fd, uint32_t shared_bytes,
                            uint8_t *vtcm, uint32_t vtcm_bytes,
                            uint32_t hmx_context_id,
                            uint32_t prepared_session_run_index);

#endif
