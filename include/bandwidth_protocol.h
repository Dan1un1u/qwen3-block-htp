#ifndef QBH_BANDWIDTH_H
#define QBH_BANDWIDTH_H
#include <stdint.h>
#define BW_MAGIC 0x42573031U
#define BW_PAYLOAD 4096U
#define BW_DRAM (128U*1024U*1024U)
#define BW_ROUNDS 12U
struct bw_header {
 uint32_t magic, mode, bytes, workers, repeats, stream, depth, bypass;
 uint32_t rounds, errors, vtcm_bytes, status;
 uint64_t payload_bytes, ticks[BW_ROUNDS], cycles[BW_ROUNDS];
 uint32_t checks[6];
 uint64_t hvx_bytes[BW_ROUNDS], hmx_bytes[BW_ROUNDS], start_delay[BW_ROUNDS], finish_delay[BW_ROUNDS];
};
#endif
