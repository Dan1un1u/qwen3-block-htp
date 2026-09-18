#ifndef LLAMA_MAPPING_PROBE_H
#define LLAMA_MAPPING_PROBE_H
#include <stdint.h>
#define LMP_MAGIC 0x4c4d5031U
#define LMP_MAX_BUFFERS 10U
#define LMP_MAX_EVENTS 2048U
#define LMP_SAMPLES 33U
#define LMP_PAGE_BYTES 4096U
struct lmp_event {
 uint32_t cycle, buffer, va, bytes;
 int32_t map_result, unmap_result;
 uint32_t checks, mismatches;
 uint64_t map_ticks, dma_ticks, unmap_ticks;
};
struct lmp_header {
 uint32_t magic, version, count, window, cycles, pinned_bytes;
 int32_t fds[LMP_MAX_BUFFERS];
 uint32_t sizes[LMP_MAX_BUFFERS];
 uint32_t event_count, mismatches, completed, vtcm_bytes;
 int32_t status, cleanup_errors;
 int32_t pinned_fd;
 uint32_t retained_checks;
 uint64_t total_ticks;
 struct lmp_event events[LMP_MAX_EVENTS];
};
_Static_assert(sizeof(struct lmp_event)==56,"event ABI");
_Static_assert(sizeof(struct lmp_header)==114832,"header ABI");
static inline uint32_t lmp_offset(uint32_t bytes,uint32_t sample) {
 return (uint32_t)(((uint64_t)(bytes/LMP_PAGE_BYTES-1U)*sample/(LMP_SAMPLES-1U))*LMP_PAGE_BYTES);
}
static inline uint32_t lmp_value(uint32_t buffer,uint32_t sample,uint32_t word) {
 return 0x6b340000U ^ (buffer*0x1020305U) ^ (sample*0x9e3779b9U) ^ (word*0x85ebca6bU);
}
int lmp_run(int fd,uint32_t bytes,void *vtcm,uint32_t vtcm_bytes);
#endif
