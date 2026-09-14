#ifndef LLAMA_SP2_PROBE_H
#define LLAMA_SP2_PROBE_H
#include <stdint.h>
#define LSP2_MAGIC 0x3250534cU
/* 0: U8 exact i32; 1: SP2 paired spatial rows; 2: SP2 two passes;
 * 3: production-style single saturating U8 conversion, timing control only.
 * 4: exhaustive LUT gather audit.
 * 7: L32-0015 full8192 dense R4 + FP16-indexed SP2 probe.
 * 8: L32-0015 dense64 R3 component probe.
 * 5/6: radix257 encoded U16 input, paired rows / two passes; signed32 output. */
struct lsp2_header {
 uint32_t magic,abi,bytes,mode,rows,k,n,input_offset,weight_offset,sum_offset,output_offset;
 int32_t status;
 uint32_t vtcm_bytes,peak_bytes,streams,conversions;
 uint64_t total_ticks,load_ticks,pack_ticks,dma_ticks,mac_ticks,convert_ticks,merge_ticks,publish_ticks;
};
_Static_assert(sizeof(struct lsp2_header)==128,"SP2 ABI");
int lsp2_run(int fd,uint32_t bytes,uint8_t *vtcm,uint32_t vtcm_bytes,uint32_t hmx_context);
#endif
