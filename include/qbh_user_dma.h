#ifndef QWEN3_BLOCK_HTP_USER_DMA_H
#define QWEN3_BLOCK_HTP_USER_DMA_H

#include <hexagon_protos.h>
#include <stdint.h>

#define QBH_DMA_STATUS_MASK UINT32_C(3)
#define QBH_DMA_STATUS_IDLE UINT32_C(0)
#define QBH_DMA_STATUS_ERROR UINT32_C(2)
#define QBH_DMA_DESC_PENDING UINT32_C(0)
#define QBH_DMA_DESC_COMPLETE UINT32_C(1)
#define QBH_DMA_TYPE_1D UINT32_C(0)
#define QBH_DMA_TYPE_2D UINT32_C(1)

struct qbh_dma_desc_1d {
    uint32_t next;
    union {
        struct {
            unsigned length : 24;
            unsigned type : 2;
            unsigned dst_dlbc : 1;
            unsigned src_dlbc : 1;
            unsigned dst_bypass : 1;
            unsigned src_bypass : 1;
            unsigned ordered : 1;
            unsigned dstate : 1;
        } __attribute__((packed));
        uint32_t control;
    };
    uint32_t src;
    uint32_t dst;
} __attribute__((packed));

struct qbh_dma_desc_2d {
    uint32_t next;
    union {
        struct {
            unsigned length : 24;
            unsigned type : 2;
            unsigned dst_dlbc : 1;
            unsigned src_dlbc : 1;
            unsigned dst_bypass : 1;
            unsigned src_bypass : 1;
            unsigned ordered : 1;
            unsigned dstate : 1;
        } __attribute__((packed));
        uint32_t control;
    };
    uint32_t src;
    uint32_t dst;
    unsigned reserved0 : 24;
    unsigned cache_alloc : 2;
    unsigned reserved1 : 6;
    uint16_t roi_width;
    uint16_t roi_height;
    uint16_t src_stride;
    uint16_t dst_stride;
    uint16_t src_width_offset;
    uint16_t dst_width_offset;
} __attribute__((packed));

struct qbh_dma_aligned_desc_1d {
    struct qbh_dma_desc_1d descriptor;
    uint8_t padding[48];
} __attribute__((aligned(64)));

_Static_assert(sizeof(struct qbh_dma_aligned_desc_1d) == 64,
               "linked DMA descriptor stride changed");

static inline int qbh_dma_wait_idle(void) {
    return (Q6_R_dmwait() & QBH_DMA_STATUS_MASK) == QBH_DMA_STATUS_IDLE
               ? 0
               : -1;
}

static inline int qbh_dma_start(void *descriptor) {
    if ((Q6_R_dmpoll() & QBH_DMA_STATUS_MASK) != QBH_DMA_STATUS_IDLE) {
        return -1;
    }
    asm volatile("release(%0):at" : : "r"(descriptor) : "memory");
    Q6_dmstart_A(descriptor);
    (void)Q6_R_dmpoll();
    return 0;
}

#endif
