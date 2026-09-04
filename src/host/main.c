#include <AEEStdErr.h>
#include <inttypes.h>
#include <limits.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hmx_u8s8_projection.h"
#include "host/session.h"
#include "probe_protocol.h"
#include "qwen3_probe.h"

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1U) / alignment * alignment;
}

static uint64_t monotonic_ns(void) {
    struct timespec now;
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
           (uint64_t)now.tv_nsec;
}

static int parse_u32(const char *text, uint32_t *value) {
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 0);
    if (text[0] == '\0' || end == NULL || *end != '\0' ||
        parsed > UINT32_MAX) {
        return -1;
    }
    *value = (uint32_t)parsed;
    return 0;
}

static int read_package_file(const char *package, const char *name,
                             void *destination, size_t expected_bytes) {
    char path[PATH_MAX];
    FILE *stream;
    long bytes;
    int written;

    if (package == NULL || name == NULL || destination == NULL) {
        return -1;
    }
    written = snprintf(path, sizeof(path), "%s/%s", package, name);
    if (written < 0 || (size_t)written >= sizeof(path)) {
        return -1;
    }
    stream = fopen(path, "rb");
    if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) {
        if (stream != NULL) {
            (void)fclose(stream);
        }
        return -1;
    }
    bytes = ftell(stream);
    if (bytes < 0 || (size_t)bytes != expected_bytes ||
        fseek(stream, 0, SEEK_SET) != 0 ||
        fread(destination, 1, expected_bytes, stream) != expected_bytes ||
        fclose(stream) != 0) {
        return -1;
    }
    return 0;
}

static const char *storage_name(uint32_t storage) {
    switch (storage) {
        case QBH_WEIGHT_EXPANDED_S8:
            return "expanded_s8_control";
        case QBH_WEIGHT_PACKED_W4:
            return "packed_w4_hvx_prescale";
        case QBH_WEIGHT_PACKED_W4_HMX_SCALE:
            return "packed_w4_hmx_postscale";
        case QBH_WEIGHT_PACKED_W4_DIRECT_N:
            return "packed_w4_direct_n";
        default:
            return "invalid";
    }
}

static int parse_storage(const char *text, uint32_t *storage) {
    uint32_t parsed;
    if (strcmp(text, "expanded_s8_control") == 0 ||
        strcmp(text, "expanded_s8") == 0 || strcmp(text, "s8") == 0 ||
        strcmp(text, "w8") == 0) {
        *storage = QBH_WEIGHT_EXPANDED_S8;
        return 0;
    }
    if (strcmp(text, "packed_w4u8") == 0 ||
        strcmp(text, "packed_w4_hvx_prescale") == 0 ||
        strcmp(text, "packed_w4") == 0 || strcmp(text, "w4") == 0) {
        *storage = QBH_WEIGHT_PACKED_W4;
        return 0;
    }
    if (strcmp(text, "packed_w4_hmx_postscale") == 0 ||
        strcmp(text, "hmx_postscale") == 0 ||
        strcmp(text, "w4_postscale") == 0) {
        *storage = QBH_WEIGHT_PACKED_W4_HMX_SCALE;
        return 0;
    }
    if (strcmp(text, "packed_w4_direct_n") == 0 ||
        strcmp(text, "direct_n") == 0 ||
        strcmp(text, "w4_direct_n") == 0) {
        *storage = QBH_WEIGHT_PACKED_W4_DIRECT_N;
        return 0;
    }
    if (parse_u32(text, &parsed) == 0 &&
        (parsed == QBH_WEIGHT_EXPANDED_S8 ||
         parsed == QBH_WEIGHT_PACKED_W4 ||
         parsed == QBH_WEIGHT_PACKED_W4_HMX_SCALE ||
         parsed == QBH_WEIGHT_PACKED_W4_DIRECT_N)) {
        *storage = parsed;
        return 0;
    }
    return -1;
}

static const char *physical_plan_name(uint32_t physical_plan,
                                      uint32_t hvx_workers,
                                      uint32_t compressed_slots,
                                      uint32_t chunk_tiles) {
    if (physical_plan ==
        QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4) {
        switch (hvx_workers) {
            case 2: return "stream32_gate_hvx2";
            case 3: return "stream32_gate_hvx3";
            case 4: return "stream32_gate_hvx4";
            case 6: return "stream32_gate_hvx6";
            default: return "invalid";
        }
    }
    if (physical_plan ==
        QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4) {
        return "stream32_gate_hvx6_cap2";
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_STREAMING_DMA_BATCH2) {
        switch (hvx_workers) {
            case 2: return "stream32_down_hvx2";
            case 3: return "stream32_down_hvx3";
            case 4: return "stream32_down_hvx4";
            case 6: return "stream32_down_hvx6";
            default: return "invalid";
        }
    }
    if (physical_plan ==
        QBH_PHYSICAL_PLAN_STREAMING_CAP2_DMA_BATCH2) {
        return "stream32_down_hvx6_cap2";
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE) {
        return "exp0005_full_bundle_control";
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_BATCH2) {
        return "expanded_s8_dma_batch2";
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_CHAIN2) {
        return "expanded_s8_dma_chain2";
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH2) {
        if (compressed_slots == 4U && chunk_tiles == 64U) {
            return "slots4e7_chunk64_dma_batch2";
        }
        if (compressed_slots == 4U && chunk_tiles == 96U) {
            return "slots4e7_chunk96_dma_batch2";
        }
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH4) {
        if (compressed_slots == 8U && chunk_tiles == 64U) {
            return "slots8e7_chunk64_dma_batch4";
        }
        if (compressed_slots == 8U && chunk_tiles == 96U) {
            return "slots8e7_chunk96_dma_batch4";
        }
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_CHAIN4) {
        if (compressed_slots == 8U && chunk_tiles == 64U) {
            return "slots8e7_chunk64_dma_chain4";
        }
        if (compressed_slots == 8U && chunk_tiles == 96U) {
            return "slots8e7_chunk96_dma_chain4";
        }
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2) {
        if (compressed_slots == 4U && chunk_tiles == 64U) {
            return "slots4_chunk64_dma_batch2";
        }
        if (compressed_slots == 4U && chunk_tiles == 96U) {
            return "slots4_chunk96_dma_batch2";
        }
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH4) {
        if (compressed_slots == 4U && chunk_tiles == 64U) {
            return "slots4_chunk64_dma_batch4";
        }
        if (compressed_slots == 4U && chunk_tiles == 96U) {
            return "slots4_chunk96_dma_batch4";
        }
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN2) {
        if (compressed_slots == 4U && chunk_tiles == 64U) {
            return "slots4_chunk64_dma_chain2";
        }
        if (compressed_slots == 4U && chunk_tiles == 96U) {
            return "slots4_chunk96_dma_chain2";
        }
    }
    if (physical_plan == QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN4) {
        if (compressed_slots == 4U && chunk_tiles == 64U) {
            return "slots4_chunk64_dma_chain4";
        }
        if (compressed_slots == 4U && chunk_tiles == 96U) {
            return "slots4_chunk96_dma_chain4";
        }
    }
    if (hvx_workers == 6U) {
        if (compressed_slots == 2U && chunk_tiles == 32U) {
            return "exp0006_slots2_chunk32_control";
        }
        if (compressed_slots == 3U && chunk_tiles == 32U) {
            return "slots3_chunk32";
        }
        if (compressed_slots == 4U && chunk_tiles == 32U) {
            return "slots4_chunk32";
        }
        if (compressed_slots == 2U && chunk_tiles == 16U) {
            return "slots2_chunk16";
        }
        if (compressed_slots == 3U && chunk_tiles == 16U) {
            return "slots3_chunk16";
        }
        if (compressed_slots == 4U && chunk_tiles == 64U) {
            return "slots4_chunk64";
        }
        if (compressed_slots == 4U && chunk_tiles == 96U) {
            return "slots4_chunk96";
        }
    }
    switch (hvx_workers) {
        case 1:
            return "chunked_hvx1";
        case 2:
            return "chunked_hvx2";
        case 4:
            return "chunked_hvx4";
        case 6:
            return "chunked_hvx6";
        default:
            return "invalid";
    }
}

static int parse_physical_plan(const char *text, uint32_t *physical_plan,
                               uint32_t *hvx_workers,
                               uint32_t *compressed_slots,
                               uint32_t *chunk_tiles) {
    if (strcmp(text, "stream32_gate_hvx2") == 0 ||
        strcmp(text, "stream32_gate_hvx3") == 0 ||
        strcmp(text, "stream32_gate_hvx4") == 0 ||
        strcmp(text, "stream32_gate_hvx6") == 0) {
        *physical_plan =
            QBH_PHYSICAL_PLAN_STREAMING_E7_DMA_CHAIN4;
        *hvx_workers = (uint32_t)(text[17] - '0');
        *compressed_slots = 8U;
        *chunk_tiles = QBH_W4_COARSE_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "stream32_down_hvx2") == 0 ||
        strcmp(text, "stream32_down_hvx3") == 0 ||
        strcmp(text, "stream32_down_hvx4") == 0 ||
        strcmp(text, "stream32_down_hvx6") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_STREAMING_DMA_BATCH2;
        *hvx_workers = (uint32_t)(text[17] - '0');
        *compressed_slots = 4U;
        *chunk_tiles = QBH_W4_WIDE_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "stream32_gate_hvx6_cap2") == 0) {
        *physical_plan =
            QBH_PHYSICAL_PLAN_STREAMING_CAP2_E7_DMA_CHAIN4;
        *hvx_workers = 6U;
        *compressed_slots = 8U;
        *chunk_tiles = QBH_W4_COARSE_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "stream32_down_hvx6_cap2") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_STREAMING_CAP2_DMA_BATCH2;
        *hvx_workers = 6U;
        *compressed_slots = 4U;
        *chunk_tiles = QBH_W4_WIDE_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "exp0005_full_bundle_control") == 0 ||
        strcmp(text, "full_bundle") == 0 ||
        strcmp(text, "control") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_FULL_BUNDLE;
        *hvx_workers = 1;
        *compressed_slots = QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT;
        *chunk_tiles = QBH_W4_DEFAULT_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "chunked_hvx1") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 1;
        *compressed_slots = QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT;
        *chunk_tiles = QBH_W4_DEFAULT_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "chunked_hvx2") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 2;
        *compressed_slots = QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT;
        *chunk_tiles = QBH_W4_DEFAULT_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "chunked_hvx4") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 4;
        *compressed_slots = QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT;
        *chunk_tiles = QBH_W4_DEFAULT_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "chunked_hvx6") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 6;
        *compressed_slots = QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT;
        *chunk_tiles = QBH_W4_DEFAULT_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "exp0006_slots2_chunk32_control") == 0 ||
        strcmp(text, "slots2_chunk32") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 6;
        *compressed_slots = 2U;
        *chunk_tiles = 32U;
        return 0;
    }
    if (strcmp(text, "slots3_chunk32") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 6;
        *compressed_slots = 3U;
        *chunk_tiles = 32U;
        return 0;
    }
    if (strcmp(text, "slots4_chunk32") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 32U;
        return 0;
    }
    if (strcmp(text, "slots2_chunk16") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 6;
        *compressed_slots = 2U;
        *chunk_tiles = 16U;
        return 0;
    }
    if (strcmp(text, "slots3_chunk16") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 6;
        *compressed_slots = 3U;
        *chunk_tiles = 16U;
        return 0;
    }
    if (strcmp(text, "slots4_chunk64") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 64U;
        return 0;
    }
    if (strcmp(text, "slots4_chunk96") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 96U;
        return 0;
    }
    if (strcmp(text, "expanded_s8_dma_batch2") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_BATCH2;
        *hvx_workers = 1;
        *compressed_slots = QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT;
        *chunk_tiles = QBH_W4_DEFAULT_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "slots4_chunk64_dma_batch2") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 64U;
        return 0;
    }
    if (strcmp(text, "slots4_chunk64_dma_batch4") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH4;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 64U;
        return 0;
    }
    if (strcmp(text, "slots4_chunk96_dma_batch2") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH2;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 96U;
        return 0;
    }
    if (strcmp(text, "slots4_chunk96_dma_batch4") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_DMA_BATCH4;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 96U;
        return 0;
    }
    if (strcmp(text, "expanded_s8_dma_chain2") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_FULL_BUNDLE_DMA_CHAIN2;
        *hvx_workers = 1;
        *compressed_slots = QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT;
        *chunk_tiles = QBH_W4_DEFAULT_CHUNK_TILES;
        return 0;
    }
    if (strcmp(text, "slots4_chunk64_dma_chain2") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN2;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 64U;
        return 0;
    }
    if (strcmp(text, "slots4_chunk64_dma_chain4") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN4;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 64U;
        return 0;
    }
    if (strcmp(text, "slots4_chunk96_dma_chain2") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN2;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 96U;
        return 0;
    }
    if (strcmp(text, "slots4_chunk96_dma_chain4") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_DMA_CHAIN4;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 96U;
        return 0;
    }
    if (strcmp(text, "slots4e7_chunk64_dma_batch2") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH2;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 64U;
        return 0;
    }
    if (strcmp(text, "slots4e7_chunk96_dma_batch2") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH2;
        *hvx_workers = 6;
        *compressed_slots = 4U;
        *chunk_tiles = 96U;
        return 0;
    }
    if (strcmp(text, "slots8e7_chunk64_dma_batch4") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH4;
        *hvx_workers = 6;
        *compressed_slots = 8U;
        *chunk_tiles = 64U;
        return 0;
    }
    if (strcmp(text, "slots8e7_chunk96_dma_batch4") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_BATCH4;
        *hvx_workers = 6;
        *compressed_slots = 8U;
        *chunk_tiles = 96U;
        return 0;
    }
    if (strcmp(text, "slots8e7_chunk64_dma_chain4") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_CHAIN4;
        *hvx_workers = 6;
        *compressed_slots = 8U;
        *chunk_tiles = 64U;
        return 0;
    }
    if (strcmp(text, "slots8e7_chunk96_dma_chain4") == 0) {
        *physical_plan = QBH_PHYSICAL_PLAN_CHUNKED_E7_DMA_CHAIN4;
        *hvx_workers = 6;
        *compressed_slots = 8U;
        *chunk_tiles = 96U;
        return 0;
    }
    return -1;
}

static const char *projection_name(uint32_t variant) {
    switch (variant) {
        case QBH_PROJECTION_GATE_UP:
            return "gate_up";
        case QBH_PROJECTION_DOWN:
            return "down";
        case QBH_PROJECTION_GATE_UP_PAIR:
            return "gate_up_pair";
        default:
            return "invalid";
    }
}

static int parse_projection(const char *text, uint32_t *variant) {
    uint32_t parsed;
    if (strcmp(text, "gate_up") == 0 || strcmp(text, "gate") == 0 ||
        strcmp(text, "up") == 0) {
        *variant = QBH_PROJECTION_GATE_UP;
        return 0;
    }
    if (strcmp(text, "down") == 0) {
        *variant = QBH_PROJECTION_DOWN;
        return 0;
    }
    if (strcmp(text, "gate_up_pair") == 0 ||
        strcmp(text, "paired_gate_up") == 0) {
        *variant = QBH_PROJECTION_GATE_UP_PAIR;
        return 0;
    }
    if (parse_u32(text, &parsed) == 0 &&
        (parsed == QBH_PROJECTION_GATE_UP ||
         parsed == QBH_PROJECTION_DOWN ||
         parsed == QBH_PROJECTION_GATE_UP_PAIR)) {
        *variant = parsed;
        return 0;
    }
    return -1;
}

static const char *output_assembly_name(uint32_t mode) {
    return mode == QBH_OUTPUT_ASSEMBLY_LINKED_2D_DMA
               ? "linked_2d_dma"
               : "scalar_memcpy";
}

static int parse_output_assembly(const char *text, uint32_t *mode) {
    if (strcmp(text, "scalar_memcpy") == 0 ||
        strcmp(text, "scalar") == 0) {
        *mode = QBH_OUTPUT_ASSEMBLY_SCALAR;
        return 0;
    }
    if (strcmp(text, "linked_2d_dma") == 0 ||
        strcmp(text, "output_dma") == 0) {
        *mode = QBH_OUTPUT_ASSEMBLY_LINKED_2D_DMA;
        return 0;
    }
    return -1;
}

static const char *resource_lifetime_name(uint32_t mode) {
    return mode == QBH_RESOURCE_LIFETIME_PREPARED_SESSION
               ? "prepared_session"
               : "transient_resources";
}

static int parse_resource_lifetime(const char *text, uint32_t *mode) {
    if (strcmp(text, "transient_resources") == 0 ||
        strcmp(text, "transient") == 0) {
        *mode = QBH_RESOURCE_LIFETIME_TRANSIENT;
        return 0;
    }
    if (strcmp(text, "prepared_session") == 0 ||
        strcmp(text, "persistent_resources") == 0 ||
        strcmp(text, "persistent") == 0) {
        *mode = QBH_RESOURCE_LIFETIME_PREPARED_SESSION;
        return 0;
    }
    return -1;
}

enum qbh_host_invocation_mode {
    QBH_HOST_SINGLE_INVOCATION = 1,
    QBH_HOST_TWO_CALL_CONTROL = 2,
};

static const char *host_invocation_name(uint32_t mode) {
    return mode == QBH_HOST_TWO_CALL_CONTROL
               ? "two_call_control"
               : "single_invocation";
}

static int parse_host_invocation(const char *text, uint32_t *mode) {
    if (strcmp(text, "single_invocation") == 0 ||
        strcmp(text, "single") == 0) {
        *mode = QBH_HOST_SINGLE_INVOCATION;
        return 0;
    }
    if (strcmp(text, "two_call_control") == 0 ||
        strcmp(text, "two_calls") == 0) {
        *mode = QBH_HOST_TWO_CALL_CONTROL;
        return 0;
    }
    return -1;
}

static const char *pattern_name(uint32_t pattern) {
    switch (pattern) {
        case QBH_PATTERN_IDENTITY:
            return "identity";
        case QBH_PATTERN_SIGNED:
            return "signed";
        case QBH_PATTERN_STRUCTURED:
            return "structured";
        case QBH_PATTERN_BOUNDARY:
            return "boundary";
        default:
            return "invalid";
    }
}

static int parse_pattern(const char *text, uint32_t *pattern) {
    uint32_t parsed;
    if (strcmp(text, "identity") == 0) {
        *pattern = QBH_PATTERN_IDENTITY;
        return 0;
    }
    if (strcmp(text, "signed") == 0) {
        *pattern = QBH_PATTERN_SIGNED;
        return 0;
    }
    if (strcmp(text, "structured") == 0) {
        *pattern = QBH_PATTERN_STRUCTURED;
        return 0;
    }
    if (strcmp(text, "boundary") == 0) {
        *pattern = QBH_PATTERN_BOUNDARY;
        return 0;
    }
    if (parse_u32(text, &parsed) == 0 &&
        parsed >= QBH_PATTERN_IDENTITY && parsed <= QBH_PATTERN_BOUNDARY) {
        *pattern = parsed;
        return 0;
    }
    return -1;
}

static uint8_t channel_scale(uint32_t pattern, uint32_t channel) {
    switch (pattern) {
        case QBH_PATTERN_IDENTITY:
            return (uint8_t)(1U + channel % 4U);
        case QBH_PATTERN_SIGNED:
            return (uint8_t)(1U + (channel * 3U) % 8U);
        case QBH_PATTERN_STRUCTURED:
            return (uint8_t)(1U + (channel * 5U) % 12U);
        default:
            return (uint8_t)(QBH_W4_MAX_INTEGER_SCALE - channel % 6U);
    }
}

static void fill_pattern(const struct qbh_projection_layout *layout,
                         uint32_t pattern, uint32_t logical_m,
                         uint32_t output_channel_base,
                         uint8_t *activation, int8_t *logical_w4,
                         int8_t *logical_s8,
                         uint8_t *channel_scales) {
    for (uint32_t output_channel = 0; output_channel < layout->n;
         ++output_channel) {
        uint32_t global_output_channel =
            output_channel_base + output_channel;
        channel_scales[output_channel] =
            channel_scale(pattern, global_output_channel);
    }

    for (uint32_t row = 0; row < layout->m; ++row) {
        for (uint32_t input_channel = 0; input_channel < layout->k;
             ++input_channel) {
            uint8_t value;
            if (row >= logical_m) {
                value = QBH_HMX_DEFAULT_ZERO_POINT;
            } else switch (pattern) {
                case QBH_PATTERN_IDENTITY:
                    value = (uint8_t)(QBH_HMX_DEFAULT_ZERO_POINT +
                                      ((row + input_channel) & 7U));
                    break;
                case QBH_PATTERN_SIGNED:
                    value = (uint8_t)(QBH_HMX_DEFAULT_ZERO_POINT +
                                      (int32_t)((row * 5U +
                                                 input_channel * 3U) %
                                                17U) -
                                      8);
                    break;
                case QBH_PATTERN_STRUCTURED:
                    value = (uint8_t)(96U +
                                      ((row * 11U + input_channel * 13U) %
                                       65U));
                    break;
                default:
                    value = ((row + input_channel) & 1U) != 0U
                                ? UINT8_MAX
                                : UINT8_C(0);
                    break;
            }
            activation[qbh_projection_activation_offset(
                layout, row, input_channel)] = value;
        }
    }

    for (uint32_t input_channel = 0; input_channel < layout->k;
         ++input_channel) {
        for (uint32_t output_channel = 0; output_channel < layout->n;
             ++output_channel) {
            int8_t q4;
            uint32_t global_output_channel =
                output_channel_base + output_channel;
            size_t offset = qbh_projection_logical_weight_offset(
                layout, input_channel, output_channel);
            switch (pattern) {
                case QBH_PATTERN_IDENTITY:
                    q4 = input_channel == global_output_channel
                             ? INT8_C(1)
                             : INT8_C(0);
                    break;
                case QBH_PATTERN_SIGNED:
                    q4 = (int8_t)((int32_t)((input_channel * 7U +
                                             global_output_channel * 5U) %
                                            7U) -
                                  3);
                    break;
                case QBH_PATTERN_STRUCTURED:
                    q4 = ((input_channel +
                           3U * global_output_channel) %
                          5U) == 0U
                             ? (int8_t)((int32_t)((input_channel +
                                                  global_output_channel) %
                                                 5U) -
                                        2)
                             : INT8_C(0);
                    break;
                default:
                    q4 = ((input_channel + global_output_channel) & 1U) != 0U
                             ? INT8_C(7)
                             : INT8_C(-7);
                    break;
            }
            logical_w4[offset] = q4;
            logical_s8[offset] = (int8_t)(
                (int32_t)q4 * channel_scales[output_channel]);
        }
    }
}

static void fill_bias_words(
    const struct qbh_projection_layout *layout,
    const int8_t *logical_s8, uint32_t output_tile,
    int32_t input_zero_point, uint32_t *bias_words) {
    for (uint32_t output_channel = 0;
         output_channel < QBH_HMX_OUTPUT_CHANNELS; ++output_channel) {
        int32_t weight_sum = 0;
        uint32_t logical_n =
            output_tile * QBH_HMX_OUTPUT_CHANNELS + output_channel;
        for (uint32_t input_channel = 0; input_channel < layout->k;
             ++input_channel) {
            weight_sum += logical_s8[
                qbh_projection_logical_weight_offset(
                    layout, input_channel, logical_n)];
        }
        bias_words[output_channel] =
            QBH_HMX_IDENTITY_CONVERT_LOWER_WORD;
        bias_words[QBH_HMX_OUTPUT_CHANNELS + output_channel] =
            (uint32_t)(-input_zero_point * weight_sum);
    }
}

/* Integer HMX uses FP16(512) as its unity output conversion scale. The
 * synthetic EXP-0008 integer scales are exactly representable after the
 * factor of 512, so the four extra mantissa bits remain zero. */
static const uint16_t qbh_hmx_integer_scale_words[19] = {
    0x0000, 0x6000, 0x6400, 0x6600, 0x6800,
    0x6900, 0x6a00, 0x6b00, 0x6c00, 0x6c80,
    0x6d00, 0x6d80, 0x6e00, 0x6e80, 0x6f00,
    0x6f80, 0x7000, 0x7040, 0x7080,
};

_Static_assert(QBH_W4_MAX_INTEGER_SCALE == 18U,
               "HMX integer-scale table must cover the experiment range");

static void fill_postscale_bias_words(
    const struct qbh_projection_layout *layout,
    const int8_t *logical_w4, const uint8_t *channel_scales,
    uint32_t output_tile, int32_t input_zero_point,
    uint32_t *bias_words) {
    for (uint32_t output_channel = 0;
         output_channel < QBH_HMX_OUTPUT_CHANNELS; ++output_channel) {
        int32_t weight_sum = 0;
        uint32_t logical_n =
            output_tile * QBH_HMX_OUTPUT_CHANNELS + output_channel;
        uint32_t scale = channel_scales[logical_n];
        for (uint32_t input_channel = 0; input_channel < layout->k;
             ++input_channel) {
            weight_sum += logical_w4[
                qbh_projection_logical_weight_offset(
                    layout, input_channel, logical_n)];
        }
        bias_words[output_channel] =
            qbh_hmx_integer_scale_words[scale];
        bias_words[QBH_HMX_OUTPUT_CHANNELS + output_channel] =
            (uint32_t)(-input_zero_point * weight_sum);
    }
}

static void pack_expanded_s8_bundles(
    const struct qbh_projection_layout *layout,
    const int8_t *logical_s8, int32_t input_zero_point,
    uint8_t *stored_weights) {
    memset(stored_weights, 0, layout->stored_weight_bytes);
    for (uint32_t output_tile = 0; output_tile < layout->n_tiles;
         ++output_tile) {
        for (uint32_t input_tile = 0; input_tile < layout->k_tiles;
             ++input_tile) {
            for (uint32_t input_channel = 0;
                 input_channel < QBH_HMX_INPUT_CHANNELS; ++input_channel) {
                for (uint32_t output_channel = 0;
                     output_channel < QBH_HMX_OUTPUT_CHANNELS;
                     ++output_channel) {
                    uint32_t logical_k =
                        input_tile * QBH_HMX_INPUT_CHANNELS + input_channel;
                    uint32_t logical_n =
                        output_tile * QBH_HMX_OUTPUT_CHANNELS +
                        output_channel;
                    stored_weights[qbh_projection_expanded_weight_offset(
                        layout, output_tile, input_tile, input_channel,
                        output_channel)] = (uint8_t)logical_s8[
                            qbh_projection_logical_weight_offset(
                                layout, logical_k, logical_n)];
                }
            }
        }
        fill_bias_words(
            layout, logical_s8, output_tile, input_zero_point,
            (uint32_t *)(stored_weights +
                         qbh_projection_expanded_bias_offset(
                             layout, output_tile)));
    }
}

static void pack_w4_bundles(
    const struct qbh_projection_layout *layout, const int8_t *logical_w4,
    const int8_t *logical_s8, const uint8_t *channel_scales,
    int32_t input_zero_point, uint32_t storage,
    uint8_t *stored_weights) {
    memset(stored_weights, 0, layout->stored_weight_bytes);
    for (uint32_t output_tile = 0; output_tile < layout->n_tiles;
         ++output_tile) {
        size_t bundle_offset =
            qbh_projection_w4_bundle_offset(layout, output_tile);
        for (uint32_t input_tile = 0; input_tile < layout->k_tiles;
             ++input_tile) {
            for (uint32_t input_channel = 0;
                 input_channel < QBH_HMX_INPUT_CHANNELS; ++input_channel) {
                for (uint32_t output_channel = 0;
                     output_channel < QBH_HMX_OUTPUT_CHANNELS;
                     ++output_channel) {
                    uint32_t logical_k =
                        input_tile * QBH_HMX_INPUT_CHANNELS + input_channel;
                    uint32_t logical_n =
                        output_tile * QBH_HMX_OUTPUT_CHANNELS +
                        output_channel;
                    size_t physical_s8_offset =
                        (size_t)input_tile * QBH_HMX_WEIGHT_BYTES +
                        qbh_packed_weight_offset(input_channel,
                                                 output_channel);
                    size_t packed_offset;
                    uint32_t high_nibble;
                    uint32_t nibble_lane;
                    uint8_t nibble = (uint8_t)logical_w4[
                        qbh_projection_logical_weight_offset(
                            layout, logical_k, logical_n)] &
                                     UINT8_C(0x0f);
                    if (storage == QBH_WEIGHT_PACKED_W4_DIRECT_N) {
                        /* A weight.n vector packs eight input channels for
                         * each of 32 output channels.  It is not obtained by
                         * simply halving the byte-weight carrier, whose
                         * vector groups only four input channels.  The extra
                         * input-channel address bit precedes the output bits,
                         * giving nibble order 0,4,1,5,2,6,3,7. */
                        nibble_lane =
                            (input_channel % 4U) * 2U +
                            (input_channel % 8U) / 4U;
                        packed_offset =
                            bundle_offset +
                            (size_t)input_tile *
                                QBH_W4_PACKED_TILE_BYTES +
                            ((size_t)(input_channel / 8U) *
                                 QBH_HMX_OUTPUT_CHANNELS +
                             output_channel) * 4U +
                            nibble_lane / 2U;
                        high_nibble = nibble_lane & 1U;
                    } else {
                        packed_offset =
                            bundle_offset + physical_s8_offset / 2U;
                        high_nibble =
                            (uint32_t)(physical_s8_offset & 1U);
                    }
                    if (high_nibble == 0U) {
                        stored_weights[packed_offset] |= nibble;
                    } else {
                        stored_weights[packed_offset] |=
                            (uint8_t)(nibble << 4U);
                    }
                }
            }
        }
        memcpy(stored_weights +
                   qbh_projection_w4_scale_offset(layout, output_tile),
               channel_scales +
                   output_tile * QBH_HMX_OUTPUT_CHANNELS,
               QBH_W4_CHANNEL_SCALE_BYTES);
        uint32_t *bias_words = (uint32_t *)(
            stored_weights + qbh_projection_w4_bias_offset(
                                 layout, output_tile));
        if (storage == QBH_WEIGHT_PACKED_W4_HMX_SCALE ||
            storage == QBH_WEIGHT_PACKED_W4_DIRECT_N) {
            fill_postscale_bias_words(
                layout, logical_w4, channel_scales, output_tile,
                input_zero_point, bias_words);
        } else {
            fill_bias_words(layout, logical_s8, output_tile,
                            input_zero_point, bias_words);
        }
    }
}

static uint64_t carrier_checksum(const int8_t *logical_s8,
                                 size_t elements) {
    uint64_t checksum = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < elements; ++index) {
        checksum ^= (uint8_t)logical_s8[index];
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static uint64_t canonical_output_checksum(
    const struct qbh_projection_layout *layout, const uint8_t *output,
    size_t output_stride, uint32_t call_count) {
    uint64_t checksum = UINT64_C(1469598103934665603);
    for (uint32_t row = 0; row < layout->m; ++row) {
        for (uint32_t call = 0; call < call_count; ++call) {
            const uint8_t *row_output =
                output + (size_t)call * output_stride +
                (size_t)row * layout->n;
            for (uint32_t channel = 0; channel < layout->n; ++channel) {
                checksum ^= row_output[channel];
                checksum *= UINT64_C(1099511628211);
            }
        }
    }
    return checksum;
}

static uint8_t clamp_to_u8(int32_t value) {
    if (value < 0) {
        return UINT8_C(0);
    }
    if (value > UINT8_MAX) {
        return UINT8_MAX;
    }
    return (uint8_t)value;
}

static uint32_t validate_output(
    const struct qbh_projection_layout *layout, const uint8_t *activation,
    const int8_t *logical_s8, const uint8_t *output,
    int32_t input_zero_point, int32_t *accumulators,
    uint8_t *reference_min, uint8_t *reference_max,
    uint64_t *reference_checksum) {
    uint32_t mismatches = 0;

    memset(accumulators, 0,
           (size_t)layout->m * layout->n * sizeof(*accumulators));
    for (uint32_t row = 0; row < layout->m; ++row) {
        int32_t *row_accumulators =
            accumulators + (size_t)row * layout->n;
        for (uint32_t input_channel = 0; input_channel < layout->k;
             ++input_channel) {
            int32_t activation_value =
                (int32_t)activation[qbh_projection_activation_offset(
                    layout, row, input_channel)] -
                input_zero_point;
            const int8_t *weight_row =
                logical_s8 + (size_t)input_channel * layout->n;
            for (uint32_t output_channel = 0;
                 output_channel < layout->n; ++output_channel) {
                row_accumulators[output_channel] +=
                    activation_value * (int32_t)weight_row[output_channel];
            }
        }
    }

    *reference_min = UINT8_MAX;
    *reference_max = 0;
    *reference_checksum = 0;
    for (uint32_t row = 0; row < layout->m; ++row) {
        for (uint32_t output_channel = 0; output_channel < layout->n;
             ++output_channel) {
            size_t offset = qbh_projection_output_offset(
                layout, row, output_channel);
            int32_t accumulator = accumulators[offset];
            uint8_t expected = clamp_to_u8(accumulator);
            if (expected < *reference_min) {
                *reference_min = expected;
            }
            if (expected > *reference_max) {
                *reference_max = expected;
            }
            *reference_checksum += expected;
            if (output[offset] != expected) {
                if (mismatches < 8U) {
                    fprintf(stderr,
                            "mismatch row=%" PRIu32 " channel=%" PRIu32
                            " expected=%u actual=%u accumulator=%" PRId32
                            "\n",
                            row, output_channel, (unsigned int)expected,
                            (unsigned int)output[offset], accumulator);
                }
                ++mismatches;
            }
        }
    }
    return mismatches;
}

static uint32_t validate_external_output(
    const struct qbh_projection_layout *layout, const uint8_t *output,
    const uint8_t *reference, uint32_t logical_m,
    uint8_t *reference_min, uint8_t *reference_max,
    uint64_t *reference_checksum) {
    uint32_t mismatches = 0U;

    *reference_min = UINT8_MAX;
    *reference_max = 0U;
    *reference_checksum = 0U;
    for (uint32_t row = 0U; row < logical_m; ++row) {
        for (uint32_t channel = 0U; channel < layout->n; ++channel) {
            size_t offset = qbh_projection_output_offset(layout, row, channel);
            uint8_t expected = reference[offset];
            if (expected < *reference_min) {
                *reference_min = expected;
            }
            if (expected > *reference_max) {
                *reference_max = expected;
            }
            *reference_checksum += expected;
            if (output[offset] != expected) {
                if (mismatches < 8U) {
                    fprintf(stderr,
                            "external mismatch row=%" PRIu32
                            " channel=%" PRIu32 " expected=%u actual=%u\n",
                            row, channel, (unsigned int)expected,
                            (unsigned int)output[offset]);
                }
                ++mismatches;
            }
        }
    }
    return mismatches;
}

int main(int argc, char **argv) {
    struct qbh_session session = {(remote_handle64)-1, 0};
    struct qbh_projection_layout layout;
    struct qbh_probe_header *header = NULL;
    struct qbh_probe_header measured_headers[2];
    uint8_t *shared = NULL;
    uint8_t *activation;
    uint8_t *stored_weights;
    uint8_t *output;
    int8_t *logical_w4 = NULL;
    int8_t *logical_s8 = NULL;
    uint8_t *channel_scales = NULL;
    uint8_t *external_reference = NULL;
    int32_t *reference_accumulators = NULL;
    uint32_t storage = QBH_WEIGHT_PACKED_W4;
    uint32_t variant = QBH_PROJECTION_GATE_UP;
    uint32_t pattern = QBH_PATTERN_IDENTITY;
    uint32_t repeats = QBH_HMX_DEFAULT_REPEATS;
    uint32_t physical_plan = QBH_PHYSICAL_PLAN_FULL_BUNDLE;
    uint32_t requested_hvx_workers = 1;
    uint32_t compressed_slot_count =
        QBH_W4_DEFAULT_COMPRESSED_SLOT_COUNT;
    uint32_t chunk_tiles = QBH_W4_DEFAULT_CHUNK_TILES;
    uint32_t output_assembly_mode = QBH_OUTPUT_ASSEMBLY_SCALAR;
    uint32_t resource_lifetime_mode =
        QBH_RESOURCE_LIFETIME_TRANSIENT;
    uint32_t host_invocation_mode = QBH_HOST_SINGLE_INVOCATION;
    uint32_t measured_rpc_calls = 1;
    uint32_t logical_m = QBH_PROJ_M;
    uint32_t input_zero_point = QBH_HMX_DEFAULT_ZERO_POINT;
    const char *external_package = NULL;
    const char *external_activation_name = NULL;
    const char *external_weight_name = NULL;
    const char *external_reference_name = NULL;
    int external_mode = 0;
    size_t activation_offset;
    size_t weight_offset;
    size_t output_offset;
    size_t weight_stride;
    size_t output_stride;
    size_t logical_weight_stride;
    size_t logical_weight_bytes;
    size_t channel_scale_bytes;
    size_t total_bytes;
    uint64_t host_start;
    uint64_t host_end;
    uint64_t measured_call_wall_ns[2] = {0, 0};
    uint64_t session_open_start;
    uint64_t session_open_end;
    uint64_t prepare_start = 0;
    uint64_t prepare_end = 0;
    uint64_t warmup_start;
    uint64_t warmup_end;
    uint64_t release_start = 0;
    uint64_t release_end = 0;
    uint64_t session_close_start;
    uint64_t session_close_end;
    uint64_t reference_start;
    uint64_t reference_end;
    uint64_t reference_checksum;
    uint64_t expanded_carrier_checksum;
    uint64_t packed_w4_checksum;
    uint64_t hmx_carrier_checksum;
    uint64_t warmup_output_checksum;
    uint64_t measured_output_checksum;
    uint32_t warmup_resource_vtcm_address;
    uint32_t warmup_resource_hmx_context_id;
    uint32_t warmup_prepared_session_run_index;
    uint8_t reference_min;
    uint8_t reference_max;
    uint32_t mismatches;
    uint32_t expected_weight_stages;
    uint32_t expected_reuses;
    uint32_t expected_dma_waits;
    uint32_t expected_dma_descriptors;
    uint32_t expected_dma_chains;
    uint32_t expected_expands;
    uint32_t expected_superchunks;
    int shared_fd = -1;
    int mapped = 0;
    int result = EXIT_FAILURE;
    int rpc_result;
    int warmup_rpc_result;
    int prepare_result = AEE_SUCCESS;
    int release_result = AEE_SUCCESS;
    int session_close_result = AEE_SUCCESS;

    if (argc > 1 && parse_storage(argv[1], &storage) != 0) {
        fprintf(stderr, "invalid weight storage: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    if (argc > 2 && parse_projection(argv[2], &variant) != 0) {
        fprintf(stderr, "invalid projection: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    if (argc > 3 && parse_pattern(argv[3], &pattern) != 0) {
        fprintf(stderr, "invalid pattern: %s\n", argv[3]);
        return EXIT_FAILURE;
    }
    if (argc > 4 && parse_u32(argv[4], &repeats) != 0) {
        fprintf(stderr, "invalid repeat count: %s\n", argv[4]);
        return EXIT_FAILURE;
    }
    if (argc > 5 &&
        parse_physical_plan(argv[5], &physical_plan,
                            &requested_hvx_workers,
                            &compressed_slot_count,
                            &chunk_tiles) != 0) {
        fprintf(stderr, "invalid physical plan: %s\n", argv[5]);
        return EXIT_FAILURE;
    }
    if (argc > 6 &&
        parse_output_assembly(argv[6], &output_assembly_mode) != 0) {
        fprintf(stderr, "invalid output assembly mode: %s\n", argv[6]);
        return EXIT_FAILURE;
    }
    if (argc > 7 &&
        parse_resource_lifetime(argv[7], &resource_lifetime_mode) != 0) {
        fprintf(stderr, "invalid resource lifetime mode: %s\n", argv[7]);
        return EXIT_FAILURE;
    }
    if (argc > 8 &&
        parse_host_invocation(argv[8], &host_invocation_mode) != 0) {
        fprintf(stderr, "invalid host invocation mode: %s\n", argv[8]);
        return EXIT_FAILURE;
    }
    if (argc > 9 && parse_u32(argv[9], &logical_m) != 0) {
        fprintf(stderr, "invalid logical M: %s\n", argv[9]);
        return EXIT_FAILURE;
    }
    if (argc > 10 && strcmp(argv[10], "-") != 0) {
        external_package = argv[10];
        external_mode = 1;
    }
    if (argc > 11 && parse_u32(argv[11], &input_zero_point) != 0) {
        fprintf(stderr, "invalid external input zero point: %s\n", argv[11]);
        return EXIT_FAILURE;
    }
    if (argc > 12) {
        fprintf(stderr,
                "usage: %s [packed_w4_hvx_prescale|"
                "packed_w4_hmx_postscale|packed_w4_direct_n|"
                "expanded_s8_control] "
                "[gate_up|down|gate_up_pair] "
                "[identity|signed|structured|boundary] [repeat] "
                "[exp0005_full_bundle_control|"
                "exp0006_slots2_chunk32_control|slots3_chunk32|"
                "slots4_chunk32|slots2_chunk16|slots3_chunk16|"
                "slots4_chunk64|slots4_chunk96|"
                "expanded_s8_dma_batch2|"
                "slots4_chunk64_dma_batch2|"
                "slots4_chunk64_dma_batch4|"
                "slots4_chunk96_dma_batch2|"
                "slots4_chunk96_dma_batch4|"
                "expanded_s8_dma_chain2|"
                "slots4_chunk64_dma_chain2|"
                "slots4_chunk64_dma_chain4|"
                "slots4_chunk96_dma_chain2|"
                "slots4_chunk96_dma_chain4|"
                "slots4e7_chunk64_dma_batch2|"
                "slots4e7_chunk96_dma_batch2|"
                "slots8e7_chunk64_dma_batch4|"
                "slots8e7_chunk96_dma_batch4|"
                "slots8e7_chunk64_dma_chain4|"
                "slots8e7_chunk96_dma_chain4|"
                "stream32_gate_hvx2|stream32_gate_hvx3|"
                "stream32_gate_hvx4|stream32_gate_hvx6|"
                "stream32_gate_hvx6_cap2|"
                "stream32_down_hvx2|stream32_down_hvx3|"
                "stream32_down_hvx4|stream32_down_hvx6|"
                "stream32_down_hvx6_cap2] "
                "[scalar_memcpy|linked_2d_dma] "
                "[transient_resources|prepared_session] "
                "[single_invocation|two_call_control] [logical_m:1|64] "
                "[external_package|-] [external_input_zero_point]\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    if (logical_m != 1U && logical_m != QBH_PROJ_M) {
        fprintf(stderr, "logical M must be 1 or %u\n",
                (unsigned int)QBH_PROJ_M);
        return EXIT_FAILURE;
    }
    if (input_zero_point > UINT8_MAX) {
        fprintf(stderr, "input zero point must be in [0, 255]\n");
        return EXIT_FAILURE;
    }
    if (repeats == 0 || repeats > QBH_HMX_MAX_REPEATS) {
        fprintf(stderr, "repeat count must be in [1, %u]\n",
                (unsigned int)QBH_HMX_MAX_REPEATS);
        return EXIT_FAILURE;
    }
    if (qbh_projection_layout_init(variant, storage, physical_plan,
                                   compressed_slot_count, chunk_tiles,
                                   &layout) != 0) {
        fprintf(stderr, "projection layout initialization failed\n");
        return EXIT_FAILURE;
    }
    if (host_invocation_mode == QBH_HOST_TWO_CALL_CONTROL &&
        (variant != QBH_PROJECTION_GATE_UP ||
         resource_lifetime_mode !=
             QBH_RESOURCE_LIFETIME_PREPARED_SESSION)) {
        fprintf(stderr,
                "two_call_control requires gate_up and prepared_session\n");
        return EXIT_FAILURE;
    }
    if (variant == QBH_PROJECTION_GATE_UP_PAIR &&
        host_invocation_mode != QBH_HOST_SINGLE_INVOCATION) {
        fprintf(stderr,
                "gate_up_pair requires single_invocation\n");
        return EXIT_FAILURE;
    }
    measured_rpc_calls =
        host_invocation_mode == QBH_HOST_TWO_CALL_CONTROL ? 2U : 1U;
    if (external_mode) {
        if (measured_rpc_calls != 1U ||
            (variant != QBH_PROJECTION_GATE_UP_PAIR &&
             variant != QBH_PROJECTION_DOWN) ||
            (storage != QBH_WEIGHT_PACKED_W4_HMX_SCALE &&
             storage != QBH_WEIGHT_PACKED_W4_DIRECT_N &&
             storage != QBH_WEIGHT_EXPANDED_S8)) {
            fprintf(stderr,
                    "external package mode requires one gate_up_pair/down "
                    "invocation and postscale W4, direct-n W4, or S8\n");
            return EXIT_FAILURE;
        }
        external_activation_name =
            variant == QBH_PROJECTION_GATE_UP_PAIR
                ? "reference_w4u8_post_attention_norm_u8.bin"
                : "reference_w4u8_middle_u8.bin";
        external_reference_name =
            variant == QBH_PROJECTION_GATE_UP_PAIR
                ? "reference_w4u8_gate_up_interleaved_u8.bin"
                : "reference_w4u8_down_u8.bin";
        if (storage == QBH_WEIGHT_PACKED_W4_DIRECT_N) {
            external_weight_name =
                variant == QBH_PROJECTION_GATE_UP_PAIR
                    ? "gate_up_direct_n_bundles.bin"
                    : "down_direct_n_bundles.bin";
        } else if (storage == QBH_WEIGHT_PACKED_W4_HMX_SCALE) {
            external_weight_name =
                variant == QBH_PROJECTION_GATE_UP_PAIR
                    ? "gate_up_packed_w4_bundles.bin"
                    : "down_packed_w4_bundles.bin";
        } else {
            external_weight_name =
                variant == QBH_PROJECTION_GATE_UP_PAIR
                    ? "gate_up_expanded_s8_bundles.bin"
                    : "down_expanded_s8_bundles.bin";
        }
    }

    activation_offset = align_up(sizeof(*header), QBH_PROBE_ALIGNMENT);
    weight_stride = align_up(layout.stored_weight_bytes,
                             QBH_PROBE_ALIGNMENT);
    output_stride = align_up(layout.output_bytes,
                             QBH_PROBE_ALIGNMENT);
    logical_weight_stride = layout.logical_weight_bytes;
    logical_weight_bytes = logical_weight_stride * measured_rpc_calls;
    channel_scale_bytes = (size_t)layout.n * measured_rpc_calls;
    weight_offset = activation_offset +
                    align_up(layout.activation_bytes,
                             QBH_PROBE_ALIGNMENT);
    output_offset = weight_offset +
                    weight_stride * measured_rpc_calls;
    total_bytes = output_offset +
                  output_stride * measured_rpc_calls;
    if (total_bytes > UINT32_MAX || total_bytes > INT_MAX) {
        fprintf(stderr, "shared allocation is too large: %zu bytes\n",
                total_bytes);
        goto cleanup;
    }

    logical_w4 = malloc(logical_weight_bytes);
    logical_s8 = malloc(logical_weight_bytes);
    channel_scales = malloc(channel_scale_bytes);
    reference_accumulators = calloc(
        (size_t)layout.m * layout.n, sizeof(*reference_accumulators));
    if (external_mode) {
        external_reference = malloc(layout.output_bytes);
    }
    if (logical_w4 == NULL || logical_s8 == NULL ||
        channel_scales == NULL || reference_accumulators == NULL ||
        (external_mode && external_reference == NULL)) {
        fprintf(stderr, "host reference allocation failed\n");
        goto cleanup;
    }
    shared = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED,
                          (int)total_bytes);
    if (shared == NULL) {
        fprintf(stderr, "rpcmem_alloc failed for %zu bytes\n", total_bytes);
        goto cleanup;
    }
    shared_fd = rpcmem_to_fd(shared);
    if (shared_fd < 0) {
        fprintf(stderr, "rpcmem_to_fd failed\n");
        goto cleanup;
    }

    memset(shared, 0, total_bytes);
    header = (struct qbh_probe_header *)shared;
    header->magic = QBH_PROBE_MAGIC;
    header->abi_version = QBH_PROBE_ABI_VERSION;
    header->header_bytes = (uint32_t)sizeof(*header);
    header->total_bytes = (uint32_t)total_bytes;
    header->pattern = pattern;
    header->projection_variant = variant;
    header->weight_storage_variant = storage;
    header->physical_plan = physical_plan;
    header->requested_hvx_workers = requested_hvx_workers;
    header->compressed_slot_count = compressed_slot_count;
    header->expanded_chunk_slot_count = layout.expanded_slot_count;
    header->chunk_tiles = chunk_tiles;
    header->output_assembly_mode = output_assembly_mode;
    header->resource_lifetime_mode = resource_lifetime_mode;
    header->activation_offset = (uint32_t)activation_offset;
    header->weight_offset = (uint32_t)weight_offset;
    header->output_offset = (uint32_t)output_offset;
    header->input_zero_point = input_zero_point;
    header->repeat_count = repeats;
    header->dsp_status = QBH_PROBE_STATUS_HOST_INITIALIZED;

    activation = shared + activation_offset;
    stored_weights = shared + weight_offset;
    output = shared + output_offset;
    if (external_mode) {
        if (read_package_file(external_package, external_activation_name,
                              activation, layout.activation_bytes) != 0 ||
            read_package_file(external_package, external_weight_name,
                              stored_weights,
                              layout.stored_weight_bytes) != 0 ||
            read_package_file(external_package, external_reference_name,
                              external_reference,
                              layout.output_bytes) != 0) {
            fprintf(stderr, "failed to read EXP-0187 external package\n");
            goto cleanup;
        }
        for (uint32_t row = logical_m; row < layout.m; ++row) {
            memset(activation + (size_t)row * layout.k,
                   (int)input_zero_point, layout.k);
        }
        memset(output, 0xa5, layout.output_bytes);
        expanded_carrier_checksum =
            storage == QBH_WEIGHT_EXPANDED_S8
                ? carrier_checksum((const int8_t *)stored_weights,
                                   layout.stored_weight_bytes)
                : 0U;
        packed_w4_checksum =
            qbh_weight_storage_is_packed_w4(storage)
                ? carrier_checksum((const int8_t *)stored_weights,
                                   layout.stored_weight_bytes)
                : 0U;
        hmx_carrier_checksum = carrier_checksum(
            (const int8_t *)stored_weights, layout.stored_weight_bytes);
    } else {
        for (uint32_t call = 0; call < measured_rpc_calls; ++call) {
            int8_t *call_w4 =
                logical_w4 + (size_t)call * logical_weight_stride;
            int8_t *call_s8 =
                logical_s8 + (size_t)call * logical_weight_stride;
            uint8_t *call_scales =
                channel_scales + (size_t)call * layout.n;
            uint8_t *call_weights =
                stored_weights + (size_t)call * weight_stride;
            fill_pattern(&layout, pattern, logical_m, call * layout.n,
                         activation, call_w4, call_s8, call_scales);
            if (qbh_weight_storage_is_packed_w4(storage)) {
                pack_w4_bundles(
                    &layout, call_w4, call_s8, call_scales,
                    (int32_t)QBH_HMX_DEFAULT_ZERO_POINT, storage,
                    call_weights);
            } else {
                pack_expanded_s8_bundles(
                    &layout, call_s8,
                    (int32_t)QBH_HMX_DEFAULT_ZERO_POINT, call_weights);
            }
            memset(output + (size_t)call * output_stride, 0xa5,
                   layout.output_bytes);
        }
        expanded_carrier_checksum = carrier_checksum(
            logical_s8, logical_weight_bytes);
        packed_w4_checksum = carrier_checksum(
            logical_w4, logical_weight_bytes);
        hmx_carrier_checksum =
            (storage == QBH_WEIGHT_PACKED_W4_HMX_SCALE ||
             storage == QBH_WEIGHT_PACKED_W4_DIRECT_N)
                ? packed_w4_checksum
                : expanded_carrier_checksum;
    }

    session_open_start = monotonic_ns();
    rpc_result = qbh_session_open(&session);
    session_open_end = monotonic_ns();
    if (rpc_result != AEE_SUCCESS) {
        goto cleanup;
    }
    rpc_result = fastrpc_mmap(CDSP_DOMAIN_ID, shared_fd, shared, 0,
                              total_bytes, FASTRPC_MAP_FD);
    if (rpc_result != AEE_SUCCESS) {
        fprintf(stderr, "fastrpc_mmap failed: 0x%08x\n",
                (unsigned int)rpc_result);
        goto cleanup;
    }
    mapped = 1;

    if (resource_lifetime_mode ==
        QBH_RESOURCE_LIFETIME_PREPARED_SESSION) {
        prepare_start = monotonic_ns();
        prepare_result = qbh_session_prepare(&session);
        prepare_end = monotonic_ns();
        if (prepare_result != AEE_SUCCESS) {
            fprintf(stderr, "qwen3_probe_prepare failed: 0x%08x\n",
                    (unsigned int)prepare_result);
            goto cleanup;
        }
    }

    warmup_start = monotonic_ns();
    warmup_rpc_result = AEE_SUCCESS;
    for (uint32_t call = 0; call < measured_rpc_calls; ++call) {
        header->weight_offset =
            (uint32_t)(weight_offset + (size_t)call * weight_stride);
        header->output_offset =
            (uint32_t)(output_offset + (size_t)call * output_stride);
        warmup_rpc_result = qwen3_probe_run(
            session.handle, shared_fd, (uint32_t)total_bytes);
        if (warmup_rpc_result != AEE_SUCCESS) {
            break;
        }
    }
    warmup_end = monotonic_ns();
    if (warmup_rpc_result != AEE_SUCCESS) {
        fprintf(stderr,
                "warm-up qwen3_probe_run failed: 0x%08x dsp_status=%d "
                "hmx_resource=%d hmx_lock=%d dma=%d sync=%d\n",
                (unsigned int)warmup_rpc_result, header->dsp_status,
                header->hmx_resource_status, header->hmx_lock_status,
                header->dma_status, header->sync_status);
        goto cleanup;
    }
    warmup_output_checksum = canonical_output_checksum(
        &layout, output, output_stride, measured_rpc_calls);
    warmup_resource_vtcm_address = header->resource_vtcm_address;
    warmup_resource_hmx_context_id = header->resource_hmx_context_id;
    warmup_prepared_session_run_index =
        header->prepared_session_run_index;

    host_start = monotonic_ns();
    rpc_result = AEE_SUCCESS;
    for (uint32_t call = 0; call < measured_rpc_calls; ++call) {
        uint64_t call_start;
        uint64_t call_end;
        header->weight_offset =
            (uint32_t)(weight_offset + (size_t)call * weight_stride);
        header->output_offset =
            (uint32_t)(output_offset + (size_t)call * output_stride);
        call_start = monotonic_ns();
        rpc_result = qwen3_probe_run(session.handle, shared_fd,
                                     (uint32_t)total_bytes);
        call_end = monotonic_ns();
        measured_call_wall_ns[call] = call_end - call_start;
        measured_headers[call] = *header;
        if (rpc_result != AEE_SUCCESS) {
            break;
        }
    }
    host_end = monotonic_ns();
    if (rpc_result != AEE_SUCCESS) {
        fprintf(stderr,
                "qwen3_probe_run failed: 0x%08x dsp_status=%d "
                "hmx_resource=%d hmx_lock=%d dma=%d sync=%d\n",
                (unsigned int)rpc_result, header->dsp_status,
                header->hmx_resource_status, header->hmx_lock_status,
                header->dma_status, header->sync_status);
        goto cleanup;
    }
    measured_output_checksum = canonical_output_checksum(
        &layout, output, output_stride, measured_rpc_calls);

    if (resource_lifetime_mode ==
        QBH_RESOURCE_LIFETIME_PREPARED_SESSION) {
        release_start = monotonic_ns();
        release_result = qbh_session_release(&session);
        release_end = monotonic_ns();
        if (release_result != AEE_SUCCESS) {
            fprintf(stderr, "qwen3_probe_release failed: 0x%08x\n",
                    (unsigned int)release_result);
            goto cleanup;
        }
    }

    session_close_start = monotonic_ns();
    session_close_result = qbh_session_close(&session);
    session_close_end = monotonic_ns();
    if (session_close_result != AEE_SUCCESS) {
        fprintf(stderr, "qwen3_probe_close failed: 0x%08x\n",
                (unsigned int)session_close_result);
        goto cleanup;
    }

    reference_start = monotonic_ns();
    mismatches = 0U;
    reference_min = UINT8_MAX;
    reference_max = 0U;
    reference_checksum = 0U;
    if (external_mode) {
        mismatches = validate_external_output(
            &layout, output, external_reference, logical_m,
            &reference_min, &reference_max, &reference_checksum);
    } else {
        for (uint32_t call = 0; call < measured_rpc_calls; ++call) {
            uint8_t call_min;
            uint8_t call_max;
            uint64_t call_checksum;
            mismatches += validate_output(
                &layout, activation,
                logical_s8 + (size_t)call * logical_weight_stride,
                output + (size_t)call * output_stride,
                (int32_t)header->input_zero_point, reference_accumulators,
                &call_min, &call_max, &call_checksum);
            if (call_min < reference_min) {
                reference_min = call_min;
            }
            if (call_max > reference_max) {
                reference_max = call_max;
            }
            reference_checksum += call_checksum;
        }
    }
    reference_end = monotonic_ns();

    uint64_t aggregate_dsp_total_ticks = 0;
    uint64_t aggregate_pipeline_ticks = 0;
    uint64_t aggregate_activation_stage_ticks = 0;
    uint64_t aggregate_weight_stage_ticks = 0;
    uint64_t aggregate_weight_expand_ticks = 0;
    uint64_t aggregate_hmx_compute_ticks = 0;
    uint64_t aggregate_hmx_ready_wait_ticks = 0;
    uint64_t aggregate_output_assembly_ticks = 0;
    uint64_t aggregate_input_cache_ticks = 0;
    uint64_t aggregate_output_cache_ticks = 0;
    uint32_t aggregate_hmx_execution_count = 0;
    uint32_t aggregate_hmx_stream_count = 0;
    uint32_t aggregate_weight_bundle_stage_count = 0;
    uint32_t aggregate_output_tile_count = 0;
    for (uint32_t call = 0; call < measured_rpc_calls; ++call) {
        const struct qbh_probe_header *call_header =
            &measured_headers[call];
        aggregate_dsp_total_ticks += call_header->dsp_total_ticks;
        aggregate_pipeline_ticks += call_header->pipeline_ticks;
        aggregate_activation_stage_ticks +=
            call_header->activation_stage_ticks;
        aggregate_weight_stage_ticks += call_header->weight_stage_ticks;
        aggregate_weight_expand_ticks += call_header->weight_expand_ticks;
        aggregate_hmx_compute_ticks += call_header->hmx_compute_ticks;
        aggregate_hmx_ready_wait_ticks +=
            call_header->hmx_ready_wait_ticks;
        aggregate_output_assembly_ticks +=
            call_header->output_assembly_ticks;
        aggregate_input_cache_ticks += call_header->input_cache_ticks;
        aggregate_output_cache_ticks += call_header->output_cache_ticks;
        aggregate_hmx_execution_count +=
            call_header->hmx_execution_count;
        aggregate_hmx_stream_count += call_header->hmx_stream_count;
        aggregate_weight_bundle_stage_count +=
            call_header->weight_bundle_stage_count;
        aggregate_output_tile_count += call_header->output_tile_count;
    }

    printf("{\"experiment\":\"%s\","
           "\"input_source\":\"%s\","
           "\"weight_storage\":\"%s\","
           "\"physical_plan\":\"%s\","
           "\"requested_hvx_workers\":%" PRIu32 ","
           "\"compressed_slot_count\":%" PRIu32 ","
           "\"expanded_chunk_slot_count\":%" PRIu32 ","
           "\"chunk_tiles\":%" PRIu32 ","
           "\"output_assembly_mode\":\"%s\","
           "\"resource_lifetime_mode\":\"%s\","
           "\"host_invocation_mode\":\"%s\","
           "\"measured_rpc_calls\":%" PRIu32 ","
           "\"dma_bundle_batch\":%" PRIu32 ","
           "\"projection\":\"%s\",\"pattern\":\"%s\","
           "\"logical_m\":%" PRIu32 ","
           "\"repeat_count\":%" PRIu32 ","
           "\"rpc_result\":%d,\"dsp_status\":%d,"
           "\"mismatches\":%" PRIu32 ","
           "\"reference_min\":%u,\"reference_max\":%u,"
           "\"reference_checksum\":%" PRIu64 ","
           "\"expanded_carrier_checksum\":%" PRIu64 ","
           "\"packed_w4_checksum\":%" PRIu64 ","
           "\"hmx_carrier_checksum\":%" PRIu64 ","
           "\"warmup_output_checksum\":%" PRIu64 ","
           "\"measured_output_checksum\":%" PRIu64 ","
           "\"warmup_resource_vtcm_address\":%" PRIu32 ","
           "\"warmup_resource_hmx_context_id\":%" PRIu32 ","
           "\"warmup_prepared_session_run_index\":%" PRIu32 ","
           "\"session_open_wall_ns\":%" PRIu64 ","
           "\"prepare_result\":%d,"
           "\"prepare_wall_ns\":%" PRIu64 ","
           "\"warmup_rpc_result\":%d,"
           "\"warmup_host_wall_ns\":%" PRIu64 ","
           "\"release_result\":%d,"
           "\"release_wall_ns\":%" PRIu64 ","
           "\"session_close_result\":%d,"
           "\"session_close_wall_ns\":%" PRIu64 ","
           "\"reference_wall_ns\":%" PRIu64 ","
           "\"host_wall_ns\":%" PRIu64 ","
           "\"first_call_host_wall_ns\":%" PRIu64 ","
           "\"second_call_host_wall_ns\":%" PRIu64 ","
           "\"projection_m\":%" PRIu32 ","
           "\"projection_k\":%" PRIu32 ","
           "\"projection_n\":%" PRIu32 ","
           "\"k_tile_count\":%" PRIu32 ","
           "\"n_tile_count\":%" PRIu32 ","
           "\"stored_weight_bundle_bytes\":%" PRIu32 ","
           "\"expanded_weight_bundle_bytes\":%" PRIu32 ","
           "\"stored_weight_bytes_per_repeat\":%" PRIu32 ","
           "\"expanded_weight_bytes_per_repeat\":%" PRIu32 ","
           "\"vtcm_plan_bytes\":%" PRIu32 ","
           "\"k_streams_per_output\":%" PRIu32 ","
           "\"qtimer_ticks\":%" PRIu64 ","
           "\"pcycles\":%" PRIu64 ","
           "\"activation_stage_ticks\":%" PRIu64 ","
           "\"weight_stage_ticks\":%" PRIu64 ","
           "\"weight_expand_ticks\":%" PRIu64 ","
           "\"hmx_compute_ticks\":%" PRIu64 ","
           "\"hmx_ready_wait_ticks\":%" PRIu64 ","
           "\"producer_slot_wait_ticks\":%" PRIu64 ","
           "\"expanded_slot_wait_ticks\":%" PRIu64 ","
           "\"pipeline_ticks\":%" PRIu64 ","
           "\"output_assembly_ticks\":%" PRIu64 ","
           "\"dsp_total_ticks\":%" PRIu64 ","
           "\"aggregate_dsp_total_ticks\":%" PRIu64 ","
           "\"aggregate_pipeline_ticks\":%" PRIu64 ","
           "\"aggregate_activation_stage_ticks\":%" PRIu64 ","
           "\"aggregate_weight_stage_ticks\":%" PRIu64 ","
           "\"aggregate_weight_expand_ticks\":%" PRIu64 ","
           "\"aggregate_hmx_compute_ticks\":%" PRIu64 ","
           "\"aggregate_hmx_ready_wait_ticks\":%" PRIu64 ","
           "\"aggregate_output_assembly_ticks\":%" PRIu64 ","
           "\"input_cache_ticks\":%" PRIu64 ","
           "\"output_cache_ticks\":%" PRIu64 ","
           "\"aggregate_input_cache_ticks\":%" PRIu64 ","
           "\"aggregate_output_cache_ticks\":%" PRIu64 ","
           "\"aggregate_hmx_execution_count\":%" PRIu32 ","
           "\"aggregate_hmx_stream_count\":%" PRIu32 ","
           "\"aggregate_weight_bundle_stage_count\":%" PRIu32 ","
           "\"aggregate_output_tile_count\":%" PRIu32 ","
           "\"vtcm_requested_bytes\":%" PRIu32 ","
           "\"vtcm_acquired_bytes\":%" PRIu32 ","
           "\"hmx_resource_status\":%d,\"hmx_lock_status\":%d,"
           "\"hmx_unlock_status\":%d,\"hmx_release_status\":%d,"
           "\"hmx_thread_create_status\":%d,"
           "\"hmx_thread_join_status\":%d,"
           "\"hmx_power_up_status\":%d,"
           "\"hmx_power_down_status\":%d,"
           "\"dcvs_power_setup_status\":%d,"
           "\"dcvs_power_reset_status\":%d,"
           "\"hmx_execution_count\":%" PRIu32 ","
           "\"hmx_stream_count\":%" PRIu32 ","
           "\"weight_expand_count\":%" PRIu32 ","
           "\"hvx_lock_status\":%d,\"hvx_unlock_status\":%d,"
           "\"activation_stage_count\":%" PRIu32 ","
           "\"weight_bundle_stage_count\":%" PRIu32 ","
           "\"output_tile_count\":%" PRIu32 ","
           "\"dma_submit_count\":%" PRIu32 ","
           "\"dma_wait_count\":%" PRIu32 ","
           "\"dma_descriptor_count\":%" PRIu32 ","
           "\"dma_chain_count\":%" PRIu32 ","
           "\"dma_descriptor_completion_count\":%" PRIu32 ","
           "\"dma_descriptor_timeout_count\":%" PRIu32 ","
           "\"output_dma_submit_count\":%" PRIu32 ","
           "\"output_dma_wait_count\":%" PRIu32 ","
           "\"output_dma_descriptor_count\":%" PRIu32 ","
           "\"output_dma_chain_count\":%" PRIu32 ","
           "\"output_dma_descriptor_completion_count\":%" PRIu32 ","
           "\"output_dma_descriptor_timeout_count\":%" PRIu32 ","
           "\"output_dma_status\":%d,"
           "\"resource_setup_in_run\":%" PRIu32 ","
           "\"resource_release_in_run\":%" PRIu32 ","
           "\"prepared_session_run_index\":%" PRIu32 ","
           "\"resource_vtcm_address\":%" PRIu32 ","
           "\"resource_hmx_context_id\":%" PRIu32 ","
           "\"weight_slot_reuse_count\":%" PRIu32 ","
           "\"expanded_chunk_slot_reuse_count\":%" PRIu32 ","
           "\"chunks_per_output\":%" PRIu32 ","
           "\"chunk_expand_count\":%" PRIu32 ","
           "\"hvx_units_128b\":%" PRIu32 ","
           "\"hvx_workers_created\":%" PRIu32 ","
           "\"hvx_workers_locked\":%" PRIu32 ","
           "\"hvx_max_active_workers\":%" PRIu32 ","
           "\"hvx_hmx_overlap_observed\":%" PRIu32 ","
           "\"hvx_parallel_overlap_observed\":%" PRIu32 ","
           "\"hvx_thread_create_status\":%d,"
           "\"hvx_thread_join_status\":%d,"
           "\"expand_window_start\":%" PRIu64 ","
           "\"expand_window_end\":%" PRIu64 ","
           "\"hmx_window_start\":%" PRIu64 ","
           "\"hmx_window_end\":%" PRIu64 ","
           "\"hvx_worker_ticks\":[%" PRIu64 ",%" PRIu64 ","
           "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "],"
           "\"dma_status\":%d,\"sync_status\":%d,"
           "\"streaming_region_publish_count\":%" PRIu32 ","
           "\"streaming_ready_timeout_count\":%" PRIu32 "}\n",
           external_mode ? "EXP-0187" : "EXP-0019",
           external_mode ? "real_layer14" : "synthetic",
           storage_name(storage),
           physical_plan_name(physical_plan, requested_hvx_workers,
                              compressed_slot_count, chunk_tiles),
           requested_hvx_workers, compressed_slot_count,
           header->expanded_chunk_slot_count, chunk_tiles,
           output_assembly_name(output_assembly_mode),
           resource_lifetime_name(resource_lifetime_mode),
           host_invocation_name(host_invocation_mode),
           measured_rpc_calls,
           qbh_physical_plan_dma_bundle_batch(physical_plan),
           projection_name(variant),
           pattern_name(pattern), logical_m, repeats, rpc_result,
           header->dsp_status,
           mismatches, (unsigned int)reference_min,
           (unsigned int)reference_max, reference_checksum,
           expanded_carrier_checksum, packed_w4_checksum,
           hmx_carrier_checksum, warmup_output_checksum,
           measured_output_checksum, warmup_resource_vtcm_address,
           warmup_resource_hmx_context_id,
           warmup_prepared_session_run_index,
           session_open_end - session_open_start,
           prepare_result, prepare_end - prepare_start,
           warmup_rpc_result, warmup_end - warmup_start,
           release_result, release_end - release_start,
           session_close_result,
           session_close_end - session_close_start,
           reference_end - reference_start,
           host_end - host_start, measured_call_wall_ns[0],
           measured_call_wall_ns[1], header->projection_m,
           header->projection_k, header->projection_n,
           header->k_tile_count, header->n_tile_count,
           header->stored_weight_bundle_bytes,
           header->expanded_weight_bundle_bytes,
           header->stored_weight_bytes_per_repeat,
           header->expanded_weight_bytes_per_repeat,
           header->vtcm_plan_bytes, header->k_streams_per_output,
           header->qtimer_elapsed,
           header->pcycles_end - header->pcycles_start,
           header->activation_stage_ticks, header->weight_stage_ticks,
           header->weight_expand_ticks, header->hmx_compute_ticks,
           header->hmx_ready_wait_ticks,
           header->producer_slot_wait_ticks,
           header->expanded_slot_wait_ticks, header->pipeline_ticks,
           header->output_assembly_ticks, header->dsp_total_ticks,
           aggregate_dsp_total_ticks, aggregate_pipeline_ticks,
           aggregate_activation_stage_ticks,
           aggregate_weight_stage_ticks,
           aggregate_weight_expand_ticks,
           aggregate_hmx_compute_ticks,
           aggregate_hmx_ready_wait_ticks,
           aggregate_output_assembly_ticks,
           header->input_cache_ticks, header->output_cache_ticks,
           aggregate_input_cache_ticks,
           aggregate_output_cache_ticks,
           aggregate_hmx_execution_count,
           aggregate_hmx_stream_count,
           aggregate_weight_bundle_stage_count,
           aggregate_output_tile_count,
           header->vtcm_requested_bytes, header->vtcm_acquired_bytes,
           header->hmx_resource_status, header->hmx_lock_status,
           header->hmx_unlock_status, header->hmx_release_status,
           header->hmx_thread_create_status,
           header->hmx_thread_join_status,
           header->hmx_power_up_status, header->hmx_power_down_status,
           header->dcvs_power_setup_status,
           header->dcvs_power_reset_status,
           header->hmx_execution_count, header->hmx_stream_count,
           header->weight_expand_count, header->hvx_lock_status,
           header->hvx_unlock_status, header->activation_stage_count,
           header->weight_bundle_stage_count, header->output_tile_count,
           header->dma_submit_count, header->dma_wait_count,
           header->dma_descriptor_count, header->dma_chain_count,
           header->dma_descriptor_completion_count,
           header->dma_descriptor_timeout_count,
           header->output_dma_submit_count,
           header->output_dma_wait_count,
           header->output_dma_descriptor_count,
           header->output_dma_chain_count,
           header->output_dma_descriptor_completion_count,
           header->output_dma_descriptor_timeout_count,
           header->output_dma_status,
           header->resource_setup_in_run,
           header->resource_release_in_run,
           header->prepared_session_run_index,
           header->resource_vtcm_address,
           header->resource_hmx_context_id,
           header->weight_slot_reuse_count,
           header->expanded_chunk_slot_reuse_count,
           header->chunks_per_output, header->chunk_expand_count,
           header->hvx_units_128b, header->hvx_workers_created,
           header->hvx_workers_locked,
           header->hvx_max_active_workers,
           header->hvx_hmx_overlap_observed,
           header->hvx_parallel_overlap_observed,
           header->hvx_thread_create_status,
           header->hvx_thread_join_status,
           header->expand_window_start, header->expand_window_end,
           header->hmx_window_start, header->hmx_window_end,
           header->hvx_worker_ticks[0], header->hvx_worker_ticks[1],
           header->hvx_worker_ticks[2], header->hvx_worker_ticks[3],
           header->hvx_worker_ticks[4], header->hvx_worker_ticks[5],
           header->dma_status,
           header->sync_status,
           header->streaming_region_publish_count,
           header->streaming_ready_timeout_count);

    expected_weight_stages = repeats * layout.n_tiles;
    expected_reuses = expected_weight_stages > compressed_slot_count
                          ? expected_weight_stages - compressed_slot_count
                          : 0U;
    expected_dma_waits =
        2U * layout.k_tiles +
        (qbh_physical_plan_uses_linked_dma(physical_plan)
             ? expected_weight_stages +
                   2U * expected_weight_stages /
                       qbh_physical_plan_dma_bundle_batch(physical_plan)
             : 2U * expected_weight_stages /
                   qbh_physical_plan_dma_bundle_batch(physical_plan));
    expected_dma_descriptors =
        layout.k_tiles +
        (qbh_physical_plan_uses_linked_dma(physical_plan)
             ? expected_weight_stages
             : expected_weight_stages /
                   qbh_physical_plan_dma_bundle_batch(physical_plan));
    expected_dma_chains =
        qbh_physical_plan_uses_linked_dma(physical_plan)
            ? expected_weight_stages /
                  qbh_physical_plan_dma_bundle_batch(physical_plan)
            : 0U;
    expected_superchunks =
        expected_weight_stages * layout.chunks_per_output;
    expected_expands =
        qbh_weight_storage_is_packed_w4(storage) &&
        !qbh_weight_storage_is_direct_n(storage)
                           ? expected_weight_stages *
                                 (qbh_physical_plan_is_streaming(physical_plan)
                                      ? layout.k_tiles /
                                            QBH_W4_STREAM_REGION_TILES
                                      : (qbh_physical_plan_is_chunked(
                                             physical_plan)
                                             ? layout.chunks_per_output
                                             : 1U))
                           : 0U;
    int all_measured_calls_valid = 1;
    for (uint32_t call = 0; call < measured_rpc_calls; ++call) {
        const struct qbh_probe_header *call_header =
            &measured_headers[call];
        if (call_header->dsp_status != QBH_PROBE_STATUS_OK ||
            call_header->resource_vtcm_address !=
                header->resource_vtcm_address ||
            call_header->resource_hmx_context_id !=
                header->resource_hmx_context_id ||
            call_header->hmx_execution_count !=
                repeats * layout.hmx_pairs_per_repeat ||
            call_header->hmx_stream_count !=
                repeats * layout.hmx_streams_per_repeat ||
            call_header->weight_bundle_stage_count !=
                expected_weight_stages ||
            call_header->output_tile_count != expected_weight_stages ||
            call_header->dma_descriptor_timeout_count != 0U ||
            call_header->output_dma_descriptor_timeout_count != 0U ||
            call_header->streaming_ready_timeout_count != 0U ||
            call_header->dma_status != 0 ||
            call_header->sync_status != 0) {
            all_measured_calls_valid = 0;
        }
    }
    if (header->dsp_status == QBH_PROBE_STATUS_OK && mismatches == 0 &&
        all_measured_calls_valid &&
        warmup_rpc_result == AEE_SUCCESS &&
        warmup_output_checksum == measured_output_checksum &&
        header->projection_m == layout.m &&
        header->projection_k == layout.k &&
        header->projection_n == layout.n &&
        header->k_tile_count == layout.k_tiles &&
        header->n_tile_count == layout.n_tiles &&
        header->compressed_slot_count == layout.compressed_slot_count &&
        header->expanded_chunk_slot_count == layout.expanded_slot_count &&
        header->output_assembly_mode == output_assembly_mode &&
        header->resource_lifetime_mode == resource_lifetime_mode &&
        header->chunk_tiles == layout.chunk_tiles &&
        header->stored_weight_bundle_bytes ==
            layout.stored_weight_bundle_bytes &&
        header->expanded_weight_bundle_bytes ==
            layout.expanded_weight_bundle_bytes &&
        header->stored_weight_bytes_per_repeat ==
            layout.stored_weight_bytes &&
        header->expanded_weight_bytes_per_repeat ==
            layout.expanded_weight_bytes &&
        header->vtcm_plan_bytes == layout.vtcm_plan_bytes &&
        header->k_streams_per_output == layout.k_streams_per_output &&
        header->qtimer_end > header->qtimer_start &&
        header->pipeline_ticks > 0 && header->hmx_compute_ticks > 0 &&
        header->vtcm_acquired_bytes >= QBH_W4U8_VTCM_BYTES &&
        header->hmx_resource_status == 0 && header->hmx_lock_status == 0 &&
        header->hmx_unlock_status == 0 && header->hmx_release_status == 0 &&
        header->hmx_thread_create_status == 0 &&
        header->hmx_thread_join_status == 0 &&
        header->hmx_power_up_status == 0 &&
        header->hmx_power_down_status == 0 &&
        header->dcvs_power_setup_status == 0 &&
        header->dcvs_power_reset_status == 0 &&
        header->hmx_execution_count ==
            repeats * layout.hmx_pairs_per_repeat &&
        header->hmx_stream_count ==
            repeats * layout.hmx_streams_per_repeat &&
        header->weight_expand_count == expected_expands &&
        header->activation_stage_count == layout.k_tiles &&
        header->weight_bundle_stage_count == expected_weight_stages &&
        header->output_tile_count == expected_weight_stages &&
        header->dma_submit_count ==
            layout.k_tiles +
                expected_weight_stages /
                    qbh_physical_plan_dma_bundle_batch(physical_plan) &&
        header->dma_wait_count == expected_dma_waits &&
        header->dma_descriptor_count == expected_dma_descriptors &&
        header->dma_chain_count == expected_dma_chains &&
        header->dma_descriptor_completion_count ==
            expected_dma_descriptors &&
        header->dma_descriptor_timeout_count == 0U &&
        header->output_dma_submit_count ==
            (output_assembly_mode ==
                     QBH_OUTPUT_ASSEMBLY_LINKED_2D_DMA
                 ? 1U
                 : 0U) &&
        header->output_dma_wait_count ==
            (output_assembly_mode ==
                     QBH_OUTPUT_ASSEMBLY_LINKED_2D_DMA
                 ? 2U
                 : 0U) &&
        header->output_dma_descriptor_count ==
            (output_assembly_mode ==
                     QBH_OUTPUT_ASSEMBLY_LINKED_2D_DMA
                 ? layout.n_tiles
                 : 0U) &&
        header->output_dma_chain_count ==
            (output_assembly_mode ==
                     QBH_OUTPUT_ASSEMBLY_LINKED_2D_DMA
                 ? 1U
                 : 0U) &&
        header->output_dma_descriptor_completion_count ==
            (output_assembly_mode ==
                     QBH_OUTPUT_ASSEMBLY_LINKED_2D_DMA
                 ? layout.n_tiles
                 : 0U) &&
        header->output_dma_descriptor_timeout_count == 0U &&
        header->output_dma_status == 0 &&
        header->resource_setup_in_run ==
            (resource_lifetime_mode == QBH_RESOURCE_LIFETIME_TRANSIENT
                 ? 1U
                 : 0U) &&
        header->resource_release_in_run ==
            (resource_lifetime_mode == QBH_RESOURCE_LIFETIME_TRANSIENT
                 ? 1U
                 : 0U) &&
        header->prepared_session_run_index ==
            (resource_lifetime_mode ==
                     QBH_RESOURCE_LIFETIME_PREPARED_SESSION
                 ? 2U * measured_rpc_calls
                 : 0U) &&
        warmup_prepared_session_run_index ==
            (resource_lifetime_mode ==
                     QBH_RESOURCE_LIFETIME_PREPARED_SESSION
                 ? measured_rpc_calls
                 : 0U) &&
        header->resource_vtcm_address != 0U &&
        header->resource_hmx_context_id != 0U &&
        (resource_lifetime_mode == QBH_RESOURCE_LIFETIME_TRANSIENT ||
         (warmup_resource_vtcm_address ==
              header->resource_vtcm_address &&
          warmup_resource_hmx_context_id ==
              header->resource_hmx_context_id)) &&
        header->weight_slot_reuse_count == expected_reuses &&
        header->chunks_per_output == layout.chunks_per_output &&
        header->chunk_expand_count ==
            (qbh_physical_plan_is_chunked(physical_plan)
                 ? expected_expands
                 : 0U) &&
        header->expanded_chunk_slot_reuse_count ==
            (qbh_physical_plan_is_chunked(physical_plan) &&
                     expected_superchunks >
                         layout.expanded_slot_count
                 ? expected_superchunks -
                       layout.expanded_slot_count
                 : 0U) &&
        header->streaming_region_publish_count ==
            (qbh_physical_plan_is_streaming(physical_plan)
                 ? expected_expands
                 : 0U) &&
        header->streaming_ready_timeout_count == 0U &&
        (qbh_physical_plan_is_full_bundle(physical_plan) ||
         (header->hvx_workers_created == requested_hvx_workers &&
          header->hvx_workers_locked == requested_hvx_workers &&
          header->hvx_thread_create_status == 0 &&
          header->hvx_thread_join_status == 0)) &&
        header->dma_status == 0 && header->sync_status == 0 &&
        header->hvx_lock_status == 0 && header->hvx_unlock_status == 0) {
        result = EXIT_SUCCESS;
    }

cleanup:
    if (mapped) {
        int unmap_result = fastrpc_munmap(CDSP_DOMAIN_ID, shared_fd, shared,
                                          total_bytes);
        if (unmap_result != AEE_SUCCESS) {
            fprintf(stderr, "fastrpc_munmap failed: 0x%08x\n",
                    (unsigned int)unmap_result);
            result = EXIT_FAILURE;
        }
    }
    (void)qbh_session_close(&session);
    if (shared != NULL) {
        rpcmem_free(shared);
    }
    free(external_reference);
    free(reference_accumulators);
    free(channel_scales);
    free(logical_s8);
    free(logical_w4);
    return result;
}
