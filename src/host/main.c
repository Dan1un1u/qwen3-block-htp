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

static const char *storage_name(uint32_t storage) {
    switch (storage) {
        case QBH_WEIGHT_EXPANDED_S8:
            return "expanded_s8_control";
        case QBH_WEIGHT_PACKED_W4:
            return "packed_w4_hvx_prescale";
        case QBH_WEIGHT_PACKED_W4_HMX_SCALE:
            return "packed_w4_hmx_postscale";
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
    if (parse_u32(text, &parsed) == 0 &&
        (parsed == QBH_WEIGHT_EXPANDED_S8 ||
         parsed == QBH_WEIGHT_PACKED_W4 ||
         parsed == QBH_WEIGHT_PACKED_W4_HMX_SCALE)) {
        *storage = parsed;
        return 0;
    }
    return -1;
}

static const char *physical_plan_name(uint32_t physical_plan,
                                      uint32_t hvx_workers,
                                      uint32_t compressed_slots,
                                      uint32_t chunk_tiles) {
    if (physical_plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE) {
        return "exp0005_full_bundle_control";
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
    return -1;
}

static const char *projection_name(uint32_t variant) {
    switch (variant) {
        case QBH_PROJECTION_GATE_UP:
            return "gate_up";
        case QBH_PROJECTION_DOWN:
            return "down";
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
    if (parse_u32(text, &parsed) == 0 &&
        (parsed == QBH_PROJECTION_GATE_UP ||
         parsed == QBH_PROJECTION_DOWN)) {
        *variant = parsed;
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
                         uint32_t pattern, uint8_t *activation,
                         int8_t *logical_w4, int8_t *logical_s8,
                         uint8_t *channel_scales) {
    for (uint32_t output_channel = 0; output_channel < layout->n;
         ++output_channel) {
        channel_scales[output_channel] =
            channel_scale(pattern, output_channel);
    }

    for (uint32_t row = 0; row < layout->m; ++row) {
        for (uint32_t input_channel = 0; input_channel < layout->k;
             ++input_channel) {
            uint8_t value;
            switch (pattern) {
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
            size_t offset = qbh_projection_logical_weight_offset(
                layout, input_channel, output_channel);
            switch (pattern) {
                case QBH_PATTERN_IDENTITY:
                    q4 = input_channel == output_channel ? INT8_C(1)
                                                         : INT8_C(0);
                    break;
                case QBH_PATTERN_SIGNED:
                    q4 = (int8_t)((int32_t)((input_channel * 7U +
                                             output_channel * 5U) %
                                            7U) -
                                  3);
                    break;
                case QBH_PATTERN_STRUCTURED:
                    q4 = ((input_channel + 3U * output_channel) % 5U) == 0U
                             ? (int8_t)((int32_t)((input_channel +
                                                  output_channel) %
                                                 5U) -
                                        2)
                             : INT8_C(0);
                    break;
                default:
                    q4 = ((input_channel + output_channel) & 1U) != 0U
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
                    size_t packed_offset =
                        bundle_offset + physical_s8_offset / 2U;
                    uint8_t nibble = (uint8_t)logical_w4[
                        qbh_projection_logical_weight_offset(
                            layout, logical_k, logical_n)] &
                                     UINT8_C(0x0f);
                    if ((physical_s8_offset & 1U) == 0U) {
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
        if (storage == QBH_WEIGHT_PACKED_W4_HMX_SCALE) {
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

int main(int argc, char **argv) {
    struct qbh_session session = {(remote_handle64)-1};
    struct qbh_projection_layout layout;
    struct qbh_probe_header *header = NULL;
    uint8_t *shared = NULL;
    uint8_t *activation;
    uint8_t *stored_weights;
    uint8_t *output;
    int8_t *logical_w4 = NULL;
    int8_t *logical_s8 = NULL;
    uint8_t *channel_scales = NULL;
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
    size_t activation_offset;
    size_t weight_offset;
    size_t output_offset;
    size_t total_bytes;
    uint64_t host_start;
    uint64_t host_end;
    uint64_t reference_start;
    uint64_t reference_end;
    uint64_t reference_checksum;
    uint64_t expanded_carrier_checksum;
    uint64_t packed_w4_checksum;
    uint64_t hmx_carrier_checksum;
    uint8_t reference_min;
    uint8_t reference_max;
    uint32_t mismatches;
    uint32_t expected_weight_stages;
    uint32_t expected_reuses;
    uint32_t expected_dma_waits;
    uint32_t expected_expands;
    int shared_fd = -1;
    int mapped = 0;
    int result = EXIT_FAILURE;
    int rpc_result;

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
    if (argc > 6) {
        fprintf(stderr,
                "usage: %s [packed_w4_hvx_prescale|"
                "packed_w4_hmx_postscale|expanded_s8_control] "
                "[gate_up|down] "
                "[identity|signed|structured|boundary] [repeat] "
                "[exp0005_full_bundle_control|"
                "exp0006_slots2_chunk32_control|slots3_chunk32|"
                "slots4_chunk32|slots2_chunk16|slots3_chunk16]\n",
                argv[0]);
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

    activation_offset = align_up(sizeof(*header), QBH_PROBE_ALIGNMENT);
    weight_offset = activation_offset +
                    align_up(layout.activation_bytes,
                             QBH_PROBE_ALIGNMENT);
    output_offset = weight_offset +
                    align_up(layout.stored_weight_bytes,
                             QBH_PROBE_ALIGNMENT);
    total_bytes = output_offset +
                  align_up(layout.output_bytes, QBH_PROBE_ALIGNMENT);
    if (total_bytes > UINT32_MAX || total_bytes > INT_MAX) {
        fprintf(stderr, "shared allocation is too large: %zu bytes\n",
                total_bytes);
        goto cleanup;
    }

    logical_w4 = malloc(layout.logical_weight_bytes);
    logical_s8 = malloc(layout.logical_weight_bytes);
    channel_scales = malloc(layout.n);
    reference_accumulators = calloc(
        (size_t)layout.m * layout.n, sizeof(*reference_accumulators));
    if (logical_w4 == NULL || logical_s8 == NULL ||
        channel_scales == NULL || reference_accumulators == NULL) {
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
    header->chunk_tiles = chunk_tiles;
    header->activation_offset = (uint32_t)activation_offset;
    header->weight_offset = (uint32_t)weight_offset;
    header->output_offset = (uint32_t)output_offset;
    header->input_zero_point = QBH_HMX_DEFAULT_ZERO_POINT;
    header->repeat_count = repeats;
    header->dsp_status = QBH_PROBE_STATUS_HOST_INITIALIZED;

    activation = shared + activation_offset;
    stored_weights = shared + weight_offset;
    output = shared + output_offset;
    fill_pattern(&layout, pattern, activation, logical_w4, logical_s8,
                 channel_scales);
    if (qbh_weight_storage_is_packed_w4(storage)) {
        pack_w4_bundles(&layout, logical_w4, logical_s8, channel_scales,
                        (int32_t)QBH_HMX_DEFAULT_ZERO_POINT,
                        storage, stored_weights);
    } else {
        pack_expanded_s8_bundles(
            &layout, logical_s8,
            (int32_t)QBH_HMX_DEFAULT_ZERO_POINT, stored_weights);
    }
    expanded_carrier_checksum = carrier_checksum(
        logical_s8, layout.logical_weight_bytes);
    packed_w4_checksum = carrier_checksum(
        logical_w4, layout.logical_weight_bytes);
    hmx_carrier_checksum =
        storage == QBH_WEIGHT_PACKED_W4_HMX_SCALE
            ? packed_w4_checksum
            : expanded_carrier_checksum;
    memset(output, 0xa5, layout.output_bytes);

    rpc_result = qbh_session_open(&session);
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

    host_start = monotonic_ns();
    rpc_result = qwen3_probe_run(session.handle, shared_fd,
                                 (uint32_t)total_bytes);
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

    reference_start = monotonic_ns();
    mismatches = validate_output(
        &layout, activation, logical_s8, output,
        (int32_t)header->input_zero_point, reference_accumulators,
        &reference_min, &reference_max, &reference_checksum);
    reference_end = monotonic_ns();

    printf("{\"experiment\":\"EXP-0008\","
           "\"weight_storage\":\"%s\","
           "\"physical_plan\":\"%s\","
           "\"requested_hvx_workers\":%" PRIu32 ","
           "\"compressed_slot_count\":%" PRIu32 ","
           "\"chunk_tiles\":%" PRIu32 ","
           "\"projection\":\"%s\",\"pattern\":\"%s\","
           "\"repeat_count\":%" PRIu32 ","
           "\"rpc_result\":%d,\"dsp_status\":%d,"
           "\"mismatches\":%" PRIu32 ","
           "\"reference_min\":%u,\"reference_max\":%u,"
           "\"reference_checksum\":%" PRIu64 ","
           "\"expanded_carrier_checksum\":%" PRIu64 ","
           "\"packed_w4_checksum\":%" PRIu64 ","
           "\"hmx_carrier_checksum\":%" PRIu64 ","
           "\"reference_wall_ns\":%" PRIu64 ","
           "\"host_wall_ns\":%" PRIu64 ","
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
           "\"dma_status\":%d,\"sync_status\":%d}\n",
           storage_name(storage),
           physical_plan_name(physical_plan, requested_hvx_workers,
                              compressed_slot_count, chunk_tiles),
           requested_hvx_workers, compressed_slot_count, chunk_tiles,
           projection_name(variant),
           pattern_name(pattern), repeats, rpc_result, header->dsp_status,
           mismatches, (unsigned int)reference_min,
           (unsigned int)reference_max, reference_checksum,
           expanded_carrier_checksum, packed_w4_checksum,
           hmx_carrier_checksum, reference_end - reference_start,
           host_end - host_start, header->projection_m,
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
           header->sync_status);

    expected_weight_stages = repeats * layout.n_tiles;
    expected_reuses = expected_weight_stages > compressed_slot_count
                          ? expected_weight_stages - compressed_slot_count
                          : 0U;
    expected_dma_waits =
        2U * (layout.k_tiles + expected_weight_stages);
    expected_expands = qbh_weight_storage_is_packed_w4(storage)
                           ? expected_weight_stages *
                                 (physical_plan == QBH_PHYSICAL_PLAN_CHUNKED
                                      ? layout.chunks_per_output
                                      : 1U)
                           : 0U;
    if (header->dsp_status == QBH_PROBE_STATUS_OK && mismatches == 0 &&
        header->projection_m == layout.m &&
        header->projection_k == layout.k &&
        header->projection_n == layout.n &&
        header->k_tile_count == layout.k_tiles &&
        header->n_tile_count == layout.n_tiles &&
        header->compressed_slot_count == layout.compressed_slot_count &&
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
            layout.k_tiles + expected_weight_stages &&
        header->dma_wait_count == expected_dma_waits &&
        header->weight_slot_reuse_count == expected_reuses &&
        header->chunks_per_output == layout.chunks_per_output &&
        header->chunk_expand_count ==
            (physical_plan == QBH_PHYSICAL_PLAN_CHUNKED
                 ? expected_expands
                 : 0U) &&
        header->expanded_chunk_slot_reuse_count ==
            (physical_plan == QBH_PHYSICAL_PLAN_CHUNKED &&
                     expected_expands >
                         QBH_W4_EXPANDED_CHUNK_SLOT_COUNT
                 ? expected_expands -
                       QBH_W4_EXPANDED_CHUNK_SLOT_COUNT
                 : 0U) &&
        (physical_plan == QBH_PHYSICAL_PLAN_FULL_BUNDLE ||
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
    qbh_session_close(&session);
    if (shared != NULL) {
        rpcmem_free(shared);
    }
    free(reference_accumulators);
    free(channel_scales);
    free(logical_s8);
    free(logical_w4);
    return result;
}
