/* CPU-side reference for the V79 integer-HMX U8 output conversion.
 *
 * Matrix accumulation is intentionally performed by the independent Python
 * reference.  This helper applies only the documented bias-word carrier and
 * the SDK libnative conversion semantics to a row-major int64 accumulator.
 * It never invokes the device, FastRPC, QNN, or project DSP code.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define QBH_HMX_REF_MAGIC UINT32_C(0x51485246)
#define QBH_HMX_REF_ABI UINT32_C(1)

struct qbh_hmx_ref_header {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t rows;
    uint32_t channels;
};

/* libnative intentionally exports the instruction-level conversion helper,
 * but its public headers expose only complete HMX instructions.  The ABI below
 * is recovered from the SDK's DWARF prototype and guarded by the project
 * conversion-probe tests. */
extern uint32_t hmx_u8_cvt(void *state_ptr, int64_t accumulator,
                           int32_t bias32, int16_t exponent,
                           int16_t zeroing, int16_t significand,
                           uint16_t output_bias, int32_t saturate,
                           int16_t legacy);

static int read_exact(void *destination, size_t bytes) {
    return fread(destination, 1U, bytes, stdin) == bytes ? 0 : -1;
}

static uint8_t convert_one(int64_t accumulator, uint32_t lower_word,
                           int32_t upper_bias) {
    const int16_t exponent = (int16_t)((lower_word >> 10U) & 0x1fU);
    const int16_t zeroing = (int16_t)(
        ((lower_word >> 17U) & 0x3U) | ((lower_word >> 13U) & 0x4U));
    const int16_t significand = (int16_t)(
        ((((lower_word & 0x3ffU) << 1U) |
           ((lower_word >> 31U) & 0x1U))) ^ 0x800U);
    const uint16_t output_bias =
        (uint16_t)((lower_word >> 19U) & 0xfffU);
    const uint32_t converted = hmx_u8_cvt(
        NULL, accumulator, upper_bias, exponent, zeroing, significand,
        output_bias, 1, 1);

    /* Integer-HMX keeps four fractional bits in the converted accumulator;
     * mxmem ... sat.ub publishes bits [11:4]. */
    return (uint8_t)((converted >> 4U) & 0xffU);
}

int qbh_hmx_u8_reference_convert(const int64_t *accumulators,
                                 uint32_t rows, uint32_t channels,
                                 const uint32_t *lower_words,
                                 const int32_t *upper_biases,
                                 uint8_t *output) {
    size_t elements;

    if (accumulators == NULL || lower_words == NULL ||
        upper_biases == NULL || output == NULL || rows == 0U ||
        channels == 0U || (size_t)rows > SIZE_MAX / (size_t)channels) {
        return -1;
    }
    elements = (size_t)rows * (size_t)channels;
    for (size_t index = 0U; index < elements; ++index) {
        const uint32_t channel =
            (uint32_t)(index % (size_t)channels);
        output[index] = convert_one(
            accumulators[index], lower_words[channel],
            upper_biases[channel]);
    }
    return 0;
}

int main(void) {
    struct qbh_hmx_ref_header header;
    int64_t *accumulators = NULL;
    uint32_t *lower_words = NULL;
    int32_t *upper_biases = NULL;
    uint8_t *output = NULL;
    size_t elements;
    int result = EXIT_FAILURE;

    if (read_exact(&header, sizeof(header)) != 0 ||
        header.magic != QBH_HMX_REF_MAGIC ||
        header.abi_version != QBH_HMX_REF_ABI || header.rows == 0U ||
        header.channels == 0U ||
        (size_t)header.rows > SIZE_MAX / (size_t)header.channels) {
        fprintf(stderr, "invalid HMX reference request\n");
        return EXIT_FAILURE;
    }
    elements = (size_t)header.rows * (size_t)header.channels;
    if (elements > SIZE_MAX / sizeof(*accumulators)) {
        fprintf(stderr, "HMX reference request is too large\n");
        return EXIT_FAILURE;
    }
    accumulators = (int64_t *)malloc(elements * sizeof(*accumulators));
    lower_words =
        (uint32_t *)malloc((size_t)header.channels * sizeof(*lower_words));
    upper_biases =
        (int32_t *)malloc((size_t)header.channels * sizeof(*upper_biases));
    output = (uint8_t *)malloc(elements);
    if (accumulators == NULL || lower_words == NULL ||
        upper_biases == NULL || output == NULL) {
        fprintf(stderr, "HMX reference allocation failed\n");
        goto cleanup;
    }
    if (read_exact(accumulators, elements * sizeof(*accumulators)) != 0 ||
        read_exact(lower_words,
                   (size_t)header.channels * sizeof(*lower_words)) != 0 ||
        read_exact(upper_biases,
                   (size_t)header.channels * sizeof(*upper_biases)) != 0) {
        fprintf(stderr, "truncated HMX reference request\n");
        goto cleanup;
    }
    if (qbh_hmx_u8_reference_convert(
            accumulators, header.rows, header.channels, lower_words,
            upper_biases, output) != 0) {
        fprintf(stderr, "HMX reference conversion failed\n");
        goto cleanup;
    }
    if (fwrite(output, 1U, elements, stdout) != elements ||
        fflush(stdout) != 0) {
        fprintf(stderr, "failed to write HMX reference output\n");
        goto cleanup;
    }
    result = EXIT_SUCCESS;

cleanup:
    free(accumulators);
    free(lower_words);
    free(upper_biases);
    free(output);
    return result;
}
