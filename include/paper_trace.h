#ifndef QBH_PAPER_TRACE_H
#define QBH_PAPER_TRACE_H
/* EXP0275 untimed diagnostic binary only. Software-observed worker envelopes
 * include their waits; DMA completion is an observed upper bound, not a
 * hardware busy timestamp. The production build removes the calls entirely. */
#ifdef QBH_PAPER_TRACE
#include <HAP_perf.h>
#include <HAP_farf.h>
#include <qurt.h>
#include <stdint.h>
#define QBH_PAPER_EVENT(event, kind, object) do { \
    unsigned long long tick=(unsigned long long)HAP_perf_get_qtimer_count(); \
    FARF(ALWAYS,"QBH_PAPER_TRACE %llu %u %s %u %x",tick, \
        (unsigned)qurt_thread_get_id(),event,(unsigned)(kind), \
        (unsigned)(uintptr_t)(object)); \
} while(0)
#else
#define QBH_PAPER_EVENT(event, kind, object) ((void)0)
#endif
#endif
