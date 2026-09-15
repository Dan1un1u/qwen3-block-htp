#ifndef QBH_PAPER_TRACE_H
#define QBH_PAPER_TRACE_H
/* Untimed diagnostic only: bounded metadata collection, no per-event logging.
 * Worker envelopes include waits; DMA observations are software bounds.
 * Production compiles every call out. No activation/weight payload is stored. */
#ifdef QBH_PAPER_TRACE
#include <stdint.h>
void qbh_paper_trace_event(const char *,uint32_t,uintptr_t);
void qbh_paper_trace_reset(void);
void qbh_paper_trace_flush(void);
#define QBH_PAPER_EVENT(event,kind,object) qbh_paper_trace_event(event,(uint32_t)(kind),(uintptr_t)(object))
#define QBH_PAPER_RESET() qbh_paper_trace_reset()
#define QBH_PAPER_FLUSH() qbh_paper_trace_flush()
#else
#define QBH_PAPER_EVENT(event,kind,object) ((void)0)
#define QBH_PAPER_RESET() ((void)0)
#define QBH_PAPER_FLUSH() ((void)0)
#endif
#endif
