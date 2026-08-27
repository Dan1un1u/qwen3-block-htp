---
status: accepted
---

# Use a fixed full-VTCM session budget for formal comparisons

The user approved querying and requiring the architecture-defined total VTCM
once per Prepared Runtime Session for formal single-client comparisons. This
keeps F16F16, W4F16, and W4U8 on the same reproducible hardware budget and
preserves freedom to design deeper project-owned pipelines; internal arenas
still allocate by tensor lifetime and report actual peak use. Requesting the
current available amount or silently accepting a smaller grant was rejected
because background DSP clients would make the experiment resource budget
non-deterministic.
