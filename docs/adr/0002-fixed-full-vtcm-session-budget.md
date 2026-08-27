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

For the current SDK, the concrete v2 request is 8 MiB with a 4 MiB minimum
page size and `min_vtcm_size=0`. The zero value is the documented
absolute-requirement mode; the returned pointer size is still checked for an
exact 8 MiB grant. An 8 MiB minimum-page argument is invalid because 8 MiB is
not one of the page sizes supported by this API.
