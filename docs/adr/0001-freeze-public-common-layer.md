# ADR-0001: Freeze the public common layer after EXP-0109

Status: Accepted

## Context

EXP-0107 found a real F16F16 interleaved MLP improvement but failed its
aggregate gate only on duplicated, identity-proven W4 parity noise. EXP-0108
improved W4F16 QKV locally but regressed F16F16 complete wall time, so it is
not a common cross-recipe plan. Continuing to mix common and recipe-specific
work would weaken comparison fairness and make baseline ancestry ambiguous.

## Decision

EXP-0109 will consolidate EXP-0106 with only the accepted EXP-0107 F16F16
component, formally revalidate all selected cells and tag the resulting source
as the Frozen Public Baseline. EXP-0108 remains rejected for the Public Common
Layer. After the freeze, new optimization work is Recipe Specialization unless
the user explicitly approves a new public physical contract.

## Consequences

Fair three-recipe results and fastest-per-recipe results remain distinct.
Specializations gain a stable parent and cannot silently alter the fair
baseline. Reopening common QKV carrier work requires either a W4F16-specific
experiment or an approved change in physical contract.
