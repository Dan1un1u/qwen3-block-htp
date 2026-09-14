# L32-0017 active: speed-first FP32 vectorization and pipeline recovery
User prioritizes speed and authorizes implementation. Read docs/experiments/L32-0017.md.
Starting source e3cf74916aebbc9254615d68b1c4c135b43fee03. L32-0016 exact FP32
control preserved; fixed scales/weights/SP2, no rotations. Current scalar
functional prefill 51.57tok/s and decode25.23 versus mode8 2003.64/45.94:
severe regression, not formal speed acceptance. Recover HVX and overlap first.
No PPL recalibration campaign; numerical implementation checks retained.
L32-0015 remains deferred. L32-0012 formal baseline immutable.
