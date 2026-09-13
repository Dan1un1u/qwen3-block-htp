# L32-0011 closed: SP2 prefill pipeline passes full-model speed gate

No active experiment. Mode5 streams SP2 SwiGLU with completed Up32-tile groups,
three permanent HVXworkers and dedicated middle low-plane allocation. Preserves
L32-0010 arithmetic/weights/LUT/Q31,exact0/7/15 blocks,3-layer andfull16greedy.
Ten fixed AB/BA M64+15 pairs:prefillU82032.36/SP21859.20token/s,+9.3135%latency,
CIupper9.7847%,PASS;decode46.1426/46.1239,+0.0405%,CIupper0.6668%,PASS.
No repeated unchanged formal;one candidate.360additiveledgers,8MiB,max8212960B,
no tensorDDR/spill/weight expansion. Defaults remain originalU8;noPPL/quality
promotion. L32-0010 failed result retained;no longcontextclaim. Read latest
handoff and L32-0011 record/report. Prefill margin only0.2153pp atCIupper.
