# L32-0013 active: hardware-priority C-RTN/SP2 adaptation

Owner no-rotation source at e03a0f8, clean/pushed on activation.
User explicitly supersedes missing historical-artifact dependency: construct new
training/folding/export from original Llama. rotation-quant remains read-only.
Keep current integer residual/nonlinear/KV and other existing hardware contracts;
only add missing support. 17.6424 is historical BF16 fakequant, not matched target.
Read docs/experiments/L32-0013.md. No model/build/device jobs at activation.
Prior L32-0012 method and all frozen results remain unchanged.
