#ifndef QBH_MODEL_CONFIG_H
#define QBH_MODEL_CONFIG_H
/* L32-0001: model arithmetic is a build identity, never a runtime weight guess. */
#ifdef QBH_MODEL_LLAMA32
#define QBH_MODEL_RMS_EPS 1.0e-5f
#define QBH_MODEL_ATTENTION_SCALE 0.125f
#define QBH_MODEL_QK_NORM 0
#else
#define QBH_MODEL_RMS_EPS 1.0e-6f
#define QBH_MODEL_ATTENTION_SCALE 0.08838834764831845f
#define QBH_MODEL_QK_NORM 1
#endif
#endif
