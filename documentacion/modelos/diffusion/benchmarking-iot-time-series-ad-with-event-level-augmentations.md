# Benchmarking IoT Time-Series AD with Event-Level Augmentations

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2602.15457v2  
> **Año:** 2026

**Autores:** Dmitry Zhevnenko, Ilya Makarov, Aleksandr Kovalenko, Fedor Meshchaninov, Anton Kozhukhov et al.

## Abstract

Anomaly detection (AD) for safety-critical IoT time series should be judged at the event level: reliability and earliness under realistic perturbations. Yet many studies still emphasize point-level results on curated base datasets, limiting value for model selection in practice. We introduce an evaluation protocol with unified event-level augmentations that simulate real-world issues: calibrated sensor dropout, linear and log drift, additive noise, and window shifts. We also perform sensor-level probing via mask-as-missing zeroing with per-channel influence estimation to support root-cause analysis. We evaluate 14 representative models on five public anomaly datasets (SWaT, WADI, SMD, SKAB, TEP) and two industrial datasets (steam turbine, nuclear turbogenerator) using unified splits and event aggregation. There is no universal winner: graph-structured models transfer best under dropout and long events (e.g., on SWaT under additive noise F1 drops 0.804->0.677 for a graph autoencoder, 0.759->0.680 for a graph-attention variant, and 0.762->0.756 for a hybrid graph attention model); density/flow models work well on clean stationary plants but can be fragile to monotone drift; spectral CNNs lead when periodicity is strong; reconstruction autoencoders become competitive after basic sensor vetting; predictive/hybrid dynamics help when faults break temporal dependencies but remain window-sensitive. The protocol also informs design choices: on SWaT under log drift, replacing normalizing flows with Gaussian density reduces high-stress F1 from ~0.75 to ~0.57, and fixing a learned DAG gives a small clean-set gain (~0.5-1.0 points) but increases drift sensitivity by ~8x.

---

_Clasificado automáticamente como **factor-riesgo** el 2026-07-17. Razón: Error LLM: litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\nPlease retry in 56.185022144s.",
    "status": "RESOURCE_EXHAUSTED",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.Help",
        "links": [
          {
            "description": "Learn more about Gemini API quotas",
            "url": "https://ai.google.dev/gemini-api/docs/rate-limits"
          }
        ]
      },
      {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [
          {
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_input_token_count",
            "quotaId": "GenerateContentInputTokensPerModelPerMinute-FreeTier",
            "quotaDimensions": {
              "location": "global",
              "model": "gemini-2.0-flash"
            }
          },
          {
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
            "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
            "quotaDimensions": {
              "location": "global",
              "model": "gemini-2.0-flash"
            }
          },
          {
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
            "quotaDimensions": {
              "location": "global",
              "model": "gemini-2.0-flash"
            }
          }
        ]
      },
      {
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "56s"
      }
    ]
  }
}
_
