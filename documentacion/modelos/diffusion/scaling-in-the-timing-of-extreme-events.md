# Scaling in the Timing of Extreme Events

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/1408.1943v1  
> **Año:** 2014

**Autores:** Alvaro Corral

## Abstract

Extreme events can come either from point processes, when the size or energy of the events is above a certain threshold, or from time series, when the intensity of a signal surpasses a threshold value. We are particularly concerned by the time between these extreme events, called respectively waiting time and quiet time. If the thresholds are high enough it is possible to justify the existence of scaling laws for the probability distribution of the times as a function of the threshold value, although the scaling functions are different in each case. For point processes, in addition to the trivial Poisson process, one can obtain double-power-law distributions with no finite mean value. This is justified in the context of renormalization-group transformations, where such distributions arise as limiting distributions after iterations of the transformation. Clear connections with the generalized central limit theorem are established from here. The non-existence of finite moments leads to a semi-parametric scaling law in terms of the sample mean waiting time, in which the (usually unkown) scale parameter is eliminated but not the exponents. In the case of time series, scaling can arise by considering random-walk-like signals with absorbing boundaries, resulting in distributions with a power-law "bulk" and a faster decay for long times. For large thresholds the moments of the quiet-time distribution show a power-law dependence with the scale parameter, and isolation of the latter and of the exponents leads to a non-parametric scaling law in terms only of the moments of the distribution. Conclusions about the projections of changes in the occurrence of natural hazards lead to the necessity of distinguishing the behavior of the mean of the distribution with the behavior of the extreme events.

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
