# FuXi: a cascade machine learning forecasting system for 15-day global weather forecast

> **Fuente:** openalex  
> **DOI:** 10.1038/s41612-023-00512-1  
> **URL:** https://doi.org/10.1038/s41612-023-00512-1  
> **Año:** 2023

**Autores:** Lei Chen, Xiaohui Zhong, Feng Zhang, Yuan Cheng, Yinghui Xu et al.

## Abstract

Abstract Over the past few years, the rapid development of machine learning (ML) models for weather forecasting has led to state-of-the-art ML models that have superior performance compared to the European Centre for Medium-Range Weather Forecasts (ECMWF)’s high-resolution forecast (HRES), which is widely considered as the world’s best physics-based weather forecasting system. Specifically, ML models have outperformed HRES in 10-day forecasts with a spatial resolution of 0.25 ∘ . However, the challenge remains in mitigating the accumulation of forecast errors for longer effective forecasts, such as achieving comparable performance to the ECMWF ensemble in 15-day forecasts. Despite various efforts to reduce accumulation errors, such as implementing autoregressive multi-time step loss, relying on a single model has been found to be insufficient for achieving optimal performance in both short and long lead times. Therefore, we present FuXi, a cascaded ML weather forecasting system that provides 15-day global forecasts at a temporal resolution of 6 hours and a spatial resolution of 0.25 ∘ . FuXi is developed using 39 years of the ECMWF ERA5 reanalysis dataset. The performance evaluation demonstrates that FuXi has forecast performance comparable to ECMWF ensemble mean (EM) in 15-day forecasts. FuXi surpasses the skillful forecast lead time achieved by ECMWF HRES by extending the lead time for Z500 from 9.25 to 10.5 days and for T2M from 10 to 14.5 days. Moreover, the FuXi ensemble is created by perturbing initial conditions and model parameters, enabling it to provide forecast uncertainty and demonstrating promising results when compared to the ECMWF ensemble.

---

_Clasificado automáticamente como **factor-riesgo** el 2026-07-17. Razón: Error LLM: litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\nPlease retry in 59.322187034s.",
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
              "model": "gemini-2.0-flash",
              "location": "global"
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
        "retryDelay": "59s"
      }
    ]
  }
}
_
