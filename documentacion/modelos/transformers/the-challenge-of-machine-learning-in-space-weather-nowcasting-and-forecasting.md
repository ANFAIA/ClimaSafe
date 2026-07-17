# The Challenge of Machine Learning in Space Weather: Nowcasting and Forecasting

> **Fuente:** openalex  
> **DOI:** 10.1029/2018sw002061  
> **URL:** https://doi.org/10.1029/2018sw002061  
> **Año:** 2019

**Autores:** Enrico Camporeale

## Abstract

Abstract The numerous recent breakthroughs in machine learning make imperative to carefully ponder how the scientific community can benefit from a technology that, although not necessarily new, is today living its golden age. This Grand Challenge review paper is focused on the present and future role of machine learning in Space Weather. The purpose is twofold. On one hand, we will discuss previous works that use machine learning for Space Weather forecasting, focusing in particular on the few areas that have seen most activity: the forecasting of geomagnetic indices, of relativistic electrons at geosynchronous orbits, of solar flares occurrence, of coronal mass ejection propagation time, and of solar wind speed. On the other hand, this paper serves as a gentle introduction to the field of machine learning tailored to the Space Weather community and as a pointer to a number of open challenges that we believe the community should undertake in the next decade. The recurring themes throughout the review are the need to shift our forecasting paradigm to a probabilistic approach focused on the reliable assessment of uncertainties, and the combination of physics‐based and machine learning approaches, known as gray box.

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
