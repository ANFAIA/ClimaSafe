# Temporal convolutional neural (TCN) network for an effective weather forecasting using time-series data from the local weather station

> **Fuente:** openalex  
> **DOI:** 10.1007/s00500-020-04954-0  
> **URL:** https://doi.org/10.1007/s00500-020-04954-0  
> **Año:** 2020

**Autores:** Pradeep Hewage, Ardhendu Behera, Marcello Trovati, Ella Pereira, Morteza Ghahremani et al.

## Abstract

Abstract Non-predictive or inaccurate weather forecasting can severely impact the community of users such as farmers. Numerical weather prediction models run in major weather forecasting centers with several supercomputers to solve simultaneous complex nonlinear mathematical equations. Such models provide the medium-range weather forecasts, i.e., every 6 h up to 18 h with grid length of 10–20 km. However, farmers often depend on more detailed short-to medium-range forecasts with higher-resolution regional forecasting models. Therefore, this research aims to address this by developing and evaluating a lightweight and novel weather forecasting system, which consists of one or more local weather stations and state-of-the-art machine learning techniques for weather forecasting using time-series data from these weather stations. To this end, the system explores the state-of-the-art temporal convolutional network (TCN) and long short-term memory (LSTM) networks. Our experimental results show that the proposed model using TCN produces better forecasting compared to the LSTM and other classic machine learning approaches. The proposed model can be used as an efficient localized weather forecasting tool for the community of users, and it could be run on a stand-alone personal computer.

---

_Clasificado automáticamente como **factor-riesgo** el 2026-07-17. Razón: Error LLM: litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\nPlease retry in 58.211901669s.",
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
        "retryDelay": "58s"
      }
    ]
  }
}
_
