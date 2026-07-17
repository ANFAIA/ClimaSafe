# Modular Deep-Learning-Based Early Warning System for Deadly Heatwave Prediction

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2512.09074v1  
> **Año:** 2025

**Autores:** Shangqing Xu, Zhiyuan Zhao, Megha Sharma, José María Martín-Olalla, Alexander Rodríguez et al.

## Abstract

Severe heatwaves in urban areas significantly threaten public health, calling for establishing early warning strategies. Despite predicting occurrence of heatwaves and attributing historical mortality, predicting an incoming deadly heatwave remains a challenge due to the difficulty in defining and estimating heat-related mortality. Furthermore, establishing an early warning system imposes additional requirements, including data availability, spatial and temporal robustness, and decision costs. To address these challenges, we propose DeepTherm, a modular early warning system for deadly heatwave prediction without requiring heat-related mortality history. By highlighting the flexibility of deep learning, DeepTherm employs a dual-prediction pipeline, disentangling baseline mortality in the absence of heatwaves and other irregular events from all-cause mortality. We evaluated DeepTherm on real-world data across Spain. Results demonstrate consistent, robust, and accurate performance across diverse regions, time periods, and population groups while allowing trade-off between missed alarms and false alarms.

---

_Clasificado automáticamente como **factor-riesgo** el 2026-07-17. Razón: Error LLM: litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\nPlease retry in 3.045815053s.",
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
              "model": "gemini-2.0-flash",
              "location": "global"
            }
          }
        ]
      },
      {
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "3s"
      }
    ]
  }
}
_
