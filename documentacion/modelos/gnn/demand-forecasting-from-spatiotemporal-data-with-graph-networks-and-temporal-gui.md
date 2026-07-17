# Demand Forecasting from Spatiotemporal Data with Graph Networks and Temporal-Guided Embedding

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/1905.10709v2  
> **Año:** 2019

**Autores:** Doyup Lee, Suehun Jung, Yeongjae Cheon, Dongil Kim, Seungil You

## Abstract

Short-term demand forecasting models commonly combine convolutional and recurrent layers to extract complex spatiotemporal patterns in data. Long-term histories are also used to consider periodicity and seasonality patterns as time series data. In this study, we propose an efficient architecture, Temporal-Guided Network (TGNet), which utilizes graph networks and temporal-guided embedding. Graph networks extract invariant features to permutations of adjacent regions instead of convolutional layers. Temporal-guided embedding explicitly learns temporal contexts from training data and is substituted for the input of long-term histories from days/weeks ago. TGNet learns an autoregressive model, conditioned on temporal contexts of forecasting targets from temporal-guided embedding. Finally, our model achieves competitive performances with other baselines on three spatiotemporal demand dataset from real-world, but the number of trainable parameters is about 20 times smaller than a state-of-the-art baseline. We also show that temporal-guided embedding learns temporal contexts as intended and TGNet has robust forecasting performances even to atypical event situations.

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
