# Analysis and forecasting of national marine litter based on coastal data in South Korea from 2009 to 2021

> **Fuente:** openalex  
> **DOI:** 10.1016/j.marpolbul.2023.114803  
> **URL:** https://doi.org/10.1016/j.marpolbul.2023.114803  
> **Año:** 2023

**Autores:** Min-Ho Park, Siljung Yeo, Seung‐Kwon Yang, Dong‐Uk Shin, Jeonghwan Kim et al.

## Abstract

In this study, statistical analysis and forecasting were performed using coastal litter data of Korea. The analysis indicated that rope and vinyl accounted for the highest proportion of coastal litter items. The statistical analysis of the national coastal litter trends revealed that the greatest concentration of litter was observed during summer months (June-August). To predict the amount of coastal litter per meter, recurrent neural network (RNN)-based models were used. Neural basis expansion analysis for interpretable time series forecasting (N-BEATS) and neural hierarchical interpolation for time series forecasting (N-HiTS), an improved model of N-BEATS recently announced, were used for comparison with RNN-based models. When predictive performance and trend followability were evaluated, overall N-BEATS and N-HiTS outperformed RNN-based models. Furthermore, we found that average of N-BEATS and N-HiTS models yielded better results than using one model.

---

_Clasificado automáticamente como **factor-riesgo** el 2026-07-17. Razón: Error LLM: litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\nPlease retry in 57.219397761s.",
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
        "retryDelay": "57s"
      }
    ]
  }
}
_
