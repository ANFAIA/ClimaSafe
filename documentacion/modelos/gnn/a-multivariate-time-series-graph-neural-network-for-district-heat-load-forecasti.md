# A multivariate time series graph neural network for district heat load forecasting

> **Fuente:** openalex  
> **DOI:** 10.1016/j.energy.2023.127911  
> **URL:** https://doi.org/10.1016/j.energy.2023.127911  
> **Año:** 2023

**Autores:** Zhijin Wang, Xiufeng Liu, Yaohui Huang, Peisong Zhang, Yonggang Fu

## Abstract

Heat load prediction is essential for energy efficiency and carbon reduction in district heating systems. However, heat load is influenced by many factors, such as building characteristics, consumption behavior, and climate, making its prediction challenging. Traditional methods based on physical models are complex and insufficiently accurate, whereas most data-driven statistical methods ignore customer energy consumption behaviors and their correlation, and do not account for the temporal inertia of consumption. This paper proposes a graph ambient intelligence (GAIN) method for heat load prediction, which classifies customers based on their load profiles and uses collaborative attention on temporal graphs to capture their associations and the weather impact on heat loads. GAIN also incorporates recursive and autoregressive methods to model the temporal inertia of consumption. The proposed method is evaluated on four metrics and compared with fifteen baseline methods. The results show that GAIN achieves the lowest daily forecasting errors in terms of RMSE, MAE, and CV-RMSE, with values of 6.972, 4.442, and 0.191, respectively. Besides, the proposed method achieves a maximum reduction of 25%, 29%, and 25% in RMSE, MAE, and CV-RMSE, respectively, compared to other methods when taking meteorological factors into account.

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
