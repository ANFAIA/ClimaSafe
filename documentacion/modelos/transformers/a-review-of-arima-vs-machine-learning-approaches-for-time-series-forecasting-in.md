# A Review of ARIMA vs. Machine Learning Approaches for Time Series Forecasting in Data Driven Networks

> **Fuente:** openalex  
> **DOI:** 10.3390/fi15080255  
> **URL:** https://doi.org/10.3390/fi15080255  
> **Año:** 2023

**Autores:** Vaia I. Kontopoulou, Athanasios D. Panagopoulos, Iοannis Kakkos, George K. Matsopoulos

## Abstract

In the broad scientific field of time series forecasting, the ARIMA models and their variants have been widely applied for half a century now due to their mathematical simplicity and flexibility in application. However, with the recent advances in the development and efficient deployment of artificial intelligence models and techniques, the view is rapidly changing, with a shift towards machine and deep learning approaches becoming apparent, even without a complete evaluation of the superiority of the new approach over the classic statistical algorithms. Our work constitutes an extensive review of the published scientific literature regarding the comparison of ARIMA and machine learning algorithms applied to time series forecasting problems, as well as the combination of these two approaches in hybrid statistical-AI models in a wide variety of data applications (finance, health, weather, utilities, and network traffic prediction). Our review has shown that the AI algorithms display better prediction performance in most applications, with a few notable exceptions analyzed in our Discussion and Conclusions sections, while the hybrid statistical-AI models steadily outperform their individual parts, utilizing the best algorithmic features of both worlds.

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
