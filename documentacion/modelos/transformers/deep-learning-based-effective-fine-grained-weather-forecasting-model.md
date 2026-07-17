# Deep learning-based effective fine-grained weather forecasting model

> **Fuente:** openalex  
> **DOI:** 10.1007/s10044-020-00898-1  
> **URL:** https://doi.org/10.1007/s10044-020-00898-1  
> **Año:** 2020

**Autores:** Pradeep Hewage, Marcello Trovati, Ella Pereira, Ardhendu Behera

## Abstract

Abstract It is well-known that numerical weather prediction (NWP) models require considerable computer power to solve complex mathematical equations to obtain a forecast based on current weather conditions. In this article, we propose a novel lightweight data-driven weather forecasting model by exploring temporal modelling approaches of long short-term memory (LSTM) and temporal convolutional networks (TCN) and compare its performance with the existing classical machine learning approaches, statistical forecasting approaches, and a dynamic ensemble method, as well as the well-established weather research and forecasting (WRF) NWP model. More specifically Standard Regression (SR), Support Vector Regression (SVR), and Random Forest (RF) are implemented as the classical machine learning approaches, and Autoregressive Integrated Moving Average (ARIMA), Vector Auto Regression (VAR), and Vector Error Correction Model (VECM) are implemented as the statistical forecasting approaches. Furthermore, Arbitrage of Forecasting Expert (AFE) is implemented as the dynamic ensemble method in this article. Weather information is captured by time-series data and thus, we explore the state-of-art LSTM and TCN models, which is a specialised form of neural network for weather prediction. The proposed deep model consists of a number of layers that use surface weather parameters over a given period of time for weather forecasting. The proposed deep learning networks with LSTM and TCN layers are assessed in two different regressions, namely multi-input multi-output and multi-input single-output. Our experiment shows that the proposed lightweight model produces better results compared to the well-known and complex WRF model, demonstrating its potential for efficient and accurate weather forecasting up to 12 h.

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
