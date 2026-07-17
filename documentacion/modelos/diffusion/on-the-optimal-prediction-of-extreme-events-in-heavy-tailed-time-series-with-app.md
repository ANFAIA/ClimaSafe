# On the optimal prediction of extreme events in heavy-tailed time series with applications to solar flare forecasting

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2407.11887v2  
> **Año:** 2024

**Autores:** Victor Verma, Stilian Stoev, Yang Chen

## Abstract

The prediction of extreme events in time series is a fundamental problem arising in many financial, scientific, engineering, and other applications. We begin by establishing a general Neyman-Pearson-type characterization of optimal extreme event predictors in terms of density ratios. This yields new insights and several closed-form optimal extreme event predictors for additive models. These results naturally extend to time series, where we study optimal extreme event prediction for both light- and heavy-tailed autoregressive and moving average models. Using a uniform law of large numbers for ergodic time series, we establish the asymptotic optimality of an empirical version of the optimal predictor for autoregressive models. Using multivariate regular variation, we obtain an expression for the optimal extremal precision in heavy-tailed infinite moving averages, which provides theoretical bounds on the ability to predict extremes in this general class of models. We address the important problem of predicting solar flares by applying our theory and methodology to a state-of-the-art time series consisting of solar soft X-ray flux measurements. Our results demonstrate the success and limitations in solar flare forecasting of long-memory autoregressive models and long-range-dependent, heavy-tailed FARIMA models.

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
