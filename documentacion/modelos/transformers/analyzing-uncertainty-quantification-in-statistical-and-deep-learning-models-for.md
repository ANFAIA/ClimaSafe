# Analyzing Uncertainty Quantification in Statistical and Deep Learning Models for Probabilistic Electricity Price Forecasting

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2509.19417v3  
> **Año:** 2025

**Autores:** Andreas Lebedev, Abhinav Das, Sven Pappert, Stephan Schlüter

## Abstract

Precise probabilistic forecasts are fundamental for energy risk management, and there is a wide range of both statistical and machine learning models for this purpose. Inherent to these probabilistic models is some form of uncertainty quantification. However, most models do not capture the full extent of uncertainty, which arises not only from the data itself but also from model and distributional choices. In this study, we examine uncertainty quantification in state-of-the-art statistical and deep learning probabilistic forecasting models for electricity price forecasting in the German market. In particular, we consider deep distributional neural networks (DDNNs) and augment them with an ensemble approach, Monte Carlo (MC) dropout, and conformal prediction to account for model uncertainty. Additionally, we consider the LASSO-estimated autoregressive (LEAR) approach combined with quantile regression averaging (QRA), generalized autoregressive conditional heteroskedasticity (GARCH), and conformal prediction. Across a range of performance metrics, we find that the LEAR-based models perform well in terms of probabilistic forecasting, irrespective of the uncertainty quantification method. Furthermore, we find that DDNNs benefit from incorporating both data and model uncertainty, improving both point and probabilistic forecasting. Uncertainty itself appears to be best captured by the models using conformal prediction. Overall, our extensive study shows that all models under consideration perform competitively. However, their relative performance depends on the choice of metrics for point and probabilistic forecasting.

---

_Clasificado automáticamente como **factor-riesgo** el 2026-07-17. Razón: Error LLM: litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash\nPlease retry in 55.178999958s.",
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
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
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
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_input_token_count",
            "quotaId": "GenerateContentInputTokensPerModelPerMinute-FreeTier",
            "quotaDimensions": {
              "location": "global",
              "model": "gemini-2.0-flash"
            }
          }
        ]
      },
      {
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "55s"
      }
    ]
  }
}
_
