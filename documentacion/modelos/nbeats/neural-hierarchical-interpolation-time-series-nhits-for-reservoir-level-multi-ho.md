# Neural Hierarchical Interpolation Time Series (NHITS) for Reservoir Level Multi-Horizon Forecasting in Hydroelectric Power Plants

> **Fuente:** openalex  
> **DOI:** 10.1109/access.2025.3554446  
> **URL:** https://doi.org/10.1109/access.2025.3554446  
> **Año:** 2025

**Autores:** Stéfano Frizzo Stefenon, Laio Oriel Seman, Cristina Keiko Yamaguchi, Leandro dos Santos Coelho, Viviana Cocco Mariani et al.

## Abstract

Energy planning in systems heavily influenced by hydroelectric power is based on assessing the availability of water in the future. In Brazil, based on the soil moisture active passive, the National Electricity System Operator defines electricity dispatch concerning a stochastic optimization problem. Currently, machine learning models are an alternative for improving forecasts, and could be a promising solution for predicting reservoir levels at hydroelectric dams. In this paper, neural hierarchical interpolation for time series (NHITS) is applied to improve forecasts and thus help decision-making in the management of electric power systems. The NHITS model achieved a root mean square error of <inline-formula xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"> <tex-math notation="LaTeX">$4.64\times 10^{-4}$ </tex-math></inline-formula> for a 1-hour forecast horizon, and <inline-formula xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink"> <tex-math notation="LaTeX">$1.03\times 10^{-3}$ </tex-math></inline-formula> for a 10-hour forecast horizon, being superior to multilayer perceptron (MLP) neural network, long short-term memory (LSTM), convolutional neural network with long short-term memory (CNN-LSTM), recurrent neural network (RNN), Dilated RNN, temporal convolutional neural (TCN), neural basis expansion analysis for interpretable time series forecasting (N-BEATS), and deep non-parametric time series forecaster (DeepNPTS) deep learning approaches.

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
