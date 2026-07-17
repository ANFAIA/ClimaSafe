# Spatio-Temporal Graph Neural Networks for Multi-Site PV Power Forecasting

> **Fuente:** openalex  
> **DOI:** 10.1109/tste.2021.3125200  
> **URL:** https://doi.org/10.1109/tste.2021.3125200  
> **Año:** 2021

**Autores:** Jelena Simeunović, Baptiste Schubnel, Pierre‐Jean Alet, Rafael E. Carrillo

## Abstract

Accurate forecasting of solar power generation with fine temporal and spatial resolution is vital for the operation of the power grid. However, state-of-the-art approaches that combine machinelearning with numerical weather predictions (NWP) have coarse resolution. In this paper, we take a graph signal processing perspective and model multi-site photovoltaic (PV) production time series as signals on a graph to capture their spatio-temporal dependencies and achieve higher spatial and temporal resolution forecasts. We present two novel graph neural network models for deterministic multi-site PV forecasting dubbed the graph-convolutional long short term memory (GCLSTM) and the graph-convolutional transformer (GCTrafo) models. These methods rely solely on production data and exploit the intuition that PV systems provide a dense network of virtual weather stations. The proposed methods were evaluated in two data sets for an entire year: 1) production data from 304 real PV systems, and 2) simulated production of 1000 PV systems, both distributed over Switzerland. The proposed models outperform state-of-the-art multi-site forecasting methods for prediction horizons of six hours ahead. Furthermore, the proposed models outperform state-of-the-art single-site methods with NWP as inputs on horizons up to four hours ahead.

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
