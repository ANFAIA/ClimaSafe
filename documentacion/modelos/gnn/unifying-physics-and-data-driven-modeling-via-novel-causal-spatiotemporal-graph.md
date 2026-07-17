# Unifying Physics- and Data-Driven Modeling via Novel Causal Spatiotemporal Graph Neural Network for Interpretable Epidemic Forecasting

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2504.05140v1  
> **Año:** 2025

**Autores:** Shuai Han, Lukas Stelz, Thomas R. Sokolowski, Kai Zhou, Horst Stöcker

## Abstract

Accurate epidemic forecasting is crucial for effective disease control and prevention. Traditional compartmental models often struggle to estimate temporally and spatially varying epidemiological parameters, while deep learning models typically overlook disease transmission dynamics and lack interpretability in the epidemiological context. To address these limitations, we propose a novel Causal Spatiotemporal Graph Neural Network (CSTGNN), a hybrid framework that integrates a Spatio-Contact SIR model with Graph Neural Networks (GNNs) to capture the spatiotemporal propagation of epidemics. Inter-regional human mobility exhibits continuous and smooth spatiotemporal patterns, leading to adjacent graph structures that share underlying mobility dynamics. To model these dynamics, we employ an adaptive static connectivity graph to represent the stable components of human mobility and utilize a temporal dynamics model to capture fluctuations within these patterns. By integrating the adaptive static connectivity graph with the temporal dynamics graph, we construct a dynamic graph that encapsulates the comprehensive properties of human mobility networks. Additionally, to capture temporal trends and variations in infectious disease spread, we introduce a temporal decomposition model to handle temporal dependence. This model is then integrated with a dynamic graph convolutional network for epidemic forecasting. We validate our model using real-world datasets at the provincial level in China and the state level in Germany. Extensive studies demonstrate that our method effectively models the spatiotemporal dynamics of infectious diseases, providing a valuable tool for forecasting and intervention strategies. Furthermore, analysis of the learned parameters offers insights into disease transmission mechanisms, enhancing the interpretability and practical applicability of our model.

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
