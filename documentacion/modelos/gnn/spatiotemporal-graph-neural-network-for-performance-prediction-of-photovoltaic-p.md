# Spatiotemporal Graph Neural Network for Performance Prediction of Photovoltaic Power Systems

> **Fuente:** openalex  
> **DOI:** 10.1609/aaai.v35i17.17799  
> **URL:** https://doi.org/10.1609/aaai.v35i17.17799  
> **Año:** 2021

**Autores:** Ahmad Maroof Karimi, Yinghui Wu, Mehmet Koyutürk, Roger H. French

## Abstract

In recent years, a large number of photovoltaic (PV) systems have been added to the electrical grid as well as installed as off-grid systems. The trend suggests that the deployment of PV systems will continue to rise in the future. Thus, accurate forecasting of PV performance is critical for the reliability of PV systems. Due to the complex non-linear variability in power output of the PV systems, forecasting PV power is a non-trivial task. This variability affects the stability and planning of a power system network, and accurate forecasting of the performance of the PV system can reduce the uncertainty caused during PV operation. In this work, we leverage spatial and temporal coherence among the power plants for PV power forecasting. Our approach is motivated by the observation that power plants in a region undergo similar environmental exposure. Thus, one power plant’s performance can help improve the forecast of other power plants' power values in the region. We utilize the relationship between PV plants to build a spatiotemporal graph neural network (st-GNN) and train machine learning models to forecast the PV power. The computational experiments on large-scale data from a network of 316 systems show that spatiotemporal forecasting of PV power performs significantly better than a model that only applies temporal convolution to isolated systems or nodes. Furthermore, the longer the future forecast time, the difference between the spatiotemporal forecasting and the isolated system forecast when only temporal convolution is applied increases further.

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
