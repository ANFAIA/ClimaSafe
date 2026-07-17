# Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2110.13179v8  
> **Año:** 2021

**Autores:** Kin G. Olivares, O. Nganba Meetei, Ruijun Ma, Rohan Reddy, Mengfei Cao et al.

## Abstract

Hierarchical forecasting problems arise when time series have a natural group structure, and predictions at multiple levels of aggregation and disaggregation across the groups are needed. In such problems, it is often desired to satisfy the aggregation constraints in a given hierarchy, referred to as hierarchical coherence in the literature. Maintaining coherence while producing accurate forecasts can be a challenging problem, especially in the case of probabilistic forecasting. We present a novel method capable of accurate and coherent probabilistic forecasts for time series when reliable hierarchical information is present. We call it Deep Poisson Mixture Network (DPMN). It relies on the combination of neural networks and a statistical model for the joint distribution of the hierarchical multivariate time series structure. By construction, the model guarantees hierarchical coherence and provides simple rules for aggregation and disaggregation of the predictive distributions. We perform an extensive empirical evaluation comparing the DPMN to other state-of-the-art methods which produce hierarchically coherent probabilistic forecasts on multiple public datasets. Comparing to existing coherent probabilistic models, we obtain a relative improvement in the overall Continuous Ranked Probability Score (CRPS) of 11.8% on Australian domestic tourism data, and 8.1% on the Favorita grocery sales dataset, where time series are grouped with geographical hierarchies or travel intent hierarchies. For San Francisco Bay Area highway traffic, where the series' hierarchical structure is randomly assigned, and their correlations are less informative, our method does not show significant performance differences over statistical baselines.

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
