# Deep Learning for Anomaly Detection in Time-Series Data: Review, Analysis, and Guidelines

> **Fuente:** openalex  
> **DOI:** 10.1109/access.2021.3107975  
> **URL:** https://doi.org/10.1109/access.2021.3107975  
> **Año:** 2021

**Autores:** Kukjin Choi, Jihun Yi, Changhwa Park, Sungroh Yoon

## Abstract

As industries become automated and connectivity technologies advance, a wide range of systems continues to generate massive amounts of data. Many approaches have been proposed to extract principal indicators from the vast sea of data to represent the entire system state. Detecting anomalies using these indicators on time prevent potential accidents and economic losses. Anomaly detection in multivariate time series data poses a particular challenge because it requires simultaneous consideration of temporal dependencies and relationships between variables. Recent deep learning-based works have made impressive progress in this field. They are highly capable of learning representations of the large-scaled sequences in an unsupervised manner and identifying anomalies from the data. However, most of them are highly specific to the individual use case and thus require domain knowledge for appropriate deployment. This review provides a background on anomaly detection in time-series data and reviews the latest applications in the real world. Also, we comparatively analyze state-of-the-art deep-anomaly-detection models for time series with several benchmark datasets. Finally, we offer guidelines for appropriate model selection and training strategy for deep learning-based time series anomaly detection.

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
