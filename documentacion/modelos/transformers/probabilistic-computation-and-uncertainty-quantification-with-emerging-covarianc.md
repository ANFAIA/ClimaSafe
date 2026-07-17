# Probabilistic computation and uncertainty quantification with emerging covariance

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2305.19265v3  
> **Año:** 2023

**Autores:** Hengyuan Ma, Yang Qi, Li Zhang, Wenlian Lu, Jianfeng Feng

## Abstract

Building robust, interpretable, and secure AI system requires quantifying and representing uncertainty under a probabilistic perspective to mimic human cognitive abilities. However, probabilistic computation presents significant challenges for most conventional artificial neural network, as they are essentially implemented in a deterministic manner. In this paper, we develop an efficient probabilistic computation framework by truncating the probabilistic representation of neural activation up to its mean and covariance and construct a moment neural network that encapsulates the nonlinear coupling between the mean and covariance of the underlying stochastic network. We reveal that when only the mean but not the covariance is supervised during gradient-based learning, the unsupervised covariance spontaneously emerges from its nonlinear coupling with the mean and faithfully captures the uncertainty associated with model predictions. Our findings highlight the inherent simplicity of probabilistic computation by seamlessly incorporating uncertainty into model prediction, paving the way for integrating it into large-scale AI systems.

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
