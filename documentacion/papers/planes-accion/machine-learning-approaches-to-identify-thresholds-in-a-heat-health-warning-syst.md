# Machine learning approaches to identify thresholds in a heat-health warning system context

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2110.10625v1  
> **Año:** 2021

**Autores:** Pierre Masselot, Fateh Chebana, Céline Campagna, Eric Lavigne, Taha B. M. J. Ouarda et al.

## Abstract

During the last two decades, a number of countries or cities established heat-health warning systems in order to alert public health authorities when some heat indicator exceeds a predetermined threshold. Different methods were considered to establish thresholds all over the world, each with its own strengths and weaknesses. The common ground is that current methods are based on exposure-response function estimates that can fail in many situations. The present paper aims at proposing several data-driven methods to establish thresholds using historical data of health issues and environmental indicators. The proposed methods are model-based regression trees (MOB), multivariate adaptive regression splines (MARS), the patient rule-induction method (PRIM) and adaptive index models (AIM). These methods focus on finding relevant splits in the association between indicators and the health outcome but do it in different fashions. A simulation study and a real-world case study hereby compare the discussed methods. Results show that proposed methods are better at predicting adverse days than current thresholds and benchmark methods. The results nonetheless suggest that PRIM is overall the more reliable method with low variability of results according to the scenario or case.

---

_Clasificado automáticamente como **factor-riesgo** el 2026-07-17. Razón: Error LLM: litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\nPlease retry in 3.045815053s.",
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
              "model": "gemini-2.0-flash",
              "location": "global"
            }
          }
        ]
      },
      {
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": "3s"
      }
    ]
  }
}
_
