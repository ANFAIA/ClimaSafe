# Heatwave attribution based on reliable operational weather forecasts

> **Fuente:** openalex  
> **DOI:** 10.1038/s41467-024-48280-7  
> **URL:** https://doi.org/10.1038/s41467-024-48280-7  
> **Año:** 2024

**Autores:** Nicholas Leach, Christopher D. Roberts, Matthias Aengenheyster, Daniel Heathcote, Dann Mitchell et al.

## Abstract

The 2021 Pacific Northwest heatwave was so extreme as to challenge conventional statistical and climate-model-based approaches to extreme weather attribution. However, state-of-the-art operational weather prediction systems are demonstrably able to simulate the detailed physics of the heatwave. Here, we leverage these systems to show that human influence on the climate made this event at least 8 [2-50] times more likely. At the current rate of global warming, the likelihood of such an event is doubling every 20 [10-50] years. Given the multi-decade lower-bound return-time implied by the length of the historical record, this rate of change in likelihood is highly relevant for decision makers. Further, forecast-based attribution can synthesise the conditional event-specific storyline and unconditional event-class probabilistic approaches to attribution. If developed as a routine service in forecasting centres, it could provide reliable estimates of human influence on extreme weather risk, which is critical to supporting effective adaptation planning.

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
