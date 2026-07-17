# A two-stage approach to heat-mortality risk assessment comparing multiple exposure-to-temperature models: the case study in Lazio, Italy

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2512.14292v1  
> **Año:** 2025

**Autores:** Emiliano Ceccarelli, Jorge Castillo-Mateo, Sandra Gudžiūnaitė, Giada Minelli, Giovanna Jona Lasinio et al.

## Abstract

This study investigates how different spatiotemporal temperature models affect the estimation of heat-related mortality in Lazio, Italy (2008--2022). First, we compare three methods to reconstruct daily maximum temperature at the municipality level: 1. a Bayesian quantile regression model with spatial interpolation, 2. a Bayesian Gaussian regression model, 3. the gridded reanalysis data from ERA5-Land. Both Bayesian models are station-based and exhibit higher and more spatially variable temperatures compared to ERA5-Land. Then, using individual mortality data for cardiovascular and respiratory causes, we estimate temperature-mortality associations through Bayesian conditional Poisson models in a case-crossover design. Exposure is defined as the mean maximum temperature over the previous three days. Additional models include heatwave definitions combining different thresholds and durations. All models exhibit a marked increase in relative risk at high temperatures; however, the temperature of minimum risk varies significantly across methods. Stratified analyses reveal higher relative risk increases in females and the elderly (80+). Heatwave effects depend on the definitions used, but all methods capture an increased mortality risk associated with prolonged heat exposure. Results confirm the importance of temperature model choice in epidemiology and provide insights for early warning systems and climate-health adaptation strategies.

---

_Clasificado automáticamente como **factor-riesgo** el 2026-07-17. Razón: Error LLM: litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash\nPlease retry in 1.68122827s.",
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
        "retryDelay": "1s"
      }
    ]
  }
}
_
