# Satellite-based time-series of sea-surface temperature since 1981 for climate applications

> **Fuente:** openalex  
> **DOI:** 10.1038/s41597-019-0236-x  
> **URL:** https://doi.org/10.1038/s41597-019-0236-x  
> **Año:** 2019

**Autores:** Christopher J. Merchant, Owen Embury, Claire E. Bulgin, T. Block, Gary K. Corlett et al.

## Abstract

Abstract A climate data record of global sea surface temperature (SST) spanning 1981–2016 has been developed from 4 × 10 12 satellite measurements of thermal infra-red radiance. The spatial area represented by pixel SST estimates is between 1 km 2 and 45 km 2 . The mean density of good-quality observations is 13 km −2 yr −1 . SST uncertainty is evaluated per datum, the median uncertainty for pixel SSTs being 0.18 K. Multi-annual observational stability relative to drifting buoy measurements is within 0.003 K yr −1 of zero with high confidence, despite maximal independence from in situ SSTs over the latter two decades of the record. Data are provided at native resolution, gridded at 0.05° latitude-longitude resolution (individual sensors), and aggregated and gap-filled on a daily 0.05° grid. Skin SSTs, depth-adjusted SSTs de-aliased with respect to the diurnal cycle, and SST anomalies are provided. Target applications of the dataset include: climate and ocean model evaluation; quantification of marine change and variability (including marine heatwaves); climate and ocean-atmosphere processes; and specific applications in ocean ecology, oceanography and geophysics.

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
