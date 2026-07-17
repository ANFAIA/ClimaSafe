# An integrated, probabilistic model for improved seasonal forecasting of agricultural crop yield under environmental uncertainty

> **Fuente:** openalex  
> **DOI:** 10.3389/fenvs.2014.00017  
> **URL:** https://doi.org/10.3389/fenvs.2014.00017  
> **Año:** 2014

**Autores:** Nathaniel K. Newlands, David Zamar, Louis Kouadio, Yinsuo Zhang, Aston Chipanshi et al.

## Abstract

We present a novel forecasting method for generating agricultural crop yield forecasts at the seasonal and regional-scale, integrating agroclimate variables and remotely-sensed indices. The method devises a multivariate statistical model to compute bias and uncertainty in forecasted yield at the Census of Agricultural Region (CAR) scale across the Canadian Prairies. The method uses robust variable-selection to select the best predictors within spatial subregions. Markov-Chain Monte Carlo (MCMC) simulation and random forest-tree machine learning techniques are then integrated to generate sequential forecasts through the growing season. Cross-validation of the model was performed by hindcasting/backcasting it and comparing its forecasts against available historical data (1987-2011) for spring wheat (Triticum aestivum L.). The model was also validated for the 2012 growing season by comparing its forecast skill at the CAR, provincial and Canadian Prairie region scales against available statistical survey data. Mean percent departures between wheat yield forecasted were under-estimated by 1-4 % in mid-season and over-estimated by 1 % at the end of the growing season. This integrated methodology offers a consistent, generalizable approach for sequentially forecasting crop yield at the regional-scale. It provides a statistically robust, yet flexible way to concurrently adjust to data-rich and data-sparse situations, adaptively select different predictors of yield to changing levels of environmental uncertainty, and to update forecasts sequentially so as to incorporate new data as it becomes available. This integrated method also provides additional statistical support for assessing the accuracy and reliability of model-based crop yield forecasts in time and space.

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
