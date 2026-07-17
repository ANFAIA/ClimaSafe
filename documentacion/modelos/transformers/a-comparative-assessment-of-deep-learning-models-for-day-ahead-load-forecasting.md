# A comparative assessment of deep learning models for day-ahead load forecasting: Investigating key accuracy drivers

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2302.12168v2  
> **Año:** 2023

**Autores:** Sotiris Pelekis, Ioannis-Konstantinos Seisopoulos, Evangelos Spiliotis, Theodosios Pountridis, Evangelos Karakolis et al.

## Abstract

Short-term load forecasting (STLF) is vital for the effective and economic operation of power grids and energy markets. However, the non-linearity and non-stationarity of electricity demand as well as its dependency on various external factors renders STLF a challenging task. To that end, several deep learning models have been proposed in the literature for STLF, reporting promising results. In order to evaluate the accuracy of said models in day-ahead forecasting settings, in this paper we focus on the national net aggregated STLF of Portugal and conduct a comparative study considering a set of indicative, well-established deep autoregressive models, namely multi-layer perceptrons (MLP), long short-term memory networks (LSTM), neural basis expansion coefficient analysis (N-BEATS), temporal convolutional networks (TCN), and temporal fusion transformers (TFT). Moreover, we identify factors that significantly affect the demand and investigate their impact on the accuracy of each model. Our results suggest that N-BEATS consistently outperforms the rest of the examined models. MLP follows, providing further evidence towards the use of feed-forward networks over relatively more sophisticated architectures. Finally, certain calendar and weather features like the hour of the day and the temperature are identified as key accuracy drivers, providing insights regarding the forecasting approach that should be used per case.

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
