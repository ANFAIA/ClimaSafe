# Numerical Weather Forecasting using Convolutional-LSTM with Attention and Context Matcher Mechanisms

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2102.00696v2  
> **Año:** 2021

**Autores:** Selim Furkan Tekin, Arda Fazla, Suleyman Serdar Kozat

## Abstract

Numerical weather forecasting using high-resolution physical models often requires extensive computational resources on supercomputers, which diminishes their wide usage in most real-life applications. As a remedy, applying deep learning methods has revealed innovative solutions within this field. To this end, we introduce a novel deep learning architecture for forecasting high-resolution spatio-temporal weather data. Our approach extends the conventional encoder-decoder structure by integrating Convolutional Long-short Term Memory and Convolutional Neural Networks. In addition, we incorporate attention and context matcher mechanisms into the model architecture. Our Weather Model achieves significant performance improvements compared to baseline deep learning models, including ConvLSTM, TrajGRU, and U-Net. Our experimental evaluation involves high-scale, real-world benchmark numerical weather datasets, namely the ERA5 hourly dataset on pressure levels and WeatherBench. Our results demonstrate substantial improvements in identifying spatial and temporal correlations with attention matrices focusing on distinct parts of the input series to model atmospheric circulations. We also compare our model with high-resolution physical models using the benchmark metrics and show that our Weather Model is accurate and easy to interpret.

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
