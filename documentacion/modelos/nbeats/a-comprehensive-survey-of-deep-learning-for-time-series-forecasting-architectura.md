# A comprehensive survey of deep learning for time series forecasting: architectural diversity and open challenges

> **Fuente:** openalex  
> **DOI:** 10.1007/s10462-025-11223-9  
> **URL:** https://doi.org/10.1007/s10462-025-11223-9  
> **Año:** 2025

**Autores:** Jongseon Kim, Hyungjoon Kim, HyunGi Kim, Dongjun Lee, Sungroh Yoon

## Abstract

Abstract Time series forecasting is a critical task that provides key information for decision-making across various fields, such as economic planning, supply chain management, and medical diagnosis. After the use of traditional statistical methodologies and machine learning in the past, various fundamental deep learning architectures such as MLPs, CNNs, RNNs, and GNNs have been developed and applied to solve time series forecasting problems. However, the structural limitations caused by the inductive biases of each deep learning architecture constrained their performance. Transformer models, which excel at handling long-term dependencies, have become significant architectural components for time series forecasting. However, recent research has shown that alternatives such as simple linear layers can outperform Transformers. These findings have opened up new possibilities for using diverse architectures, ranging from fundamental deep learning models to emerging architectures and hybrid approaches. In this context of exploration into various models, the architectural modeling of time series forecasting has now entered a renaissance. This survey not only provides a historical context for time series forecasting but also offers comprehensive and timely analysis of the movement toward architectural diversification. By comparing and re-examining various deep learning models, we uncover new perspectives and present the latest trends in time series forecasting, including the emergence of hybrid models, diffusion models, Mamba models, and foundation models. By focusing on the inherent characteristics of time series data, we also address open challenges that have gained attention in time series forecasting, such as channel dependency, distribution shift, causality, and feature extraction. This survey explores vital elements that can enhance forecasting performance through diverse approaches. These contributions help lower entry barriers for newcomers by providing a systematic understanding of the diverse research areas in time series forecasting (TSF), while offering seasoned researchers broader perspectives and new opportunities through in-depth exploration of TSF challenges.

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
