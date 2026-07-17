# Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution

> **Fuente:** openalex  
> **DOI:** 10.1145/3532611  
> **URL:** https://doi.org/10.1145/3532611  
> **Año:** 2022

**Autores:** Fuxian Li, Jie Feng, Huan Yan, Guangyin Jin, Fan Yang et al.

## Abstract

Traffic prediction is the cornerstone of intelligent transportation system. Accurate traffic forecasting is essential for the applications of smart cities, i.e., intelligent traffic management and urban planning. Although various methods are proposed for spatio-temporal modeling, they ignore the dynamic characteristics of correlations among locations on road network. Meanwhile, most Recurrent Neural Network based works are not efficient enough due to their recurrent operations. Additionally, there is a severe lack of fair comparison among different methods on the same datasets. To address the above challenges, in this article, we propose a novel traffic prediction framework, named Dynamic Graph Convolutional Recurrent Network (DGCRN). In DGCRN, hyper-networks are designed to leverage and extract dynamic characteristics from node attributes, while the parameters of dynamic filters are generated at each time step. We filter the node embeddings and then use them to generate dynamic graph, which is integrated with pre-defined static graph. As far as we know, we are first to employ a generation method to model fine topology of dynamic graph at each time step. Furthermore, to enhance efficiency and performance, we employ a training strategy for DGCRN by restricting the iteration number of decoder during forward and backward propagation. Finally, a reproducible standardized benchmark and a brand new representative traffic dataset are opened for fair comparison and further research. Extensive experiments on three datasets demonstrate that our model outperforms 15 baselines consistently. Source codes are available at https://github.com/tsinghua-fib-lab/Traffic-Benchmark .

---

_Clasificado automáticamente como **factor-riesgo** el 2026-07-17. Razón: Error LLM: litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\nPlease retry in 58.211901669s.",
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
        "retryDelay": "58s"
      }
    ]
  }
}
_
