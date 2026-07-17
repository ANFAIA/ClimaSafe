# Multivariate Temporal Convolutional Network: A Deep Neural Networks Approach for Multivariate Time Series Forecasting

> **Fuente:** openalex  
> **DOI:** 10.3390/electronics8080876  
> **URL:** https://doi.org/10.3390/electronics8080876  
> **Año:** 2019

**Autores:** Renzhuo Wan, Shuping Mei, Jun Wang, Min Liu, Fan Yang

## Abstract

Multivariable time series prediction has been widely studied in power energy, aerology, meteorology, finance, transportation, etc. Traditional modeling methods have complex patterns and are inefficient to capture long-term multivariate dependencies of data for desired forecasting accuracy. To address such concerns, various deep learning models based on Recurrent Neural Network (RNN) and Convolutional Neural Network (CNN) methods are proposed. To improve the prediction accuracy and minimize the multivariate time series data dependence for aperiodic data, in this article, Beijing PM2.5 and ISO-NE Dataset are analyzed by a novel Multivariate Temporal Convolution Network (M-TCN) model. In this model, multi-variable time series prediction is constructed as a sequence-to-sequence scenario for non-periodic datasets. The multichannel residual blocks in parallel with asymmetric structure based on deep convolution neural network is proposed. The results are compared with rich competitive algorithms of long short term memory (LSTM), convolutional LSTM (ConvLSTM), Temporal Convolution Network (TCN) and Multivariate Attention LSTM-FCN (MALSTM-FCN), which indicate significant improvement of prediction accuracy, robust and generalization of our model.

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
