# Forecasting of Solar Power Using GRU–Temporal Fusion Transformer Model and DILATE Loss Function

> **Fuente:** openalex  
> **DOI:** 10.3390/en16248105  
> **URL:** https://doi.org/10.3390/en16248105  
> **Año:** 2023

**Autores:** Fatma Mazen Ali Mazen, Yomna Shaker, Rania Ahmed Abdel Azeem Abul Seoud

## Abstract

Solar power is a clean and sustainable energy source that does not emit greenhouse gases or other atmospheric pollutants. The inherent variability in solar energy due to random fluctuations introduces novel attributes to the power generation and load dynamics of the grid. Consequently, there has been growing attention to developing an accurate forecast model using various machine and deep learning techniques. Temporal attention mechanisms enable the model to concentrate on the critical components of the input sequence at each time step, thereby enhancing the accuracy of the prediction. The suggested GRU–temporal fusion transformer (GRU-TFT) model was trained and validated employing the “Daily Power Production of Solar Panels” Kaggle dataset. Furthermore, an innovative loss function termed DILATE is introduced to train the proposed model specifically for multistep and nonstationary time series forecasting. The outcomes have been subjected to a comparative analysis with alternative algorithms, such as neural basis expansion analysis for interpretable time series (N-BEATS), neural hierarchical interpolation for time series (N-HiTS), and extreme gradient boosting (XGBoost), using several evaluation metrics, including the absolute percentage error (MAE), mean square error (MSE), and root mean square error (RMSE). The model presented in this study exhibited significant performance improvements compared with traditional statistical and machine learning techniques. This is evident from the achieved values of MAE, MSE, and RMSE, which were 1.19, 2.08, and 1.44, respectively. In contrast, the machine learning approach utilizing the Holt–Winters method for time series forecasting in additive mode yielded MAE, MSE, and RMSE scores of 4.126, 29.105, and 5.3949, respectively.

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
