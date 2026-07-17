# From physical surfaces to human-centric heat stress: LST and UTCI heat mapping reveals nonlinear effects of urban morphology

> **Fuente:** arxiv  
> **DOI:** N/A  
> **URL:** http://arxiv.org/abs/2604.22433v2  
> **Año:** 2026

**Autores:** Yuan Wang, Shengao Yi, Xiaojiang Li, Pengyuan Liu, Zhiwei Yang et al.

## Abstract

Heat exposure connects the built environment and public health, directly shaping the livability and sustainability of urban areas. Understanding the spatial heterogeneity of heat exposure and its drivers is vital for climate-adaptive urban planning. However, most planning-oriented studies rely on land surface temperature (LST), and whether LST adequately represents human heat exposure and how it differs from physiologically relevant heat stress remains insufficiently examined. Here, using Landsat-retrieved 30-m LST and GPU-accelerated 1-m universal thermal climate index (UTCI) in Singapore, this study establishes a comprehensive "Modeling-Comparing-Assessing" framework to systematically evaluate the spatial and mechanistic differences between these two metrics. We further investigate their pronounced non-stationary and threshold-based relationships with urban factors using a novel geographically weighted XGBoost (GW-XGBoost) and generalized additive model (GAM) workflow. Our results reveal substantial differences in the spatial patterns of LST and UTCI, along with marked spatial heterogeneity in how 2D and 3D urban factors impact these thermal metrics, as demonstrated by explainable GW-XGBoost models (test R2 = 0.855 for LST and 0.905 for UTCI). Crucially, spatially explicit SHAP shows that sky view factor plays a central role in explaining UTCI variability but exhibits a comparatively marginal independent contribution to LST, indicating that LST inadequately captures shading-driven and radiative processes governing actual human heat stress. Moreover, SHAP-GAM analysis indicates that higher albedo is associated with increased UTCI. These findings provide model-informed planning implications for integrating physiologically relevant thermal indices to support targeted heat risk management and human-centric urban planning.

---

_Clasificado automáticamente como **factor-riesgo** el 2026-07-17. Razón: Error LLM: litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash\nPlease retry in 4.14316937s.",
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
        "retryDelay": "4s"
      }
    ]
  }
}
_
