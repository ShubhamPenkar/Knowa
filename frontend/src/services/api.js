/**
 * Legacy demo API client (fixed churn schema under /api/predict, /api/explain, etc.).
 * The SaaS product UI uses authenticated fetch() against /api/projects, /api/datasets, ...
 * Keep this file for scripts, demos, or future rewiring of a demo console.
 */
import axios from 'axios'

const API_BASE = '/api'

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Prediction API
export const predictionApi = {
  create: (data) => api.post('/predict', data),
  get: (id) => api.get(`/predict/${id}`),
  getCustomerPredictions: (customerId, limit = 10) =>
    api.get(`/predict/customer/${customerId}?limit=${limit}`),
  getLatest: (customerId) => api.get(`/predict/customer/${customerId}/latest`),
  createCustomer: (data) => api.post('/predict/customer', data),
}

// Explanation API
export const explanationApi = {
  get: (predictionId) => api.get(`/explain/${predictionId}`),
  generate: (predictionId) => api.post(`/explain/${predictionId}/generate`),
}

// Insight API
export const insightApi = {
  get: (predictionId) => api.get(`/insights/${predictionId}`),
  generate: (predictionId) => api.post(`/insights/${predictionId}/generate`),
}

// Recommendation API
export const recommendationApi = {
  get: (predictionId, topN = 5) =>
    api.get(`/recommend/${predictionId}?top_n=${topN}`),
  generate: (predictionId, topN = 5) =>
    api.post(`/recommend/${predictionId}/generate?top_n=${topN}`),
}

// Simulation API
export const simulationApi = {
  run: (data) => api.post('/simulate', data),
  fromPrediction: (predictionId, modifiedFeatures) =>
    api.post(`/simulate/from-prediction/${predictionId}`, modifiedFeatures),
}

// Feedback API
export const feedbackApi = {
  submit: (data) => api.post('/feedback', data),
  get: (predictionId) => api.get(`/feedback/${predictionId}`),
  getSummary: (days = 30) => api.get(`/feedback/stats/summary?days=${days}`),
}

// Model API
export const modelApi = {
  train: (modelType = 'ensemble') =>
    api.post(`/model/train?model_type=${modelType}`),
  getMetrics: (limit = 10) => api.get(`/model/metrics?limit=${limit}`),
  getVersionMetrics: (version) => api.get(`/model/metrics/${version}`),
  getActionEffectiveness: () => api.get('/model/actions/effectiveness'),
  getStatus: () => api.get('/model/status'),
}

export default api
