import React, { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { 
  AlertTriangle, 
  Shield, 
  TrendingDown, 
  Lightbulb,
  CheckCircle,
  RefreshCw
} from 'lucide-react'
import { 
  predictionApi, 
  explanationApi, 
  insightApi, 
  recommendationApi 
} from '../services/api'

function CustomerDetail() {
  const { predictionId, customerId } = useParams()
  const [prediction, setPrediction] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [insights, setInsights] = useState(null)
  const [recommendations, setRecommendations] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadData()
  }, [predictionId, customerId])

  const loadData = async () => {
    setLoading(true)
    setError(null)
    
    try {
      let pred
      if (predictionId) {
        pred = (await predictionApi.get(predictionId)).data
      } else if (customerId) {
        pred = (await predictionApi.getLatest(customerId)).data
      }
      setPrediction(pred)

      if (pred?.id) {
        const [expRes, insRes, recRes] = await Promise.all([
          explanationApi.get(pred.id).catch(() => ({ data: null })),
          insightApi.get(pred.id).catch(() => ({ data: null })),
          recommendationApi.get(pred.id).catch(() => ({ data: null })),
        ])
        setExplanation(expRes.data)
        setInsights(insRes.data)
        setRecommendations(recRes.data)
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (level) => {
    const colors = {
      critical: 'risk-critical',
      high: 'risk-high',
      medium: 'risk-medium',
      low: 'risk-low',
    }
    return colors[level] || 'risk-medium'
  }

  const getSeverityClass = (severity) => {
    const classes = {
      critical: 'severity-critical',
      warning: 'severity-warning',
      info: 'severity-info',
      positive: 'severity-positive',
    }
    return classes[severity] || 'severity-info'
  }

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center min-h-screen">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-8">
        <div className="card bg-red-50 border-red-200">
          <p className="text-red-800">{error}</p>
        </div>
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="p-8">
        <div className="card">
          <p className="text-gray-500">No prediction found</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Prediction Details</h1>
          <p className="text-gray-500">ID: {prediction.id}</p>
        </div>
        <button onClick={loadData} className="btn btn-secondary flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Prediction Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <div className="card">
          <h3 className="text-sm text-gray-500 mb-2">Churn Probability</h3>
          <div className="flex items-center gap-4">
            <div className="text-4xl font-bold">
              {(prediction.churn_probability * 100).toFixed(1)}%
            </div>
            <span className={`badge ${getRiskColor(prediction.churn_risk_level)}`}>
              {prediction.churn_risk_level}
            </span>
          </div>
        </div>
        
        <div className="card">
          <h3 className="text-sm text-gray-500 mb-2">Confidence Score</h3>
          <div className="text-4xl font-bold">
            {(prediction.confidence_score * 100).toFixed(1)}%
          </div>
          <p className="text-sm text-gray-400 mt-1">Model certainty</p>
        </div>

        <div className="card">
          <h3 className="text-sm text-gray-500 mb-2">Explanation Trust</h3>
          <div className="text-4xl font-bold capitalize">
            {explanation?.trust_level || 'N/A'}
          </div>
          <p className="text-sm text-gray-400 mt-1">
            Consistency: {explanation?.consistency_score 
              ? `${(explanation.consistency_score * 100).toFixed(1)}%` 
              : 'N/A'}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Insights */}
        <div className="card">
          <h2 className="card-header flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-yellow-500" />
            Business Insights
          </h2>
          {insights?.insights?.length > 0 ? (
            <div className="space-y-3">
              {insights.insights.map((insight, idx) => (
                <div 
                  key={idx}
                  className={`p-4 rounded-lg ${getSeverityClass(insight.severity)}`}
                >
                  <p className="text-sm">{insight.text}</p>
                  <p className="text-xs text-gray-500 mt-1">
                    Feature: {insight.display_name || insight.feature}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No insights available</p>
          )}
          
          {insights?.summary && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <p className="text-sm font-medium">Summary</p>
              <p className="text-sm text-gray-600 mt-1">{insights.summary}</p>
            </div>
          )}
        </div>

        {/* Recommendations */}
        <div className="card">
          <h2 className="card-header flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-500" />
            Recommended Actions
          </h2>
          {recommendations?.recommendations?.length > 0 ? (
            <div className="space-y-4">
              {recommendations.recommendations.map((rec, idx) => (
                <div 
                  key={idx}
                  className="p-4 border border-gray-200 rounded-lg"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-lg font-bold text-blue-600">
                          #{rec.rank}
                        </span>
                        <span className="font-medium">{rec.action_name}</span>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">{rec.description}</p>
                    </div>
                    <span className="badge bg-blue-100 text-blue-800">
                      Score: {(rec.final_score * 100).toFixed(0)}
                    </span>
                  </div>
                  
                  <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                    <div className="text-center p-2 bg-gray-50 rounded">
                      <p className="text-gray-500">Impact</p>
                      <p className="font-medium">{(rec.impact_score * 100).toFixed(0)}%</p>
                    </div>
                    <div className="text-center p-2 bg-gray-50 rounded">
                      <p className="text-gray-500">Cost</p>
                      <p className="font-medium">{(rec.cost_score * 100).toFixed(0)}%</p>
                    </div>
                    <div className="text-center p-2 bg-gray-50 rounded">
                      <p className="text-gray-500">Relevance</p>
                      <p className="font-medium">{(rec.relevance_score * 100).toFixed(0)}%</p>
                    </div>
                  </div>
                  
                  <p className="text-xs text-gray-500 mt-2 italic">{rec.reasoning}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No recommendations available</p>
          )}
        </div>
      </div>

      {/* Feature Explanation */}
      {explanation && (
        <div className="card mt-6">
          <h2 className="card-header flex items-center gap-2">
            <TrendingDown className="w-5 h-5 text-purple-500" />
            Feature Importance (SHAP)
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium text-red-600 mb-2 flex items-center gap-1">
                <AlertTriangle className="w-4 h-4" />
                Risk Factors
              </h4>
              <div className="space-y-2">
                {explanation.top_risk_factors?.map((factor, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-sm">
                    <span className="w-4 h-4 bg-red-100 text-red-600 rounded flex items-center justify-center text-xs">
                      {idx + 1}
                    </span>
                    {factor}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="text-sm font-medium text-green-600 mb-2 flex items-center gap-1">
                <Shield className="w-4 h-4" />
                Protective Factors
              </h4>
              <div className="space-y-2">
                {explanation.top_protective_factors?.map((factor, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-sm">
                    <span className="w-4 h-4 bg-green-100 text-green-600 rounded flex items-center justify-center text-xs">
                      {idx + 1}
                    </span>
                    {factor}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default CustomerDetail
