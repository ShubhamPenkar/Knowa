import React, { useState, useEffect } from 'react'
import { BarChart3, RefreshCw } from 'lucide-react'
import { modelApi, feedbackApi } from '../services/api'

function Analytics() {
  const [actionEffectiveness, setActionEffectiveness] = useState([])
  const [feedbackStats, setFeedbackStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [actionsRes, feedbackRes] = await Promise.all([
        modelApi.getActionEffectiveness(),
        feedbackApi.getSummary(90),
      ])
      setActionEffectiveness(actionsRes.data || [])
      setFeedbackStats(feedbackRes.data)
    } catch (error) {
      console.error('Failed to load analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center min-h-screen">
        <RefreshCw className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <BarChart3 className="w-7 h-7 text-blue-600" />
            Analytics
          </h1>
          <p className="text-gray-500">Action effectiveness and system performance</p>
        </div>
        <button onClick={loadData} className="btn btn-secondary flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Feedback Summary */}
      {feedbackStats && (
        <div className="card mb-6">
          <h2 className="card-header">Feedback Summary (Last 90 Days)</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Total Feedback</p>
              <p className="text-2xl font-bold">{feedbackStats.total_feedback}</p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Model Accuracy</p>
              <p className="text-2xl font-bold">
                {feedbackStats.model_accuracy 
                  ? `${(feedbackStats.model_accuracy * 100).toFixed(1)}%`
                  : 'N/A'}
              </p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Retained</p>
              <p className="text-2xl font-bold text-green-600">
                {feedbackStats.outcome_distribution?.retained || 0}
              </p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Churned</p>
              <p className="text-2xl font-bold text-red-600">
                {feedbackStats.outcome_distribution?.churned || 0}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Action Effectiveness */}
      <div className="card">
        <h2 className="card-header">Action Effectiveness</h2>
        {actionEffectiveness.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 border-b">
                  <th className="pb-3 font-medium">Action</th>
                  <th className="pb-3 font-medium text-right">Times Recommended</th>
                  <th className="pb-3 font-medium text-right">Times Taken</th>
                  <th className="pb-3 font-medium text-right">Adoption Rate</th>
                  <th className="pb-3 font-medium text-right">Success Rate</th>
                </tr>
              </thead>
              <tbody>
                {actionEffectiveness.map((action) => (
                  <tr key={action.action_code} className="border-b last:border-0">
                    <td className="py-3">
                      <p className="font-medium">{action.action_name}</p>
                      <p className="text-xs text-gray-500">{action.action_code}</p>
                    </td>
                    <td className="py-3 text-right">{action.times_recommended}</td>
                    <td className="py-3 text-right">{action.times_taken}</td>
                    <td className="py-3 text-right">
                      <span className={`badge ${
                        action.adoption_rate > 0.5 ? 'bg-green-100 text-green-800' :
                        action.adoption_rate > 0.2 ? 'bg-yellow-100 text-yellow-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {(action.adoption_rate * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 text-right">
                      <span className={`badge ${
                        action.success_rate > 0.7 ? 'bg-green-100 text-green-800' :
                        action.success_rate > 0.4 ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {(action.success_rate * 100).toFixed(1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-center py-8 text-gray-500">
            No action effectiveness data available yet
          </p>
        )}
      </div>
    </div>
  )
}

export default Analytics
