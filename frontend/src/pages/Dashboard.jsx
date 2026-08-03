import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { 
  Users, 
  AlertTriangle, 
  TrendingUp, 
  Activity,
  ChevronRight,
  RefreshCw
} from 'lucide-react'
import { modelApi, feedbackApi } from '../services/api'

function Dashboard() {
  const [modelStatus, setModelStatus] = useState(null)
  const [feedbackStats, setFeedbackStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    setLoading(true)
    try {
      const [statusRes, feedbackRes] = await Promise.all([
        modelApi.getStatus(),
        feedbackApi.getSummary(30),
      ])
      setModelStatus(statusRes.data)
      setFeedbackStats(feedbackRes.data)
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const StatCard = ({ icon: Icon, title, value, subtitle, color }) => (
    <div className="card">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500">{title}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </div>
  )

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
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-gray-500">Overview of churn prediction and recommendations</p>
        </div>
        <button 
          onClick={loadDashboardData}
          className="btn btn-secondary flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          icon={Activity}
          title="Model Status"
          value={modelStatus?.model_loaded ? 'Active' : 'Not Loaded'}
          subtitle={modelStatus?.model_version || 'No version'}
          color={modelStatus?.model_loaded ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'}
        />
        <StatCard
          icon={Users}
          title="Total Feedback"
          value={feedbackStats?.total_feedback || 0}
          subtitle={`Last ${feedbackStats?.period_days || 30} days`}
          color="bg-blue-100 text-blue-600"
        />
        <StatCard
          icon={TrendingUp}
          title="Model Accuracy"
          value={feedbackStats?.model_accuracy 
            ? `${(feedbackStats.model_accuracy * 100).toFixed(1)}%` 
            : 'N/A'}
          subtitle="Based on feedback"
          color="bg-purple-100 text-purple-600"
        />
        <StatCard
          icon={AlertTriangle}
          title="Retrain Recommended"
          value={feedbackStats?.retrain_recommended ? 'Yes' : 'No'}
          subtitle="Based on performance"
          color={feedbackStats?.retrain_recommended 
            ? 'bg-orange-100 text-orange-600' 
            : 'bg-gray-100 text-gray-600'}
        />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="card">
          <h2 className="card-header">Quick Actions</h2>
          <div className="space-y-3">
            <Link 
              to="/simulation"
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div>
                <p className="font-medium">Run What-If Simulation</p>
                <p className="text-sm text-gray-500">Test how changes affect churn risk</p>
              </div>
              <ChevronRight className="w-5 h-5 text-gray-400" />
            </Link>
            <Link 
              to="/model"
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div>
                <p className="font-medium">Model Management</p>
                <p className="text-sm text-gray-500">Train models and view metrics</p>
              </div>
              <ChevronRight className="w-5 h-5 text-gray-400" />
            </Link>
            <Link 
              to="/analytics"
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div>
                <p className="font-medium">View Analytics</p>
                <p className="text-sm text-gray-500">Action effectiveness and trends</p>
              </div>
              <ChevronRight className="w-5 h-5 text-gray-400" />
            </Link>
          </div>
        </div>

        <div className="card">
          <h2 className="card-header">Outcome Distribution</h2>
          {feedbackStats?.outcome_distribution && 
           Object.keys(feedbackStats.outcome_distribution).length > 0 ? (
            <div className="space-y-4">
              {Object.entries(feedbackStats.outcome_distribution).map(([outcome, count]) => (
                <div key={outcome}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="capitalize">{outcome}</span>
                    <span className="font-medium">{count}</span>
                  </div>
                  <div className="progress-bar">
                    <div 
                      className={`progress-fill ${
                        outcome === 'retained' ? 'bg-green-500' :
                        outcome === 'churned' ? 'bg-red-500' : 'bg-gray-400'
                      }`}
                      style={{ 
                        width: `${(count / feedbackStats.total_feedback) * 100}%` 
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">
              No feedback data available yet
            </p>
          )}
        </div>
      </div>

      {/* Model Info */}
      {modelStatus?.latest_metrics && (
        <div className="card">
          <h2 className="card-header">Latest Model Performance</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {[
              { label: 'Accuracy', value: modelStatus.latest_metrics.accuracy },
              { label: 'Precision', value: modelStatus.latest_metrics.precision },
              { label: 'Recall', value: modelStatus.latest_metrics.recall },
              { label: 'F1 Score', value: modelStatus.latest_metrics.f1_score },
              { label: 'AUC-ROC', value: modelStatus.latest_metrics.auc_roc },
            ].map((metric) => (
              <div key={metric.label} className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-500">{metric.label}</p>
                <p className="text-xl font-bold mt-1">
                  {metric.value ? `${(metric.value * 100).toFixed(1)}%` : 'N/A'}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default Dashboard
