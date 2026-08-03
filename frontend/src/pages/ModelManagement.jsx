import React, { useState, useEffect } from 'react'
import { Settings, Play, RefreshCw, CheckCircle, XCircle } from 'lucide-react'
import { modelApi } from '../services/api'

function ModelManagement() {
  const [status, setStatus] = useState(null)
  const [metrics, setMetrics] = useState([])
  const [training, setTraining] = useState(false)
  const [trainingResult, setTrainingResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState('ensemble')

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [statusRes, metricsRes] = await Promise.all([
        modelApi.getStatus(),
        modelApi.getMetrics(5),
      ])
      setStatus(statusRes.data)
      setMetrics(metricsRes.data)
    } catch (error) {
      console.error('Failed to load model data:', error)
    } finally {
      setLoading(false)
    }
  }

  const trainModel = async () => {
    setTraining(true)
    setTrainingResult(null)
    try {
      const res = await modelApi.train(selectedModel)
      setTrainingResult({ success: true, message: res.data.message })
      // Reload data after training
      setTimeout(loadData, 2000)
    } catch (error) {
      setTrainingResult({ 
        success: false, 
        message: error.response?.data?.detail || 'Training failed' 
      })
    } finally {
      setTraining(false)
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
            <Settings className="w-7 h-7 text-gray-600" />
            Model Management
          </h1>
          <p className="text-gray-500">Train models and monitor performance</p>
        </div>
        <button onClick={loadData} className="btn btn-secondary flex items-center gap-2">
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </div>

      {/* Model Status */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="card">
          <h2 className="card-header">Current Model Status</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-gray-500">Status</span>
              <span className={`badge ${
                status?.model_loaded 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {status?.model_loaded ? 'Loaded' : 'Not Loaded'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-500">Model Type</span>
              <span className="font-medium">{status?.model_type || 'N/A'}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-500">Version</span>
              <span className="font-medium">{status?.model_version || 'N/A'}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-500">Model Path</span>
              <span className="font-mono text-sm">{status?.model_path || 'N/A'}</span>
            </div>
          </div>
        </div>

        <div className="card">
          <h2 className="card-header">Train New Model</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Type
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="input"
                disabled={training}
              >
                <option value="ensemble">Ensemble (Recommended)</option>
                <option value="xgboost">XGBoost</option>
                <option value="lightgbm">LightGBM</option>
                <option value="random_forest">Random Forest</option>
                <option value="logistic">Logistic Regression</option>
              </select>
            </div>
            
            <button
              onClick={trainModel}
              disabled={training}
              className="btn btn-primary w-full flex items-center justify-center gap-2"
            >
              {training ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Train Model
                </>
              )}
            </button>

            {trainingResult && (
              <div className={`p-4 rounded-lg flex items-center gap-2 ${
                trainingResult.success 
                  ? 'bg-green-50 text-green-800' 
                  : 'bg-red-50 text-red-800'
              }`}>
                {trainingResult.success 
                  ? <CheckCircle className="w-5 h-5" />
                  : <XCircle className="w-5 h-5" />
                }
                {trainingResult.message}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Performance History */}
      <div className="card">
        <h2 className="card-header">Performance History</h2>
        {metrics.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 border-b">
                  <th className="pb-3 font-medium">Version</th>
                  <th className="pb-3 font-medium">Date</th>
                  <th className="pb-3 font-medium text-right">Accuracy</th>
                  <th className="pb-3 font-medium text-right">Precision</th>
                  <th className="pb-3 font-medium text-right">Recall</th>
                  <th className="pb-3 font-medium text-right">F1 Score</th>
                  <th className="pb-3 font-medium text-right">AUC-ROC</th>
                </tr>
              </thead>
              <tbody>
                {metrics.map((m, idx) => (
                  <tr key={idx} className="border-b last:border-0">
                    <td className="py-3 font-mono text-sm">{m.model_version}</td>
                    <td className="py-3 text-sm">
                      {new Date(m.evaluation_date).toLocaleDateString()}
                    </td>
                    <td className="py-3 text-right">
                      {m.accuracy ? `${(m.accuracy * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="py-3 text-right">
                      {m.precision ? `${(m.precision * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="py-3 text-right">
                      {m.recall ? `${(m.recall * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="py-3 text-right">
                      {m.f1_score ? `${(m.f1_score * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                    <td className="py-3 text-right">
                      {m.auc_roc ? `${(m.auc_roc * 100).toFixed(1)}%` : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-center py-8 text-gray-500">
            No performance history available. Train a model to see metrics.
          </p>
        )}
      </div>
    </div>
  )
}

export default ModelManagement
