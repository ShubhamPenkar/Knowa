import React, { useState } from 'react'
import { FlaskConical, ArrowRight, TrendingDown, TrendingUp } from 'lucide-react'
import { simulationApi, predictionApi } from '../services/api'

const defaultFeatures = {
  tenure: 12,
  monthly_charges: 65,
  total_charges: 780,
  contract_type: 'month-to-month',
  payment_method: 'electronic_check',
  internet_service: 'fiber_optic',
  online_security: 'no',
  tech_support: 'no',
  streaming_tv: 'yes',
  streaming_movies: 'yes',
  num_support_tickets: 2,
  days_since_last_interaction: 30,
  num_complaints: 1,
  satisfaction_score: 3.0,
}

const featureConfig = {
  tenure: { label: 'Tenure (months)', type: 'number', min: 0, max: 72 },
  monthly_charges: { label: 'Monthly Charges ($)', type: 'number', min: 20, max: 120 },
  contract_type: { 
    label: 'Contract Type', 
    type: 'select', 
    options: ['month-to-month', 'one_year', 'two_year']
  },
  payment_method: {
    label: 'Payment Method',
    type: 'select',
    options: ['electronic_check', 'mailed_check', 'bank_transfer', 'credit_card']
  },
  tech_support: { label: 'Tech Support', type: 'select', options: ['yes', 'no'] },
  online_security: { label: 'Online Security', type: 'select', options: ['yes', 'no'] },
  satisfaction_score: { label: 'Satisfaction Score', type: 'number', min: 1, max: 5, step: 0.1 },
  num_complaints: { label: 'Complaints', type: 'number', min: 0, max: 10 },
}

function Simulation() {
  const [baseFeatures, setBaseFeatures] = useState(defaultFeatures)
  const [modifiedFeatures, setModifiedFeatures] = useState({})
  const [result, setResult] = useState(null)
  const [basePrediction, setBasePrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleBaseChange = (feature, value) => {
    setBaseFeatures(prev => ({ ...prev, [feature]: value }))
  }

  const handleModifiedChange = (feature, value) => {
    if (value === '' || value === baseFeatures[feature]) {
      const newModified = { ...modifiedFeatures }
      delete newModified[feature]
      setModifiedFeatures(newModified)
    } else {
      setModifiedFeatures(prev => ({ ...prev, [feature]: value }))
    }
  }

  const runBasePrediction = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await predictionApi.create({ features: baseFeatures })
      setBasePrediction(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction')
    } finally {
      setLoading(false)
    }
  }

  const runSimulation = async () => {
    if (Object.keys(modifiedFeatures).length === 0) {
      setError('Please modify at least one feature')
      return
    }
    
    setLoading(true)
    setError(null)
    try {
      const res = await simulationApi.run({
        base_features: baseFeatures,
        modified_features: modifiedFeatures,
      })
      setResult(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Simulation failed')
    } finally {
      setLoading(false)
    }
  }

  const renderInput = (feature, config, value, onChange) => {
    if (config.type === 'select') {
      return (
        <select
          value={value}
          onChange={(e) => onChange(feature, e.target.value)}
          className="input"
        >
          <option value="">-- Keep Original --</option>
          {config.options.map(opt => (
            <option key={opt} value={opt}>{opt.replace(/_/g, ' ')}</option>
          ))}
        </select>
      )
    }
    return (
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(feature, parseFloat(e.target.value) || e.target.value)}
        min={config.min}
        max={config.max}
        step={config.step || 1}
        className="input"
      />
    )
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <FlaskConical className="w-7 h-7 text-purple-600" />
          What-If Simulation
        </h1>
        <p className="text-gray-500">
          Modify features to see how changes affect churn prediction
        </p>
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-800">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Base Features */}
        <div className="card">
          <h2 className="card-header">Base Features</h2>
          <div className="space-y-4">
            {Object.entries(featureConfig).map(([feature, config]) => (
              <div key={feature}>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {config.label}
                </label>
                {renderInput(feature, config, baseFeatures[feature], handleBaseChange)}
              </div>
            ))}
          </div>
          <button
            onClick={runBasePrediction}
            disabled={loading}
            className="btn btn-primary w-full mt-4"
          >
            Get Base Prediction
          </button>
          
          {basePrediction && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Base Churn Probability</p>
              <p className="text-2xl font-bold">
                {(basePrediction.churn_probability * 100).toFixed(1)}%
              </p>
              <span className={`badge ${
                basePrediction.churn_risk_level === 'low' ? 'bg-green-100 text-green-800' :
                basePrediction.churn_risk_level === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {basePrediction.churn_risk_level} risk
              </span>
            </div>
          )}
        </div>

        {/* Modified Features */}
        <div className="card">
          <h2 className="card-header">Modified Features</h2>
          <p className="text-sm text-gray-500 mb-4">
            Change values to simulate different scenarios
          </p>
          <div className="space-y-4">
            {Object.entries(featureConfig).map(([feature, config]) => (
              <div key={feature}>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {config.label}
                </label>
                {renderInput(
                  feature, 
                  config, 
                  modifiedFeatures[feature] ?? '', 
                  handleModifiedChange
                )}
                {modifiedFeatures[feature] !== undefined && (
                  <p className="text-xs text-blue-600 mt-1">
                    Changed from: {baseFeatures[feature]}
                  </p>
                )}
              </div>
            ))}
          </div>
          <button
            onClick={runSimulation}
            disabled={loading || Object.keys(modifiedFeatures).length === 0}
            className="btn btn-success w-full mt-4"
          >
            Run Simulation
          </button>
        </div>

        {/* Results */}
        <div className="card">
          <h2 className="card-header">Simulation Results</h2>
          {result ? (
            <div className="space-y-6">
              {/* Probability Comparison */}
              <div className="flex items-center justify-between">
                <div className="text-center">
                  <p className="text-sm text-gray-500">Original</p>
                  <p className="text-2xl font-bold">
                    {(result.original_probability * 100).toFixed(1)}%
                  </p>
                </div>
                <ArrowRight className="w-6 h-6 text-gray-400" />
                <div className="text-center">
                  <p className="text-sm text-gray-500">Modified</p>
                  <p className="text-2xl font-bold">
                    {(result.modified_probability * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Change Indicator */}
              <div className={`p-4 rounded-lg text-center ${
                result.risk_level_change === 'improved' 
                  ? 'bg-green-50 text-green-800' 
                  : result.risk_level_change === 'worsened'
                  ? 'bg-red-50 text-red-800'
                  : 'bg-gray-50 text-gray-800'
              }`}>
                <div className="flex items-center justify-center gap-2 mb-2">
                  {result.risk_level_change === 'improved' 
                    ? <TrendingDown className="w-5 h-5" />
                    : result.risk_level_change === 'worsened'
                    ? <TrendingUp className="w-5 h-5" />
                    : null
                  }
                  <span className="font-medium capitalize">{result.risk_level_change}</span>
                </div>
                <p className="text-3xl font-bold">
                  {result.probability_change_percent > 0 ? '+' : ''}
                  {result.probability_change_percent.toFixed(1)}%
                </p>
              </div>

              {/* Key Changes */}
              {result.key_changes?.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2">Key Insights</h4>
                  <ul className="space-y-2">
                    {result.key_changes.map((change, idx) => (
                      <li key={idx} className="text-sm text-gray-600 flex items-start gap-2">
                        <span className="text-blue-500">•</span>
                        {change}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Recommendation */}
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="text-sm font-medium text-blue-800">Recommendation</p>
                <p className="text-sm text-blue-600 mt-1">{result.recommendation}</p>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <FlaskConical className="w-12 h-12 mx-auto mb-3 text-gray-300" />
              <p>Modify features and run simulation to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Simulation
