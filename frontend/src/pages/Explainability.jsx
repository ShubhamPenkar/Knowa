import { useState, useEffect } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Explainability() {
  const { projectId } = useParams();
  const [searchParams] = useSearchParams();
  const predictionId = searchParams.get('prediction');
  
  const { token } = useAuth();
  const navigate = useNavigate();
  
  const [project, setProject] = useState(null);
  const [testData, setTestData] = useState(null);
  const [selectedRow, setSelectedRow] = useState(null);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (token && projectId) {
      fetchProject();
    }
  }, [projectId, token]);

  const fetchProject = async () => {
    try {
      const res = await fetch(`/api/projects/${projectId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setProject(data);
        if (data.status === 'trained' || data.status === 'ready') {
          fetchTestData();
        }
      }
    } catch (err) {
      console.error('Error fetching project:', err);
    }
    setLoading(false);
  };

  const fetchTestData = async () => {
    try {
      const res = await fetch(`/api/projects/${projectId}/test-data?limit=50`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setTestData(data.rows);
      }
    } catch (err) {
      console.error('Error fetching test data:', err);
    }
  };

  const handleAnalyze = async (row, idx) => {
    setSelectedRow(row);
    setSelectedIdx(idx);
    setAnalyzing(true);
    setError('');
    setPrediction(null);

    const features = {};
    project.feature_columns.forEach(col => {
      features[col] = row[col];
    });

    try {
      const res = await fetch(`/api/projects/${projectId}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          features,
          include_explanations: true,
          include_recommendations: true
        })
      });

      if (res.ok) {
        const data = await res.json();
        // Wrap prediction data properly
        setPrediction({
          prediction: data,
          features: features,
          actualValue: row[project.target_column]
        });
      } else {
        const errData = await res.json();
        setError(errData.detail || 'Analysis failed');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    }
    setAnalyzing(false);
  };

  if (loading) {
    return (
      <div className="p-6 flex justify-center">
        <div className="animate-spin h-8 w-8 border-4 border-brand-teal-dark border-t-transparent rounded-full"></div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="p-6">
        <p className="text-gray-600">Project not found</p>
      </div>
    );
  }

  if (project.status !== 'trained' && project.status !== 'ready') {
    return (
      <div className="p-6">
        <button
          onClick={() => navigate(`/projects/${projectId}`)}
          className="text-gray-600 hover:text-gray-900 mb-4 flex items-center gap-1"
        >
          ← Back to Project
        </button>
        <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 text-center">
          <h2 className="text-xl font-semibold text-yellow-800 mb-2">Model Not Trained</h2>
          <p className="text-yellow-700">
            Please train your model first to use the explainability features.
          </p>
          <button
            onClick={() => navigate(`/projects/${projectId}`)}
            className="mt-4 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700"
          >
            Go to Project to Train
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <button
          onClick={() => navigate(`/projects/${projectId}`)}
          className="text-gray-600 hover:text-gray-900 mb-2 flex items-center gap-1"
        >
          ← Back to Project
        </button>
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              🔍 Explainability Analysis
            </h1>
            <p className="text-gray-600 mt-1">
              Understand why the model makes its predictions using SHAP values
            </p>
          </div>
          <button
            onClick={() => navigate(`/projects/${projectId}/whatif${selectedRow ? `?row=${selectedIdx}` : ''}`)}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
          >
            Open What-If Analysis →
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Left: Row Selection */}
        <div className="bg-white rounded-xl shadow-sm border">
          <div className="p-4 border-b">
            <h2 className="text-lg font-semibold text-gray-900">Select a Record to Analyze</h2>
            <p className="text-gray-600 text-sm">Click on any row to see detailed SHAP explanations</p>
          </div>
          <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left text-gray-600">#</th>
                  <th className="px-3 py-2 text-left text-gray-600">Actual</th>
                  {project.feature_columns.slice(0, 3).map(col => (
                    <th key={col} className="px-3 py-2 text-left text-gray-600 truncate max-w-20">
                      {col}
                    </th>
                  ))}
                  <th className="px-3 py-2 text-left text-gray-600">Action</th>
                </tr>
              </thead>
              <tbody>
                {(testData || []).map((row, idx) => {
                  const isSelected = selectedIdx === idx;
                  const actualValue = row[project.target_column];
                  
                  return (
                    <tr
                      key={idx}
                      className={`border-t cursor-pointer transition-colors ${
                        isSelected ? 'bg-brand-teal/10' : 'hover:bg-gray-50'
                      }`}
                      onClick={() => handleAnalyze(row, idx)}
                    >
                      <td className="px-3 py-2 text-gray-500">{idx + 1}</td>
                      <td className="px-3 py-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          project.problem_type === 'regression'
                            ? 'bg-brand-teal/20 text-brand-teal-dark'
                            : String(actualValue) === project.target_positive_label
                              ? 'bg-red-100 text-red-700'
                              : 'bg-green-100 text-green-700'
                        }`}>
                          {project.problem_type === 'regression'
                            ? Number(actualValue).toFixed(2)
                            : String(actualValue) === project.target_positive_label ? 'Yes' : 'No'
                          }
                        </span>
                      </td>
                      {project.feature_columns.slice(0, 3).map(col => (
                        <td key={col} className="px-3 py-2 truncate max-w-20 text-gray-600">
                          {String(row[col] ?? '').substring(0, 15)}
                        </td>
                      ))}
                      <td className="px-3 py-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleAnalyze(row, idx);
                          }}
                          disabled={analyzing && selectedIdx === idx}
                          className="px-2 py-1 bg-brand-teal/20 text-brand-teal-dark rounded hover:bg-blue-200 text-xs disabled:opacity-50"
                        >
                          {analyzing && selectedIdx === idx ? '...' : 'Analyze'}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right: SHAP Explanation */}
        <div className="space-y-6">
          {/* Loading */}
          {analyzing && (
            <div className="bg-brand-teal/10 rounded-xl p-6 flex items-center justify-center gap-3">
              <div className="animate-spin h-6 w-6 border-3 border-brand-teal-dark border-t-transparent rounded-full"></div>
              <span className="text-brand-teal-dark">Analyzing prediction with SHAP...</span>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* No Selection */}
          {!prediction && !analyzing && !error && (
            <div className="bg-gray-50 rounded-xl p-8 text-center">
              <div className="text-6xl mb-4">🔬</div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Select a Record
              </h3>
              <p className="text-gray-600">
                Click on any row in the table to see a detailed SHAP explanation
                of why the model made that prediction.
              </p>
            </div>
          )}

          {/* Prediction Result */}
          {prediction && !analyzing && (
            <>
              {/* Prediction Summary */}
              <div className="bg-white rounded-xl shadow-sm border p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  📊 Prediction Summary
                </h3>
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-gray-500 text-sm mb-1">Actual</div>
                    <div className={`text-2xl font-bold ${
                      project.problem_type === 'regression'
                        ? 'text-brand-teal-dark'
                        : String(prediction.actualValue) === project.target_positive_label
                          ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {project.problem_type === 'regression'
                        ? Number(prediction.actualValue).toFixed(2)
                        : String(prediction.actualValue) === project.target_positive_label ? 'Yes' : 'No'
                      }
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-gray-500 text-sm mb-1">Predicted</div>
                    <div className={`text-2xl font-bold ${
                      project.problem_type === 'regression'
                        ? 'text-brand-teal-dark'
                        : prediction.prediction.probability > 0.5
                          ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {project.problem_type === 'regression'
                        ? prediction.prediction.predicted_value?.toFixed(2)
                        : `${(prediction.prediction.probability * 100).toFixed(1)}%`
                      }
                    </div>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-gray-500 text-sm mb-1 flex items-center gap-1">
                      Confidence
                      <span className="cursor-help text-gray-400" title="Model confidence based on agreement between multiple models (XGBoost, LightGBM, etc.) in the ensemble. Higher confidence means models agree on this prediction.">ⓘ</span>
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {(prediction.prediction.confidence * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-400 mt-1">Model agreement</div>
                  </div>
                </div>
              </div>

              {/* SHAP Explanation */}
              {prediction.prediction.explanations?.shap?.top_features && (
                <div className="bg-white rounded-xl shadow-sm border p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    🎯 SHAP Feature Importance
                  </h3>
                  <p className="text-gray-600 text-sm mb-4">
                    Features driving this prediction. Green = decreases risk, Red = increases risk.
                  </p>
                  
                  {/* Waterfall-style visualization */}
                  <div className="space-y-3">
                    {prediction.prediction.explanations.shap.top_features.map((f, i) => {
                      const maxImpact = Math.max(
                        ...prediction.prediction.explanations.shap.top_features.map(x => Math.abs(x.impact))
                      );
                      const barWidth = Math.abs(f.impact) / maxImpact * 100;
                      
                      return (
                        <div key={i} className="flex items-center gap-3">
                          <div className="w-32 text-sm text-gray-700 truncate font-medium">
                            {f.feature}
                          </div>
                          <div className="flex-1 h-8 bg-gray-100 rounded relative overflow-hidden">
                            {/* Center line */}
                            <div className="absolute inset-y-0 left-1/2 w-px bg-gray-300"></div>
                            {/* Impact bar */}
                            <div
                              className={`absolute inset-y-0 h-full transition-all ${
                                f.impact > 0 ? 'bg-red-400' : 'bg-green-400'
                              }`}
                              style={{
                                width: `${barWidth / 2}%`,
                                left: f.impact > 0 ? '50%' : undefined,
                                right: f.impact < 0 ? '50%' : undefined,
                              }}
                            />
                          </div>
                          <div className={`w-20 text-sm font-medium text-right ${
                            f.impact > 0 ? 'text-red-600' : 'text-green-600'
                          }`}>
                            {f.impact > 0 ? '+' : ''}{(f.impact * 100).toFixed(1)}%
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Base value explanation */}
                  {prediction.prediction.explanations.shap.base_value !== undefined && (
                    <div className="mt-4 pt-4 border-t text-sm text-gray-600">
                      <strong>Base prediction:</strong> {(prediction.prediction.explanations.shap.base_value * 100).toFixed(1)}%
                      (average model output before considering this record's features)
                    </div>
                  )}
                </div>
              )}

              {/* Feature Values */}
              <div className="bg-white rounded-xl shadow-sm border p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  📋 Feature Values for this Record
                </h3>
                <div className="grid grid-cols-2 gap-3 max-h-60 overflow-y-auto">
                  {Object.entries(prediction.features).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-sm p-2 bg-gray-50 rounded">
                      <span className="text-gray-600 truncate mr-2">{key}</span>
                      <span className="font-medium text-gray-900">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Business Insights - Improved with reasons and suggestions */}
              {prediction.prediction.insights?.length > 0 && (
                <div className="bg-white rounded-xl shadow-sm border p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    💡 Why This Prediction? (SHAP Analysis)
                  </h3>
                  <p className="text-gray-600 text-sm mb-4">
                    Understanding the key factors driving this {project.target_description || 'prediction'} risk:
                  </p>
                  <div className="space-y-4">
                    {prediction.prediction.insights.map((insight, i) => {
                      const severityStyles = {
                        critical: 'bg-red-50 border-red-200',
                        warning: 'bg-yellow-50 border-yellow-200',
                        positive: 'bg-green-50 border-green-200',
                        info: 'bg-brand-teal/10 border-blue-200'
                      };
                      const severityText = {
                        critical: 'text-red-800',
                        warning: 'text-yellow-800',
                        positive: 'text-green-800',
                        info: 'text-blue-800'
                      };
                      const severityBadge = {
                        critical: 'bg-red-100 text-red-700',
                        warning: 'bg-yellow-100 text-yellow-700',
                        positive: 'bg-green-100 text-green-700',
                        info: 'bg-brand-teal/20 text-brand-teal-dark'
                      };
                      
                      return (
                        <div key={i} className={`rounded-lg border p-4 ${severityStyles[insight.severity] || severityStyles.info}`}>
                          <div className="flex items-start justify-between mb-2">
                            <span className={`font-semibold ${severityText[insight.severity] || severityText.info}`}>
                              {insight.feature?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                            </span>
                            <div className="flex items-center gap-2">
                              <span className={`text-xs px-2 py-1 rounded ${severityBadge[insight.severity] || severityBadge.info}`}>
                                {insight.impact_strength || (Math.abs(insight.impact) > 0.1 ? 'High' : 'Medium')} Impact
                              </span>
                              <span className={`text-xs px-2 py-1 rounded ${insight.direction === 'increasing' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                                {insight.direction === 'increasing' ? '↑ Risk' : '↓ Risk'}
                              </span>
                            </div>
                          </div>
                          
                          {/* Current Value */}
                          <div className="text-sm text-gray-600 mb-2">
                            <span className="font-medium">Current Value:</span> {String(insight.value)}
                          </div>
                          
                          {/* Reason */}
                          <div className={`text-sm mb-3 ${severityText[insight.severity] || severityText.info}`}>
                            <span className="font-medium">📊 Analysis:</span> {insight.reason || insight.message}
                          </div>
                          
                          {/* Suggestion */}
                          {insight.suggestion && (
                            <div className="bg-white bg-opacity-60 rounded p-3 text-sm">
                              <span className="font-medium text-gray-800">💡 Recommendation:</span>
                              <span className="text-gray-700 ml-1">{insight.suggestion}</span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {prediction.prediction.recommendations?.length > 0 && (
                <div className="bg-brand-teal/10 rounded-xl border border-blue-200 p-6">
                  <h3 className="text-lg font-semibold text-blue-800 mb-3">
                    Recommended Actions
                  </h3>
                  <p className="text-xs text-blue-800/80 mb-3 leading-relaxed">
                    Rank scores and impact figures are illustrative catalog heuristics —
                    not re-simulated outcomes for this case.
                  </p>
                  <div className="space-y-3">
                    {prediction.prediction.recommendations.map((rec, i) => (
                      <div key={i} className="bg-white rounded-lg p-3 border border-blue-100">
                        <div className="flex justify-between items-start mb-1">
                          <span className="font-medium text-gray-900">{rec.action_name || rec.name}</span>
                          <span className="text-xs bg-brand-teal/20 text-brand-teal-dark px-2 py-1 rounded">
                            Rank score: {(((rec.final_score ?? rec.score) || 0) * 100).toFixed(0)}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600">{rec.reasoning || rec.reason}</p>
                        {rec.expected_probability_reduction > 0.01 && (
                          <p className="text-xs text-gray-500 mt-1">
                            Illustrative est. ~−
                            {(Number(rec.expected_probability_reduction) * 100).toFixed(0)} pp
                            {' '}(not re-simulated)
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-3">
                <button
                  onClick={() => navigate(`/projects/${projectId}/whatif?row=${selectedIdx}`)}
                  className="flex-1 px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-medium"
                >
                  🔮 Try What-If Scenario
                </button>
                <button
                  onClick={() => {
                    setPrediction(null);
                    setSelectedRow(null);
                    setSelectedIdx(null);
                  }}
                  className="px-4 py-3 border rounded-lg hover:bg-gray-50"
                >
                  Clear
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
