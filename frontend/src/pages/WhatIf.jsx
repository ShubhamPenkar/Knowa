import { useState, useEffect } from 'react';
import { useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function WhatIf() {
  const { projectId } = useParams();
  const [searchParams] = useSearchParams();
  const rowIdx = searchParams.get('row');
  
  const { token } = useAuth();
  const navigate = useNavigate();
  
  const [project, setProject] = useState(null);
  const [testData, setTestData] = useState(null);
  const [baseRow, setBaseRow] = useState(null);
  const [modifiedValues, setModifiedValues] = useState({});
  const [baselinePrediction, setBaselinePrediction] = useState(null);
  const [simulationResult, setSimulationResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);

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
          fetchTestData(data);
        }
      }
    } catch (err) {
      console.error('Error fetching project:', err);
    }
    setLoading(false);
  };

  const fetchTestData = async (proj) => {
    try {
      const res = await fetch(`/api/projects/${projectId}/test-data?limit=50`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setTestData(data.rows);
        
        // Auto-select row if specified in URL
        if (rowIdx !== null && data.rows[parseInt(rowIdx)]) {
          const row = data.rows[parseInt(rowIdx)];
          selectBaseRow(row, proj);
        }
      }
    } catch (err) {
      console.error('Error fetching test data:', err);
    }
  };

  const selectBaseRow = async (row, proj = project) => {
    setBaseRow(row);
    setModifiedValues({});
    setSimulationResult(null);
    setError('');

    // Get baseline prediction
    const features = {};
    proj.feature_columns.forEach(col => {
      features[col] = row[col];
    });

    try {
      const res = await fetch(`/api/projects/${projectId}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ features, include_explanations: true })
      });

      if (res.ok) {
        const data = await res.json();
        setBaselinePrediction(data);
      }
    } catch (err) {
      console.error('Error getting baseline:', err);
    }
  };

  const handleFeatureChange = (feature, value) => {
    // Parse as number if it looks like one
    const parsedValue = value === '' ? undefined : 
      !isNaN(Number(value)) ? Number(value) : value;
    
    setModifiedValues(prev => ({
      ...prev,
      [feature]: parsedValue
    }));
  };

  const runSimulation = async () => {
    setSimulating(true);
    setError('');

    const baseFeatures = {};
    project.feature_columns.forEach(col => {
      baseFeatures[col] = baseRow[col];
    });

    try {
      const res = await fetch(`/api/projects/${projectId}/simulate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          base_features: baseFeatures,
          modified_features: modifiedValues
        })
      });

      if (res.ok) {
        const data = await res.json();
        setSimulationResult(data);
        
        // Add to history
        setHistory(prev => [{
          timestamp: new Date().toISOString(),
          changes: { ...modifiedValues },
          result: data
        }, ...prev.slice(0, 9)]);
      } else {
        const errData = await res.json();
        setError(errData.detail || 'Simulation failed');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    }
    setSimulating(false);
  };

  const resetModifications = () => {
    setModifiedValues({});
    setSimulationResult(null);
  };

  const applyHistoryItem = (item) => {
    setModifiedValues(item.changes);
    setSimulationResult(item.result);
  };

  if (loading) {
    return (
      <div className="p-6 flex justify-center">
        <div className="animate-spin h-8 w-8 border-4 border-purple-500 border-t-transparent rounded-full"></div>
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
            Please train your model first to use what-if analysis.
          </p>
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
              🔮 What-If Analysis
            </h1>
            <p className="text-gray-600 mt-1">
              Modify feature values to see how the prediction changes
            </p>
          </div>
          <button
            onClick={() => navigate(`/projects/${projectId}/explainability`)}
            className="px-4 py-2 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal"
          >
            ← Back to Explainability
          </button>
        </div>
      </div>

      {/* Step 1: Select Base Record */}
      {!baseRow && (
        <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Step 1: Select a Base Record
          </h2>
          <p className="text-gray-600 mb-4">
            Choose a record as your starting point for the what-if analysis
          </p>
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-3 py-2 text-left text-gray-600">#</th>
                  <th className="px-3 py-2 text-left text-gray-600">Actual</th>
                  {project.feature_columns.slice(0, 4).map(col => (
                    <th key={col} className="px-3 py-2 text-left text-gray-600 truncate max-w-24">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(testData || []).map((row, idx) => (
                  <tr
                    key={idx}
                    className="border-t cursor-pointer hover:bg-purple-50 transition-colors"
                    onClick={() => selectBaseRow(row)}
                  >
                    <td className="px-3 py-2 text-gray-500">{idx + 1}</td>
                    <td className="px-3 py-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        project.problem_type === 'regression'
                          ? 'bg-brand-teal/20 text-brand-teal-dark'
                          : String(row[project.target_column]) === project.target_positive_label
                            ? 'bg-red-100 text-red-700'
                            : 'bg-green-100 text-green-700'
                      }`}>
                        {project.problem_type === 'regression'
                          ? Number(row[project.target_column]).toFixed(2)
                          : String(row[project.target_column]) === project.target_positive_label ? 'Yes' : 'No'
                        }
                      </span>
                    </td>
                    {project.feature_columns.slice(0, 4).map(col => (
                      <td key={col} className="px-3 py-2 truncate max-w-24 text-gray-600">
                        {String(row[col] ?? '').substring(0, 12)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* What-If Interface */}
      {baseRow && (
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left: Feature Modifications */}
          <div className="lg:col-span-2 space-y-6">
            {/* Current Prediction Banner */}
            {baselinePrediction && (
              <div className="bg-gradient-to-r from-purple-500 to-blue-500 text-white rounded-xl p-6">
                <div className="flex justify-between items-center">
                  <div>
                    <div className="text-purple-100 text-sm">Current {project.target_description || 'Prediction'}</div>
                    <div className="text-4xl font-bold">
                      {project.problem_type === 'regression'
                        ? baselinePrediction.predicted_value?.toFixed(2)
                        : `${(baselinePrediction.probability * 100).toFixed(1)}%`
                      }
                    </div>
                    <div className="text-purple-100 text-sm mt-1">
                      Confidence: {(baselinePrediction.confidence * 100).toFixed(0)}%
                      {baselinePrediction.risk_level && ` • Risk: ${baselinePrediction.risk_level}`}
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      setBaseRow(null);
                      setBaselinePrediction(null);
                      setSimulationResult(null);
                      setModifiedValues({});
                    }}
                    className="px-3 py-2 bg-white/20 hover:bg-white/30 rounded-lg text-sm"
                  >
                    Change Record
                  </button>
                </div>
              </div>
            )}

            {/* Feature Modification Panel */}
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <div className="flex justify-between items-center mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">
                    Modify Features
                  </h3>
                  <p className="text-gray-600 text-sm">
                    Change values to see how it affects the prediction
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={resetModifications}
                    className="px-3 py-2 text-gray-600 hover:bg-gray-100 rounded-lg text-sm"
                  >
                    Reset All
                  </button>
                  <button
                    onClick={runSimulation}
                    disabled={simulating || Object.keys(modifiedValues).filter(k => modifiedValues[k] !== undefined).length === 0}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 text-sm font-medium"
                  >
                    {simulating ? 'Running...' : 'Run Simulation'}
                  </button>
                </div>
              </div>

              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
                  {error}
                </div>
              )}

              <div className="grid md:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
                {project.feature_columns.map(col => {
                  const originalValue = baseRow[col];
                  const isModified = modifiedValues[col] !== undefined;
                  
                  return (
                    <div key={col} className={`p-3 rounded-lg border ${
                      isModified ? 'border-purple-300 bg-purple-50' : 'border-gray-200'
                    }`}>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {col}
                        {isModified && (
                          <span className="ml-2 text-purple-600 text-xs">modified</span>
                        )}
                      </label>
                      <div className="flex gap-2 items-center">
                        <input
                          type="text"
                          value={modifiedValues[col] !== undefined ? modifiedValues[col] : originalValue}
                          onChange={(e) => handleFeatureChange(col, e.target.value)}
                          className={`flex-1 px-3 py-2 border rounded-lg text-sm ${
                            isModified ? 'border-purple-400' : 'border-gray-300'
                          }`}
                        />
                        {isModified && (
                          <button
                            onClick={() => {
                              setModifiedValues(prev => {
                                const next = { ...prev };
                                delete next[col];
                                return next;
                              });
                            }}
                            className="p-2 text-gray-400 hover:text-gray-600"
                            title="Reset to original"
                          >
                            ↩
                          </button>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Original: {String(originalValue)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Simulation Result */}
            {simulationResult && (
              <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
                <div className="bg-gradient-to-r from-green-500 to-emerald-500 text-white p-6">
                  <h3 className="text-lg font-semibold mb-4">Simulation Result</h3>
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div className="bg-white/20 rounded-lg p-4">
                      <div className="text-green-100 text-sm">Before</div>
                      <div className="text-3xl font-bold">
                        {project.problem_type === 'regression'
                          ? simulationResult.original.predicted_value?.toFixed(2)
                          : `${(simulationResult.original.probability * 100).toFixed(1)}%`
                        }
                      </div>
                    </div>
                    <div className="bg-white/20 rounded-lg p-4">
                      <div className="text-green-100 text-sm">After</div>
                      <div className="text-3xl font-bold">
                        {project.problem_type === 'regression'
                          ? simulationResult.modified.predicted_value?.toFixed(2)
                          : `${(simulationResult.modified.probability * 100).toFixed(1)}%`
                        }
                      </div>
                    </div>
                    <div className="bg-white/20 rounded-lg p-4">
                      <div className="text-green-100 text-sm">Impact</div>
                      <div className={`text-3xl font-bold ${
                        simulationResult.impact < 0 ? 'text-green-200' : 'text-red-200'
                      }`}>
                        {simulationResult.impact > 0 ? '+' : ''}
                        {(simulationResult.impact * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* Impact Interpretation */}
                <div className="p-6">
                  <div className={`p-4 rounded-lg ${
                    simulationResult.impact < 0
                      ? 'bg-green-50 border border-green-200'
                      : simulationResult.impact > 0
                        ? 'bg-red-50 border border-red-200'
                        : 'bg-gray-50 border border-gray-200'
                  }`}>
                    <div className="flex items-center gap-3">
                      <div className="text-3xl">
                        {simulationResult.impact < 0 ? '✅' : simulationResult.impact > 0 ? '⚠️' : '➡️'}
                      </div>
                      <div>
                        <div className={`font-semibold ${
                          simulationResult.impact < 0 ? 'text-green-800' : 
                          simulationResult.impact > 0 ? 'text-red-800' : 'text-gray-800'
                        }`}>
                          {simulationResult.impact < -0.1
                            ? 'Significant positive impact!'
                            : simulationResult.impact < 0
                              ? 'Slight improvement'
                              : simulationResult.impact > 0.1
                                ? 'This change increases risk'
                                : simulationResult.impact > 0
                                  ? 'Slight increase in risk'
                                  : 'No significant change'
                          }
                        </div>
                        <div className={`text-sm ${
                          simulationResult.impact < 0 ? 'text-green-700' : 
                          simulationResult.impact > 0 ? 'text-red-700' : 'text-gray-700'
                        }`}>
                          {simulationResult.impact < 0
                            ? `These changes would reduce ${project.target_description || 'the outcome'} probability by ${Math.abs(simulationResult.impact * 100).toFixed(1)} percentage points.`
                            : simulationResult.impact > 0
                              ? `These changes would increase ${project.target_description || 'the outcome'} probability by ${(simulationResult.impact * 100).toFixed(1)} percentage points.`
                              : 'The changes have minimal effect on the prediction.'
                          }
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Changes Made */}
                  <div className="mt-4">
                    <h4 className="font-medium text-gray-900 mb-2">Changes Applied:</h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(modifiedValues)
                        .filter(([_, v]) => v !== undefined)
                        .map(([key, value]) => (
                          <span key={key} className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm">
                            {key}: {String(baseRow[key])} → {String(value)}
                          </span>
                        ))
                      }
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right: Simulation History */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <div className="bg-white rounded-xl shadow-sm border p-4">
              <h3 className="font-semibold text-gray-900 mb-3">Quick Actions</h3>
              <div className="space-y-2">
                <button
                  onClick={() => navigate(`/projects/${projectId}/explainability`)}
                  className="w-full px-3 py-2 text-left bg-brand-teal/10 text-brand-teal-dark rounded-lg hover:bg-brand-teal/20 text-sm"
                >
                  🔍 View SHAP Explanation
                </button>
                <button
                  onClick={() => navigate(`/projects/${projectId}`)}
                  className="w-full px-3 py-2 text-left bg-gray-50 text-gray-700 rounded-lg hover:bg-gray-100 text-sm"
                >
                  📊 Back to Project
                </button>
              </div>
            </div>

            {/* Simulation History */}
            {history.length > 0 && (
              <div className="bg-white rounded-xl shadow-sm border p-4">
                <h3 className="font-semibold text-gray-900 mb-3">
                  Simulation History ({history.length})
                </h3>
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {history.map((item, idx) => (
                    <button
                      key={idx}
                      onClick={() => applyHistoryItem(item)}
                      className="w-full p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors"
                    >
                      <div className="flex justify-between items-start mb-1">
                        <span className={`text-sm font-medium ${
                          item.result.impact < 0 ? 'text-green-700' : 'text-red-700'
                        }`}>
                          Impact: {item.result.impact > 0 ? '+' : ''}{(item.result.impact * 100).toFixed(1)}%
                        </span>
                        <span className="text-xs text-gray-500">
                          {new Date(item.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="text-xs text-gray-600">
                        {Object.keys(item.changes).filter(k => item.changes[k] !== undefined).length} changes
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Tips - Dynamic based on baseline prediction */}
            <div className="bg-purple-50 rounded-xl border border-purple-200 p-4">
              <h3 className="font-semibold text-purple-800 mb-2">💡 Suggestions for This Record</h3>
              {baselinePrediction?.explanations?.shap?.top_features ? (
                <ul className="text-sm text-purple-700 space-y-2">
                  {baselinePrediction.explanations.shap.top_features.slice(0, 4).map((f, i) => (
                    <li key={i}>
                      {f.impact > 0 ? (
                        <span>• <strong>{f.feature.replace(/_/g, ' ')}</strong>: Try {typeof f.value === 'number' ? 'decreasing' : 'changing'} this value to reduce {project.target_description || 'risk'}</span>
                      ) : (
                        <span>• <strong>{f.feature.replace(/_/g, ' ')}</strong>: This is helping — current value is favorable</span>
                      )}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-purple-700">Select a row to see personalized suggestions</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
