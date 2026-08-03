import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function AnalyticsSaaS() {
  const { token } = useAuth();
  const [projects, setProjects] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedProject, setSelectedProject] = useState(null);
  const [projectDetails, setProjectDetails] = useState(null);

  useEffect(() => {
    if (token) {
      fetchProjects();
    }
  }, [token]);

  const fetchProjects = async () => {
    try {
      const res = await fetch('/api/projects', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setProjects(data);
        // Auto-select first trained project (check both 'trained' and 'ready' status)
        const trainedProject = data.find(p => p.status === 'trained' || p.status === 'ready');
        if (trainedProject) {
          setSelectedProject(trainedProject.id);
          fetchProjectDetails(trainedProject.id);
          fetchPredictions(trainedProject.id);
        }
      }
    } catch (err) {
      console.error('Error fetching projects:', err);
    }
    setLoading(false);
  };

  const fetchProjectDetails = async (projectId) => {
    try {
      const res = await fetch(`/api/projects/${projectId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        setProjectDetails(await res.json());
      }
    } catch (err) {
      console.error('Error fetching project details:', err);
    }
  };

  const fetchPredictions = async (projectId) => {
    try {
      const res = await fetch(`/api/projects/${projectId}/predictions?limit=100`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        setPredictions(await res.json());
      }
    } catch (err) {
      console.error('Error fetching predictions:', err);
    }
  };

  const handleProjectChange = (projectId) => {
    setSelectedProject(projectId);
    setProjectDetails(null);
    if (projectId) {
      fetchProjectDetails(projectId);
      fetchPredictions(projectId);
    } else {
      setPredictions([]);
    }
  };

  // Calculate stats - check for both 'trained' and 'ready' status
  const trainedProjects = projects.filter(p => p.status === 'trained' || p.status === 'ready');
  const totalPredictions = predictions.length;
  const highRiskCount = predictions.filter(p => p.risk_level === 'high' || p.risk_level === 'critical').length;
  const avgProbability = predictions.length > 0 
    ? predictions.reduce((sum, p) => sum + p.probability, 0) / predictions.length 
    : 0;

  // Risk distribution
  const riskDistribution = {
    critical: predictions.filter(p => p.risk_level === 'critical').length,
    high: predictions.filter(p => p.risk_level === 'high').length,
    medium: predictions.filter(p => p.risk_level === 'medium').length,
    low: predictions.filter(p => p.risk_level === 'low').length,
  };

  if (loading) {
    return (
      <div className="p-6 flex justify-center">
        <div className="animate-spin h-8 w-8 border-4 border-brand-teal-dark border-t-transparent rounded-full"></div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-brand-dark">Analytics</h1>
        <p className="text-gray-600">Overview of your prediction models and insights</p>
      </div>

      {trainedProjects.length === 0 ? (
        <div className="bg-brand-teal/10 rounded-xl p-8 text-center">
          <h3 className="text-lg font-medium text-brand-dark mb-2">No Trained Models Yet</h3>
          <p className="text-gray-600 mb-4">
            Train a model first to see analytics and insights.
          </p>
          <Link
            to="/projects"
            className="px-4 py-2 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal inline-block"
          >
            Go to Projects
          </Link>
        </div>
      ) : (
        <>
          {/* Project Selector */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">Select Project</label>
            <select
              value={selectedProject || ''}
              onChange={(e) => handleProjectChange(e.target.value)}
              className="px-4 py-2 border rounded-lg bg-white min-w-64"
            >
              {trainedProjects.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-xl p-6 shadow-sm border-l-4 border-l-brand-teal-dark">
              <div className="text-gray-500 text-sm">Trained Models</div>
              <div className="text-3xl font-bold text-brand-dark">{trainedProjects.length}</div>
            </div>
            <div className="bg-white rounded-xl p-6 shadow-sm border-l-4 border-l-brand-teal">
              <div className="text-gray-500 text-sm">Total Predictions</div>
              <div className="text-3xl font-bold text-brand-teal-dark">{totalPredictions}</div>
            </div>
            <div className="bg-white rounded-xl p-6 shadow-sm border-l-4 border-l-red-500">
              <div className="text-gray-500 text-sm">High Risk Cases</div>
              <div className="text-3xl font-bold text-red-600">{highRiskCount}</div>
            </div>
            <div className="bg-white rounded-xl p-6 shadow-sm border-l-4 border-l-brand-teal group relative">
              <div className="text-gray-500 text-sm flex items-center gap-1">
                Avg Probability
                <span className="cursor-help text-gray-400" title="Average prediction probability across all analyzed records. Higher values indicate the model is predicting more positive cases (e.g., more likely to churn).">ⓘ</span>
              </div>
              <div className="text-3xl font-bold text-brand-teal-dark">{(avgProbability * 100).toFixed(1)}%</div>
              <div className="text-xs text-gray-400 mt-1">Across all predictions</div>
            </div>
          </div>

          {/* Risk Distribution */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="bg-white rounded-xl p-6 shadow-sm border">
              <h3 className="text-lg font-semibold text-brand-dark mb-4">Risk Distribution</h3>
              {totalPredictions === 0 ? (
                <p className="text-gray-500">No predictions yet. Make some predictions to see distribution.</p>
              ) : (
                <div className="space-y-3">
                  {[
                    { level: 'Critical', count: riskDistribution.critical, color: 'bg-red-600' },
                    { level: 'High', count: riskDistribution.high, color: 'bg-brand-teal-dark' },
                    { level: 'Medium', count: riskDistribution.medium, color: 'bg-brand-teal' },
                    { level: 'Low', count: riskDistribution.low, color: 'bg-green-500' },
                  ].map(({ level, count, color }) => (
                    <div key={level} className="flex items-center gap-3">
                      <span className="w-16 text-sm text-gray-600">{level}</span>
                      <div className="flex-1 h-6 bg-brand-light rounded-full overflow-hidden">
                        <div
                          className={`h-full ${color} transition-all`}
                          style={{ width: `${totalPredictions > 0 ? (count / totalPredictions) * 100 : 0}%` }}
                        />
                      </div>
                      <span className="w-12 text-sm text-brand-dark text-right">{count}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Model Performance */}
            {selectedProject && (
              <div className="bg-white rounded-xl p-6 shadow-sm border">
                <h3 className="text-lg font-semibold text-brand-dark mb-4">Model Performance</h3>
                {(() => {
                  const project = projects.find(p => p.id === selectedProject);
                  if (!project) return <p className="text-gray-500">Select a project</p>;
                  
                  // Use projectDetails.active_model for metrics
                  const model = projectDetails?.active_model;
                  const isRegression = projectDetails?.problem_type === 'regression';
                  
                  return (
                    <div className="grid grid-cols-2 gap-4">
                      {isRegression ? (
                        <>
                          <div className="bg-green-50 rounded-lg p-4 text-center">
                            <div className="text-2xl font-bold text-green-700">
                              {model?.mae ? model.mae.toFixed(4) : '--'}
                            </div>
                            <div className="text-sm text-green-600">MAE</div>
                          </div>
                          <div className="bg-brand-teal/10 rounded-lg p-4 text-center">
                            <div className="text-2xl font-bold text-brand-teal-dark">
                              {model?.r2_score ? (model.r2_score * 100).toFixed(1) : '--'}%
                            </div>
                            <div className="text-sm text-brand-teal-dark">R² Score</div>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="bg-green-50 rounded-lg p-4 text-center">
                            <div className="text-2xl font-bold text-green-700">
                              {model?.accuracy ? (model.accuracy * 100).toFixed(1) : '--'}%
                            </div>
                            <div className="text-sm text-green-600">Accuracy</div>
                          </div>
                          <div className="bg-brand-teal/10 rounded-lg p-4 text-center">
                            <div className="text-2xl font-bold text-brand-teal-dark">
                              {model?.f1_score ? (model.f1_score * 100).toFixed(1) : '--'}%
                            </div>
                            <div className="text-sm text-brand-teal-dark">F1 Score</div>
                          </div>
                        </>
                      )}
                      <div className="bg-brand-teal/10 rounded-lg p-4 text-center">
                        <div className="text-2xl font-bold text-brand-teal-dark">
                          {projectDetails?.feature_columns?.length ?? project.feature_count ?? '--'}
                        </div>
                        <div className="text-sm text-brand-teal-dark">Features</div>
                      </div>
                      <div className="bg-brand-light rounded-lg p-4 text-center">
                        <div className="text-2xl font-bold text-brand-dark text-base">
                          {project.target_column || '--'}
                        </div>
                        <div className="text-sm text-gray-600">Target</div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>

          {/* Feature Importance & Model Insights */}
          {projectDetails?.active_model && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              {/* Feature Importance */}
              <div className="bg-white rounded-xl p-6 shadow-sm border">
                <h3 className="text-lg font-semibold text-brand-dark mb-4">Feature Importance</h3>
                {projectDetails.active_model.feature_importance && Object.keys(projectDetails.active_model.feature_importance).length > 0 ? (
                  <div className="space-y-3">
                    {Object.entries(projectDetails.active_model.feature_importance)
                      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                      .slice(0, 8)
                      .map(([feature, importance]) => (
                        <div key={feature} className="flex items-center gap-3">
                          <span className="w-32 text-sm text-gray-600 truncate" title={feature}>{feature}</span>
                          <div className="flex-1 h-4 bg-brand-light rounded-full overflow-hidden">
                            <div
                              className={`h-full ${importance > 0 ? 'bg-brand-teal-dark' : 'bg-brand-teal'}`}
                              style={{ width: `${Math.min(Math.abs(importance) * 100, 100)}%` }}
                            />
                          </div>
                          <span className="w-16 text-sm text-brand-dark text-right">
                            {(Math.abs(importance) * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                  </div>
                ) : (
                  <p className="text-gray-500">Feature importance not available for this model.</p>
                )}
              </div>

              {/* Quick Insights */}
              <div className="bg-white rounded-xl p-6 shadow-sm border">
                <h3 className="text-lg font-semibold text-brand-dark mb-4">Model Insights</h3>
                <div className="space-y-3">
                  {/* Top driver */}
                  {projectDetails.active_model.feature_importance && (
                    <div className="p-3 bg-brand-teal/10 rounded-lg">
                      <div className="text-sm font-medium text-blue-800">🎯 Top Predictor</div>
                      <div className="text-brand-teal-dark">
                        {Object.entries(projectDetails.active_model.feature_importance)
                          .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))[0]?.[0] || 'N/A'}
                      </div>
                    </div>
                  )}
                  
                  {/* Model type */}
                  <div className="p-3 bg-brand-teal/10 rounded-lg">
                    <div className="text-sm font-medium text-brand-dark">🤖 Model Type</div>
                    <div className="text-brand-teal-dark">Ensemble (XGBoost + LightGBM + RF + LR)</div>
                  </div>
                  
                  {/* Problem type */}
                  <div className="p-3 bg-brand-teal/10 rounded-lg">
                    <div className="text-sm font-medium text-brand-dark">📊 Problem Type</div>
                    <div className="text-brand-teal-dark capitalize">
                      {projectDetails.problem_type?.replace('_', ' ') || 'Classification'}
                    </div>
                  </div>

                  {/* Training date */}
                  <div className="p-3 bg-brand-light rounded-lg">
                    <div className="text-sm font-medium text-brand-dark">📅 Trained On</div>
                    <div className="text-gray-700">
                      {projectDetails.active_model.trained_at 
                        ? new Date(projectDetails.active_model.trained_at).toLocaleDateString()
                        : 'N/A'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Prediction Accuracy Summary - if we have predictions */}
          {predictions.length > 0 && (
            <div className="bg-white rounded-xl p-6 shadow-sm border mb-6">
              <h3 className="text-lg font-semibold text-brand-dark mb-4">Prediction Distribution</h3>
              <div className="grid grid-cols-5 gap-2">
                {[
                  { range: '0-20%', color: 'bg-green-500' },
                  { range: '20-40%', color: 'bg-brand-teal' },
                  { range: '40-60%', color: 'bg-brand-teal-dark' },
                  { range: '60-80%', color: 'bg-brand-dark' },
                  { range: '80-100%', color: 'bg-red-500' },
                ].map(({ range, color }, i) => {
                  const low = i * 0.2;
                  const high = (i + 1) * 0.2;
                  const count = predictions.filter(p => p.probability >= low && p.probability < high).length;
                  const pct = predictions.length > 0 ? (count / predictions.length) * 100 : 0;
                  
                  return (
                    <div key={range} className="text-center">
                      <div className={`h-24 ${color} rounded-t-lg flex items-end justify-center relative`}
                           style={{ opacity: 0.3 + (pct / 100) * 0.7 }}>
                        <div className={`${color} w-full rounded-t-lg absolute bottom-0`} 
                             style={{ height: `${Math.max(pct, 5)}%` }} />
                      </div>
                      <div className="text-xs text-gray-600 mt-1">{range}</div>
                      <div className="text-sm font-medium">{count}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Recent Predictions */}
          {predictions.length > 0 && (
            <div className="bg-white rounded-xl shadow-sm border">
              <div className="p-4 border-b">
                <h3 className="text-lg font-semibold text-brand-dark">Recent Predictions</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-2 text-left text-gray-600">Date</th>
                      <th className="px-4 py-2 text-left text-gray-600">Probability</th>
                      <th className="px-4 py-2 text-left text-gray-600">Risk Level</th>
                      <th className="px-4 py-2 text-left text-gray-600">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictions.slice(0, 10).map((pred, idx) => (
                      <tr key={pred.id || idx} className="border-t">
                        <td className="px-4 py-2 text-gray-600">
                          {new Date(pred.created_at).toLocaleDateString()}
                        </td>
                        <td className="px-4 py-2 font-medium">
                          {(pred.probability * 100).toFixed(1)}%
                        </td>
                        <td className="px-4 py-2">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            pred.risk_level === 'critical' ? 'bg-red-100 text-red-700' :
                            pred.risk_level === 'high' ? 'bg-orange-100 text-orange-700' :
                            pred.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                            'bg-green-100 text-green-700'
                          }`}>
                            {pred.risk_level}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-gray-600">
                          {(pred.confidence * 100).toFixed(0)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
