import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProjectDetail() {
  const { id } = useParams();
  const { token, user } = useAuth();
  const navigate = useNavigate();
  const [project, setProject] = useState(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState('');
  const [predicting, setPredicting] = useState(false);
  const [dataPreview, setDataPreview] = useState(null);
  const [selectedRow, setSelectedRow] = useState(null);
  const [selectedRowIdx, setSelectedRowIdx] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predictions, setPredictions] = useState({}); // Store predictions by row index
  const [showSimulation, setShowSimulation] = useState(false);
  const [simulationData, setSimulationData] = useState({});
  const [simulationResult, setSimulationResult] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState('');

  useEffect(() => {
    if (token && id) {
      fetchProject();
    }
  }, [id, token]);

  const fetchProject = async () => {
    try {
      const res = await fetch(`/api/projects/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setProject(data);
        if (data.dataset_id) {
          fetchDataPreview(data.dataset_id);
        }
      } else {
        console.error('Failed to fetch project:', res.status);
      }
    } catch (err) {
      console.error('Error fetching project:', err);
    }
    setLoading(false);
  };

  const [testData, setTestData] = useState(null);

  const fetchDataPreview = async (datasetId) => {
    try {
      const res = await fetch(`/api/datasets/${datasetId}/preview?rows=20`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        setDataPreview(await res.json());
      }
    } catch (err) {
      console.error('Error fetching data preview:', err);
    }
  };

  const fetchTestData = async () => {
    try {
      const res = await fetch(`/api/projects/${id}/test-data?limit=30`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setTestData(data);
      }
    } catch (err) {
      console.error('Error fetching test data:', err);
    }
  };

  // Fetch test data when project is trained
  useEffect(() => {
    if (project?.status === 'trained' || project?.status === 'ready') {
      fetchTestData();
    }
  }, [project?.status]);

  const handleTrain = async () => {
    setTraining(true);
    setTrainError('');
    try {
      const res = await fetch(`/api/projects/${id}/train`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        // Refresh project data to get updated status and model info
        await fetchProject();
      } else {
        const data = await res.json();
        setTrainError(data.detail || 'Training failed');
      }
    } catch (err) {
      console.error('Training error:', err);
      setTrainError('Training failed: ' + err.message);
    }
    setTraining(false);
  };

  const [predictError, setPredictError] = useState('');

  const handlePredict = async (row) => {
    setSelectedRow(row);
    setPredicting(true);
    setPrediction(null);
    setPredictError('');
    
    // Build features from selected row
    const features = {};
    project.feature_columns.forEach(col => {
      features[col] = row[col];
    });
    
    console.log('Predicting with features:', features);
    
    try {
      const res = await fetch(`/api/projects/${id}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ features })
      });
      
      const data = await res.json();
      console.log('Prediction response:', res.status, data);
      
      if (res.ok) {
        setPrediction(data);
      } else {
        setPredictError(data.detail || 'Prediction failed');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setPredictError('Network error: ' + err.message);
    }
    setPredicting(false);
  };

  // Predict a specific row and store result by index
  const handlePredictRow = async (row, idx) => {
    setSelectedRow(row);
    setSelectedRowIdx(idx);
    setPredicting(true);
    setPredictError('');
    
    const features = {};
    project.feature_columns.forEach(col => {
      features[col] = row[col];
    });
    
    try {
      const res = await fetch(`/api/projects/${id}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ features })
      });
      
      const data = await res.json();
      
      if (res.ok) {
        setPrediction(data);
        // Store prediction by row index for inline display
        setPredictions(prev => ({
          ...prev,
          [idx]: data.prediction
        }));
      } else {
        setPredictError(data.detail || 'Prediction failed');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setPredictError('Network error: ' + err.message);
    }
    setPredicting(false);
  };

  // Predict all rows at once
  const handlePredictAll = async () => {
    const rows = testData?.rows || dataPreview?.rows || [];
    setPredicting(true);
    
    for (let i = 0; i < Math.min(rows.length, 20); i++) {
      const row = rows[i];
      const features = {};
      project.feature_columns.forEach(col => {
        features[col] = row[col];
      });
      
      try {
        const res = await fetch(`/api/projects/${id}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`
          },
          body: JSON.stringify({ features })
        });
        
        if (res.ok) {
          const data = await res.json();
          setPredictions(prev => ({ ...prev, [i]: data.prediction }));
        }
      } catch (err) {
        console.error(`Prediction error for row ${i}:`, err);
      }
    }
    setPredicting(false);
  };

  const handleSimulate = async () => {
    const res = await fetch(`/api/projects/${id}/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify({
        base_features: Object.fromEntries(
          project.feature_columns.map(col => [col, selectedRow[col]])
        ),
        modified_features: simulationData
      })
    });
    
    if (res.ok) {
      setSimulationResult(await res.json());
    }
  };

  const handleDeleteProject = async () => {
    const confirmed = window.confirm(
      `Delete project \"${project.name}\"? This will deactivate the project and remove it from the projects list.`
    );
    if (!confirmed) return;

    setDeleting(true);
    setDeleteError('');

    try {
      const res = await fetch(`/api/projects/${id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });

      if (res.ok) {
        navigate('/projects');
        return;
      }

      const data = await res.json();
      setDeleteError(data.detail || 'Failed to delete project');
    } catch (err) {
      console.error('Delete project error:', err);
      setDeleteError('Network error while deleting project');
    } finally {
      setDeleting(false);
    }
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

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <button
          onClick={() => navigate('/projects')}
          className="text-gray-600 hover:text-brand-dark mb-2 flex items-center gap-1"
        >
          ← Back to Projects
        </button>
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-bold text-brand-dark">{project.name}</h1>
            {project.description && (
              <p className="text-gray-600 mt-1">{project.description}</p>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className={`px-3 py-1 rounded-full text-sm ${
              project.status === 'trained'
                ? 'bg-green-100 text-green-700'
                : project.status === 'training'
                ? 'bg-yellow-100 text-yellow-700'
                : 'bg-gray-100 text-gray-700'
            }`}>
              {project.status}
            </span>
            {(user?.role === 'owner' || user?.role === 'admin') && (
              <button
                onClick={handleDeleteProject}
                disabled={deleting}
                className="px-3 py-1 text-sm border border-red-200 text-red-600 rounded-lg hover:bg-red-50 disabled:opacity-50"
              >
                {deleting ? 'Deleting...' : 'Delete Project'}
              </button>
            )}
          </div>
        </div>
        {deleteError && (
          <div className="mt-3 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            {deleteError}
          </div>
        )}
      </div>

      {/* Project Info */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white rounded-xl p-4 shadow-sm border">
          <div className="text-gray-500 text-sm">Target Column</div>
          <div className="font-semibold text-brand-dark">{project.target_column}</div>
        </div>
        <div className="bg-white rounded-xl p-4 shadow-sm border">
          <div className="text-gray-500 text-sm">Features</div>
          <div className="font-semibold text-brand-dark">
            {project.feature_columns && project.feature_columns.length > 0 
              ? `${project.feature_columns.length} columns` 
              : 'Not configured'}
          </div>
        </div>
        <div className="bg-white rounded-xl p-4 shadow-sm border">
          <div className="text-gray-500 text-sm">Problem Type</div>
          <div className="font-semibold text-brand-dark">{project.problem_type?.replace('_', ' ')}</div>
        </div>
      </div>

      {/* Training Section */}
      {(project.status === 'created' || project.status === 'draft') && (
        <div className="bg-brand-teal/10 rounded-xl p-6 mb-6">
          <h2 className="text-lg font-semibold text-brand-dark mb-2">Ready to Train</h2>
          <p className="text-gray-600 mb-4">
            Your project is configured. Click the button below to train the ML model.
          </p>
          {trainError && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              {trainError}
            </div>
          )}
          <button
            onClick={handleTrain}
            disabled={training}
            className="px-6 py-3 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal disabled:opacity-50 flex items-center gap-2"
          >
            {training ? (
              <>
                <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>
                Training Model...
              </>
            ) : (
              <>
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Train Model
              </>
            )}
          </button>
        </div>
      )}

      {/* Model Performance (after training) */}
      {(project.status === 'trained' || project.status === 'ready') && project.active_model && (
        <div className="bg-green-50 rounded-xl p-6 mb-6">
          <h2 className="text-lg font-semibold text-brand-dark mb-4">Model Performance</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {project.problem_type === 'regression' ? (
              // Regression metrics
              [
                ['MAE', project.active_model.mae],
                ['MSE', project.active_model.mse],
                ['RMSE', project.active_model.rmse],
                ['R² Score', project.active_model.r2_score]
              ].filter(([k, v]) => v !== undefined && v !== null).map(([key, value]) => (
                <div key={key} className="bg-white rounded-lg p-3">
                  <div className="text-gray-500 text-sm">{key}</div>
                  <div className="text-2xl font-bold text-brand-dark">
                    {key === 'R² Score' ? (value * 100).toFixed(1) + '%' : value?.toFixed(4)}
                  </div>
                </div>
              ))
            ) : (
              // Classification metrics with color coding
              [
                ['Accuracy', project.active_model.accuracy],
                ['Precision', project.active_model.precision],
                ['Recall', project.active_model.recall],
                ['F1 Score', project.active_model.f1_score],
                ['AUC-ROC', project.active_model.auc_roc]
              ].filter(([k, v]) => v !== undefined && v !== null).map(([key, value]) => {
                const pct = Math.min(value, 1) * 100;
                const isLow = pct < 50;
                return (
                  <div key={key} className="bg-white rounded-lg p-3">
                    <div className="text-gray-500 text-sm">{key}</div>
                    <div className={`text-2xl font-bold ${isLow ? 'text-orange-600' : 'text-brand-dark'}`}>
                      {pct.toFixed(1)}%
                    </div>
                    {isLow && key !== 'AUC-ROC' && (
                      <div className="text-xs text-orange-500">Consider retraining</div>
                    )}
                  </div>
                );
              })
            )}
            <div className="bg-white rounded-lg p-3">
              <div className="text-gray-500 text-sm">Version</div>
              <div className="text-2xl font-bold text-brand-dark">{project.active_model.version}</div>
            </div>
          </div>
          
          {/* Low metrics warning */}
          {project.problem_type !== 'regression' && 
           project.active_model.recall !== undefined && 
           project.active_model.recall < 0.5 && (
            <div className="mt-4 p-3 bg-orange-100 border border-orange-300 rounded-lg text-orange-800 text-sm">
              ⚠️ <strong>Low Recall/F1:</strong> The model may be missing many positive cases. 
              Try retraining with more balanced data or adjust the threshold.
            </div>
          )}
          
          {/* Quick Action Buttons */}
          <div className="mt-4 pt-4 border-t border-green-200 flex flex-wrap gap-3">
            <button
              onClick={() => navigate(`/projects/${id}/explainability`)}
              className="px-4 py-2 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal font-medium flex items-center gap-2"
            >
              🔍 Explainability Analysis
            </button>
            <button
              onClick={() => navigate(`/projects/${id}/whatif`)}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-medium flex items-center gap-2"
            >
              🔮 What-If Simulation
            </button>
            <button
              onClick={handleTrain}
              disabled={training}
              className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 font-medium flex items-center gap-2 disabled:opacity-50"
            >
              🔄 Retrain Model
            </button>
          </div>
        </div>
      )}

      {/* Test Data Predictions Section - shows after training */}
      {(project.status === 'trained' || project.status === 'ready') && (testData || dataPreview) && (
        <div className="bg-white rounded-xl shadow-sm border mb-6">
          <div className="p-4 border-b">
            <h2 className="text-lg font-semibold text-brand-dark">
              Test Data Preview
            </h2>
            <p className="text-gray-600 text-sm">
              Sample of held-out test data. Click on a row to see prediction details.
            </p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left text-gray-600">#</th>
                  <th className="px-3 py-2 text-left text-gray-600">Actual</th>
                  {(testData ? Object.keys(testData.rows?.[0] || {}) : dataPreview?.columns || [])
                    .filter(col => col !== project.target_column)
                    .slice(0, 5)
                    .map(col => (
                      <th key={col} className="px-3 py-2 text-left text-gray-600 truncate max-w-24">
                        {col}
                      </th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {(testData?.rows || dataPreview?.rows || dataPreview?.data || []).slice(0, 15).map((row, idx) => {
                  const actual = String(row[project.target_column]) === project.target_positive_label;
                  
                  return (
                    <tr key={idx} className="border-t hover:bg-gray-50">
                      <td className="px-3 py-2 text-gray-500">{idx + 1}</td>
                      <td className="px-3 py-2">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          actual ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                        }`}>
                          {actual ? 'Yes' : 'No'}
                        </span>
                      </td>
                      {(testData ? Object.keys(testData.rows?.[0] || {}) : dataPreview?.columns || [])
                        .filter(col => col !== project.target_column)
                        .slice(0, 6)
                        .map(col => (
                          <td key={col} className="px-3 py-2 truncate max-w-24 text-gray-600">
                            {String(row[col] ?? '')}
                          </td>
                        ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
}
