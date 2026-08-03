import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Projects() {
  const { token, user } = useAuth();
  const [projects, setProjects] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [showCreate, setShowCreate] = useState(false);
  const [newProject, setNewProject] = useState({
    name: '',
    description: '',
    dataset_id: '',
    target_column: '',
    feature_columns: [],
    problem_type: 'binary_classification'
  });
  const [selectedDataset, setSelectedDataset] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (token) {
      fetchProjects();
      fetchDatasets();
    }
  }, [token]);

  const fetchProjects = async () => {
    try {
      const res = await fetch('/api/projects', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) setProjects(await res.json());
    } catch (err) {
      console.error('Error fetching projects:', err);
    }
  };

  const fetchDatasets = async () => {
    try {
      const res = await fetch('/api/datasets', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) setDatasets(await res.json());
    } catch (err) {
      console.error('Error fetching datasets:', err);
    }
  };

  const handleDatasetSelect = async (datasetId) => {
    setNewProject(prev => ({ ...prev, dataset_id: datasetId, target_column: '', feature_columns: [] }));
    const dataset = datasets.find(d => d.id === datasetId);
    if (dataset) {
      // Fetch columns
      const res = await fetch(`/api/datasets/${datasetId}/preview?rows=0`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setSelectedDataset({ ...dataset, columns: data.columns });
      }
    }
  };

  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');
  const [deleteError, setDeleteError] = useState('');
  const [deletingProjectId, setDeletingProjectId] = useState(null);

  const handleCreateProject = async (e) => {
    e.preventDefault();
    setCreating(true);
    setError('');
    
    try {
      console.log('Creating project with:', newProject);
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify(newProject)
      });
      
      const data = await res.json();
      console.log('Response:', res.status, data);
      
      if (res.ok) {
        navigate(`/projects/${data.id}`);
      } else {
        setError(data.detail || 'Failed to create project');
      }
    } catch (err) {
      console.error('Create project error:', err);
      setError('Network error. Please try again.');
    } finally {
      setCreating(false);
    }
  };

  const toggleFeature = (col) => {
    setNewProject(prev => ({
      ...prev,
      feature_columns: prev.feature_columns.includes(col)
        ? prev.feature_columns.filter(c => c !== col)
        : [...prev.feature_columns, col]
    }));
  };

  const selectAllFeatures = () => {
    if (selectedDataset) {
      const cols = selectedDataset.columns.filter(c => c !== newProject.target_column);
      setNewProject(prev => ({ ...prev, feature_columns: cols }));
    }
  };

  const handleDeleteProject = async (projectId, projectName) => {
    const confirmed = window.confirm(
      `Delete project \"${projectName}\"? This will deactivate the project and hide it from your list.`
    );
    if (!confirmed) return;

    setDeleteError('');
    setDeletingProjectId(projectId);

    try {
      const res = await fetch(`/api/projects/${projectId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });

      if (res.ok) {
        setProjects(prev => prev.filter(project => project.id !== projectId));
      } else {
        const data = await res.json();
        setDeleteError(data.detail || 'Failed to delete project');
      }
    } catch (err) {
      console.error('Delete project error:', err);
      setDeleteError('Network error while deleting project');
    } finally {
      setDeletingProjectId(null);
    }
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-brand-dark">Projects</h1>
          <p className="text-gray-600">Create and manage prediction models</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="px-4 py-2 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          New Project
        </button>
      </div>

      {deleteError && (
        <div className="mb-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {deleteError}
        </div>
      )}

      {/* Create Modal */}
      {showCreate && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 overflow-y-auto py-8">
          <div className="bg-white rounded-xl p-6 w-full max-w-2xl mx-4">
            <h2 className="text-xl font-bold mb-4">Create New Project</h2>
            
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
                {error}
              </div>
            )}
            
            <form onSubmit={handleCreateProject} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Project Name</label>
                <input
                  type="text"
                  value={newProject.name}
                  onChange={(e) => setNewProject(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-brand-teal"
                  placeholder="Customer Churn Prediction"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Description (optional)</label>
                <textarea
                  value={newProject.description}
                  onChange={(e) => setNewProject(prev => ({ ...prev, description: e.target.value }))}
                  className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-brand-teal"
                  placeholder="Predict which customers are likely to churn..."
                  rows={2}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Select Dataset</label>
                {datasets.length === 0 ? (
                  <div className="text-gray-500 p-4 bg-gray-50 rounded-lg">
                    No datasets uploaded yet.{' '}
                    <Link to="/datasets" className="text-brand-teal-dark hover:underline">Upload a dataset first</Link>
                  </div>
                ) : (
                  <select
                    value={newProject.dataset_id}
                    onChange={(e) => handleDatasetSelect(e.target.value)}
                    className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-brand-teal"
                    required
                  >
                    <option value="">Choose a dataset...</option>
                    {datasets.map(ds => (
                      <option key={ds.id} value={ds.id}>{ds.name} ({ds.row_count} rows)</option>
                    ))}
                  </select>
                )}
              </div>

              {selectedDataset && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Problem Type
                    </label>
                    <div className="grid grid-cols-2 gap-3">
                      <label className={`flex items-center gap-3 p-3 border rounded-lg cursor-pointer ${
                        newProject.problem_type === 'binary_classification' ? 'border-brand-teal-dark bg-brand-teal/10' : 'hover:bg-gray-50'
                      }`}>
                        <input
                          type="radio"
                          name="problem_type"
                          value="binary_classification"
                          checked={newProject.problem_type === 'binary_classification'}
                          onChange={(e) => setNewProject(prev => ({ ...prev, problem_type: e.target.value }))}
                          className="text-brand-teal-dark"
                        />
                        <div>
                          <div className="font-medium text-brand-dark">Classification</div>
                          <div className="text-xs text-gray-500">Predict categories (Yes/No, High/Low)</div>
                        </div>
                      </label>
                      <label className={`flex items-center gap-3 p-3 border rounded-lg cursor-pointer ${
                        newProject.problem_type === 'regression' ? 'border-brand-teal-dark bg-brand-teal/10' : 'hover:bg-gray-50'
                      }`}>
                        <input
                          type="radio"
                          name="problem_type"
                          value="regression"
                          checked={newProject.problem_type === 'regression'}
                          onChange={(e) => setNewProject(prev => ({ ...prev, problem_type: e.target.value }))}
                          className="text-brand-teal-dark"
                        />
                        <div>
                          <div className="font-medium text-brand-dark">Regression</div>
                          <div className="text-xs text-gray-500">Predict numbers (price, quantity)</div>
                        </div>
                      </label>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Target Column <span className="text-gray-500">(what to predict)</span>
                    </label>
                    <select
                      value={newProject.target_column}
                      onChange={(e) => {
                        const target = e.target.value;
                        setNewProject(prev => ({
                          ...prev,
                          target_column: target,
                          feature_columns: prev.feature_columns.filter(c => c !== target)
                        }));
                      }}
                      className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-brand-teal"
                      required
                    >
                      <option value="">Select target column...</option>
                      {selectedDataset.columns.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <label className="block text-sm font-medium text-gray-700">
                        Feature Columns <span className="text-gray-500">(inputs for prediction)</span>
                      </label>
                      <button
                        type="button"
                        onClick={selectAllFeatures}
                        className="text-sm text-brand-teal-dark hover:text-blue-700"
                      >
                        Select All
                      </button>
                    </div>
                    <div className="border rounded-lg p-3 max-h-48 overflow-y-auto">
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                        {selectedDataset.columns
                          .filter(col => col !== newProject.target_column)
                          .map(col => (
                            <label key={col} className="flex items-center gap-2 text-sm">
                              <input
                                type="checkbox"
                                checked={newProject.feature_columns.includes(col)}
                                onChange={() => toggleFeature(col)}
                                className="rounded"
                              />
                              <span className="truncate">{col}</span>
                            </label>
                          ))}
                      </div>
                    </div>
                    <p className="text-sm text-gray-500 mt-1">
                      {newProject.feature_columns.length} features selected
                    </p>
                  </div>
                </>
              )}

              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => {
                    setShowCreate(false);
                    setSelectedDataset(null);
                    setError('');
                    setNewProject({
                      name: '',
                      description: '',
                      dataset_id: '',
                      target_column: '',
                      feature_columns: [],
                      problem_type: 'binary_classification'
                    });
                  }}
                  className="flex-1 px-4 py-2 border rounded-lg hover:bg-gray-50"
                  disabled={creating}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={creating || !newProject.name || !newProject.dataset_id || !newProject.target_column || newProject.feature_columns.length === 0}
                  className="flex-1 px-4 py-2 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal disabled:opacity-50"
                >
                  {creating ? 'Creating...' : 'Create Project'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Project List */}
      {projects.length === 0 ? (
        <div className="bg-gray-50 rounded-xl p-12 text-center">
          <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
          <h3 className="text-lg font-medium text-brand-dark mb-2">No projects yet</h3>
          <p className="text-gray-600 mb-4">Create a project to start making predictions</p>
          {datasets.length > 0 ? (
            <button
              onClick={() => setShowCreate(true)}
              className="px-4 py-2 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal"
            >
              Create Your First Project
            </button>
          ) : (
            <Link
              to="/datasets"
              className="px-4 py-2 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal inline-block"
            >
              Upload a Dataset First
            </Link>
          )}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {projects.map((project) => (
            <div
              key={project.id}
              className="bg-white rounded-xl p-6 shadow-sm border hover:shadow-md transition cursor-pointer"
              onClick={() => navigate(`/projects/${project.id}`)}
            >
              <div className="flex justify-between items-start mb-3">
                <h3 className="text-lg font-semibold text-brand-dark">{project.name}</h3>
                <div className="flex items-center gap-2">
                  <span className={`px-3 py-1 rounded-full text-sm ${
                    project.status === 'trained' || project.status === 'ready'
                      ? 'bg-brand-teal/20 text-brand-teal-dark'
                      : project.status === 'training'
                      ? 'bg-brand-teal/10 text-brand-teal-dark'
                      : project.status === 'error'
                      ? 'bg-red-100 text-red-700'
                      : 'bg-brand-light text-brand-dark'
                  }`}>
                    {project.status}
                  </span>
                  {(user?.role === 'owner' || user?.role === 'admin') && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteProject(project.id, project.name);
                      }}
                      disabled={deletingProjectId === project.id}
                      className="px-2 py-1 text-xs border border-red-200 text-red-600 rounded hover:bg-red-50 disabled:opacity-50"
                    >
                      {deletingProjectId === project.id ? 'Deleting...' : 'Delete'}
                    </button>
                  )}
                </div>
              </div>
              {project.description && (
                <p className="text-gray-600 text-sm mb-3">{project.description}</p>
              )}
              <div className="flex gap-4 text-sm text-gray-500">
                <span>Target: <strong className="text-brand-dark">{project.target_column}</strong></span>
                <span>{project.feature_count || 0} features</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
