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
    problem_type: 'binary_classification',
    target_positive_label: '',
  });
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [targetLabels, setTargetLabels] = useState([]);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');
  const [deleteError, setDeleteError] = useState('');
  const [deletingProjectId, setDeletingProjectId] = useState(null);
  const navigate = useNavigate();

  const preferredPositive = [
    'yes', 'y', 'true', 't', '1', 'positive', 'pos', 'high',
    'churn', 'churned', 'attrition', 'left', 'exited',
  ];

  const pickPositiveLabel = (values) => {
    const byLower = Object.fromEntries(values.map((v) => [String(v).toLowerCase(), String(v)]));
    for (const key of preferredPositive) {
      if (byLower[key]) return byLower[key];
    }
    return values[0] || '';
  };

  useEffect(() => {
    if (token) {
      fetchProjects();
      fetchDatasets();
    }
  }, [token]);

  const fetchProjects = async () => {
    try {
      const res = await fetch('/api/projects', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setProjects(await res.json());
    } catch (err) {
      console.error(err);
    }
  };

  const fetchDatasets = async () => {
    try {
      const res = await fetch('/api/datasets', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) setDatasets(await res.json());
    } catch (err) {
      console.error(err);
    }
  };

  const handleDatasetSelect = async (datasetId) => {
    setNewProject((prev) => ({
      ...prev,
      dataset_id: datasetId,
      target_column: '',
      feature_columns: [],
      target_positive_label: '',
    }));
    setTargetLabels([]);
    const dataset = datasets.find((d) => d.id === datasetId);
    if (dataset) {
      const res = await fetch(`/api/datasets/${datasetId}/preview?rows=0`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setSelectedDataset({ ...dataset, columns: data.columns });
      }
    }
  };

  const handleTargetSelect = async (targetColumn) => {
    setNewProject((prev) => ({
      ...prev,
      target_column: targetColumn,
      feature_columns: prev.feature_columns.filter((c) => c !== targetColumn),
      target_positive_label: '',
    }));
    setTargetLabels([]);
    if (!targetColumn || !newProject.dataset_id) return;
    if (newProject.problem_type === 'regression') return;
    try {
      const res = await fetch(
        `/api/datasets/${newProject.dataset_id}/columns/${encodeURIComponent(targetColumn)}`,
        { headers: { Authorization: `Bearer ${token}` } },
      );
      if (!res.ok) return;
      const stats = await res.json();
      let values = [];
      if (stats.value_counts) {
        values = Object.keys(stats.value_counts);
      } else if (Array.isArray(stats.categories)) {
        values = stats.categories.map(String);
      }
      values = values.map(String).sort();
      setTargetLabels(values);
      setNewProject((prev) => ({
        ...prev,
        target_positive_label: pickPositiveLabel(values),
      }));
    } catch (err) {
      console.error(err);
    }
  };

  const handleCreateProject = async (e) => {
    e.preventDefault();
    setCreating(true);
    setError('');
    try {
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(newProject),
      });
      const data = await res.json();
      if (res.ok) navigate(`/projects/${data.id}`);
      else setError(data.detail || 'Failed to create project');
    } catch {
      setError('Network error. Please try again.');
    } finally {
      setCreating(false);
    }
  };

  const toggleFeature = (col) => {
    setNewProject((prev) => ({
      ...prev,
      feature_columns: prev.feature_columns.includes(col)
        ? prev.feature_columns.filter((c) => c !== col)
        : [...prev.feature_columns, col],
    }));
  };

  const selectAllFeatures = () => {
    if (selectedDataset) {
      const cols = selectedDataset.columns.filter((c) => c !== newProject.target_column);
      setNewProject((prev) => ({ ...prev, feature_columns: cols }));
    }
  };

  const handleDeleteProject = async (projectId, projectName) => {
    if (!window.confirm(`Delete project "${projectName}"?`)) return;
    setDeletingProjectId(projectId);
    setDeleteError('');
    try {
      const res = await fetch(`/api/projects/${projectId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) fetchProjects();
      else {
        const data = await res.json();
        setDeleteError(data.detail || 'Failed to delete');
      }
    } catch {
      setDeleteError('Network error while deleting project');
    } finally {
      setDeletingProjectId(null);
    }
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Workspace</p>
          <h1 className="page-title">Projects</h1>
          <p className="page-sub">
            Connect a dataset, prepare guidance, then review cases and choose next steps.
          </p>
        </div>
        <button type="button" onClick={() => setShowCreate(true)} className="btn-primary">
          New project
        </button>
      </div>

      {deleteError && (
        <div className="mb-4 text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control">
          {deleteError}
        </div>
      )}

      {showCreate && (
        <div className="fixed inset-0 z-50 bg-ink/40 flex items-start justify-center overflow-y-auto py-10 px-4">
          <div className="bg-surface border border-mist rounded-control w-full max-w-2xl p-6 animate-page-in">
            <h2 className="font-display text-xl font-semibold text-ink mb-1">Create project</h2>
            <p className="text-sm text-[var(--muted)] mb-4">
              Name it, pick your data, and tell us what you want to predict.
            </p>
            {error && (
              <div className="mb-4 text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control">
                {error}
              </div>
            )}
            <form onSubmit={handleCreateProject} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Project name</label>
                <input
                  className="input"
                  value={newProject.name}
                  onChange={(e) => setNewProject((p) => ({ ...p, name: e.target.value }))}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Description</label>
                <textarea
                  className="input"
                  rows={2}
                  value={newProject.description}
                  onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Dataset</label>
                {datasets.length === 0 ? (
                  <p className="text-sm text-[var(--muted)]">
                    No datasets yet.{' '}
                    <Link to="/datasets" className="text-teal font-medium">
                      Upload one
                    </Link>
                  </p>
                ) : (
                  <select
                    className="input"
                    value={newProject.dataset_id}
                    onChange={(e) => handleDatasetSelect(e.target.value)}
                    required
                  >
                    <option value="">Choose…</option>
                    {datasets.map((ds) => (
                      <option key={ds.id} value={ds.id}>
                        {ds.name} ({ds.row_count} rows)
                      </option>
                    ))}
                  </select>
                )}
              </div>
              {selectedDataset && (
                <>
                  <div>
                    <label className="block text-sm font-medium mb-1">What kind of question?</label>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        ['binary_classification', 'Will it happen?', 'Yes / No risk (churn, attrition…)'],
                        ['regression', 'How much?', 'A number (spend, score…)'],
                      ].map(([val, label, hint]) => (
                        <label
                          key={val}
                          className={`border rounded-control p-3 text-sm cursor-pointer ${
                            newProject.problem_type === val
                              ? 'border-teal bg-teal-soft/30'
                              : 'border-mist'
                          }`}
                        >
                          <input
                            type="radio"
                            className="sr-only"
                            name="problem_type"
                            value={val}
                            checked={newProject.problem_type === val}
                            onChange={(e) => {
                              const problem_type = e.target.value;
                              setNewProject((p) => ({ ...p, problem_type }));
                              if (problem_type === 'binary_classification' && newProject.target_column) {
                                handleTargetSelect(newProject.target_column);
                              } else if (problem_type === 'regression') {
                                setTargetLabels([]);
                                setNewProject((p) => ({
                                  ...p,
                                  problem_type,
                                  target_positive_label: '',
                                }));
                              }
                            }}
                          />
                          <span className="font-medium text-ink block">{label}</span>
                          <span className="text-xs text-[var(--muted)] mt-0.5 block">{hint}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">Target column</label>
                    <select
                      className="input"
                      value={newProject.target_column}
                      onChange={(e) => handleTargetSelect(e.target.value)}
                      required
                    >
                      <option value="">Choose…</option>
                      {selectedDataset.columns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>
                  {newProject.problem_type === 'binary_classification' &&
                    newProject.target_column && (
                      <div>
                        <label className="block text-sm font-medium mb-1">
                          Positive label
                        </label>
                        <p className="text-xs text-[var(--muted)] mb-1">
                          Which target value means the outcome you want to predict (e.g. attrition = Yes)?
                        </p>
                        {targetLabels.length > 0 ? (
                          <select
                            className="input"
                            value={newProject.target_positive_label}
                            onChange={(e) =>
                              setNewProject((p) => ({
                                ...p,
                                target_positive_label: e.target.value,
                              }))
                            }
                            required
                          >
                            {targetLabels.map((v) => (
                              <option key={v} value={v}>
                                {v}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            className="input"
                            value={newProject.target_positive_label}
                            onChange={(e) =>
                              setNewProject((p) => ({
                                ...p,
                                target_positive_label: e.target.value,
                              }))
                            }
                            placeholder="e.g. Yes"
                            required
                          />
                        )}
                      </div>
                    )}
                  {newProject.target_column && (
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <label className="text-sm font-medium">What signals to use</label>
                        <button type="button" className="text-xs text-teal font-medium" onClick={selectAllFeatures}>
                          Select all
                        </button>
                      </div>
                      <div className="max-h-40 overflow-y-auto border border-mist rounded-control p-2 space-y-1">
                        {selectedDataset.columns
                          .filter((c) => c !== newProject.target_column)
                          .map((c) => (
                            <label key={c} className="flex items-center gap-2 text-sm px-1 py-0.5">
                              <input
                                type="checkbox"
                                checked={newProject.feature_columns.includes(c)}
                                onChange={() => toggleFeature(c)}
                              />
                              {c}
                            </label>
                          ))}
                      </div>
                    </div>
                  )}
                </>
              )}
              <div className="flex gap-2 pt-2">
                <button type="button" className="btn-secondary flex-1" onClick={() => setShowCreate(false)}>
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn-primary flex-1"
                  disabled={
                    creating ||
                    !newProject.feature_columns.length ||
                    (newProject.problem_type === 'binary_classification' &&
                      !newProject.target_positive_label)
                  }
                >
                  {creating ? 'Creating…' : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {projects.length === 0 ? (
        <div className="empty-state">
          <h3 className="font-display text-xl font-semibold text-ink">Start with a project</h3>
          <p className="text-[var(--muted)] mt-2 mb-6 max-w-sm mx-auto text-sm">
            A project links your data to guidance you can act on — who needs attention, why, and what
            to try.
          </p>
          {datasets.length > 0 ? (
            <button type="button" onClick={() => setShowCreate(true)} className="btn-primary">
              Create your first project
            </button>
          ) : (
            <Link to="/datasets" className="btn-primary">
              Upload a dataset first
            </Link>
          )}
        </div>
      ) : (
        <ul className="divide-y divide-mist border-y border-mist">
          {projects.map((project) => {
            const ready = project.status === 'ready' || project.status === 'trained';
            const statusLabel = ready
              ? 'Ready'
              : project.status === 'error'
                ? 'Needs attention'
                : project.status === 'created' || project.status === 'draft'
                  ? 'Needs setup'
                  : String(project.status || '').replace(/_/g, ' ');
            return (
            <li key={project.id}>
              <div
                className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 py-4 cursor-pointer group"
                onClick={() => navigate(`/projects/${project.id}`)}
                onKeyDown={(e) => e.key === 'Enter' && navigate(`/projects/${project.id}`)}
                role="link"
                tabIndex={0}
              >
                <div className="min-w-0">
                  <h3 className="font-display text-lg font-semibold text-ink group-hover:text-teal transition-colors">
                    {project.name}
                  </h3>
                  <p className="text-sm text-[var(--muted)] mt-0.5">
                    Watching{' '}
                    {(project.target_description || project.target_column || 'outcome')
                      .replace(/[_-]+/g, ' ')}
                    {project.feature_count != null ? ` · ${project.feature_count} factors` : ''}
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span
                    className={`badge ${
                      ready
                        ? 'bg-teal-soft/50 text-ink border border-teal/20'
                        : project.status === 'error'
                          ? 'bg-coral-soft text-ink'
                          : 'bg-mist text-ink'
                    }`}
                  >
                    {statusLabel}
                  </span>
                  {(user?.role === 'owner' || user?.role === 'admin') && (
                    <button
                      type="button"
                      className="btn-danger text-xs py-1"
                      disabled={deletingProjectId === project.id}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteProject(project.id, project.name);
                      }}
                    >
                      {deletingProjectId === project.id ? '…' : 'Delete'}
                    </button>
                  )}
                </div>
              </div>
            </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
