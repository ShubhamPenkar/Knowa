import { useState, useEffect, useMemo } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import Spinner from '../components/common/Spinner';
import { useToast } from '../components/ui/Toast';
import Badge from '../components/ui/Badge';
import EmptyState from '../components/ui/EmptyState';
import { SkeletonCard } from '../components/ui/Skeleton';
import {
  FolderKanban,
  Plus,
  Sparkles,
  Trash2,
  ArrowRight,
  Briefcase,
  AlertCircle,
  CheckCircle2,
  Clock,
  Layers,
  Activity,
  SlidersHorizontal,
} from 'lucide-react';
import { useViewMode } from '../context/ViewModeContext';

export default function Projects() {
  const { token, user } = useAuth();
  const { plainEnglish, term } = useViewMode();
  const toast = useToast();
  const [projects, setProjects] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('all'); // all | ready | needs_setup
  const [showCreate, setShowCreate] = useState(false);
  const [newProject, setNewProject] = useState({
    name: '',
    description: '',
    dataset_id: '',
    target_column: '',
    feature_columns: [],
    problem_type: 'binary_classification',
    target_positive_label: '',
    target_description: '',
  });
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [targetLabels, setTargetLabels] = useState([]);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');
  const [deleteError, setDeleteError] = useState('');
  const [deletingProjectId, setDeletingProjectId] = useState(null);
  const [intentDraft, setIntentDraft] = useState(null);
  const [suggesting, setSuggesting] = useState(false);
  const [showManualSetup, setShowManualSetup] = useState(false);
  const navigate = useNavigate();

  const resetCreateForm = () => {
    setNewProject({
      name: '',
      description: '',
      dataset_id: '',
      target_column: '',
      feature_columns: [],
      problem_type: 'binary_classification',
      target_positive_label: '',
      target_description: '',
    });
    setSelectedDataset(null);
    setTargetLabels([]);
    setIntentDraft(null);
    setShowManualSetup(false);
    setError('');
  };

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
    } finally {
      setLoading(false);
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
      target_description: '',
    }));
    setTargetLabels([]);
    setIntentDraft(null);
    setShowManualSetup(false);
    const dataset = datasets.find((d) => d.id === datasetId);
    if (dataset) {
      const res = await fetch(`/api/datasets/${datasetId}/preview?rows=0`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        const cols = (data.columns || [])
          .map((c) => (typeof c === 'string' ? c : c?.name))
          .filter(Boolean);
        setSelectedDataset({ ...dataset, columns: cols });
      }
    }
  };

  const applyIntentDraft = async (draft) => {
    if (!draft) return;
    const features = Array.isArray(draft.feature_columns)
      ? draft.feature_columns.map((c) => (typeof c === 'string' ? c : c?.name)).filter(Boolean)
      : [];
    setNewProject((prev) => ({
      ...prev,
      name: prev.name?.trim() ? prev.name : draft.suggested_name || prev.name,
      description: prev.description?.trim()
        ? prev.description
        : draft.problem_description || prev.description,
      problem_type: draft.problem_type || prev.problem_type,
      target_column: draft.target_column || '',
      target_positive_label: draft.target_positive_label || '',
      target_description: draft.target_description || '',
      feature_columns: features,
    }));
    if (Array.isArray(draft.present_target_labels) && draft.present_target_labels.length) {
      setTargetLabels(draft.present_target_labels.map(String));
    } else if (
      draft.target_column &&
      draft.problem_type !== 'regression' &&
      newProject.dataset_id
    ) {
      // Fetch labels if suggest payload omitted them
      try {
        const res = await fetch(
          `/api/datasets/${newProject.dataset_id}/columns/${encodeURIComponent(draft.target_column)}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (res.ok) {
          const stats = await res.json();
          let values = [];
          if (stats.value_counts) values = Object.keys(stats.value_counts);
          else if (Array.isArray(stats.categories)) values = stats.categories.map(String);
          setTargetLabels(values.map(String).sort());
        }
      } catch {
        /* ignore */
      }
    }
    setShowManualSetup(true);
  };

  const handleSuggestSetup = async () => {
    if (!newProject.dataset_id) {
      setError('Choose a dataset first.');
      return;
    }
    if (!newProject.description?.trim()) {
      setError('Describe the decision problem in plain language first.');
      return;
    }
    setSuggesting(true);
    setError('');
    try {
      const res = await fetch('/api/projects/suggest-config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          dataset_id: newProject.dataset_id,
          problem_description: newProject.description,
          project_name: newProject.name || null,
        }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail;
        setIntentDraft(null);
        setShowManualSetup(true);
        setError(
          typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
              : 'Could not suggest a setup'
        );
        return;
      }
      setIntentDraft(data);
      applyIntentDraft(data);
      toast.success('Configuration suggested based on dataset semantics!');
    } catch {
      setError('Network error while suggesting setup.');
      toast.error('Network error while suggesting setup.');
    } finally {
      setSuggesting(false);
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
      const payload = {
        ...newProject,
        name:
          newProject.name?.trim() ||
          intentDraft?.suggested_name ||
          selectedDataset?.name ||
          'New project',
        target_description:
          newProject.target_description?.trim() ||
          newProject.target_column ||
          'outcome',
        description: newProject.description?.trim() || null,
      };
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok) {
        toast.success(`Project "${payload.name}" created successfully!`);
        navigate(`/projects/${data.id}`);
      } else {
        const detail = data.detail;
        const msg =
          typeof detail === 'string'
            ? detail
            : detail?.message ||
              (Array.isArray(detail)
                ? detail.map((d) => d.msg || JSON.stringify(d)).join('; ')
                : 'Failed to create project');
        setError(msg);
        toast.error(msg);
      }
    } catch {
      setError('Network error. Please try again.');
      toast.error('Network error. Please try again.');
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
      if (res.ok) {
        toast.success(`Project "${projectName}" removed.`);
        fetchProjects();
      } else {
        const data = await res.json();
        const msg = data.detail || 'Failed to delete';
        setDeleteError(msg);
        toast.error(msg);
      }
    } catch {
      const msg = 'Network error while deleting project';
      setDeleteError(msg);
      toast.error(msg);
    } finally {
      setDeletingProjectId(null);
    }
  };

  const filteredProjects = useMemo(() => {
    return projects.filter((project) => {
      const ready = project.status === 'ready' || project.status === 'trained';
      const needsSetup =
        project.status === 'created' ||
        project.status === 'draft' ||
        project.status === 'error';
      if (statusFilter === 'ready') return ready;
      if (statusFilter === 'needs_setup') return needsSetup || !ready;
      return true;
    });
  }, [projects, statusFilter]);

  if (loading) {
    return (
      <div className="page space-y-4">
        <div className="h-8 w-44 bg-mist/60 rounded animate-pulse mb-6" />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Workspace</p>
          <h1 className="page-title">Projects</h1>
          <p className="page-sub">
            Create and prepare projects here. Day-to-day review lives under Cases, Follow-ups, and What-if.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Link to="/follow-ups" className="btn-secondary">
            Follow-ups
          </Link>
          <Link to="/cases" className="btn-ghost">
            Cases
          </Link>
          <button
            type="button"
            onClick={() => {
              resetCreateForm();
              setShowCreate(true);
            }}
            className="btn-primary inline-flex items-center gap-1.5"
          >
            <Plus className="w-4 h-4" />
            New project
          </button>
        </div>
      </div>

      {deleteError && (
        <div className="mb-4 text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-coral shrink-0" />
          <span>{deleteError}</span>
        </div>
      )}

      {showCreate && (
        <div className="fixed inset-0 z-50 bg-ink/40 flex items-start justify-center overflow-y-auto py-10 px-4">
          <div className="bg-surface border border-mist rounded-control w-full max-w-2xl p-6 animate-page-in">
            <h2 className="font-display text-xl font-semibold text-ink mb-1">Create project</h2>
            <p className="text-sm text-[var(--muted)] mb-4">
              Describe the decision in plain language — we&apos;ll draft the setup for you to confirm.
            </p>
            {error && (
              <div className="mb-4 text-sm border border-coral/40 bg-coral-soft px-4 py-3 rounded-control">
                {typeof error === 'string' ? error : JSON.stringify(error)}
              </div>
            )}
            <form onSubmit={handleCreateProject} className="space-y-4">
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
              <div>
                <label className="block text-sm font-medium mb-1">
                  What decision are you trying to make?
                </label>
                <textarea
                  className="input"
                  rows={3}
                  value={newProject.description}
                  onChange={(e) => setNewProject((p) => ({ ...p, description: e.target.value }))}
                  placeholder="e.g. Spot telecom customers likely to churn so we can retain them"
                  required
                  autoFocus
                />
                <p className="text-xs text-[var(--muted)] mt-1">
                  Must sound like a business decision we can score — churn, attrition, conversion,
                  fraud, default, or spend. Vague prompts (e.g. “help me with bananas”) are rejected;
                  you can still set up manually.
                </p>
              </div>
              {selectedDataset && (
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    className="btn-primary text-sm"
                    disabled={suggesting || !newProject.description.trim()}
                    onClick={handleSuggestSetup}
                  >
                    {suggesting ? 'Suggesting…' : 'Suggest setup'}
                  </button>
                  <button
                    type="button"
                    className="btn-ghost text-sm"
                    onClick={() => setShowManualSetup((v) => !v)}
                  >
                    {showManualSetup || intentDraft
                      ? showManualSetup
                        ? 'Hide manual fields'
                        : 'Edit setup fields'
                      : 'Set up manually'}
                  </button>
                </div>
              )}
              {(intentDraft || showManualSetup) && (
                <div>
                  <label className="block text-sm font-medium mb-1">Project name</label>
                  <input
                    className="input"
                    value={newProject.name}
                    onChange={(e) => setNewProject((p) => ({ ...p, name: e.target.value }))}
                    placeholder={intentDraft?.suggested_name || 'Optional — we can suggest one'}
                  />
                </div>
              )}
              {intentDraft && (
                <div className="border border-mist px-4 py-3 space-y-2 bg-mist/15">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-ink">
                    Suggested setup
                    {intentDraft.confidence != null ? (
                      <span className="ml-2 font-normal normal-case tracking-normal text-[var(--muted)]">
                        confidence {Math.round(Number(intentDraft.confidence) * 100)}%
                      </span>
                    ) : null}
                  </div>
                  <p className="text-sm text-ink leading-relaxed">{intentDraft.rationale}</p>
                  <ul className="text-xs text-[var(--muted)] space-y-1">
                    <li>
                      Question:{' '}
                      {intentDraft.problem_type === 'regression' ? 'How much?' : 'Will it happen?'}
                    </li>
                    <li>Target column: {intentDraft.target_column}</li>
                    <li>Business label: {intentDraft.target_description}</li>
                    {intentDraft.target_positive_label ? (
                      <li>Positive label: {intentDraft.target_positive_label}</li>
                    ) : null}
                    <li>
                      Features: {(intentDraft.feature_columns || []).length} columns (IDs/constants
                      removed)
                    </li>
                  </ul>
                  <p className="text-xs text-[var(--muted)]">
                    Review the fields below, then create — nothing is saved until you confirm.
                  </p>
                </div>
              )}
              {selectedDataset && showManualSetup && (
                <>
                  <div>
                    <label className="block text-sm font-medium mb-1">Business outcome label</label>
                    <input
                      className="input"
                      value={newProject.target_description}
                      onChange={(e) =>
                        setNewProject((p) => ({ ...p, target_description: e.target.value }))
                      }
                      placeholder="e.g. churn, attrition"
                    />
                    <p className="text-xs text-[var(--muted)] mt-1">
                      Used in Cases, insights, and follow-up language.
                    </p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      {plainEnglish ? 'What kind of question do you want to answer?' : 'Problem Type / Learning Task'}
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        [
                          'binary_classification',
                          plainEnglish ? 'Will it happen?' : 'Binary Classification',
                          plainEnglish ? 'Yes / No risk (churn, conversion, fraud)' : 'Discrete 2-class probability distribution',
                        ],
                        [
                          'regression',
                          plainEnglish ? 'How much?' : 'Continuous Regression',
                          plainEnglish ? 'A continuous number (spend, revenue, days)' : 'Continuous numerical target estimation',
                        ],
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
                    <label className="block text-sm font-medium mb-1">
                      {plainEnglish ? 'Outcome column you want to predict' : 'Target Column (y Label)'}
                    </label>
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
                          {plainEnglish ? 'What counts as a positive match?' : 'Target Positive Class (y=1)'}
                        </label>
                        <p className="text-xs text-[var(--muted)] mb-1">
                          {plainEnglish
                            ? 'Which value means the event of concern (e.g. Yes, Churned, 1)?'
                            : 'Specifies the ground-truth positive class encoded as 1 for classification metrics.'}
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
              {!newProject.target_column || !newProject.feature_columns.length ? (
                <p className="text-xs text-[var(--muted)]">
                  {newProject.dataset_id && newProject.description?.trim()
                    ? 'Click “Suggest setup” (or set up manually) before creating.'
                    : 'Choose a dataset and describe the decision to continue.'}
                </p>
              ) : null}
              <div className="flex gap-2 pt-2">
                <button
                  type="button"
                  className="btn-secondary flex-1"
                  onClick={() => {
                    setShowCreate(false);
                    resetCreateForm();
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn-primary flex-1"
                  disabled={
                    creating ||
                    !newProject.target_column ||
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
        <EmptyState
          icon={FolderKanban}
          title="Start with a project"
          description="A project links your data to guidance you can act on — who needs attention, why, and what to try."
          action={
            datasets.length > 0 ? (
              <button
                type="button"
                onClick={() => {
                  resetCreateForm();
                  setShowCreate(true);
                }}
                className="btn-primary inline-flex items-center gap-2"
              >
                <Sparkles className="w-4 h-4" />
                Describe a decision
              </button>
            ) : (
              <Link to="/datasets" className="btn-primary inline-flex items-center gap-2">
                <Plus className="w-4 h-4" />
                Upload a dataset first
              </Link>
            )
          }
        />
      ) : (
        <>
          <div className="flex flex-wrap gap-2 mb-4" role="tablist" aria-label="Filter projects">
            {[
              { id: 'all', label: 'All' },
              { id: 'ready', label: 'Ready' },
              { id: 'needs_setup', label: 'Needs setup' },
            ].map((chip) => (
              <button
                key={chip.id}
                type="button"
                role="tab"
                aria-selected={statusFilter === chip.id}
                onClick={() => setStatusFilter(chip.id)}
                className={`px-3 py-1.5 text-xs border rounded-control transition-colors ${
                  statusFilter === chip.id
                    ? 'border-teal bg-teal-soft/30 text-ink font-medium shadow-xs'
                    : 'border-mist text-[var(--muted)] hover:text-ink hover:bg-mist/20'
                }`}
              >
                {chip.label}
              </button>
            ))}
          </div>
          {filteredProjects.length === 0 ? (
            <div className="border border-mist px-5 py-8 text-sm text-[var(--muted)] text-center rounded-control">
              No projects match this filter.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredProjects.map((project) => {
                const ready = project.status === 'ready' || project.status === 'trained';
                const statusLabel = ready
                  ? 'Ready'
                  : project.status === 'error'
                    ? 'Needs attention'
                    : project.status === 'created' || project.status === 'draft'
                      ? 'Needs setup'
                      : String(project.status || '').replace(/_/g, ' ');
                const badgeVariant = ready
                  ? 'safe'
                  : project.status === 'error'
                    ? 'risk'
                    : 'neutral';

                return (
                  <div
                    key={project.id}
                    className="p-5 rounded-control border border-mist hover:border-teal/50 bg-surface decision-card cursor-pointer group flex flex-col justify-between transition-all"
                    onClick={() => navigate(`/projects/${project.id}`)}
                    onKeyDown={(e) => e.key === 'Enter' && navigate(`/projects/${project.id}`)}
                    role="link"
                    tabIndex={0}
                  >
                    <div>
                      <div className="flex items-start justify-between gap-3 mb-3">
                        <div className="w-10 h-10 rounded-control bg-teal-soft/40 border border-teal/20 flex items-center justify-center shrink-0 text-teal group-hover:scale-105 transition-transform">
                          <FolderKanban className="w-5 h-5" />
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={badgeVariant}>{statusLabel}</Badge>
                          {(user?.role === 'owner' || user?.role === 'admin') && (
                            <button
                              type="button"
                              className="p-1.5 rounded-control text-[var(--muted)] hover:text-coral hover:bg-coral-soft/50 transition-colors"
                              disabled={deletingProjectId === project.id}
                              title="Delete project"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDeleteProject(project.id, project.name);
                              }}
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <h3 className="font-display text-base font-semibold text-ink group-hover:text-teal transition-colors">
                          {project.name}
                        </h3>
                        <ArrowRight className="w-4 h-4 text-[var(--muted)] opacity-0 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                      </div>
                      <p className="text-xs text-[var(--muted)] mt-1.5 line-clamp-2">
                        {project.description || 'Target outcome and predictive factors configured.'}
                      </p>
                    </div>

                    <div className="pt-4 mt-4 border-t border-mist/60 flex items-center justify-between text-xs text-[var(--muted)]">
                      <span className="flex items-center gap-1.5">
                        <Activity className="w-3.5 h-3.5 text-teal" />
                        {plainEnglish ? 'Goal:' : 'Target (y):'}{' '}
                        <strong className="text-ink font-medium">
                          {(project.target_description || project.target_column || 'outcome').replace(/[_-]+/g, ' ')}
                        </strong>
                      </span>
                      {project.feature_count != null && (
                        <span className="flex items-center gap-1">
                          <Layers className="w-3.5 h-3.5" />
                          {project.feature_count} {plainEnglish ? 'factors' : 'features'}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
