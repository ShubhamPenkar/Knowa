import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import ProductIntelligenceOverview from '../components/common/ProductIntelligenceOverview';

const ROLE_OPTIONS = [
  { value: 'dimension', label: 'Dimension (Categorical/Grouping)' },
  { value: 'measure', label: 'Measure (Numeric metric/Quantity)' },
  { value: 'identifier', label: 'Identifier (Row key/ID)' },
  { value: 'timestamp', label: 'Timestamp (Date/Time)' },
  { value: 'unknown', label: 'Unknown / Ambiguous' },
];

const SEMANTIC_TYPE_OPTIONS = [
  { value: 'categorical', label: 'Categorical' },
  { value: 'measure', label: 'Measure / Metric' },
  { value: 'numeric_measure', label: 'Numeric Measure (Legacy)' },
  { value: 'identifier', label: 'Identifier / Key' },
  { value: 'id', label: 'Identifier (Legacy)' },
  { value: 'boolean', label: 'Boolean Flag' },
  { value: 'datetime', label: 'Date / Timestamp' },
  { value: 'date', label: 'Date' },
  { value: 'currency', label: 'Currency / Monetary' },
  { value: 'percentage', label: 'Percentage' },
  { value: 'ordinal', label: 'Ordinal' },
  { value: 'duration', label: 'Duration' },
  { value: 'email', label: 'Email' },
  { value: 'phone', label: 'Phone' },
  { value: 'geographic', label: 'Geographic' },
  { value: 'text', label: 'Unstructured Text' },
  { value: 'unknown', label: 'Unknown / Ambiguous' },
];

const BUSINESS_ROLE_OPTIONS = [
  { value: 'unknown', label: 'None / Generic' },
  { value: 'revenue', label: 'Revenue / Sales' },
  { value: 'profit', label: 'Profit / Margin' },
  { value: 'cost', label: 'Cost / Expense' },
  { value: 'price', label: 'Price / Fee' },
  { value: 'customer', label: 'Customer' },
  { value: 'product', label: 'Product' },
  { value: 'order', label: 'Order / Transaction' },
  { value: 'user', label: 'User / Account' },
  { value: 'churn', label: 'Churn Indicator' },
  { value: 'conversion', label: 'Conversion' },
  { value: 'age', label: 'Age' },
  { value: 'tenure', label: 'Tenure' },
  { value: 'rating', label: 'Rating / Score' },
  { value: 'quantity', label: 'Quantity' },
  { value: 'percentage', label: 'Percentage' },
  { value: 'country', label: 'Country' },
  { value: 'state', label: 'State' },
  { value: 'city', label: 'City' },
  { value: 'date', label: 'Date' },
];

export default function Datasets() {
  const { token } = useAuth();
  const [datasets, setDatasets] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadName, setUploadName] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('datasets');

  // Semantic Inspection & Correction State
  const [inspectingDatasetId, setInspectingDatasetId] = useState(null);
  const [datasetDetail, setDatasetDetail] = useState(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [editingColumn, setEditingColumn] = useState(null);
  const [editForm, setEditForm] = useState({ role: '', semantic_type: '', business_role: '', notes: '' });
  const [savingCorrection, setSavingCorrection] = useState(false);
  const [correctionSuccess, setCorrectionSuccess] = useState('');
  const [correctionError, setCorrectionError] = useState('');
  const [expandedColumn, setExpandedColumn] = useState(null);

  useEffect(() => {
    if (token) fetchDatasets();
  }, [token]);

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

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile || !uploadName) return;
    setUploading(true);
    setError('');
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('name', uploadName);
    try {
      const res = await fetch('/api/datasets', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: formData,
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || 'Upload failed');
      }
      setShowUpload(false);
      setSelectedFile(null);
      setUploadName('');
      fetchDatasets();
    } catch (err) {
      setError(err.message);
    }
    setUploading(false);
  };

  const openInspector = async (datasetId) => {
    if (inspectingDatasetId === datasetId) {
      setInspectingDatasetId(null);
      setDatasetDetail(null);
      setEditingColumn(null);
      setExpandedColumn(null);
      return;
    }
    setInspectingDatasetId(datasetId);
    setLoadingDetail(true);
    setEditingColumn(null);
    setExpandedColumn(null);
    setCorrectionSuccess('');
    setCorrectionError('');
    try {
      const [detailRes, profileRes] = await Promise.all([
        fetch(`/api/datasets/${datasetId}`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
        fetch(`/api/datasets/${datasetId}/semantic-profile`, {
          headers: { Authorization: `Bearer ${token}` },
        }),
      ]);

      if (detailRes.ok) {
        const detail = await detailRes.json();
        let semanticProfile = null;
        if (profileRes.ok) {
          semanticProfile = await profileRes.json();
        }
        setDatasetDetail({ ...detail, semantic_profile: semanticProfile });
      }
    } catch (err) {
      console.error(err);
    }
    setLoadingDetail(false);
  };

  const startEditColumn = (col) => {
    setEditingColumn(col.name);
    setEditForm({
      role: col.role || 'dimension',
      semantic_type: col.semantic_type || 'categorical',
      business_role: col.business_role || 'unknown',
      notes: '',
    });
    setCorrectionSuccess('');
  };

  const handleSaveCorrection = async (columnName) => {
    if (!inspectingDatasetId) return;
    setSavingCorrection(true);
    setCorrectionError('');
    setCorrectionSuccess('');
    try {
      const res = await fetch(`/api/datasets/${inspectingDatasetId}/columns/${encodeURIComponent(columnName)}/semantic`, {
        method: 'PATCH',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          role: editForm.role,
          semantic_type: editForm.semantic_type,
          business_role: editForm.business_role || undefined,
          notes: editForm.notes || undefined,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        const updatedCol = data.updated;
        if (updatedCol && datasetDetail) {
          const updatedCols = (datasetDetail.columns || []).map((c) =>
            c.name === columnName ? { ...c, ...updatedCol } : c
          );
          setDatasetDetail({
            ...datasetDetail,
            columns: updatedCols,
            semantic_profile: datasetDetail.semantic_profile
              ? {
                  ...datasetDetail.semantic_profile,
                  columns: (datasetDetail.semantic_profile.columns || []).map((c) =>
                    c.name === columnName ? { ...c, ...updatedCol } : c
                  ),
                }
              : null,
          });
        }
        setEditingColumn(null);
        setCorrectionSuccess(`Corrected column '${columnName}' successfully.`);
      } else {
        const errData = await res.json().catch(() => ({}));
        setCorrectionError(errData.detail || 'Failed to update semantic column');
      }
    } catch (err) {
      console.error(err);
      setCorrectionError('Error updating column semantics');
    }
    setSavingCorrection(false);
  };

  const getRoleBadgeClass = (role) => {
    switch (role) {
      case 'measure':
        return 'bg-blue-500/20 text-blue-300 border border-blue-500/30';
      case 'identifier':
        return 'bg-purple-500/20 text-purple-300 border border-purple-500/30';
      case 'timestamp':
        return 'bg-amber-500/20 text-amber-300 border border-amber-500/30';
      case 'dimension':
        return 'bg-teal-500/20 text-teal-300 border border-teal-500/30';
      default:
        return 'bg-gray-500/20 text-gray-300 border border-gray-500/30';
    }
  };

  const toggleExpandColumn = (colName) => {
    setExpandedColumn(expandedColumn === colName ? null : colName);
  };

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Data</p>
          <h1 className="page-title">Datasets</h1>
          <p className="page-sub">Evidence-based semantic data understanding, profiling, and schema models.</p>
        </div>
        <button type="button" onClick={() => setShowUpload(true)} className="btn-primary">
          Upload dataset
        </button>
      </div>

      {/* View Switcher Tabs */}
      <div className="flex border-b border-mist mb-6 gap-6">
        <button
          type="button"
          onClick={() => setActiveTab('datasets')}
          className={`pb-3 text-sm font-medium transition-colors border-b-2 ${
            activeTab === 'datasets'
              ? 'border-teal text-ink'
              : 'border-transparent text-[var(--muted)] hover:text-ink'
          }`}
        >
          Datasets & Semantics ({datasets.length})
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('intelligence')}
          className={`pb-3 text-sm font-medium transition-colors border-b-2 ${
            activeTab === 'intelligence'
              ? 'border-teal text-ink'
              : 'border-transparent text-[var(--muted)] hover:text-ink'
          }`}
        >
          Product Intelligence
        </button>
      </div>

      {showUpload && (
        <div className="fixed inset-0 z-50 bg-ink/40 flex items-center justify-center p-4">
          <div className="bg-surface border border-mist rounded-control w-full max-w-md p-6 animate-page-in">
            <h2 className="font-display text-xl font-semibold mb-4">Upload dataset</h2>
            {error && (
              <div className="mb-3 text-sm border border-coral/40 bg-coral-soft px-3 py-2 rounded-control">
                {error}
              </div>
            )}
            <form onSubmit={handleUpload} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Name</label>
                <input
                  className="input"
                  value={uploadName}
                  onChange={(e) => setUploadName(e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">CSV file</label>
                <div className="border border-dashed border-mist rounded-control p-5 text-center">
                  <input
                    type="file"
                    accept=".csv"
                    id="file-upload"
                    className="sr-only"
                    onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                  />
                  <label htmlFor="file-upload" className="cursor-pointer text-sm text-teal font-medium">
                    {selectedFile ? selectedFile.name : 'Choose CSV…'}
                  </label>
                </div>
              </div>
              <div className="flex gap-2">
                <button type="button" className="btn-secondary flex-1" onClick={() => setShowUpload(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn-primary flex-1" disabled={uploading || !selectedFile}>
                  {uploading ? 'Uploading…' : 'Upload'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {activeTab === 'intelligence' ? (
        <ProductIntelligenceOverview token={token} />
      ) : datasets.length === 0 ? (
        <div className="empty-state">
          <h3 className="font-display text-xl font-semibold">No datasets yet</h3>
          <p className="text-sm text-[var(--muted)] mt-2 mb-6">Upload a CSV to get started.</p>
          <button type="button" className="btn-primary" onClick={() => setShowUpload(true)}>
            Upload dataset
          </button>
        </div>
      ) : (
        <ul className="divide-y divide-mist border-y border-mist">
          {datasets.map((dataset) => {
            const isInspecting = inspectingDatasetId === dataset.id;
            const profile = datasetDetail?.semantic_profile;
            return (
              <li key={dataset.id} className="py-4">
                <div className="flex justify-between gap-3 items-start">
                  <div>
                    <h3 className="font-display text-lg font-semibold text-ink">{dataset.name}</h3>
                    <p className="text-sm text-[var(--muted)] mt-0.5">
                      {dataset.row_count?.toLocaleString()} rows · {dataset.column_count} columns
                    </p>
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {(dataset.columns || []).slice(0, 6).map((col) => {
                        const name = typeof col === 'string' ? col : col?.name;
                        return name ? (
                          <span key={name} className="badge bg-mist text-ink/80">
                            {name}
                          </span>
                        ) : null;
                      })}
                      {dataset.column_count > 6 && (
                        <span className="text-xs text-[var(--muted)]">+{dataset.column_count - 6}</span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => openInspector(dataset.id)}
                      className="btn-secondary text-xs px-3 py-1.5"
                    >
                      {isInspecting ? 'Close Semantics' : 'Inspect Semantics'}
                    </button>
                    <span className="badge bg-teal-soft/50 text-ink border border-teal/20">Ready</span>
                  </div>
                </div>

                {isInspecting && (
                  <div className="mt-4 p-4 rounded-control bg-surface border border-mist animate-page-in">
                    <div className="flex justify-between items-center mb-3">
                      <div>
                        <h4 className="font-display text-base font-semibold text-ink">Semantic Data Understanding</h4>
                        <p className="text-xs text-[var(--muted)]">
                          Evidence-based inferences: types, business roles, data quality, targets, and relationships.
                        </p>
                      </div>
                      {correctionSuccess && (
                        <span className="text-xs text-teal bg-teal-soft/40 px-2.5 py-1 rounded-control border border-teal/30">
                          {correctionSuccess}
                        </span>
                      )}
                    </div>

                    {correctionError && (
                      <div className="mb-3 text-xs border border-coral/40 bg-coral-soft px-3 py-2 rounded-control text-coral">
                        {correctionError}
                      </div>
                    )}

                    {loadingDetail ? (
                      <div className="text-sm text-[var(--muted)] py-4 text-center">Loading semantic profile…</div>
                    ) : (
                      <>
                        {/* Step 9: Dataset Overview Cards */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                          <div className="p-3 bg-mist/20 rounded-control border border-mist/40">
                            <div className="text-[10px] uppercase font-semibold text-[var(--muted)]">Dataset Shape</div>
                            <div className="text-base font-semibold text-ink mt-0.5">
                              {(profile?.dataset?.row_count ?? dataset.row_count)?.toLocaleString()} rows
                            </div>
                            <div className="text-xs text-[var(--muted)]">
                              {profile?.dataset?.column_count ?? dataset.column_count} columns
                            </div>
                          </div>

                          <div className="p-3 bg-mist/20 rounded-control border border-mist/40">
                            <div className="text-[10px] uppercase font-semibold text-[var(--muted)]">Quality Score</div>
                            <div className="flex items-center gap-1.5 mt-0.5">
                              <span className="text-base font-semibold text-ink">
                                {profile?.dataset?.quality_score ?? 100}/100
                              </span>
                              <span
                                className={`badge text-[10px] py-0 px-1.5 ${
                                  (profile?.dataset?.quality_score ?? 100) >= 90
                                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                                    : (profile?.dataset?.quality_score ?? 100) >= 75
                                    ? 'bg-blue-500/20 text-blue-300 border-blue-500/30'
                                    : (profile?.dataset?.quality_score ?? 100) >= 60
                                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                                    : 'bg-coral-soft text-coral border-coral/30'
                                }`}
                              >
                                {profile?.quality_summary?.grade || 'Good'}
                              </span>
                            </div>
                            <div className="text-xs text-[var(--muted)]">
                              {profile?.quality_summary?.missing_cells_ratio != null
                                ? `${(profile.quality_summary.missing_cells_ratio * 100).toFixed(1)}% missing cells`
                                : '0% missing'}
                            </div>
                          </div>

                          <div className="p-3 bg-mist/20 rounded-control border border-mist/40">
                            <div className="text-[10px] uppercase font-semibold text-[var(--muted)]">Duplicates & Hygiene</div>
                            <div className="text-base font-semibold text-ink mt-0.5">
                              {profile?.dataset?.duplicate_rows_count ?? 0} duplicates
                            </div>
                            <div className="text-xs text-[var(--muted)]">
                              {profile?.dataset?.duplicate_rows_ratio != null
                                ? `${(profile.dataset.duplicate_rows_ratio * 100).toFixed(1)}% duplication`
                                : '0% duplication'}
                            </div>
                          </div>

                          <div className="p-3 bg-mist/20 rounded-control border border-mist/40">
                            <div className="text-[10px] uppercase font-semibold text-[var(--muted)]">Target Candidate</div>
                            {profile?.potential_targets?.length > 0 ? (
                              <>
                                <div className="text-base font-semibold text-teal truncate mt-0.5" title={profile.potential_targets[0].column_name}>
                                  {profile.potential_targets[0].column_name}
                                </div>
                                <div className="text-xs text-[var(--muted)] capitalize">
                                  {profile.potential_targets[0].target_type} ({Math.round(profile.potential_targets[0].confidence * 100)}%)
                                </div>
                              </>
                            ) : (
                              <>
                                <div className="text-base font-semibold text-[var(--muted)] mt-0.5">None detected</div>
                                <div className="text-xs text-[var(--muted)]">Define target during training</div>
                              </>
                            )}
                          </div>
                        </div>

                        {/* Ranked Target Candidates & Relationships */}
                        {profile?.potential_targets?.length > 1 && (
                          <div className="mb-3 p-2.5 rounded-control bg-teal/5 border border-teal/20 text-xs flex items-center gap-2">
                            <span className="font-semibold text-teal">Ranked Target Candidates:</span>
                            <div className="flex flex-wrap gap-1.5">
                              {profile.potential_targets.map((t, idx) => (
                                <span key={t.column_name} className="badge bg-surface border border-mist text-ink text-[11px]">
                                  #{idx + 1} {t.column_name} ({t.target_type}, {Math.round(t.confidence * 100)}%)
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {profile?.column_relationships?.length > 0 && (
                          <div className="mb-3 p-2.5 rounded-control bg-amber-500/10 border border-amber-500/30 text-xs text-amber-200">
                            <span className="font-semibold text-amber-300">Feature Relationships: </span>
                            {profile.column_relationships.slice(0, 3).map((r, i) => (
                              <span key={i} className="mr-3 inline-block">
                                {r.description}
                              </span>
                            ))}
                          </div>
                        )}

                        {/* Step 9: Semantic Column Table */}
                        <div className="overflow-x-auto">
                          <table className="w-full text-left text-xs border-collapse">
                            <thead>
                              <tr className="border-b border-mist text-[var(--muted)] uppercase tracking-wider text-[10px]">
                                <th className="py-2 px-2.5">Column</th>
                                <th className="py-2 px-2.5">Type</th>
                                <th className="py-2 px-2.5">Business Meaning</th>
                                <th className="py-2 px-2.5">Role</th>
                                <th className="py-2 px-2.5">Missing / Uniq</th>
                                <th className="py-2 px-2.5" title="Evidence-based heuristic score, not a calibrated probability">Confidence</th>
                                <th className="py-2 px-2.5">Explainability</th>
                                <th className="py-2 px-2.5 text-right">Action</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-mist/60">
                              {(profile?.columns || datasetDetail?.columns || []).map((col) => {
                                const isEditing = editingColumn === col.name;
                                const isExpanded = expandedColumn === col.name;
                                const confPercent = Math.round((col.confidence || 0) * 100);
                                const missingPercent = Math.round((col.missing_ratio ?? col.null_frac ?? 0) * 100);
                                return (
                                  <tr key={col.name} className="hover:bg-mist/30 transition-colors group">
                                    <td className="py-2.5 px-2.5 font-medium text-ink font-mono text-xs">
                                      <div className="flex items-center gap-1.5 flex-wrap">
                                        <span>{col.name}</span>
                                        {col.is_user_corrected && (
                                          <span className="badge bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[9px] py-0 px-1">
                                            Corrected
                                          </span>
                                        )}
                                        {col.is_target_candidate && (
                                          <span className="badge bg-teal-500/20 text-teal-300 border border-teal-500/30 text-[9px] py-0 px-1">
                                            Target Candidate
                                          </span>
                                        )}
                                        {col.is_identifier && (
                                          <span className="badge bg-purple-500/20 text-purple-300 border border-purple-500/30 text-[9px] py-0 px-1">
                                            Identifier
                                          </span>
                                        )}
                                      </div>
                                    </td>
                                    <td className="py-2.5 px-2.5 text-[var(--muted)] font-mono text-[11px]">
                                      {col.physical_type || col.dtype || '—'}
                                    </td>
                                    <td className="py-2.5 px-2.5">
                                      {isEditing ? (
                                        <select
                                          className="input py-1 text-xs"
                                          value={editForm.business_role}
                                          onChange={(e) => setEditForm({ ...editForm, business_role: e.target.value })}
                                        >
                                          {BUSINESS_ROLE_OPTIONS.map((opt) => (
                                            <option key={opt.value} value={opt.value}>
                                              {opt.label}
                                            </option>
                                          ))}
                                        </select>
                                      ) : (
                                        <span className="badge bg-mist text-ink/90 font-mono text-[11px] capitalize">
                                          {col.business_role && col.business_role !== 'unknown'
                                            ? col.business_role.replace(/_/g, ' ')
                                            : '—'}
                                        </span>
                                      )}
                                    </td>
                                    <td className="py-2.5 px-2.5">
                                      {isEditing ? (
                                        <div className="space-y-1">
                                          <select
                                            className="input py-1 text-xs"
                                            value={editForm.role}
                                            onChange={(e) => setEditForm({ ...editForm, role: e.target.value })}
                                          >
                                            {ROLE_OPTIONS.map((opt) => (
                                              <option key={opt.value} value={opt.value}>
                                                {opt.label}
                                              </option>
                                            ))}
                                          </select>
                                          <select
                                            className="input py-1 text-xs"
                                            value={editForm.semantic_type}
                                            onChange={(e) => setEditForm({ ...editForm, semantic_type: e.target.value })}
                                          >
                                            {SEMANTIC_TYPE_OPTIONS.map((opt) => (
                                              <option key={opt.value} value={opt.value}>
                                                {opt.label}
                                              </option>
                                            ))}
                                          </select>
                                        </div>
                                      ) : (
                                        <div className="flex flex-col gap-0.5">
                                          <span className={`badge ${getRoleBadgeClass(col.role)} capitalize`}>
                                            {col.role || 'unknown'}
                                          </span>
                                          <span className="text-[10px] text-[var(--muted)] font-mono">
                                            {col.semantic_type || 'unknown'}
                                          </span>
                                        </div>
                                      )}
                                    </td>
                                    <td className="py-2.5 px-2.5 text-[11px] text-[var(--muted)]">
                                      <div>{missingPercent}% null</div>
                                      <div className="text-[10px] text-[var(--muted)]/80">
                                        {col.cardinality ?? col.n_unique ?? '—'} uniques
                                      </div>
                                    </td>
                                    <td className="py-2.5 px-2.5">
                                      <div
                                        className="flex items-center gap-1.5"
                                        title={
                                          col.is_user_corrected
                                            ? 'User-assigned certainty (100%)'
                                            : `Evidence-based heuristic score: ${confPercent}%`
                                        }
                                      >
                                        <div className="w-12 h-1.5 bg-mist rounded-full overflow-hidden">
                                          <div
                                            className={`h-full ${
                                              confPercent >= 85
                                                ? 'bg-teal'
                                                : confPercent >= 60
                                                ? 'bg-amber-400'
                                                : 'bg-red-400'
                                            }`}
                                            style={{ width: `${confPercent}%` }}
                                          />
                                        </div>
                                        <span className="text-[11px] text-[var(--muted)]">{confPercent}%</span>
                                        <span className="text-[9px] uppercase tracking-wider text-[var(--muted)]/60 font-mono">
                                          {col.is_user_corrected ? 'user' : 'heur.'}
                                        </span>
                                      </div>
                                    </td>
                                    <td className="py-2.5 px-2.5 text-[11px] text-[var(--muted)] max-w-xs">
                                      {isEditing ? (
                                        <input
                                          className="input py-1 text-xs w-full"
                                          placeholder="Correction note…"
                                          value={editForm.notes}
                                          onChange={(e) => setEditForm({ ...editForm, notes: e.target.value })}
                                        />
                                      ) : (
                                        <div>
                                          <button
                                            type="button"
                                            onClick={() => toggleExpandColumn(col.name)}
                                            className="text-teal hover:underline text-[11px] flex items-center gap-1"
                                          >
                                            {isExpanded ? 'Hide explanation' : 'Why this role?'}
                                          </button>
                                          {isExpanded && (
                                            <div className="mt-1 p-2 bg-mist/30 rounded border border-mist/60 text-[10px] space-y-1">
                                              <div className="font-semibold text-ink">Evidence:</div>
                                              <ul className="list-disc list-inside space-y-0.5">
                                                {(col.evidence || [col.profile_reason]).filter(Boolean).map((ev, i) => (
                                                  <li key={i}>{ev}</li>
                                                ))}
                                              </ul>
                                              {col.quality_issues?.length > 0 && (
                                                <div className="mt-1 pt-1 border-t border-mist/40">
                                                  <span className="text-amber-400 font-semibold">Quality notes: </span>
                                                  <span>{col.quality_issues.join(', ')}</span>
                                                </div>
                                              )}
                                              {col.distribution_stats && Object.keys(col.distribution_stats).length > 0 && (
                                                <div className="mt-1 pt-1 border-t border-mist/40 text-[9px] text-[var(--muted)]">
                                                  {col.distribution_stats.min !== undefined && (
                                                    <span>Min: {col.distribution_stats.min} · Max: {col.distribution_stats.max} · </span>
                                                  )}
                                                  {col.distribution_stats.mean !== undefined && (
                                                    <span>Mean: {col.distribution_stats.mean.toFixed(2)} · </span>
                                                  )}
                                                  {col.distribution_stats.cardinality !== undefined && (
                                                    <span>Card: {col.distribution_stats.cardinality}</span>
                                                  )}
                                                </div>
                                              )}
                                            </div>
                                          )}
                                        </div>
                                      )}
                                    </td>
                                    <td className="py-2.5 px-2.5 text-right whitespace-nowrap">
                                      {isEditing ? (
                                        <div className="flex gap-1 justify-end">
                                          <button
                                            type="button"
                                            onClick={() => handleSaveCorrection(col.name)}
                                            disabled={savingCorrection}
                                            className="btn-primary text-[11px] py-1 px-2"
                                          >
                                            {savingCorrection ? 'Saving…' : 'Save'}
                                          </button>
                                          <button
                                            type="button"
                                            onClick={() => setEditingColumn(null)}
                                            className="btn-secondary text-[11px] py-1 px-2"
                                          >
                                            Cancel
                                          </button>
                                        </div>
                                      ) : (
                                        <button
                                          type="button"
                                          onClick={() => startEditColumn(col)}
                                          className="text-teal hover:underline text-[11px]"
                                        >
                                          Correct
                                        </button>
                                      )}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
