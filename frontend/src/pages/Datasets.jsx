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
  { value: 'numeric_measure', label: 'Numeric Measure' },
  { value: 'id', label: 'Identifier / Key' },
  { value: 'boolean', label: 'Boolean Flag' },
  { value: 'datetime', label: 'Date / Timestamp' },
  { value: 'geographic', label: 'Geographic (Country/State/Zip/Coords)' },
  { value: 'text', label: 'Unstructured Text' },
  { value: 'unknown', label: 'Unknown / Ambiguous' },
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
  const [editForm, setEditForm] = useState({ role: '', semantic_type: '', notes: '' });
  const [savingCorrection, setSavingCorrection] = useState(false);
  const [correctionSuccess, setCorrectionSuccess] = useState('');
  const [correctionError, setCorrectionError] = useState('');

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
      return;
    }
    setInspectingDatasetId(datasetId);
    setLoadingDetail(true);
    setEditingColumn(null);
    setCorrectionSuccess('');
    setCorrectionError('');
    try {
      const res = await fetch(`/api/datasets/${datasetId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const detail = await res.json();
        setDatasetDetail(detail);
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
          notes: editForm.notes || undefined,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        const updatedCol = data.updated;
        if (updatedCol && datasetDetail) {
          setDatasetDetail({
            ...datasetDetail,
            columns: datasetDetail.columns.map((c) =>
              c.name === columnName ? { ...c, ...updatedCol } : c
            ),
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

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Data</p>
          <h1 className="page-title">Datasets</h1>
          <p className="page-sub">Spreadsheets, CSVs, and inferred semantic column models.</p>
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
                        <h4 className="font-display text-base font-semibold text-ink">Semantic Column Understanding</h4>
                        <p className="text-xs text-[var(--muted)]">
                          Inferred semantic roles, types, heuristic confidence scores (not calibrated probabilities), and privacy-preserving pattern explanations. Corrections are preserved.
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
                      <div className="text-sm text-[var(--muted)] py-4 text-center">Loading semantic model…</div>
                    ) : (
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs border-collapse">
                          <thead>
                            <tr className="border-b border-mist text-[var(--muted)] uppercase tracking-wider text-[10px]">
                              <th className="py-2 px-2.5">Column</th>
                              <th className="py-2 px-2.5">Native Dtype</th>
                              <th className="py-2 px-2.5">Inferred Role</th>
                              <th className="py-2 px-2.5">Semantic Type</th>
                              <th className="py-2 px-2.5" title="Rule-based heuristic score, not a calibrated probability">Heuristic Score</th>
                              <th className="py-2 px-2.5">Pattern Evidence</th>
                              <th className="py-2 px-2.5 text-right">Action</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-mist/60">
                            {(datasetDetail?.columns || []).map((col) => {
                              const isEditing = editingColumn === col.name;
                              const confPercent = Math.round((col.confidence || 0) * 100);
                              return (
                                <tr key={col.name} className="hover:bg-mist/30 transition-colors">
                                  <td className="py-2.5 px-2.5 font-medium text-ink font-mono text-xs">
                                    {col.name}
                                    {col.is_user_corrected && (
                                      <span className="ml-1.5 badge bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[9px] py-0 px-1">
                                        Corrected
                                      </span>
                                    )}
                                  </td>
                                  <td className="py-2.5 px-2.5 text-[var(--muted)] font-mono text-[11px]">
                                    {col.dtype || '—'}
                                  </td>
                                  <td className="py-2.5 px-2.5">
                                    {isEditing ? (
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
                                    ) : (
                                      <span className={`badge ${getRoleBadgeClass(col.role)} capitalize`}>
                                        {col.role || 'unknown'}
                                      </span>
                                    )}
                                  </td>
                                  <td className="py-2.5 px-2.5">
                                    {isEditing ? (
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
                                    ) : (
                                      <span className="badge bg-mist text-ink/90 font-mono text-[11px]">
                                        {col.semantic_type || 'unknown'}
                                      </span>
                                    )}
                                  </td>
                                  <td className="py-2.5 px-2.5">
                                    <div
                                      className="flex items-center gap-1.5"
                                      title={
                                        col.is_user_corrected
                                          ? 'User-assigned certainty (100%)'
                                          : `Rule-based heuristic score: ${confPercent}% (not a calibrated probability)`
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
                                        placeholder="Optional correction note…"
                                        value={editForm.notes}
                                        onChange={(e) => setEditForm({ ...editForm, notes: e.target.value })}
                                      />
                                    ) : (
                                      <ul className="list-disc list-inside space-y-0.5">
                                        {(col.evidence || [col.profile_reason]).filter(Boolean).map((ev, i) => (
                                          <li key={i} className="truncate" title={ev}>
                                            {ev}
                                          </li>
                                        ))}
                                      </ul>
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
