import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';

export default function Datasets() {
  const { token } = useAuth();
  const [datasets, setDatasets] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadName, setUploadName] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState('');

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

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="page-kicker">Data</p>
          <h1 className="page-title">Datasets</h1>
          <p className="page-sub">Spreadsheets and CSVs you connect to decision projects.</p>
        </div>
        <button type="button" onClick={() => setShowUpload(true)} className="btn-primary">
          Upload dataset
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

      {datasets.length === 0 ? (
        <div className="empty-state">
          <h3 className="font-display text-xl font-semibold">No datasets yet</h3>
          <p className="text-sm text-[var(--muted)] mt-2 mb-6">Upload a CSV to get started.</p>
          <button type="button" className="btn-primary" onClick={() => setShowUpload(true)}>
            Upload dataset
          </button>
        </div>
      ) : (
        <ul className="divide-y divide-mist border-y border-mist">
          {datasets.map((dataset) => (
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
                <span className="badge bg-teal-soft/50 text-ink border border-teal/20">Ready</span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
