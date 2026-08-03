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
    if (token) {
      fetchDatasets();
    }
  }, [token]);

  const fetchDatasets = async () => {
    try {
      const res = await fetch('/api/datasets', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        setDatasets(await res.json());
      }
    } catch (err) {
      console.error('Error fetching datasets:', err);
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
        body: formData
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
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-brand-dark">Datasets</h1>
          <p className="text-gray-600">Upload and manage your data</p>
        </div>
        <button
          onClick={() => setShowUpload(true)}
          className="px-4 py-2 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Upload Dataset
        </button>
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 w-full max-w-md">
            <h2 className="text-xl font-bold mb-4">Upload Dataset</h2>
            
            {error && (
              <div className="bg-red-100 text-red-700 px-4 py-2 rounded-lg mb-4">{error}</div>
            )}
            
            <form onSubmit={handleUpload} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Dataset Name</label>
                <input
                  type="text"
                  value={uploadName}
                  onChange={(e) => setUploadName(e.target.value)}
                  className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-brand-teal"
                  placeholder="Customer Churn Data"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">CSV File</label>
                <div className="border-2 border-dashed rounded-lg p-6 text-center">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setSelectedFile(e.target.files[0])}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    {selectedFile ? (
                      <span className="text-brand-teal-dark">{selectedFile.name}</span>
                    ) : (
                      <>
                        <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                        <p className="mt-2 text-gray-600">Click to select a CSV file</p>
                      </>
                    )}
                  </label>
                </div>
              </div>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => setShowUpload(false)}
                  className="flex-1 px-4 py-2 border rounded-lg hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={uploading || !selectedFile}
                  className="flex-1 px-4 py-2 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal disabled:opacity-50"
                >
                  {uploading ? 'Uploading...' : 'Upload'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Dataset List */}
      {datasets.length === 0 ? (
        <div className="bg-gray-50 rounded-xl p-12 text-center">
          <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <h3 className="text-lg font-medium text-brand-dark mb-2">No datasets yet</h3>
          <p className="text-gray-600 mb-4">Upload a CSV file to get started with predictions</p>
          <button
            onClick={() => setShowUpload(true)}
            className="px-4 py-2 bg-brand-teal-dark text-white rounded-lg hover:bg-brand-teal"
          >
            Upload Your First Dataset
          </button>
        </div>
      ) : (
        <div className="grid gap-4">
          {datasets.map((dataset) => (
            <div key={dataset.id} className="bg-white rounded-xl p-6 shadow-sm border">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-lg font-semibold text-brand-dark">{dataset.name}</h3>
                  <p className="text-gray-500 text-sm mt-1">
                    {dataset.row_count.toLocaleString()} rows • {dataset.column_count} columns
                  </p>
                </div>
                <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">
                  Ready
                </span>
              </div>
              <div className="mt-4 flex gap-2 flex-wrap">
                {dataset.columns?.slice(0, 5).map(col => (
                  <span key={col} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-sm">
                    {col}
                  </span>
                ))}
                {dataset.column_count > 5 && (
                  <span className="px-2 py-1 text-gray-500 text-sm">
                    +{dataset.column_count - 5} more
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
