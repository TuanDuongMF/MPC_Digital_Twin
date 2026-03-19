import React, { useState } from 'react';

function GenerateModelFromDbButton({ onComplete }) {
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState(null);
  const [stages, setStages] = useState(null);
  const [downloadAvailable, setDownloadAvailable] = useState(false);

  const [siteId, setSiteId] = useState('1');
  const [siteName, setSiteName] = useState('DefaultSite');
  const [fidelity, setFidelity] = useState('Low');

  const handleRunPipeline = async () => {
    try {
      setRunning(true);
      setStatus('processing');
      setMessage('Generating model from database...');
      setError(null);
      setStages(null);

      const response = await fetch('/api/model/generate-from-db', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          site_id: siteId,
          site_name: siteName,
          fidelity: fidelity || undefined,
        }),
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Model generation failed');
      }

      setStatus('completed');
      setMessage('Model generated successfully from database.');
      setStages(data.stages || {});
      setDownloadAvailable(true);
      if (onComplete) {
        onComplete(data);
      }
    } catch (err) {
      setStatus('error');
      setError(err.message);
    } finally {
      setRunning(false);
    }
  };

  const handleDownload = async (fileType) => {
    try {
      const url = `/api/model/download/${fileType}`;
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Download failed');
      }
      const blob = await response.blob();
      const dlUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = dlUrl;
      a.download = 'model.json';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(dlUrl);
      document.body.removeChild(a);
    } catch (err) {
      alert(`Failed to download ${fileType}: ${err.message}`);
    }
  };

  const getStatusBadgeClass = () => {
    switch (status) {
      case 'completed':
        return 'bg-success';
      case 'processing':
        return 'bg-info';
      case 'error':
        return 'bg-danger';
      default:
        return 'bg-secondary';
    }
  };

  return (
    <div className="card mb-4">
      <div className="card-header">Generate Model from Database</div>
      <div className="card-body">
        <div className="mb-3">
          <label className="form-label">Site ID</label>
          <input
            type="text"
            className="form-control"
            value={siteId}
            onChange={(e) => setSiteId(e.target.value)}
            placeholder="e.g. 1"
          />
        </div>
        <div className="mb-3">
          <label className="form-label">Site Name</label>
          <input
            type="text"
            className="form-control"
            value={siteName}
            onChange={(e) => setSiteName(e.target.value)}
            placeholder="e.g. DefaultSite"
          />
        </div>
        <div className="mb-3">
          <label className="form-label">Fidelity</label>
          <select
            className="form-select"
            value={fidelity}
            onChange={(e) => setFidelity(e.target.value)}
          >
            <option value="Low">Low</option>
            <option value="High">High</option>
          </select>
        </div>
        {status && (
          <div className="mb-4 p-3 bg-light rounded">
            <div className="d-flex justify-content-between align-items-center mb-2">
              <div className="d-flex align-items-center">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted me-2">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                <span className="text-muted me-2">Status:</span>
                <span className={`badge ${getStatusBadgeClass()} status-badge`}>
                  {status}
                </span>
              </div>
            </div>
            {message && (
              <p className="text-muted small mb-0">{message}</p>
            )}
            {stages && (
              <ul className="mt-2 mb-0 small text-muted">
                {Object.entries(stages).map(([name, ok]) => (
                  <li key={name}>
                    {name}: {ok ? 'OK' : 'FAILED'}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {error && (
          <div className="alert alert-danger mb-4" role="alert">
            <strong>Error:</strong> {error}
          </div>
        )}

        <button
          className="btn btn-primary w-100 mb-2"
          onClick={handleRunPipeline}
          disabled={running}
        >
          {running ? (
            <>
              <span
                className="spinner-border spinner-border-sm me-2"
                role="status"
                aria-hidden="true"
              ></span>
              Generating...
            </>
          ) : (
            'Generate Model from DB'
          )}
        </button>

        {downloadAvailable && status === 'completed' && (
          <div className="d-flex gap-2 mt-2">
            <button
              className="btn btn-outline-primary btn-sm flex-fill"
              onClick={() => handleDownload('model')}
            >
              Download Model
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default GenerateModelFromDbButton;

