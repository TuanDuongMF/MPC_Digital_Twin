import React, { useState, useEffect, useRef } from 'react';

function ExportButton({ site, onExportComplete }) {
  const [exporting, setExporting] = useState(false);
  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [files, setFiles] = useState({});
  const eventSourceRef = useRef(null);

  // Export file type selection
  const [exportModel, setExportModel] = useState(true);
  const [exportSimulation, setExportSimulation] = useState(true);

  useEffect(() => {
    // Check if there's an existing export in progress on mount
    checkInitialStatus();
  }, [site]);

  useEffect(() => {
    // Cleanup SSE on unmount
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);

  const checkInitialStatus = async () => {
    try {
      const response = await fetch(`/api/export/status/${encodeURIComponent(site.site_name)}`);
      if (response.ok) {
        const data = await response.json();
        setStatus(data.status);
        setProgress(data.progress || 0);
        setMessage(data.message || '');
        setFiles(data.files || {});

        if (data.status === 'processing') {
          setExporting(true);
          startSSE();
        }
      }
    } catch (err) {
      console.error('Error checking status:', err);
    }
  };

  const startSSE = () => {
    // Close existing connection if any
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const eventSource = new EventSource(`/api/export/events/${encodeURIComponent(site.site_name)}`);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setStatus(data.status);
        setProgress(data.progress || 0);
        setMessage(data.message || '');
        setFiles(data.files || {});

        if (data.status === 'completed') {
          setExporting(false);
          eventSource.close();
          eventSourceRef.current = null;
          if (onExportComplete) {
            onExportComplete();
          }
        } else if (data.status === 'error') {
          setExporting(false);
          eventSource.close();
          eventSourceRef.current = null;
        }
      } catch (err) {
        console.error('Error parsing SSE data:', err);
      }
    };

    eventSource.onerror = () => {
      // SSE connection closed or error
      eventSource.close();
      eventSourceRef.current = null;
    };
  };

  const handleExport = async () => {
    try {
      setExporting(true);
      setStatus('processing');
      setProgress(0);
      setMessage('Starting export...');
      setFiles({});

      const response = await fetch('/api/export', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          site_name: site.site_name,
          config: {
            limit: 100000,
            sample_interval: 5,
          },
          export_model: exportModel,
          export_simulation: exportSimulation,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Export failed');
      }

      // Start SSE for real-time status updates
      startSSE();
    } catch (err) {
      setExporting(false);
      setStatus('error');
      setMessage(err.message);
    }
  };

  const handleDownload = async (fileType) => {
    try {
      const response = await fetch(
        `/api/export/download/${encodeURIComponent(site.site_name)}/${fileType}`
      );
      if (!response.ok) {
        throw new Error('Download failed');
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = files[fileType] || `${fileType}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
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
    <div className="card">
      <div className="card-header">Export Files</div>
      <div className="card-body">
        <div className="mb-4">
          <div className="d-flex align-items-center mb-2">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted me-2">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
              <circle cx="12" cy="10" r="3"></circle>
            </svg>
            <strong className="text-muted">Site:</strong>
          </div>
          <div className="ms-4">
            <span className="fs-5 fw-semibold">{site.site_name}</span>
            {site.site_short && (
              <span className="text-muted ms-2">({site.site_short})</span>
            )}
          </div>
        </div>

        {/* Export Options */}
        <div className="export-options">
          <div className="export-options-header">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="me-2">
              <path d="M4 22h14a2 2 0 0 0 2-2V7.5L14.5 2H6a2 2 0 0 0-2 2v4"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <path d="M3 15h6"></path>
              <path d="M6 12v6"></path>
            </svg>
            <span className="export-options-title">Export Options</span>
          </div>
          <div className="export-options-grid">
            <div
              className={`export-option-item ${exportModel ? 'active' : ''} ${exporting || (exportModel && !exportSimulation) ? 'disabled' : ''}`}
              onClick={() => {
                if (exporting) return;
                // Prevent turning off if it's the only one enabled
                if (exportModel && !exportSimulation) return;
                setExportModel(!exportModel);
              }}
            >
              <div className="export-option-info">
                <div className="export-option-icon model">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <path d="M12 18v-6"></path>
                    <path d="M9 15h6"></path>
                  </svg>
                </div>
                <div className="export-option-text">
                  <span className="export-option-label">Model</span>
                  <span className="export-option-desc">Site structure & config</span>
                </div>
              </div>
              <label className="toggle-switch" onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={exportModel}
                  onChange={(e) => {
                    // Prevent turning off if it's the only one enabled
                    if (!e.target.checked && !exportSimulation) return;
                    setExportModel(e.target.checked);
                  }}
                  disabled={exporting}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
            <div
              className={`export-option-item ${exportSimulation ? 'active' : ''} ${exporting || (exportSimulation && !exportModel) ? 'disabled' : ''}`}
              onClick={() => {
                if (exporting) return;
                // Prevent turning off if it's the only one enabled
                if (exportSimulation && !exportModel) return;
                setExportSimulation(!exportSimulation);
              }}
            >
              <div className="export-option-info">
                <div className="export-option-icon simulation">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 3v18h18"></path>
                    <path d="M18 17V9"></path>
                    <path d="M13 17V5"></path>
                    <path d="M8 17v-3"></path>
                  </svg>
                </div>
                <div className="export-option-text">
                  <span className="export-option-label">Simulation</span>
                  <span className="export-option-desc">DES inputs & events</span>
                </div>
              </div>
              <label className="toggle-switch" onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={exportSimulation}
                  onChange={(e) => {
                    // Prevent turning off if it's the only one enabled
                    if (!e.target.checked && !exportModel) return;
                    setExportSimulation(e.target.checked);
                  }}
                  disabled={exporting}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
          </div>
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
              {progress > 0 && <span className="fw-semibold">{progress}%</span>}
            </div>
            {message && (
              <p className="text-muted small mb-2 mb-0">{message}</p>
            )}
            {status === 'processing' && (
              <div className="progress mt-3">
                <div
                  className="progress-bar progress-bar-striped progress-bar-animated"
                  role="progressbar"
                  style={{ width: `${progress}%` }}
                  aria-valuenow={progress}
                  aria-valuemin="0"
                  aria-valuemax="100"
                >
                  {progress}%
                </div>
              </div>
            )}
          </div>
        )}

        {status === 'error' && (
          <div className="alert alert-danger" role="alert">
            <strong>Error:</strong> {message}
          </div>
        )}

        {status === 'completed' && Object.keys(files).length > 0 && (
          <div className="mb-4 p-3 bg-success bg-opacity-10 rounded">
            <div className="d-flex align-items-center mb-3">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-success me-2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                <polyline points="22 4 12 14.01 9 11.01"></polyline>
              </svg>
              <strong className="text-success">Generated Files:</strong>
            </div>
            <div className="d-flex flex-wrap gap-2">
              {files.model && (
                <button
                  className="btn btn-outline-primary btn-sm"
                  onClick={() => handleDownload('model')}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="me-1">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                  </svg>
                  Model
                </button>
              )}
              {files.des_inputs && (
                <button
                  className="btn btn-outline-primary btn-sm"
                  onClick={() => handleDownload('des_inputs')}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="me-1">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                  </svg>
                  DES Inputs
                </button>
              )}
              {files.ledger && (
                <button
                  className="btn btn-outline-primary btn-sm"
                  onClick={() => handleDownload('ledger')}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="me-1">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                  </svg>
                  Ledger
                </button>
              )}
            </div>
          </div>
        )}

        <button
          className="btn btn-primary btn-export"
          onClick={handleExport}
          disabled={exporting}
        >
          {exporting ? (
            <>
              <span
                className="spinner-border spinner-border-sm me-2"
                role="status"
                aria-hidden="true"
              ></span>
              Exporting...
            </>
          ) : (
            'Export Files'
          )}
        </button>

        {status === 'completed' && (
          <button
            className="btn btn-secondary btn-export"
            onClick={() => {
              setStatus(null);
              setProgress(0);
              setMessage('');
              setFiles({});
              if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
              }
            }}
          >
            Reset
          </button>
        )}
      </div>
    </div>
  );
}

export default ExportButton;
