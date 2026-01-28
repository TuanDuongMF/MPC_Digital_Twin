import React, { useState, useRef, useEffect } from 'react';

function ImportButton({ site, onImportComplete }) {
  const [importing, setImporting] = useState(false);
  const [status, setStatus] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState(null);
  const [exportStatus, setExportStatus] = useState(null);
  const [exportFiles, setExportFiles] = useState(null);
  const fileInputRef = useRef(null);
  const statusPollIntervalRef = useRef(null);
  
  const DEFAULT_SITE_NAME = 'DefaultSite';

  const handleFileSelect = (event) => {
    const files = event.target.files;
    if (files.length === 0) {
      return;
    }
    handleImport(files);
  };

  useEffect(() => {
    // Cleanup polling on unmount
    return () => {
      if (statusPollIntervalRef.current) {
        clearInterval(statusPollIntervalRef.current);
      }
    };
  }, []);

  const pollImportStatus = async () => {
    try {
      const response = await fetch(`/api/import/status/${encodeURIComponent(DEFAULT_SITE_NAME)}`);
      if (response.ok) {
        const data = await response.json();
        setExportStatus(data.status);
        setMessage(data.message || '');
        
        if (data.status === 'completed') {
          setStatus('completed');
          setExportFiles(data.files || {});
          if (statusPollIntervalRef.current) {
            clearInterval(statusPollIntervalRef.current);
            statusPollIntervalRef.current = null;
          }
        } else if (data.status === 'error') {
          setStatus('error');
          setError(data.message || 'Export failed');
          if (statusPollIntervalRef.current) {
            clearInterval(statusPollIntervalRef.current);
            statusPollIntervalRef.current = null;
          }
        }
      }
    } catch (err) {
      console.error('Error polling import status:', err);
    }
  };

  const handleDownloadFile = async (fileType, filename) => {
    try {
      const response = await fetch(
        `/api/import/download/${encodeURIComponent(DEFAULT_SITE_NAME)}/${fileType}`
      );
      if (!response.ok) {
        throw new Error('Download failed');
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename || `${fileType}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      alert(`Failed to download ${fileType}: ${err.message}`);
    }
  };

  const handleImport = async (files) => {
    try {
      setImporting(true);
      setStatus('uploading');
      setMessage('Uploading files...');
      setError(null);
      setExportStatus(null);
      setExportFiles(null);

      // Create FormData for multipart/form-data upload
      const formData = new FormData();
      formData.append('site_name', DEFAULT_SITE_NAME);
      formData.append('export', 'true');
      
      // Add all files to FormData
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }

      // Upload files with progress tracking
      const xhr = new XMLHttpRequest();

      // Track upload progress
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          const percentComplete = Math.round((e.loaded / e.total) * 100);
          setMessage(`Uploading... ${percentComplete}%`);
        }
      });

      // Handle response
      xhr.addEventListener('load', () => {
        if (xhr.status === 202) {
          // Export started (always expected since export=true)
          try {
            JSON.parse(xhr.responseText);
          } catch (err) {
            setStatus('error');
            setError('Failed to parse response');
            setImporting(false);
            return;
          }

          setStatus('processing');
          setMessage('Import completed. Exporting simulation files...');

          // Start polling for export status
          if (statusPollIntervalRef.current) {
            clearInterval(statusPollIntervalRef.current);
          }
          statusPollIntervalRef.current = setInterval(() => {
            pollImportStatus();
          }, 2000);

          // Initial poll
          pollImportStatus();
          setImporting(false);
        } else {
          let serverMessage = `Server error: ${xhr.status}`;
          try {
            const errorResponse = JSON.parse(xhr.responseText);
            serverMessage = errorResponse.error || serverMessage;
          } catch (err) {
            // ignore JSON parse errors and keep default message
          }
          setStatus('error');
          setError(serverMessage);
          setImporting(false);
        }
      });

      xhr.addEventListener('error', () => {
        setImporting(false);
        setStatus('error');
        setError('Network error occurred');
      });

      xhr.addEventListener('abort', () => {
        setImporting(false);
        setStatus('error');
        setError('Upload cancelled');
      });

      // Start upload
      xhr.open('POST', '/api/import');
      setStatus('parsing');
      setMessage('Parsing files...');
      xhr.send(formData);

    } catch (err) {
      setImporting(false);
      setStatus('error');
      setError(err.message);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      handleImport(files);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDragEnter = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const getStatusBadgeClass = () => {
    switch (status) {
      case 'completed':
        return 'bg-success';
      case 'uploading':
      case 'parsing':
        return 'bg-info';
      case 'error':
        return 'bg-danger';
      default:
        return 'bg-secondary';
    }
  };

  return (
    <div className="card">
      <div className="card-header">Import Raw Data</div>
      <div className="card-body">
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
          </div>
        )}

        {error && (
          <div className="alert alert-danger mb-4" role="alert">
            <strong>Error:</strong> {error}
          </div>
        )}

        {exportStatus === 'completed' && exportFiles && (
          <div className="mb-3 p-2 bg-success bg-opacity-10 rounded">
            <div className="d-flex align-items-center justify-content-between flex-wrap gap-2">
              <div className="d-flex align-items-center">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-success me-2">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                  <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
                <strong className="text-success small">Export Completed</strong>
              </div>
              <div className="d-flex flex-wrap gap-2">
                {exportFiles.model && (
                  <button
                    className="btn btn-sm btn-outline-primary"
                    onClick={() => handleDownloadFile('model', exportFiles.model)}
                    title="Download Model File"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="me-1">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="7 10 12 15 17 10"></polyline>
                      <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Model
                  </button>
                )}
                {exportFiles.des_inputs && (
                  <button
                    className="btn btn-sm btn-outline-primary"
                    onClick={() => handleDownloadFile('des_inputs', exportFiles.des_inputs)}
                    title="Download DES Inputs File"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="me-1">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="7 10 12 15 17 10"></polyline>
                      <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    DES Inputs
                  </button>
                )}
                {exportFiles.ledger && (
                  <button
                    className="btn btn-sm btn-outline-primary"
                    onClick={() => handleDownloadFile('ledger', exportFiles.ledger)}
                    title="Download Events Ledger File"
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="me-1">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="7 10 12 15 17 10"></polyline>
                      <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Ledger
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        <div
          className="border border-2 border-dashed rounded p-4 text-center mb-3"
          style={{
            borderColor: importing ? '#0d6efd' : '#dee2e6',
            backgroundColor: importing ? '#f0f7ff' : '#f8f9fa',
            cursor: importing ? 'not-allowed' : 'pointer',
            transition: 'all 0.3s ease'
          }}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragEnter={handleDragEnter}
          onClick={() => !importing && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            style={{ display: 'none' }}
            onChange={handleFileSelect}
            disabled={importing}
          />
          {importing ? (
            <div>
              <div className="spinner-border text-primary mb-2" role="status">
                <span className="visually-hidden">Loading...</span>
              </div>
              <p className="text-muted mb-0">{message}</p>
            </div>
          ) : (
            <div>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted mb-2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              <p className="mb-1">
                <strong>Click to upload</strong> or drag and drop
              </p>
              <p className="text-muted small mb-0">
                ZIP file or multiple raw data files (supports large files up to 5GB)
              </p>
              <p className="text-muted small mt-1 mb-0">
                <em>Note: For folder upload, compress it as ZIP first</em>
              </p>
            </div>
          )}
        </div>

        {(status === 'completed' || exportStatus === 'completed') && (
          <button
            className="btn btn-secondary w-100"
            onClick={() => {
              setStatus(null);
              setMessage('');
              setError(null);
              setExportStatus(null);
              setExportFiles(null);
              if (statusPollIntervalRef.current) {
                clearInterval(statusPollIntervalRef.current);
                statusPollIntervalRef.current = null;
              }
              if (fileInputRef.current) {
                fileInputRef.current.value = '';
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

export default ImportButton;
