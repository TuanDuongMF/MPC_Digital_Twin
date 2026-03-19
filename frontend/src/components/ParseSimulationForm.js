import React, { useEffect, useRef, useState } from 'react';

function ParseSimulationForm() {
  const [fidelity, setFidelity] = useState('Low');
  const [file, setFile] = useState(null);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState(null);
  const [files, setFiles] = useState({});
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);

  const fileInputRef = useRef(null);
  const logsRef = useRef(null);

  useEffect(() => {
    if (!logsRef.current) return;
    // Always scroll to end when logs update
    logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [logs]);

  const handleFileSelect = (event) => {
    const selected = event.target.files && event.target.files[0];
    setFile(selected || null);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();

    if (running) return;

    const droppedFiles = event.dataTransfer.files;
    if (droppedFiles && droppedFiles.length > 0) {
      const selected = droppedFiles[0];
      setFile(selected);
      if (fileInputRef.current) {
        // Sync input element for consistency
        fileInputRef.current.files = droppedFiles;
      }
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

  const handleSubmit = async (event) => {
    event.preventDefault();
    setRunning(true);
    setStatus(null);
    setMessage('');
    setError(null);
    setFiles({});
    setJobId(null);
    setProgress(0);
    setLogs([]);

    try {
      const formData = new FormData();
      formData.append('fidelity', fidelity);
      if (file) {
        formData.append('file', file);
      }

      const response = await fetch('/api/simulation/parse', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || data.error || data.success === false || !data.job_id) {
        throw new Error(data.error || 'Failed to start parse job');
      }

      setJobId(data.job_id);
      setStatus('running');
      setMessage('Job started. Waiting for progress…');

      // Poll job status until completed/error
      const pollIntervalMs = 1000;
      const startedAt = Date.now();
      while (true) {
        // Avoid very long runaway polls (safety)
        if (Date.now() - startedAt > 6 * 60 * 60 * 1000) {
          throw new Error('Job polling timed out');
        }

        // eslint-disable-next-line no-await-in-loop
        const sResp = await fetch(`/api/simulation/parse/status/${data.job_id}`);
        // eslint-disable-next-line no-await-in-loop
        const sData = await sResp.json().catch(() => ({}));
        if (!sResp.ok) {
          throw new Error(sData.error || 'Failed to fetch job status');
        }

        setProgress(typeof sData.progress === 'number' ? sData.progress : 0);
        setLogs(Array.isArray(sData.logs) ? sData.logs : []);

        if (sData.status === 'completed') {
          setStatus('completed');
          setRunning(false);
          setMessage('Simulation data parsed successfully.');
          setFiles(sData.files || {});
          break;
        }
        if (sData.status === 'error') {
          setStatus('error');
          setRunning(false);
          throw new Error(sData.error || 'Job failed');
        }

        // eslint-disable-next-line no-await-in-loop
        await new Promise((r) => setTimeout(r, pollIntervalMs));
      }
    } catch (err) {
      setStatus('error');
      setError(err.message || 'Unknown error');
    } finally {
      // running is cleared on completion/error above; keep as-is here
      setRunning(false);
    }
  };

  const handleDownload = async (fileType) => {
    try {
      const url = `/api/simulation/download/${fileType}`;
      const response = await fetch(url);
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Failed to download file');
      }

      const blob = await response.blob();
      const disposition = response.headers.get('Content-Disposition');
      let filename = files[fileType] || (fileType === 'model' ? 'model.json' : `${fileType}.json`);

      if (disposition) {
        const match = disposition.match(/filename="?([^"]+)"?/);
        if (match && match[1]) {
          filename = match[1];
        }
      }

      const urlObject = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = urlObject;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(urlObject);
    } catch (err) {
      alert(err.message || 'Failed to download file');
    }
  };

  const canSubmit = !running && !!file;
  const hasOutputs = status === 'completed';

  return (
    <div className="card mb-4">
      <div className="card-header d-flex align-items-center justify-content-between">
        <div>Parse Simulation Data</div>
        <div className="parse-pill">
          <span className="parse-pill__dot" data-state={running ? 'running' : status || 'idle'} />
          <span className="parse-pill__text">
            {running ? 'Running' : status === 'completed' ? 'Completed' : status === 'error' ? 'Error' : 'Idle'}
          </span>
        </div>
      </div>

      <div className="card-body">
        <form onSubmit={handleSubmit}>
          <div className="row g-4 parse-grid">
            {/* Left: Inputs */}
            <div className="col-12 col-lg-5">
              <div className="parse-panel">
                <div className="parse-panel__title">Inputs</div>

                <div className="mb-3">
                  <label className="form-label">Fidelity</label>
                  <select
                    className="form-select"
                    value={fidelity}
                    onChange={(e) => setFidelity(e.target.value)}
                    disabled={running}
                  >
                    <option value="Low">Low</option>
                    <option value="High">High</option>
                  </select>
                </div>

                <div className="mb-0">
                  <label className="form-label">Upload Raw ZIP (required)</label>
                  <div
                    className="border border-2 border-dashed rounded p-4 text-center"
                    style={{
                      borderColor: running ? '#0d6efd' : '#dee2e6',
                      backgroundColor: running ? '#f0f7ff' : '#f8f9fa',
                      cursor: running ? 'not-allowed' : 'pointer',
                      transition: 'all 0.3s ease',
                    }}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragEnter={handleDragEnter}
                    onClick={() => !running && fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".zip"
                      style={{ display: 'none' }}
                      onChange={handleFileSelect}
                      disabled={running}
                    />

                    {running ? (
                      <div>
                        <div className="spinner-border text-primary mb-2" role="status">
                          <span className="visually-hidden">Loading...</span>
                        </div>
                        <p className="text-muted mb-0">{message || 'Processing...'}</p>
                      </div>
                    ) : (
                      <div>
                        <svg
                          width="48"
                          height="48"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          className="text-muted mb-2"
                        >
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                          <polyline points="17 8 12 3 7 8"></polyline>
                          <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <p className="mb-1">
                          <strong>Click to upload</strong> or drag and drop
                        </p>
                        <p className="text-muted small mb-0">
                          ZIP file only (supports large files up to 5GB)
                        </p>
                        {file && (
                          <p className="text-muted small mt-2 mb-0">
                            Selected:&nbsp;
                            <span className="fw-semibold">{file.name}</span>
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Middle: Parse button */}
            <div className="col-12 col-lg-2">
              <div className="parse-center">
                <button
                  type="submit"
                  className="btn btn-primary btn-lg w-100 parse-cta"
                  disabled={!canSubmit}
                >
                  {running ? 'Processing…' : 'Parse Data'}
                </button>
                <div className="text-muted small mt-2 text-center">
                  Generates model, DES inputs, and ledger
                </div>
              </div>
            </div>

            {/* Right: Outputs */}
            <div className="col-12 col-lg-5">
              <div className="parse-panel">
                <div className="parse-panel__title d-flex align-items-center justify-content-between">
                  <span>Outputs</span>
                  <span className="text-muted small">{hasOutputs ? 'Ready' : 'Not ready'}</span>
                </div>

                {(status || message || error) && (
                  <div className="mb-3">
                    {status === 'completed' && (
                      <div className="alert alert-success mb-0">
                        {message || 'Completed.'}
                      </div>
                    )}
                    {status === 'error' && (
                      <div className="alert alert-danger mb-0">
                        {error || 'Error occurred.'}
                      </div>
                    )}
                    {running && (
                      <div className="alert alert-info mb-0">
                        {message || 'Running pipeline...'}
                      </div>
                    )}
                  </div>
                )}

                {(running || status === 'running') && (
                  <div className="mb-3">
                    <div className="d-flex align-items-center justify-content-between mb-2">
                      <div className="fw-semibold">Progress</div>
                      <div className="text-muted small">{progress}%</div>
                    </div>
                    <div className="progress" style={{ height: '10px' }}>
                      <div
                        className="progress-bar"
                        role="progressbar"
                        style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
                        aria-valuenow={progress}
                        aria-valuemin="0"
                        aria-valuemax="100"
                      />
                    </div>
                    {jobId && (
                      <div className="text-muted small mt-2">
                        Job: <span className="font-monospace">{jobId.slice(0, 12)}</span>
                      </div>
                    )}
                  </div>
                )}

                {logs && logs.length > 0 && (
                  <div className="mb-3">
                    <div className="fw-semibold mb-2">Process</div>
                    <div className="parse-logs" ref={logsRef}>
                      {logs.slice(-12).map((line) => (
                        <div key={line} className="parse-logs__line">
                          {line}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {hasOutputs ? (
                  <div className="download-grid">
                    <button
                      type="button"
                      className="download-tile"
                      onClick={() => handleDownload('model')}
                      disabled={!files.model}
                    >
                      <span className="download-tile__icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="7 10 12 15 17 10" />
                          <line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                      </span>
                      <span className="download-tile__text">
                        <span className="download-tile__title">Model</span>
                        <span className="download-tile__meta">{files.model || 'model.json'}</span>
                      </span>
                    </button>

                    <button
                      type="button"
                      className="download-tile"
                      onClick={() => handleDownload('des_inputs')}
                      disabled={!files.des_inputs}
                    >
                      <span className="download-tile__icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                          <line x1="8" y1="13" x2="16" y2="13" />
                          <line x1="8" y1="17" x2="16" y2="17" />
                        </svg>
                      </span>
                      <span className="download-tile__text">
                        <span className="download-tile__title">DES Inputs</span>
                        <span className="download-tile__meta">{files.des_inputs || 'simulation_des_inputs.json.gz'}</span>
                      </span>
                    </button>

                    <button
                      type="button"
                      className="download-tile"
                      onClick={() => handleDownload('ledger')}
                      disabled={!files.ledger}
                    >
                      <span className="download-tile__icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M3 3v18h18" />
                          <path d="M7 14l3-3 3 3 5-6" />
                        </svg>
                      </span>
                      <span className="download-tile__text">
                        <span className="download-tile__title">Events</span>
                        <span className="download-tile__meta">{files.ledger || 'simulation_ledger.json.gz'}</span>
                      </span>
                    </button>
                  </div>
                ) : (
                  <div className="text-muted small">
                    Downloads will appear here when the job completes.
                  </div>
                )}
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

export default ParseSimulationForm;

