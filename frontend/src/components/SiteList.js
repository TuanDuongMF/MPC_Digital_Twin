import React from 'react';

function SiteList({ sites, loading, selectedSite, onSiteSelect }) {
  if (loading) {
    return (
      <div className="card">
        <div className="card-header">Available Sites</div>
        <div className="card-body">
          <div className="text-center">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
            <p className="mt-2 text-muted">Loading sites...</p>
          </div>
        </div>
      </div>
    );
  }

  if (sites.length === 0) {
    return (
      <div className="card">
        <div className="card-header">Available Sites</div>
        <div className="card-body">
          <p className="text-muted mb-0">No sites found</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">Available Sites ({sites.length})</div>
      <ul className="list-group list-group-flush">
        {sites.map((site, index) => (
          <li
            key={index}
            className={`list-group-item ${
              selectedSite && selectedSite.site_name === site.site_name
                ? 'active'
                : ''
            }`}
            onClick={() => onSiteSelect(site)}
          >
            <div className="d-flex justify-content-between align-items-center">
              <div>
                <strong>{site.site_name}</strong>
                {site.site_short && (
                  <span className="text-muted ms-2">({site.site_short})</span>
                )}
              </div>
              {selectedSite && selectedSite.site_name === site.site_name && (
                <span className="badge bg-white text-primary border-0 shadow-sm fw-semibold">✓ Selected</span>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default SiteList;
