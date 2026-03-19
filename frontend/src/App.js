import React from 'react';
// import SiteList from './components/SiteList';
// import ExportButton from './components/ExportButton';
import ParseSimulationForm from './components/ParseSimulationForm';
import './App.css';

function App() {
  // Temporarily hidden: export from database functionality
  // const [sites, setSites] = useState([]);
  // const [loading, setLoading] = useState(true);
  // const [error, setError] = useState(null);
  // const [selectedSite, setSelectedSite] = useState(null);

  // useEffect(() => {
  //   fetchSites();
  // }, []);

  // const fetchSites = async () => {
  //   try {
  //     setLoading(true);
  //     setError(null);
  //     const response = await fetch('/api/sites');
  //     if (!response.ok) {
  //       throw new Error('Failed to fetch sites');
  //     }
  //     const data = await response.json();
  //     setSites(data.sites || []);
  //   } catch (err) {
  //     setError(err.message);
  //   } finally {
  //     setLoading(false);
  //   }
  // };

  // const handleSiteSelect = (site) => {
  //   setSelectedSite(site);
  // };

  return (
    <div className="App">
      <div className="container">
        <div className="row mb-4">
          <div className="col-12">
            <h1>AMT Cycle Productivity Reader</h1>
            <p className="text-muted mb-0">Import raw data files and generate simulation files</p>
          </div>
        </div>

        {/* Temporarily hidden: export from database functionality */}
        {/* {error && (
          <div className="row">
            <div className="col-12">
              <div className="alert alert-danger" role="alert">
                <strong>Error:</strong> {error}
                <button
                  className="btn btn-sm btn-outline-danger ms-2"
                  onClick={fetchSites}
                >
                  Retry
                </button>
              </div>
            </div>
          </div>
        )} */}

        <div className="row">
          <div className="col-12">
            <ParseSimulationForm />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
