import React from 'react';
import './Footer.css';

function Footer() {
  return (
    <footer className="app-footer">
      <div className="footer-content">
        <div className="footer-brand">
          <div className="footer-logo">
            MovieNex <span>RecSys</span>
          </div>
          <p className="footer-desc">
            This is a personal demonstration project showcasing modern Movie Recommendation Systems. 
            It implements personal and similar item recommenders to suggest movies in real time.
          </p>
        </div>
        
        <div className="footer-links-grid" style={{ display: 'flex', flexWrap: 'wrap', gap: '40px', flex: '2 1 600px' }}>
          <div className="footer-links-col" style={{ flex: '1 1 200px' }}>
            <h3>Project Resources</h3>
            <ul>
              <li>
                <a href="https://github.com" target="_blank" rel="noopener noreferrer" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <svg viewBox="0 0 24 24" fill="currentColor" style={{ width: '16px', height: '16px' }}>
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                  </svg>
                  GitHub Repository
                </a>
              </li>
              <li>
                <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer">
                  Backend API Docs (Swagger)
                </a>
              </li>
            </ul>
          </div>
          
          <div className="footer-links-col" style={{ flex: '2 1 350px', maxWidth: '500px' }}>
            <h3>Contributors</h3>
            <div style={{ overflowX: 'auto', marginTop: '10px' }}>
              <table className="footer-contributors-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13.5px', color: 'rgba(255,255,255,0.85)' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.15)', textAlign: 'left', color: 'var(--tmdbLightBlue)' }}>
                    <th style={{ padding: '0 12px 8px 0', fontWeight: 700 }}>Name</th>
                    <th style={{ padding: '0 0 8px 12px', fontWeight: 700 }}>Role</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    <td style={{ padding: '10px 12px 10px 0', fontWeight: 700, whiteSpace: 'nowrap' }}>Le Dinh Duc</td>
                    <td style={{ padding: '10px 0 10px 12px', color: 'rgba(255,255,255,0.75)' }}>Big Data Engineer</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '10px 12px 10px 0', fontWeight: 700, whiteSpace: 'nowrap' }}>Huynh Le Duy Khanh</td>
                    <td style={{ padding: '10px 0 10px 12px', color: 'rgba(255,255,255,0.75)' }}>Data Scientist</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
      
      <div className="footer-bottom">
        <p>&copy; {new Date().getFullYear()} MovieNex RecSys Demo. Personal Project.</p>
      </div>
    </footer>
  );
}

export default Footer;