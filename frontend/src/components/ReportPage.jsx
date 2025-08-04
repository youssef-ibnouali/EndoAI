import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import logo from '../assets/logo1.png';
import homeIcon from '../assets/home_icon.png';
import docIcon from '../assets/doc_icon.png';
import returnIcon from '../assets/return_icon.png';
import Footer from "./Footer";
import Header from './Header';

const ReportPage = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const name = location.state?.name || 'Unknown';
  const dominantClass = location.state?.dominantClass || 'Uncertain';

    const handleDownload = () => {
    window.open(`http://localhost:5000/generate_report?name=${name}&diagnosis=${dominantClass}`, '_blank');
    };

  return (
    <>
    <Header />
    <div style={{
      backgroundColor: '#e6f6ff',
      fontFamily: 'Arial, sans-serif',
      height: '100vh',
      width: '100vw',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between'
    }}>
      {/* Centered Content */}
      <div className="centered-content" style={{
        flexGrow: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center'
      }}>
        <img src={logo} alt="logo" style={{ width: 'clamp(100px, 20vw, 350px)', marginBottom: '3vh' }} />

        <p className="report-label" style={{
          fontSize: 'clamp(1rem, 1.8vw, 1.5rem)',
          fontWeight: 'bold',
          marginBottom: '2vh',
          color: '#484747'
        }}>Check the medical report :</p>

        <button onClick={handleDownload} style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer'
        }}>
          <img src={docIcon} alt="Download PDF" title="Download PDF Report" style={{
            width: 'clamp(40px, 6vw, 80px)',
            marginBottom: '4vh'
          }} />
        </button>

        {/* Return Button */}
        <button onClick={() => navigate(-1)} style={{
          position: 'fixed',
          bottom: '5vh',
          left: '4vw',
          background: 'none',
          border: 'none',
          cursor: 'pointer'
        }}>
          <img src={returnIcon} alt="Return" style={{ width: 'clamp(40px, 5vw, 90px)' }} />
        </button>
      </div>
    </div>
    <Footer />
    </>
  );
};

export default ReportPage;
