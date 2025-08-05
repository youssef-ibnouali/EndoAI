import React, { useState } from 'react';
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
  const diagnosis = location.state?.diagnosis|| 'Uncertain';
  const confidence = location.state?.confidence || null; 
  const resultImgUrl = location.state?.resultImgUrl || null;
  const [showPopup, setShowPopup] = useState(false);


    const handleDownload = () => {
    window.open(`http://localhost:5000/generate_report?name=${name}&diagnosis=${diagnosis}`, '_blank');
    };

  return (
    <>
    <Header />
    <div style={{
      backgroundColor: '#e6f6ff',
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
        {/*<img src={logo} alt="logo" style={{ width: 'clamp(70px, 15vw, 350px)', marginBottom: '5vh' }} />*/}
        {diagnosis  && (
          <p style={{ fontSize: 'clamp(1rem, 1.8vw, 1.5rem)', fontWeight: 'bold', color: '#000000' }}>
            Diagnosis:{" "}
            <span style={{
              color:
                diagnosis === 'Cancer' || diagnosis === 'Start of Cancer' ? '#890089' :
                diagnosis === 'Dysplasia' ? '#c20000' :
                diagnosis === 'IM' || diagnosis === 'AG with early signs of IM' ? '#ae9714' :
                diagnosis === 'Start of AG' || diagnosis === 'AG' ? '#009cc7' :
                diagnosis === 'Normal' ? '#018f48' :
                '#000000'
            }}>
              {diagnosis}
            </span>
          </p>
        )}

      {resultImgUrl && (
        <>
          {/* Small Image Preview */}
          <img
            src={resultImgUrl}
            alt="result"
            onClick={() => setShowPopup(true)}
            style={{
              width: 'clamp(30px, 7vw, 800px)',
              borderRadius: 200,
              marginTop: '0vh',
              cursor: 'pointer',
              boxShadow: '0 0 8px rgba(0, 0, 0, 0.2)',
              transition: 'transform 0.2s'
            }}
            title="Click to view full size"
          />

          {/* Fullscreen Popup */}
          {showPopup && (
            <div
              onClick={() => setShowPopup(false)}
              style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100vw',
                height: '100vh',
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                zIndex: 9999,
                cursor: 'zoom-out'
              }}
            >
              <img
                src={resultImgUrl}
                alt="Full Size Result"
                style={{
                  maxWidth: '90vw',
                  maxHeight: '90vh',
                  borderRadius: '12px',
                  boxShadow: '0 0 20px rgba(255,255,255,0.4)'
                }}
              />
            </div>
          )}
        </>
      )}




      {confidence !== null && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          width: '100%',
          maxWidth: '400px' 
        }}>
          <p style={{
            fontSize: 'clamp(1rem, 1.8vw, 1.5rem)',
            fontWeight: 'bold',
            color: '#000000',
            whiteSpace: 'nowrap',
            marginRight: '1rem'
          }}>
            AI confidence level : {confidence.toFixed(1)}%
          </p>
          <div style={{
            flexGrow: 1,
            backgroundColor: '#949494ff',
            borderRadius: '10px',
            overflow: 'hidden',
            height: '20px',
            minWidth: '200px'
          }}>
            <div style={{
              width: `${confidence}%`,
              backgroundColor:
                confidence > 75 ? '#4caf50' :
                confidence > 50 ? '#ff9800' :
                '#f44336',
              height: '100%',
              transition: 'width 0.4s ease'
            }} />
          </div>
        </div>
      )}

      <div style={{
        display: 'flex',
        alignItems: 'center',
         // spacing between text and icon
          marginBottom: '2vh'
              }}>
          <p className="report-label" style={{
            fontSize: 'clamp(1rem, 1.8vw, 1.5rem)',
            fontWeight: 'bold',
            color: '#000000',
            margin: 0
          }}>
            Check the medical report:
          </p>

          <button
            onClick={handleDownload}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: 0,
              marginLeft: '2.5rem'
            }}
            onMouseOver={e => (e.currentTarget.firstChild.style.transform = 'scale(1.1)')}
            onMouseOut={e => (e.currentTarget.firstChild.style.transform = 'scale(1)')}
          >
            <img src={docIcon} alt="Download PDF" title="Download PDF Report" style={{
              width: 'clamp(40px, 4vw, 60px)',
              transition: 'transform 0.2s'
            }} />
          </button>
              </div>


          {/* Return Button */}

        <button
            onClick={() => navigate(-1)}
            style={{
                background: 'none',
                border: 'none',
                position: 'absolute',
                bottom: '2vh',
                left: '5vw',
                cursor: 'pointer',
                transition: 'transform 0.2s'
            }}
            onMouseOver={e => (e.currentTarget.firstChild.style.transform = 'scale(1.1)')}
            onMouseOut={e => (e.currentTarget.firstChild.style.transform = 'scale(1)')}
        >
            <img
                src={returnIcon}
                alt="return"
                style={{ width: 'clamp(40px, 5vw, 90px)', transition: 'transform 0.2s' }}
            />
        </button>

      </div>
    </div>
    <Footer />
    </>
  );
};

export default ReportPage;
