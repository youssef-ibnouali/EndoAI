import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import nextIcon from '../assets/next_red_icon.png';
import returnIcon from '../assets/return_icon.png';
import Footer from "./Footer";
import Header from './Header';
import Loading from '../assets/loading.gif';

const UploadPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const name = location.state?.name || 'Unknown';

  const [image, setImage] = useState(null);
  const [imageName, setImageName] = useState('');
  const [resultImgUrl, setResultImgUrl] = useState(null);
  const [scores, setScores] = useState(null);
  const [diagnosis, setDiagnosis] = useState('');
  const [confidence, setConfidence] = useState(null);

  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setImageName(file?.name || '');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;

    setLoading(true);
    setScores(null);
    const formData = new FormData();
    formData.append('image', image);
    formData.append('name', name);

    try {
      const response = await axios.post('http://localhost:5000/classify', formData);
      setScores(response.data.scores);
      setResultImgUrl(`http://localhost:5000${response.data.result_img}?t=${Date.now()}`);
      setDiagnosis(response.data.diagnosis);
      setConfidence(response.data.confidence);
      

    } catch (err) {
      console.error('Upload error:', err);
    } finally {
      setLoading(false);
    }
  };


  const formatPercent = (val) => `${(val || 0).toFixed(2)}%`;


  return (
    <>
    <Header />
    <div className="background" style={{ 
        backgroundColor: '#e6f6ff',
        minHeight: '100vh',
        width: '100vw',
        padding: '2vh 3vw',
        display: 'flex',
        flexDirection: 'column'
        }}>

      {/* CONTENT SECTION */}
      <div className="content-section" style={{
        display: 'flex', gap: '2vw', flexWrap: 'wrap', marginTop: '10vh'
      }}>
        {/* FORM */}
        <form
          onSubmit={handleSubmit}
          encType="multipart/form-data"
          style={{
            flex: 1,
            maxWidth: '12vw',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            marginLeft: '2vw'
          }}
        >
          {/* Upload Button */}
          <label
            className="custom-upload"
            style={{
              width: 'clamp(150px, 18vw, 170px)',
              padding: '12px 18px',
              borderRadius: '20px',
              backgroundColor: '#424141',
              color: 'white',
              fontWeight: 'bold',
              textAlign: 'center',
              cursor: 'pointer',
              boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
              transition: 'all 0.3s ease'
            }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#2c2c2c'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#424141'}
          >
            upload image
            <input
              type="file"
              hidden
              required
              accept="image/*"
              onChange={handleImageChange}
            />
          </label>

          {/* File name */}
          <div style={{
            fontStyle: 'italic',
            marginTop: '1vh',
            color: 'gray',
            textAlign: 'center',
            maxWidth: '100%'
          }}>
            {imageName || 'No file selected'}
          </div>

          {/* Process Button */}
          <button
            type="submit"
            className="process-btn"
            style={{
              marginTop: '4vh',
              height: 50,
              width: 'clamp(150px, 18vw, 210px)',
              borderRadius: '22px',
              backgroundColor: '#ab3c3e',
              color: 'white',
              fontWeight: 'bold',
              fontSize: 'clamp(1rem, 1.2vw, 1.3rem)',
              border: 'none',
              cursor: 'pointer',
              boxShadow: '0 6px 12px rgba(0, 0, 0, 0.2)',
              transition: 'all 0.3s ease'
            }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#922e32'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#ab3c3e'}
          >
            Process
          </button>
        </form>

        {loading && (
          <div style={{ marginTop: '3vh', textAlign: 'center' }}>
            <img src={Loading} alt="Loading..." style={{ width: '100px', marginLeft : '35vw', marginTop: '10vh' }} />
          </div>
        )}

        {/* RESULT DISPLAY */}
        {!loading && scores && (
          <div className="result-box" style={{ flex: 2, textAlign: 'center' }}>
            <img src={resultImgUrl} alt="result" style={{
              width: 'clamp(50px, 43vw, 1000px)', borderRadius:2, marginTop: '0vh'
            }} />
            <div className="score-bar" style={{
                /*backgroundColor: 'white',*/ 
                padding: 12, borderRadius: 6,
                marginTop: '2vh', fontWeight: 'bold', fontSize: 'clamp(0.8rem, 1.2vw, 1.3rem)', marginBottom: '1vh'
              }}>
                <span style={{ color: '#ae9714' }}> IM: {formatPercent(scores.IM)}</span> &nbsp;|
                <span style={{ color: '#009cc7' }}> AG: {formatPercent(scores.AG)}</span> &nbsp;|
                <span style={{ color: '#018f48' }}> Normal: {formatPercent(scores.Normal)}</span> &nbsp;|
                <span style={{ color: '#c20000' }}> Dysplasia: {formatPercent(scores.Dysplasia)}</span> &nbsp;|
                <span style={{ color: '#890089' }}> Cancer: {formatPercent(scores.Cancer)}</span>
              </div>

              {/* NEXT BUTTON */}
              <button
                onClick={() => navigate('/report', {
                state: { name, diagnosis, confidence, resultImgUrl }
                })}
                style={{
                marginTop: '1vh',
                background: 'none',
                border: 'none',
                cursor: 'pointer'
                }}
                onMouseOver={e => (e.currentTarget.firstChild.style.transform = 'scale(1.1)')}
                onMouseOut={e => (e.currentTarget.firstChild.style.transform = 'scale(1)')}
              >
                <img src={nextIcon} alt="continue" style={{ width: 'clamp(40px, 8vw, 130px)', transition: 'transform 0.2s' }} />
              </button>
              </div>
            )}
            </div>

            {/* RETURN BUTTON */}

          <button
              onClick={() => navigate('/home')}
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
    <Footer />
    </>
  );
};

export default UploadPage;
