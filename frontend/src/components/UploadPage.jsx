import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import logo from '../assets/logo2.png';
import homeIcon from '../assets/home_icon.png';
import nextIcon from '../assets/next_red_icon.png';
import returnIcon from '../assets/return_icon.png';
import Footer from "./Footer";
import Header from './Header';

const UploadPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const name = location.state?.name || 'Unknown';

  const [image, setImage] = useState(null);
  const [imageName, setImageName] = useState('');
  const [resultImgUrl, setResultImgUrl] = useState(null);
  const [scores, setScores] = useState(null);
  const [dominantClass, setDominantClass] = useState('');

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setImageName(file?.name || '');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;

    const formData = new FormData();
    formData.append('image', image);
    formData.append('name', name);

    try {
      const response = await axios.post('http://localhost:5000/classify', formData);
      setScores(response.data.scores);
      setResultImgUrl(`http://localhost:5000${response.data.result_img}?t=${Date.now()}`);
      setDominantClass(response.data.diagnosis);
    } catch (err) {
      console.error('Upload error:', err);
    }
  };

  const formatPercent = (val) => `${(val || 0).toFixed(2)}%`;

  return (
    <>
    <Header />
    <div className="background" style={{ 
        backgroundColor: '#e6f6ff',
        height: '100vh',
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
        <form onSubmit={handleSubmit} encType="multipart/form-data" style={{ flex: 1, maxWidth: '12vw' }}>
          <label className="custom-upload" style={{
            width: 'clamp(150px, 18vw, 170px)', padding: '12px 18px',
            borderRadius: '20px', backgroundColor: '#424141', color: 'white',
            fontWeight: 'bold', display: 'block', cursor: 'pointer', textAlign: 'center'
          }}>
            upload image
            <input type="file" hidden required accept="image/*" onChange={handleImageChange} />
          </label>
          <div style={{ fontStyle: 'italic', marginTop: '1vh', color:"gray" }}>{imageName || 'No file selected'}</div>
          <button type="submit" className="process-btn" style={{
            marginTop: '4vh', height: 50, width: 'clamp(150px, 18vw, 210px)',
            borderRadius: '22px', backgroundColor: '#ab3c3e', color: 'white',
            fontWeight: 'bold', fontSize: 'clamp(1rem, 1.2vw, 1.3rem)', border: 'none'
          }}>Process</button>
        </form>

        {/* RESULT DISPLAY */}
        {scores && (
          <div className="result-box" style={{ flex: 2, textAlign: 'center' }}>
            <img src={resultImgUrl} alt="result" style={{
              width: 'clamp(50px, 50vw, 1200px)', borderRadius: 8, marginTop: '0vh'
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
            <button onClick={() => navigate('/report', { state: { name, dominantClass } })}
              style={{
                marginTop: '1vh',
                background: 'none',
                border: 'none',
                cursor: 'pointer'
              }}>
              <img src={nextIcon} alt="continue" style={{ width: 'clamp(40px, 7vw, 120px)' }} />
            </button>
          </div>
        )}
      </div>

      {/* RETURN BUTTON */}
      <button onClick={() => navigate('/home')} style={{
        position: 'relative', bottom: '-40vh', left: '-47vw', background: 'none', border: 'none', zIndex: 1000
      }}>
        <img src={returnIcon} alt="return" style={{ width: 'clamp(40px, 5vw, 90px)' }} />
      </button>
    </div>
    <Footer />
    </>
  );
};

export default UploadPage;
