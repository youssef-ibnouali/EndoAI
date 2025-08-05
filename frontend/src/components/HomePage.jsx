import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logo1.png';
import nextIcon from '../assets/next_black_icon.png';
import Footer from "./Footer";
import Header from './Header';


const HomePage = () => {
  const [name, setName] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (name.trim()) {
      navigate('/upload', { state: { name } });
    }
  };
  const handleCheckRecords = () => {
    if (name.trim()) {
      navigate(`/patient-data?name=${encodeURIComponent(name.trim())}`);
    }
  };

  return (
    <>
    <Header />
    <div style={{
      backgroundColor: '#e6f6ff',
      height: '100vh',
      width: '100vw',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '2vh'
    }}>
      <div style={{ textAlign: 'center', width: '100%', maxWidth: 600 }}>
        <img src={logo} alt="logo" style={{ width: 'clamp(100px, 25vw, 400px)' }} />
        <div style={{ marginTop: '5vh' }}>
          <p style={{
            fontSize: 'clamp(1rem, 2vw, 1.3rem)',
            fontWeight: 'bold',
            color: '#484747',
            marginBottom: '1.5vh',
            marginRight: '8vh'
          }}>
            Enter your patient's full name:
          </p>
          <form onSubmit={handleSubmit}>
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '1.2vw'
            }}>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                style={{
                  padding: '0.8em',
                  width: 'clamp(200px, 60vw, 400px)',
                  fontSize: 'clamp(1rem, 2vw, 1.2rem)',
                  borderRadius: 8,
                  border: '2px solid #ccc'
                }}
              />

              <button
                type="button"
                onClick={handleCheckRecords}
                className="custom-btn-record"
              >
                Patient's Record
              </button>

              <button
                type="submit"
                className="custom-btn-start"
              >
                Start
              </button>

            </div>
          </form>
        </div>
      </div>
    </div>
    <Footer />
    <style>
      {`
        .custom-btn-record {
          background-color: #484747;
          color: white;
          padding: 0.5em 1.5em;
          border-radius: 8px;
          border: none;
          cursor: pointer;
          transition: background 0.2s;
          box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        }
        .custom-btn-record:hover {
          background-color: #333;
        }
        .custom-btn-start {
          background-color: #ab3c3e;
          color: white;
          padding: 0.5em 1.5em;
          border-radius: 8px;
          border: none;
          cursor: pointer;
          transition: background 0.2s;
          box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        }
        .custom-btn-start:hover {
          background-color: #8a2a2c;
        }
      `}
    </style>
    </>
  );
};

export default HomePage;
