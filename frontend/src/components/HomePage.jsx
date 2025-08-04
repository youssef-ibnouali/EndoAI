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

  return (
    <>
    <Header />
    <div style={{
      backgroundColor: '#e6f6ff',
      fontFamily: 'Arial, sans-serif',
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
              <button type="submit" style={{ background: 'none', border: 'none', padding: 0 }}>
                <img src={nextIcon} alt="next" style={{ width: 'clamp(35px, 10vw, 80px)', cursor: 'pointer' }} />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
    <Footer />
    </>
    
  );
};

export default HomePage;
