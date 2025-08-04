import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import videoIntro from '../assets/logo_intro.mp4';

const SplashScreen = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigate('/login');
    }, 4000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div style={{
      backgroundColor: '#e6f6ff',
      height: '100vh',
      width: '100vw',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      <video autoPlay muted playsInline style={{ width: '90vw', maxWidth: 1500 }}>
        <source src={videoIntro} type="video/mp4" />
      </video>
    </div>
  );
};

export default SplashScreen;
