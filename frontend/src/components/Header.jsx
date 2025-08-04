import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import logo from '../assets/logo2.png';
import homeIcon from '../assets/home_icon.png';

const Header = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Get patient's name from location if provided
  const name = location.state?.name || '';

  return (
    <div className="top-bar" style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '1.5vh 10vw',
      backgroundColor: 'white',
      borderBottom: '2px solid #ccc'
    }}>
      <img
        src={homeIcon}
        alt="home"
        onClick={() => navigate('/')}
        style={{ width: 'clamp(40px, 6vw, 90px)', cursor: 'pointer' }}
      />
      <img src={logo} alt="logo" style={{ width: 'clamp(100px, 25vw, 400px)' }} />
      <div style={{
        fontWeight: 'bold',
        fontSize: 'clamp(1rem, 1.5vw, 1.6rem)',
        color: '#4b4b4b'
      }}>
        {name}
      </div>
    </div>
  );
};

export default Header;
