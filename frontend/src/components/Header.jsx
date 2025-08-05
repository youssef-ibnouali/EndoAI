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
        onClick={() => navigate('/home')}
        style={{ width: 'clamp(40px, 6vw, 90px)', cursor: 'pointer' }}
        onMouseOver={e => e.currentTarget.style.filter = 'brightness(0.85)'}
        onMouseOut={e => e.currentTarget.style.filter = ''}
      />
      <img src={logo} style={{ width: 'clamp(100px, 25vw, 400px)', cursor: 'pointer' }} alt="logo" onClick={() => navigate('/')}  />
      <div style={{
        fontWeight: 'bold',
        fontSize: 'clamp(1rem, 1.5vw, 1.6rem)',
        color: '#000000ff'
      }}>
        {name}
      </div>
    </div>
  );
};

export default Header;
