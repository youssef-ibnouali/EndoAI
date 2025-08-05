import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logo1.png';
import Footer from "./Footer";
import Header from './Header';
import axios from 'axios';


const LoginPage = () => {
  const [organization, setOrg] = useState('');
  const [organizations, setOrganizations] = useState([]);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

    useEffect(() => {
    const fetchOrganizations = async () => {
        try {
        const response = await axios.get('http://localhost:5000/organizations');
        const orgs = Array.isArray(response.data) ? response.data : [];
        setOrganizations(orgs);
        setOrg(orgs[0] || '');
        } catch (err) {
        console.error("Failed to load organizations", err);
        const fallback = [
            "Military Technical Academy 'Ferdinand I' of Bucharest",
            "Central Military Universitary Emergency Hospital 'Carol Davila' of Bucharest",
            "Other"
        ];
        setOrganizations(fallback);
        setOrg(fallback[0]);
        }
    };
    fetchOrganizations();
    }, []);
    
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/login', {
        organization,
        username,
        password,
      });

      if (response.data.success) {
        navigate('/home', { state: { organization, username } });
      } else {
        setError('Invalid credentials');
      }
    } catch (err) {
      setError('Login failed. Please try again.');
    }
  };

  return (
    <>
      <Header />
      <div style={{
        backgroundColor: '#e6f6ff',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        position: 'relative'
      }}>
        <a
          href="https://www.youtube.com/watch?v=zR6D5o5bIdU&list=PLeP7OZvLMPPfgWw8kHGZzPNhurUBbTbz3"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            position: 'absolute',
            top: '18px',
            right: '32px',
            marginTop: 0,
            display: 'inline-block',
            color: '#d10d10ff',
            textDecoration: 'underline',
            fontSize: '0.95rem',
            background: 'transparent',
            border: 'none',
            cursor: 'pointer',
            transition: 'color 0.2s'
          }}
          onMouseEnter={e => e.target.style.color = '#525252ff'}
          onMouseLeave={e => e.target.style.color = '#d10d10ff'}
        >
          Watch Tutorial
        </a>

        <div style={{ textAlign: 'center', width: '100%', maxWidth: '480px' }}>
          <img src={logo} alt="logo" style={{ width: 'clamp(100px, 25vw, 400px)', marginBottom: '3vh' }} />

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.2vh' }}>
            <select value={organization} onChange={(e) => setOrg(e.target.value)} required style={inputStyle}>
              {Array.isArray(organizations) && organizations.map((option, idx) => (
                <option key={idx} value={option}>{option}</option>
              ))}
            </select>

            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              style={inputStyle}
            />

            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={inputStyle}
            />

            <button
              type="submit"
              style={{
                ...buttonStyle,
                transition: 'background 0.2s',
              }}
              onMouseEnter={e => e.target.style.backgroundColor = '#8c2c2e'}
              onMouseLeave={e => e.target.style.backgroundColor = '#ab3c3e'}
            >
              Login
            </button>
            {error && <p style={{ color: 'red' }}>{error}</p>}
          </form>
        </div>
      </div>
      <Footer />
    </>
  );
};

const inputStyle = {
  padding: '10px',
  fontSize: '1rem',
  borderRadius: '6px',
  border: '1px solid #aaa'
};

const buttonStyle = {
  padding: '12px',
  fontWeight: 'bold',
  backgroundColor: '#ab3c3e',
  color: 'white',
  border: 'none',
  borderRadius: '6px',
  cursor: 'pointer'
};

export default LoginPage;
