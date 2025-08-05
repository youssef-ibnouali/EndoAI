// src/pages/PatientDataPage.jsx
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import Footer from "./Footer";
import Header from './Header';
import returnIcon from '../assets/return_icon.png';

const diagnoses = ['Normal', 'AG', 'IM', 'Dysplasia', 'Cancer'];

const PatientDataPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const query = new URLSearchParams(location.search);
  const name = query.get("name");
  const patientName = name || 'Unknown';


  const [records, setRecords] = useState([]);
  const [newDiagnosis, setNewDiagnosis] = useState('Normal');
  const [newComment, setNewComment] = useState('');

  const fetchRecords = async () => {
    try {
      const res = await axios.get(`https://8vh7qbwt-5000.euw.devtunnels.ms/patient_records?name=${encodeURIComponent(patientName)}`);
      setRecords(res.data.records || []);
    } catch (err) {
      console.error(err);
    }
  };

  const handleAddRecord = async () => {
    try {
      await axios.post('https://8vh7qbwt-5000.euw.devtunnels.ms/patient_records', {
        name: patientName,
        diagnosis: newDiagnosis,
        comments: newComment
      });
      setNewComment('');
      setNewDiagnosis('Normal');
      fetchRecords();
    } catch (err) {
      console.error(err);
    }
  };

  const handleDelete = async (index) => {
    try {
      await axios.delete(`https://8vh7qbwt-5000.euw.devtunnels.ms/patient_records?name=${encodeURIComponent(patientName)}&index=${index}`);
      fetchRecords();
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    if (patientName && patientName !== 'Unknown') {
    fetchRecords();
  }

  }, [patientName]);

return (
    <>
    <Header />
    <div
        style={{
            padding: 30,
            backgroundColor: '#e6f6ff',
            minHeight: '100vh',
            width: '100vw',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center'
        }}
    >
        <h2 style={{ color: 'black', fontSize: 'clamp(30px, 3vh, 70px)' }}>
            Patient: {patientName}
        </h2>

        <div
            style={{
                marginTop: 20,
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'center',
                gap: '10px',
                flexWrap: 'wrap'
            }}
        >
            <h4
                style={{
                    color: '#ed5356',
                    fontSize: 'clamp(20px, 2vh, 55px)',
                    marginRight: '2vw'
                }}
            >
                Add new record :
            </h4>
            <select
                style={{
                    height: 'clamp(30px, 2vh, 30px)',
                    fontSize: 'clamp(18px, 2vh, 20px)'
                }}
                value={newDiagnosis}
                onChange={(e) => setNewDiagnosis(e.target.value)}
            >
                {diagnoses.map((d) => (
                    <option key={d} value={d}>
                        {d}
                    </option>
                ))}
            </select>
            <input
                type="text"
                placeholder="Comments (optional)"
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                style={{
                    marginLeft: 10,
                    padding: 4,
                    width: '300px',
                    height: 'clamp(20px, 2vh, 30px)'
                }}
            />
            <button
                onClick={handleAddRecord}
                style={{
                    marginLeft: 15,
                    backgroundColor: '#093c00ff',
                    color: 'white',
                    border: 'none',
                    padding: '8px 16px',
                    borderRadius: '4px',
                    transition: 'background 0.2s',
                    cursor: 'pointer'
                }}
                onMouseOver={e => (e.currentTarget.style.backgroundColor = '#145c00')}
                onMouseOut={e => (e.currentTarget.style.backgroundColor = '#093c00ff')}
            >
                Add
            </button>
        </div>

        <h4
            style={{
                marginTop: 40,
                color: '#ed5356',
                fontSize: 'clamp(25px, 2vh, 60px)'
            }}
        >
            History :
        </h4>
        <table border="1" cellPadding={8}>
            <thead>
                <tr style={{ color: 'black' }}>
                    <th>Date</th>
                    <th>Diagnosis</th>
                    <th>Comments</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {records.map((record, index) => (
                    <tr key={index} style={{ color: '#343333ff' }}>
                        <td>{record.date}</td>
                        <td>{record.diagnosis}</td>
                        <td>{record.comments}</td>
                        <td>
                            <button
                                style={{
                                    backgroundColor: '#5a0000ff',
                                    color: 'white',
                                    border: 'none',
                                    padding: '6px 12px',
                                    borderRadius: '4px',
                                    transition: 'background 0.2s',
                                    cursor: 'pointer'
                                }}
                                onClick={() => handleDelete(index)}
                                onMouseOver={e => (e.currentTarget.style.backgroundColor = '#a00000')}
                                onMouseOut={e => (e.currentTarget.style.backgroundColor = '#5a0000ff')}
                            >
                                Delete
                            </button>
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>

        <button
            onClick={() => navigate('/home')}
            style={{
                background: 'none',
                border: 'none',
                position: 'absolute',
                bottom: '2vh',
                left: '4vw',
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

export default PatientDataPage;
