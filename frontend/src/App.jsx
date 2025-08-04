import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import SplashScreen from "./components/SplashScreen";
import LoginPage from './components/LoginPage';
import HomePage from "./components/HomePage";
import UploadPage from "./components/UploadPage";
import ReportPage from "./components/ReportPage";

function App() {
  return (
    <div style={{ fontFamily: "'Potta One', cursive" }}>
      {/* Load Potta One from Google Fonts */}
      <link
        href="https://fonts.googleapis.com/css2?family=Potta+One&display=swap"
        rel="stylesheet"
      />

      <Router>
        <Routes>
          <Route path="/" element={<SplashScreen />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/report" element={<ReportPage />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
