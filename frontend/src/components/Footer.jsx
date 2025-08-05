import React, { useState } from "react";
import "./Footer.css";

export default function Footer() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: ""
  });

  const [messageSent, setMessageSent] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Here you could send the data via fetch or axios if needed
    console.log("Message sent:", formData);

    setMessageSent(true);
    setFormData({ name: "", email: "", subject: "", message: "" });

    setTimeout(() => setMessageSent(false), 3000); // remove confirmation after 3 seconds
  };

  return (
    <footer className="footer">
      {/* RIGHT side: Credits & Partners */}
      <div className="footer-credits">
        <div className="footer-section partners">
          <br />
          <div className="footer-column about">
            <img src="./src/assets/logo2.png" alt="EndoAI" className="footer-logo" />
            <p>
              EndoAI is a medical image analysis tool using AI to detect gastric anomalies from endoscopic images. Built for diagnostic assistance.
            </p>
          </div>
          <br />
          <a href="https://polytech.univ-nantes.fr" target="_blank" rel="noopener noreferrer">
            <img src="./src/assets/polynantes_logo.png" alt="Partner 1" />
          </a>
          <a href="https://mta.ro" target="_blank" rel="noopener noreferrer">
            <img src="./src/assets/mta_logo.png" alt="Partner 2" />
          </a>
          <a href="https://www.scumc.ro" target="_blank" rel="noopener noreferrer">
            <img src="./src/assets/hospital_logo.jpg" alt="Partner 3" />
          </a>
        </div>
        <div className="footer-section credits">
          <p>© 2025 Youssef IBNOUALI – EndoAI Project @ MTA Bucharest</p>
          <p>All rights reserved | Legal mentions to be added</p>
        </div>
      </div>
      {/* LEFT side: Contact Form */}
      <div className="footer-contact">
        <div className="footer-section contact">
          <p><b> contact@endo.ai |  +33 6 12 34 56 78 |  Bucharest, ROMANIA</b></p>
          <p>Contact the support team :</p>
        </div>
        <form className="contact-form" onSubmit={handleSubmit}>
          <input
            type="text"
            name="name"
            placeholder="Name"
            value={formData.name}
            onChange={handleChange}
            required
          />
          <input
            type="email"
            name="email"
            placeholder="Email"
            value={formData.email}
            onChange={handleChange}
            required
          />
          <input
            type="text"
            name="subject"
            placeholder="Subject"
            value={formData.subject}
            onChange={handleChange}
            required
          />
          <textarea
            name="message"
            placeholder="Message"
            rows="3"
            value={formData.message}
            onChange={handleChange}
            required
          ></textarea>
          <button type="submit">Send</button>
          {messageSent && <p style={{ color: "green", marginTop: "0.5rem" }}>✅ Message sent!</p>}
        </form>
      </div>
    </footer>
  );
}
