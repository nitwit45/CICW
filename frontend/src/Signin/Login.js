import React, { useState } from "react";
import "./login.css";
import Img from "../assets/login.png";
import FloatingLabel from "react-bootstrap/FloatingLabel";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import { useNavigate, Link } from "react-router-dom";
import Modal from "react-bootstrap/Modal";

const Login = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showAlert, setShowAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState("");

  const handleSignIn = () => {
    if (email === "testuser@gmail.com" && password === "abc@123") {
      navigate("/dashboard");
    } else if (email === "" || password === "") {
      setAlertMessage("Please enter email and password.");
      setShowAlert(true);
    } else {
      setAlertMessage("Wrong email or password. Please try again.");
      setShowAlert(true);
    }
  };

  const handleCloseAlert = () => {
    setShowAlert(false);
  };

  return (
    <div className="login-container">
      <div className="heading">
        <h1>Welcome back</h1>
      </div>

      <div className="logo">
        <img src={Img} alt="logo" />
      </div>
      <div className="centered-form">
        <div className="form-container">
          <FloatingLabel
            controlId="floatingInput"
            label="📧 Email address"
            className="mb-3"
          >
            <Form.Control
              type="email"
              placeholder="name@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </FloatingLabel>
          <FloatingLabel controlId="floatingPassword" label="🔑 Password">
            <Form.Control
              size="sm"
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </FloatingLabel>
          <div className="button-container">
            <h3>Sign In</h3>
            <Button
              variant="outline-light"
              style={{ borderRadius: "50px" }}
              className="button"
              onClick={handleSignIn}
            >
              →
            </Button>
          </div>
        </div>
        <div className="signup">
          <Link to="/signup">
            <h6>Sign Up</h6>
          </Link>
          <h6>Forgot Password</h6>
        </div>
      </div>

      {/* Alert Modal */}
      <Modal show={showAlert} onHide={handleCloseAlert} centered>
        <Modal.Header closeButton>
          <Modal.Title>Alert</Modal.Title>
        </Modal.Header>
        <Modal.Body>{alertMessage}</Modal.Body>
        <Modal.Footer>
          <Button variant="primary" onClick={handleCloseAlert}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
};

export default Login;
