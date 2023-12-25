import React, { useState } from "react";
import "./signup.css";
import Img from "../assets/login.png";
import FloatingLabel from "react-bootstrap/FloatingLabel";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import { useNavigate, Link } from "react-router-dom";

const SignUp = () => {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");

  const handleSignIn = () => {
    if (email && password && name) {
      navigate("/dashboard");
    } else {
      alert("Please fill in both email and password fields.");
    }
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
            label="🧑 Name"
            className="mb-3"
          >
            <Form.Control
              type="text"
              placeholder="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </FloatingLabel>
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
            <h3>SignUp</h3>
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
          <Link to="/login">
            <h6>Login</h6>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default SignUp;
