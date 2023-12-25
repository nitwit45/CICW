import "./App.css";
import { BrowserRouter as Router, Route, Routes, Navigate } from "react-router-dom";
import Login from "./Signin/Login";  
import SignUp from "./Signin/SignUp";
import Dashboard from "./Dashboard/Dashboard";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  return (
    <div className="App">
    <Router>
      <Routes>
        <Route path="/" element={<Navigate to="/login" />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Router>
    </div>
  );
}

export default App;
