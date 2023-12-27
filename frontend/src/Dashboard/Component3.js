import React, { useEffect, useState } from "react";
import Maps from "../Map/Maps";
import { Form, Button } from "react-bootstrap";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Papa from "papaparse";
import CsvTable from "../Dashboard/CsvTable"; // Make sure to use the correct path
// import "./form.css";

const Component3 = () => {
  const [loading, setLoading] = useState(false);

  

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
        textAlign: "center",
      }}
    >
      <div style={{ flex: 1, textAlign: "center" }}>
        <h1 className="comp-heading">Interface3</h1>
        <Maps />
      </div>
    </div>
  );
};

export default Component3;