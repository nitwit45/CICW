import React, { useState } from "react";
import { Form, Button } from "react-bootstrap";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Papa from "papaparse";
import CsvTable from "../Dashboard/CsvTable"; // Make sure to use the correct path
import "./form.css";

const Forms = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [csvData, setCsvData] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      console.log("No file selected");
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.text();

      // Parse the CSV data using papaparse
      Papa.parse(result, {
        header: true,
        complete: function (parsedResult) {
          console.log("Parsed CSV Data:", parsedResult.data);

          // Set the parsed CSV data to state
          setCsvData(parsedResult.data);
        },
        error: function (error) {
          console.error('Error parsing CSV:', error.message);
        },
      });
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handleButton = async () => {
    if (!selectedFile) {
      console.log("No file selected");
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/button', {
        method: 'POST',
        body: formData,
      });
      const result = await response.text();

      // Parse the CSV data using papaparse
      Papa.parse(result, {
        header: true,
        complete: function (parsedResult) {
          console.log("Parsed CSV Data:", parsedResult.data);

          // Set the parsed CSV data to state
          setCsvData(parsedResult.data);
        },
        error: function (error) {
          console.error('Error parsing CSV:', error.message);
        },
      });
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };
  const handleButton2 = async () => {
    try {
      const response = await fetch('http://localhost:5000/button2', {
        method: 'POST',
      });
  
      if (!response.ok) {
        console.error('Error fetching graph:', response.statusText);
        return;
      }
  
      const blob = await response.blob();
      const objectURL = URL.createObjectURL(blob);
  
      // Display the graph by opening it in a new tab
      window.open(objectURL, '_blank');
    } catch (error) {
      console.error('Error fetching graph:', error);
    }
  };
  const handleButton3 = async () => {
    try {
      const response = await fetch('http://localhost:5000/button3', {
        method: 'POST',
      });
  
      if (!response.ok) {
        console.error('Error fetching graph:', response.statusText);
        return;
      }
  
      const blob = await response.blob();
      const objectURL = URL.createObjectURL(blob);
  
      // Display the graph by opening it in a new tab
      window.open(objectURL, '_blank');
    } catch (error) {
      console.error('Error fetching graph:', error);
    }
  };

  return (
    <div className="form1">
      <Container>
        <Row>
          <Col></Col>
          <Col xs={6}>
            <div className="sub-container">
              <Form>
                <Form.Group controlId="formFile" className="mb-3">
                  <Form.Label>Choose a file</Form.Label>
                  <Form.Control type="file" onChange={handleFileChange} />
                </Form.Group>
                <Button variant="primary" onClick={handleUpload}>
                  Upload
                </Button>
              </Form>
            </div>
          </Col>
          <Col></Col>
        </Row>
      </Container>

      {/* Display the CSV data in a table */}
      {csvData && <CsvTable csvData={csvData} />}
      <Button className="button3" variant="primary" onClick={handleButton}>
        View GA RESULTS for Different Countries
      </Button>
      <Col></Col>
      <Button className="button2" variant="primary" onClick={handleButton2}>
        View Convergence Plot
      </Button>      
      
      <Button className="button2" variant="primary" onClick={handleButton3}>
        View Pareto Front Plot
      </Button>
    </div>
  );
};

export default Forms;
