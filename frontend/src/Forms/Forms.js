import React, { useState } from "react";
import { Form, Button, Alert } from "react-bootstrap";
import Papa from "papaparse";
import CsvTable from "../Dashboard/CsvTable"; // Make sure to use the correct path

const Forms = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const [uploadMessage, setUploadMessage] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    // Reset messages when a new file is selected
    setUploadMessage(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadMessage("No file selected");
      return;
    }

    setUploadMessage("Please Wait, Making Predictions...");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setUploadMessage("Predictions Made, view Table Below");

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
            console.error("Error parsing CSV:", error.message);
            setUploadMessage("Error parsing CSV: " + error.message);
          },
        });
      } else {
        setUploadMessage("Error uploading file. Please try again.");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setUploadMessage("Error uploading file: " + error.message);
    }
  };

  return (
    <div className="form1">
      <Form>
        <Form.Group controlId="formFile" className="mb-3">
          <Form.Label>Choose a file</Form.Label>
          <Form.Control type="file" onChange={handleFileChange} />
        </Form.Group>
        <Button variant="primary" onClick={handleUpload}>
          Upload
        </Button>
      </Form>

      {/* Display upload message */}
      {uploadMessage && <Alert variant="info">{uploadMessage}</Alert>}

      {/* Display the CSV data in a table */}
      {csvData && <CsvTable csvData={csvData} />}
    </div>
  );
};

export default Forms;
