import React, { useState } from "react";
import { Form, Button } from "react-bootstrap";
import Papa from "papaparse";
import CsvTable from "../Dashboard/CsvTable"; // Make sure to use the correct path

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

      {/* Display the CSV data in a table */}
      {csvData && <CsvTable csvData={csvData} />}
    </div>
  );
};

export default Forms;
