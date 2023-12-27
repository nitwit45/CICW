import React, { useEffect, useState } from "react";
import Maps from "../Map/Maps";
import { Form, Button } from "react-bootstrap";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Papa from "papaparse";
import CsvTable from "../Dashboard/CsvTable"; // Make sure to use the correct path
// import "./form.css";

const Component1 = () => {
  const [loading, setLoading] = useState(false);

  const handleButton3 = async () => {
    try {
      setLoading(true); // Set loading state to true when starting the request

      const response = await fetch('http://localhost:5000/route', {
        method: 'POST',
      });

      if (!response.ok) {
        console.error('Error fetching graph:', response.statusText);
        return;
      }

      const truck_1_coordinates = await response.json();

      console.log('Truck1 Coordinates:', truck_1_coordinates);
      // window.openc(objectURL, '_blank');
    } catch (error) {
      console.error('Error fetching graph:', error);
    } finally {
      setLoading(false); // Set loading state to false when the request is complete
    }
  };

  const handleButton4 = async () => {
    try {
      setLoading(true); // Set loading state to true when starting the request

      const response = await fetch('http://localhost:5000/truck2', {
        method: 'POST',
      });

      if (!response.ok) {
        console.error('Error fetching graph:', response.statusText);
        return;
      }

      const truck_2_coordinates = await response.json();

      console.log('Truck2 Coordinates:', truck_2_coordinates);
      // window.openc(objectURL, '_blank');
    } catch (error) {
      console.error('Error fetching graph:', error);
    } finally {
      setLoading(false); // Set loading state to false when the request is complete
    }
  };

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
        <Button className="button2" variant="primary" onClick={handleButton3}>
          Generate Route for Truck 1
        </Button>
        <Button className="button2" variant="primary" onClick={handleButton4}>
          Generate Route for Truck 2
        </Button>
        {loading && <p>Please Wait Generating...</p>}
        <Maps />
      </div>
    </div>
  );
};

export default Component1;