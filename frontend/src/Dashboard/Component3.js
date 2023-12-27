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
  const [truck1Capacity, setTruck1Capacity] = useState(null);
  const [truck2Capacity, setTruck2Capacity] = useState(null);

  const handleButton3 = async () => {
    try {
      setLoading(true);

      const response = await fetch('http://localhost:5000/capacity1', {
        method: 'POST',
      });

      if (!response.ok) {
        console.error('Error fetching graph:', response.statusText);
        return;
      }

      const truck_1_coordinates = await response.json();
      setTruck1Capacity(truck_1_coordinates);
    } catch (error) {
      console.error('Error fetching graph:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleButton4 = async () => {
    try {
      setLoading(true);

      const response = await fetch('http://localhost:5000/capacity2', {
        method: 'POST',
      });

      if (!response.ok) {
        console.error('Error fetching graph:', response.statusText);
        return;
      }

      const truck_2_coordinates = await response.json();
      setTruck2Capacity(truck_2_coordinates);
    } catch (error) {
      console.error('Error fetching graph:', error);
    } finally {
      setLoading(false);
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
        <Maps />
        <Button className="button2" variant="primary" onClick={handleButton3}>
          Generate Total Capacity for Truck 1
        </Button>
        <Button className="button2" variant="primary" onClick={handleButton4}>
          Generate Total Capacity for Truck 2
        </Button>
        {loading && <p>Please Wait Generating...</p>}
        {!loading && (
          <>
            {truck1Capacity && (
              <div>
                <h3>Truck 1 Capacity:</h3>
                <h1>{truck1Capacity}</h1>
                {/* You can add creative elements here for Truck 1 Capacity */}
              </div>
            )}
            {truck2Capacity && (
              <div>
                <h3>Truck 2 Capacity:</h3>
                <h1>{truck2Capacity}</h1>
                {/* You can add creative elements here for Truck 2 Capacity */}
              </div>
            )}
            <p>Generation Complete</p>
          </>
        )}
      </div>
    </div>
  );
};

export default Component3;
