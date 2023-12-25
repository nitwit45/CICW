// src/components/DistanceCalculator.js

import React, { useState } from 'react';
import { Form, Button } from 'react-bootstrap';
import axios from 'axios';

const DistanceCalculator = () => {
  const [locationA, setLocationA] = useState('');
  const [locationB, setLocationB] = useState('');
  const [distance, setDistance] = useState(null);

  const calculateDistance = async () => {
    try {
      const response = await axios.get(
        `https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins=${locationA}&destinations=${locationB}&key=${process.env.key}`
      );

      const result = response.data.rows[0].elements[0];
      setDistance(result.distance.text);
    } catch (error) {
      console.error('Error calculating distance:', error);
    }
  };

  return (
    <div>
      <Form>
        <Form.Group controlId="locationA">
          <Form.Label>Location A</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter location A"
            value={locationA}
            onChange={(e) => setLocationA(e.target.value)}
          />
        </Form.Group>

        <Form.Group controlId="locationB">
          <Form.Label>Location B</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter location B"
            value={locationB}
            onChange={(e) => setLocationB(e.target.value)}
          />
        </Form.Group>

        <Button variant="primary" onClick={calculateDistance}>
          Calculate Distance
        </Button>
      </Form>

      {distance && (
        <div className="mt-3">
          <p>Distance: {distance}</p>
        </div>
      )}
    </div>
  );
};

export default DistanceCalculator;
