import React, { useState } from 'react';

const OptimizationComponent = () => {
  const [result, setResult] = useState(null);

  const handleOptimizeClick = async () => {
    try {
      const response = await fetch('http://localhost:5000/optimize', {
        method: 'GET',
        mode: 'cors',
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error fetching optimization data:', error);
    }
  };

  return (
    <div>
      <h2>Optimization Component</h2>
      <button onClick={handleOptimizeClick}>Optimize</button>

      {result && (
        <div>
          <h3>Optimization Results:</h3>
          <p>Truck 1 Coordinates: {JSON.stringify(result.truck1_coordinates)}</p>
          <p>Truck 2 Coordinates: {JSON.stringify(result.truck2_coordinates)}</p>
          <p>Best Fitness: {result.best_fitness}</p>
          {/* Add other result display components as needed */}
        </div>
      )}
    </div>
  );
};

export default OptimizationComponent;
