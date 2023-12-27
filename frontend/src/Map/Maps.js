import React, { useEffect, useState } from "react";
import { Map, GoogleApiWrapper, Marker } from "google-maps-react";
import "./maps.css";
import { Form, Button } from "react-bootstrap";

const mapStyles = {
  width: "90%",
  height: "700px",
  margin: "auto",
};

const locations = [
  { lat: 6.964029, lng: 79.897132, label: "A" },
  { lat: 6.986521, lng: 79.936442, label: "B" },
  { lat: 7.051093, lng: 79.866576, label: "C" },
  { lat: 7.113161, lng: 79.883399, label: "D" },
  { lat: 7.178056, lng: 79.892326, label: "E" },
  { lat: 7.163409, lng: 79.983821, label: "F" },
  { lat: 6.950222, lng: 79.976905, label: "G" },
  { lat: 6.852967, lng: 80.084421, label: "H" },
  { lat: 6.795612, lng: 80.142464, label: "I" },
  { lat: 6.612645, lng: 80.220264, label: "J" },
];

const MapContainer = (props) => {
  const [legDistances, setLegDistances] = useState([]);

  useEffect(() => {
    const map = new props.google.maps.Map(document.getElementById("map"), {
      zoom: 10,
      center: { lat: 6.92696, lng: 79.86138 },
    });

    const directionsService = new props.google.maps.DirectionsService();
    const directionsRenderer = new props.google.maps.DirectionsRenderer();

    directionsRenderer.setMap(map);

    const waypoints = locations.map((location) => ({
      location: new props.google.maps.LatLng(location.lat, location.lng),
      stopover: true,
    }));

    const request = {
      origin: waypoints[0].location,
      destination: waypoints[waypoints.length - 1].location,
      waypoints: waypoints.slice(1, waypoints.length - 1),
      travelMode: props.google.maps.TravelMode.DRIVING,
    };

    directionsService.route(request, (result, status) => {
      if (status === props.google.maps.DirectionsStatus.OK) {
        directionsRenderer.setDirections(result);

        // Calculate and set leg distances
        const distances = result.routes[0].legs.map((leg) => {
          return {
            distance: leg.distance.text,
            startLocation: leg.start_address,
            endLocation: leg.end_address,
          };
        });

        setLegDistances(distances);
      } else {
        console.error("Error fetching directions:", status);
      }
    });

    locations.forEach((location) => {
      new props.google.maps.Marker({
        position: { lat: location.lat, lng: location.lng },
        map: map,
        label: location.label,
      });
    });
  }, [props.google.maps.Map]);

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
    <div>
        <Button className="button2" variant="primary" onClick={handleButton3}>
          Generate Route for Truck 1
        </Button>
        <Button className="button2" variant="primary" onClick={handleButton4}>
          Generate Route for Truck 2
        </Button>
        {loading && <p>Please Wait Generating...</p>}
        {!loading && <p>Generation Complete</p>}
      <div id="map" style={mapStyles}></div>
      <div className="leg-distances">
        <p>Leg Distances:</p>
        <ul>
          {legDistances.map((leg, index) => (
            <li key={index}>
              <span className="start-location">{leg.startLocation}</span>
              <span className="separator"> to </span>
              <span className="end-location">{leg.endLocation}</span>
              <span className="distance">: {leg.distance}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default GoogleApiWrapper({
  apiKey: "AIzaSyD1-kK-XpZFXmP9H3IoCRMr2qtblWLp1tE",
})(MapContainer);
