import React, { useEffect, useState } from "react";
import { Map, GoogleApiWrapper, Marker } from "google-maps-react";
import "./maps.css";

const mapStyles = {
  width: "90%",
  height: "700px",
  margin: "auto",
};

const locations = [
  { lat: 40.7018013, lng: -84.88657379, label: "A" },
  { lat: 47.6038321, lng: -122.330062, label: "B" },
  { lat: 29.7589382, lng: -95.3676974, label: "C" },
  { lat: 41.7571701, lng: -88.3147539, label: "D" },
  { lat: 39.9622601, lng: -83.0007065, label: "E" },
];

const locations2 = [
  { lat: 40.7018013, lng: -84.88657379, label: "A" },
  { lat: 34.0536909, lng: -118.242766, label: "B" },
  { lat: 40.735657, lng: -74.1723667, label: "C" },
  { lat: 32.7174202, lng: -117.1627728, label: "D" },
  { lat: 39.9622601, lng: -83.0007065, label: "E" },
];

const truck1Info = {
  truckNumber: 1,
  orders: [
    { orderId: 3071, capacity: 3200 },
    { orderId: 7936, capacity: 2700 },
    { orderId: 31411, capacity: 1200 },
    { orderId: 5056, capacity: 800 },
    { orderId: 31411, capacity: 300 },
  ],
  totalCapacity: 8200,
};

const truck2Info = {
  truckNumber: 2,
  orders: [
    { orderId: 4929, capacity: 3000 },
    { orderId: 5655, capacity: 2400 },
    { orderId: 31411, capacity: 1500 },
    { orderId: 1589, capacity: 700 },
    { orderId: 5068, capacity: 600 },
  ],
  totalCapacity: 8200,
};

const MapContainer = (props) => {
  const [legDistances, setLegDistances] = useState([]);
  const [legDistances2, setLegDistances2] = useState([]);
  const [departmentId, setDepartmentId] = useState("3");
  const [date, setDate] = useState("2016-04-05")

  useEffect(() => {
    const map = new props.google.maps.Map(document.getElementById("map"), {
      zoom: 10,
      center: { lat: 40.7018013, lng: -84.88657379 },
    });

    const map2 = new props.google.maps.Map(document.getElementById("map2"), {
      zoom: 10,
      center: { lat: 40.7018013, lng: -84.88657379 },
    });

    const directionsService = new props.google.maps.DirectionsService();
    const directionsRenderer = new props.google.maps.DirectionsRenderer();
    const directionsRenderer2 = new props.google.maps.DirectionsRenderer();
    directionsRenderer.setMap(map);
    directionsRenderer2.setMap(map2);

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

    const waypoints2 = locations2.map((location) => ({
      location: new props.google.maps.LatLng(location.lat, location.lng),
      stopover: true,
    }));

    const request2 = {
      origin: waypoints2[0].location,
      destination: waypoints2[waypoints2.length - 1].location,
      waypoints: waypoints2.slice(1, waypoints2.length - 1),
      travelMode: props.google.maps.TravelMode.DRIVING,
    };

    directionsService.route(request2, (result, status) => {
      if (status === props.google.maps.DirectionsStatus.OK) {
        directionsRenderer2.setDirections(result);

        const distances2 = result.routes[0].legs.map((leg) => {
          return {
            distance: leg.distance.text,
            startLocation: leg.start_address,
            endLocation: leg.end_address,
          };
        });

        setLegDistances2(distances2);
      } else {
        console.error("Error fetching directions:", status);
      }
    });

    locations2.forEach((location) => {
      new props.google.maps.Marker({
        position: { lat: location.lat, lng: location.lng },
        map: map2,
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
