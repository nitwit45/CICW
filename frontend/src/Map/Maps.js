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

  return (
    <div className="container">
      <div className="truck-info-container">
        <div className="truck-info">
          <h2>Truck 1 Info:</h2>
          <table className="center-table">
            <thead>
              <tr>
                <th>Order ID</th>
                <th>Capacity</th>
              </tr>
            </thead>
            <tbody>
              {truck1Info.orders.map((order) => (
                <tr key={order.orderId}>
                  <td>{order.orderId}</td>
                  <td>{order.capacity}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="capacity">Total capacity of Truck 1: {truck1Info.totalCapacity}</p>
        </div>
      </div>
      <div id="map" style={mapStyles}></div>
      <div className="leg-distances">
        <p>Distances covered(by truck 1):</p>
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
      <div className="truck-info-container">
        <div className="truck-info">
          <h2>Truck 2 Info:</h2>
          <table className="center-table">
            <thead>
              <tr>
                <th>Order ID</th>
                <th>Capacity</th>
              </tr>
            </thead>
            <tbody>
              {truck2Info.orders.map((order) => (
                <tr key={order.orderId}>
                  <td>{order.orderId}</td>
                  <td>{order.capacity}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="capacity">Total capacity of Truck 2: {truck2Info.totalCapacity}</p>
        </div>
      </div>
      <div id="map2" style={mapStyles}></div>
      <div className="leg-distances">
        <p>Distances covered(by truck 2):</p>
        <ul>
          {legDistances2.map((leg, index) => (
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
