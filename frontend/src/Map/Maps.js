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
  { lat: 39.9622601, lng:  -83.0007065, label: "E" },
];

const MapContainer = (props) => {
  const [legDistances, setLegDistances] = useState([]);

  useEffect(() => {
    const map = new props.google.maps.Map(document.getElementById("map"), {
      zoom: 10,
      center: { lat: 40.7018013, lng: -84.88657379 },
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
