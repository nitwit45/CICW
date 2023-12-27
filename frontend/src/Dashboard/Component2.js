import React from "react";
import "./dashboard.css";
import Maps from "../Map/Maps";
import TruckInfo from "../Dashboard/TruckInfo"

const Component2 = () => {
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
        <h2 className="comp-heading" >Interface2</h2>
        {/* <Maps /> */}
        <TruckInfo />
      </div>
    </div>
  );
};

export default Component2;
