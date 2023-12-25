import React from "react";
import Maps from "../Map/Maps";

const Component1 = () => {
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
      </div>
    </div>
  );
};

export default Component1;
