import React from "react";
import "./dashboard.css";
import Forms from "../Forms/Forms";

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
        <h1 className="comp-heading">ddd</h1>
        <Forms />
      </div>
    </div>
  );
};

export default Component1;
