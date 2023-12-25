import React, { useState } from "react";
import "./dashboard.css";
import Component1 from "./Component1";
import Component2 from "./Component2";
import Component3 from "./Component3";
import { useNavigate } from "react-router-dom";

const Content = ({ selectedTab }) => {
  const navigate = useNavigate();

  switch (selectedTab) {
    case 0:
      return <Component1 />;
    case 1:
      return <Component2 />;
    case 2:
      return <Component3 />;
    case 3:
      navigate("/login");
      return null;
    default:
      return null;
  }
};

const Tabs = ({ tabs, selectedTab, onTabChange }) => {
  const icons = ["fa-chart-bar", "fa-search", "fa-map", "fa-sign-out-alt"];

  return (
    <div className="tabs">
      {tabs.map((tab, index) => (
        <div
          key={index}
          className={`tab ${selectedTab === index ? "active" : ""}`}
          onClick={() => onTabChange(index)}
        >
          <i className={`fas ${icons[index]}`}></i>&nbsp;{" "}
          {/* Add a space here */}
          {tab}
        </div>
      ))}
    </div>
  );
};

const Dashboard = () => {
  const tabs = ["Interface1", "Interface2", "Interface3", "Logout"];
  const [selectedTab, setSelectedTab] = useState(0);

  return (
    <div className="dashboard">
      <Tabs
        tabs={tabs}
        selectedTab={selectedTab}
        onTabChange={setSelectedTab}
      />
      <div className="content-container">
        <Content selectedTab={selectedTab} />
      </div>
    </div>
  );
};

export default Dashboard;
