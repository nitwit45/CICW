import React, { useState } from "react";
import { Table, Button } from "react-bootstrap";

const CsvTable = ({ csvData }) => {
  const [showMore, setShowMore] = useState(false);
  const initialRowCount = 5;

  if (!csvData || csvData.length === 0) {
    return <div>No data to display</div>;
  }

  const headers = Object.keys(csvData[0]);
  const visibleRows = showMore ? csvData.length : initialRowCount;

  return (
    <div className="table-container">
      <Table responsive striped bordered hover>
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {csvData.slice(0, visibleRows).map((row, index) => (
            <tr key={index}>
              {headers.map((header) => (
                <td key={header}>{row[header]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </Table>
      {csvData.length > initialRowCount && (
        <Button onClick={() => setShowMore(!showMore)}>
          {showMore ? "Show Less" : "Show More"}
        </Button>
      )}
    </div>
  );
};

export default CsvTable;
