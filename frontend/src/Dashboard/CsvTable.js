import React, { useState } from "react";
import { Table, Button } from "react-bootstrap";

const CsvTable = ({ csvData }) => {
  const [startIndex, setStartIndex] = useState(0);
  const initialRowCount = 5;

  if (!csvData || csvData.length === 0) {
    return <div>No data to display</div>;
  }

  const headers = Object.keys(csvData[0]);
  const visibleRows = csvData.slice(startIndex, startIndex + initialRowCount);

  const showMoreRows = () => {
    if (startIndex + initialRowCount < csvData.length) {
      setStartIndex(startIndex + initialRowCount);
    }
  };

  const showLessRows = () => {
    if (startIndex - initialRowCount >= 0) {
      setStartIndex(startIndex - initialRowCount);
    }
  };

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
          {visibleRows.map((row, index) => (
            <tr key={index}>
              {headers.map((header) => (
                <td key={header}>{row[header]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </Table>
      {csvData.length > initialRowCount && (
        <div>
          <Button onClick={showLessRows}>Back</Button>{" "}
          <Button onClick={showMoreRows}>Next</Button>
        </div>
      )}
    </div>
  );
};

export default CsvTable;
