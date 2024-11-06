// frontend/src/components/Filters.jsx
import React from 'react';

const Filters = ({ setStartDate, setEndDate }) => {
  const handleStartDateChange = (e) => setStartDate(e.target.value);
  const handleEndDateChange = (e) => setEndDate(e.target.value);

  return (
    <div className="filters">
      <label>Start Date: </label>
      <input type="date" onChange={handleStartDateChange} />
      <label>End Date: </label>
      <input type="date" onChange={handleEndDateChange} />
    </div>
  );
};

export default Filters;
