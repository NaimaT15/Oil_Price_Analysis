import React, { useState, useEffect } from 'react';
import { fetchData, fetchModelMetrics, fetchEvents, fetchForecast } from '../api';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './Dashboard.css';

const Dashboard = () => {
  const [oilData, setOilData] = useState([]);
  const [modelMetrics, setModelMetrics] = useState({});
  const [events, setEvents] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [filteredOilData, setFilteredOilData] = useState([]);
  const [filteredForecast, setFilteredForecast] = useState([]);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  useEffect(() => {
    const fetchDataFromAPI = async () => {
      try {
        const oilData = await fetchData();
        setOilData(oilData);
        setFilteredOilData(oilData);

        const metrics = await fetchModelMetrics();
        setModelMetrics(metrics);

        const eventsData = await fetchEvents();
        setEvents(eventsData);

        const forecastData = await fetchForecast();
        setForecast(forecastData);
        setFilteredForecast(forecastData);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchDataFromAPI();
  }, []);

  const filterDataByDate = () => {
    const filteredOil = oilData.filter((data) => {
      const date = new Date(data.date);
      return date >= new Date(startDate) && date <= new Date(endDate);
    });
    setFilteredOilData(filteredOil);

    const filteredForecastData = forecast.filter((data) => {
      const date = new Date(data.date);
      return date >= new Date(startDate) && date <= new Date(endDate);
    });
    setFilteredForecast(filteredForecastData);
  };

  const handleStartDateChange = (e) => setStartDate(e.target.value);
  const handleEndDateChange = (e) => setEndDate(e.target.value);

  return (
    <div className="dashboard">
      <h1>Brent Oil Price Dashboard</h1>

      <div className="date-filter">
        <label>Start Date:</label>
        <input type="date" value={startDate} onChange={handleStartDateChange} />
        <label>End Date:</label>
        <input type="date" value={endDate} onChange={handleEndDateChange} />
        <button onClick={filterDataByDate}>Filter</button>
      </div>

      <div className="main-content">
        <div className="chart-container">
          <h2>Historical Oil Prices</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={filteredOilData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="price" stroke="#8884d8" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="forecast-container">
          <h2>Forecasted Prices</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={filteredForecast}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="price" stroke="#82ca9d" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="sidebar">
        <div className="model-metrics">
          <h2>Model Metrics</h2>
          <p>RMSE: {modelMetrics.rmse || 'Loading...'}</p>
          <p>MAE: {modelMetrics.mae || 'Loading...'}</p>
        </div>
        <div className="events">
          <h2>Events Impacting Oil Prices</h2>
          <ul>
            {events.length > 0 ? (
              events.map((event, index) => (
                <li key={index}>
                  <strong>{event.date}</strong>: {event.description}
                </li>
              ))
            ) : (
              <p>No events found for the selected date range.</p>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
