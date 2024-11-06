// frontend/src/api/api.js
import axios from 'axios';

export const fetchData = async () => {
  const response = await axios.get('http://localhost:5000/api/data');
  return response.data;
};

export const fetchModelMetrics = async () => {
  const response = await axios.get('http://localhost:5000/api/metrics');
  return response.data;
};

export const fetchEvents = async () => {
  const response = await axios.get('http://localhost:5000/api/events');
  return response.data;
};

// Add the fetchForecast function
export const fetchForecast = async () => {
  const response = await axios.get('http://localhost:5000/api/forecast'); // Replace with correct endpoint if different
  return response.data;
};
