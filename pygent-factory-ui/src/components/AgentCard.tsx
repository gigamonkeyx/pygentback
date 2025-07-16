import React from 'react';

interface AgentCardProps {
  name: string;
  type: string;
  status: string;
  performanceMetrics: any[]; // You can replace this with your actual metrics type
}

const AgentCard: React.FC<AgentCardProps> = ({ name, type, status, performanceMetrics }) => {
  return (
    <div className="flex flex-col rounded overflow-hidden shadow-lg mx-auto max-w-sm">
      <div className="px-6 py-4">
        <h2 className="font-bold text-xl mb-2">{name}</h2>
        <p className="text-gray-700 text-base">Type: {type}</p>
        <p className="text-gray-700 text-base">Status: 
          <span className={`px-2 py-1 rounded ${status === 'active' ? 'bg-green-500 text-white' : status === 'inactive' ? 'bg-red-500 text-white' : 'bg-gray-300'}`}>
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </span>
        </p>
      </div>
      {performanceMetrics && (
        <ul className="px-6 py-2 text-gray-700 text-base">
          {performanceMetrics.map((metric, index) => (
            <li key={index}>
              <strong>{metric.name}:</strong> {metric.value}
            </li>
          ))}
        </ul>
      )}
      <div className="px-6 pt-4 pb-2">
        <button className="mr-2 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Start</button>
        <button className="mr-2 bg-red-500 hover:bg-red-700 text-white font-bold py-2 px<｜begin▁of▁sentence｜>.