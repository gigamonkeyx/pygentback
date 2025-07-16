import React from 'react';

interface Agent {
    id: number;
    name: string;
    status: string;
}

interface Props {
    agents: Agent[];
    onAddAgent: () => void;
    onDeleteAgent: (id: number) => void;
    onEditAgent: (agent: Agent) => void;
}

const Agents: React.FC<Props> = ({agents, onAddAgent, onDeleteAgent, onEditAgent}) => {
    return (
        <div className="p-4">
            <h1 className="text-2xl mb-4">Agents</h1>
            
            {/* Agent list */}
            <table className="w-full text-left">
                <thead>
                    <tr>
                        <th className="px-4 py-2">Name</th>
                        <th className="px-4 py-2">Status</th>
                        <th className="px-4 py-2">Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {agents.map(agent => (
                        <tr key={agent.id}>
                            <td className="border px-4 py-2">{agent.name}</td>
                            <td className="border px-4 py-2">{agent.status}</td>
                            <td className="border px-4 py-2 flex space-x-2">
                                <button onClick={() => onEditAgent(agent)}>Edit</button>
                                <button onClick={() => onDeleteAgent(agent.id)}>Delete</button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
            
            {/* Add new agent button */}
            <div className="mt-4">
                <button onClick={onAddAgent} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                    Add New Agent
                </button>
            </div>
        </div>
    );
};

export default Agents;