// Agent interface 
interface Agent {
    id: string;
    name: string;
    type: string;
    status: AgentStatus;
    capabilities: any[]; // This can be replaced with the actual capability types if they exist.
}

// Task interface 
interface Task {
    id: string;
    title: string;

    description: string;
    status: TaskStatus;
    agent_id?: string; // The ? makes it optional
}

// API response types
type ApiResponse = {
    data: any,  // This can be replaced with the actual data type if they exist.
    error?: ErrorType, // The ? makes it optional
};

// Component prop types
interface ComponentProps {
    agentData: Agent;
    taskData: Task[];
}

// Enum types for status values
enum AgentStatus {
    Available = "Available",
    Busy = "Busy"
}

enum TaskStatus {
  Pending = 'Pending',
  InProgress = 'In Progress',
  Completed = 'Completed'
}

type ErrorType = {
  message: string;
};