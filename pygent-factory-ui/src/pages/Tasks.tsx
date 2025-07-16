jsx
import React from 'react';

interface Task {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
}

const Tasks = () => {
  const [tasks, setTasks] = React.useState<Task[]>([]);
  
  // Add your form handling and task creation logic here...

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-200">
      <h1 className="text-3xl font-bold mb-8">Tasks</h1>
      
      {/* Add your task list, form, and other components here... */}
    </div>
  );
};

export default Tasks;