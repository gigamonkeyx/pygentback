import React from 'react';

const Layout: React.FC = () => {
  return (
    <div className="flex flex-col min-h-screen bg-gray-100 text-black transition-all duration-300 dark:bg-gray-900 dark:text-white">
      <header className="sticky top-0 z-50 w-full bg-white border-b border-gray-200 shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <nav className="px-6 py-4 mx-auto max-w-screen-lg flex items-center justify-between">
          <a href="#" className="text-2xl font-bold text-black no-underline dark:text-white">Logo</a>
          <ul className="flex items-center space-x-4">
            <li><a href="#dashboard" className="hover:underline">Dashboard</a></li>
            <li><a href="#agents" className="hover:underline">Agents</a></li>
            <li><a href="#tasks" className="hover:underline">Tasks</a></li>
          </ul>
        </nav>
      </header>
      
      <div className="flex flex-col md:flex-row min-h-screen">
        <aside className="w-64 px-8 py-4 bg-white border-r border-gray-200 dark:border-gray-700 dark:bg-gray-800 md:sticky md:top-16">
          <h3 className="mb-4 text-lg font-bold">Sidebar</h3>
        </aside>
        
        <main className="flex-grow px-8 py-4 bg-white dark:bg-gray-900">
          <h1 className="text-2xl font-bold mb-6">Main Content</h1>
          {/* Main content goes here */}
        </main>
      </div>
      
      {/* Dark/light theme toggle button */}
    </div>
  );
};

export default Layout;