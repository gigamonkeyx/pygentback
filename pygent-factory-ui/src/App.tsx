import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Dashboard from './Dashboard';
import Agents from './Agents';
import Tasks from './Tasks';

const theme = createTheme();

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/agents" element={<Agents />} />
            <Route path="/tasks" element={<Tasks />} />
          </Routes>
        </Layout>
      </Router>
    </ThemeProvider>
  );
}

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <Box sx={{ display: 'flex' }}>
      {/* Navbar goes here */}
      <Container component="main" sx={{ mt: 8, mb: 2 }} maxWidth="md">
        {children}
      </Container>
    </Box>
  );
}

export default App;