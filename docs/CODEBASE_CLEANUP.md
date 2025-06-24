
# Codebase Cleanup Plan

This document outlines the plan to remove mock code and address `TODO` items throughout the codebase.

## Phase 1: Critical Mock Implementations

These are the highest priority items to ensure the core application logic is sound.

1.  **`src/orchestration/historical_research_agent.py`**
    *   **Issue:** The agent uses mock data for historical archives and a mock geocoding service.
    *   **Plan:**
        *   **Task 1.1:** Replace the mock database with a real database connection. I will define a standard interface and implement a connection to a local SQLite database for development and testing.
        *   **Task 1.2:** Replace the mock geocoding service with a real one. I will integrate with a free, public API like the Nominatim API.
        *   **Task 1.3:** Add configuration settings for the database path and any necessary API endpoints.

2.  **`agent_swarm_vision.py`**
    *   **Issue:** The `create_agent` function returns a hardcoded, mock agent.
    *   **Plan:**
        *   **Task 2.1:** Refactor the function to use the `AgentFactory` to dynamically create a complete, production-ready agent based on the provided vision and requirements.

3.  **`ui/src/components/chat/ChatInterface.tsx`**
    *   **Issue:** The UI indicates that the agent is receiving mock responses. This is a critical issue that breaks the user's trust.
    *   **Plan:**
        *   **Task 3.1:** Investigate the data flow to the chat interface to identify the source of the mock data.
        *   **Task 3.2:** Trace the backend calls to ensure that only real, production-level services are being used.
        *   **Task 3.3:** Remove the "MOCK RESPONSES" warning from the UI once the underlying issue is fixed.

## Phase 2: Test Environment Improvements

These items will improve the quality and reliability of our testing suite.

4.  **`test_tot_integration.py` & `test_tot_implementation.py`**
    *   **Issue:** These tests rely on mock LLM backends, generators, and evaluators.
    *   **Plan:**
        *   **Task 4.1:** Create a dedicated test configuration that can (optionally) connect to a real, sandboxed LLM endpoint for more realistic integration testing.
        *   **Task 4.2:** Keep the mock objects for fast, isolated unit tests, but make the integration tests more robust.

## Phase 3: TODOs and Minor Enhancements

These are lower-priority items that will improve the overall quality of the application.

5.  **`TODO_DOCUMENT_SEARCH.md`**
    *   **Issue:** This file outlines a needed enhancement for document search.
    *   **Plan:**
        *   **Task 5.1:** Read the markdown file to fully understand the requirements.
        *   **Task 5.2:** Implement the specified document search functionality.

6.  **`ui/src/pages/MCPMarketplacePage.tsx`**
    *   **Issue:** A `TODO` item exists to merge the status of registered and discovered servers.
    *   **Plan:**
        *   **Task 6.1:** Implement the client-side logic to merge the two data sources and display them correctly in the UI.

7.  **`ui/src/components/chat/ChatInterface.tsx`**
    *   **Issue:** A `TODO` item exists to add an error notification.
    *   **Plan:**
        *   **Task 7.1:** Implement a user-friendly error notification component (e.g., a toast or an alert) to display any errors that occur during the chat session.

## Utility Scripts

8.  **`update_mcp_servers.py`**
    *   **Status:** This appears to be a utility script for clearing out old mock server configurations.
    *   **Plan:** No action is needed at this time, but I will keep it in mind as a useful tool for database maintenance.
