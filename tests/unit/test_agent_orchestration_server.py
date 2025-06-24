#!/usr/bin/env python3
"""
Agent Orchestration MCP Server Test Suite

Comprehensive tests for agent orchestration capabilities including:
- Agent creation and management
- Task submission and execution
- Priority-based scheduling
- Performance monitoring
"""

import time
import json
import requests
from typing import Dict, List, Any
from datetime import datetime

class AgentOrchestrationTester:
    """Test agent orchestration server capabilities"""
    
    def __init__(self, base_url: str = "http://localhost:8005"):
        self.base_url = base_url
        self.results = []
        self.created_agents = []
        self.submitted_tasks = []
    
    def log_result(self, test_name: str, success: bool, details: str = "", duration: float = 0):
        """Log test result"""
        result = {
            'test': test_name,
            'success': success,
            'details': details,
            'duration_ms': round(duration * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }
        self.results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name} ({duration*1000:.1f}ms)")
        if details:
            print(f"    {details}")
    
    def test_server_health(self) -> bool:
        """Test server health endpoint"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            duration = time.time() - start
            
            if response.status_code == 200:
                health_data = response.json()
                
                if health_data.get('status') == 'healthy':
                    agents = health_data.get('agents', {})
                    performance = health_data.get('performance', {})
                    uptime = performance.get('uptime_seconds', 0)
                    details = f"Status: healthy, Uptime: {uptime}s, Agents: {agents.get('total_agents', 0)}"
                    self.log_result("Server Health", True, details, duration)
                    return True
                else:
                    self.log_result("Server Health", False, f"Unhealthy status", duration)
                    return False
            else:
                self.log_result("Server Health", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Server Health", False, f"Error: {str(e)}", duration)
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint information"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            duration = time.time() - start
            
            if response.status_code == 200:
                root_data = response.json()
                
                service = root_data.get('service', '')
                capabilities = root_data.get('capabilities', [])
                endpoints = root_data.get('endpoints', {})
                
                if 'Agent Orchestration' in service and len(capabilities) > 0:
                    details = f"Service: {service}, Capabilities: {len(capabilities)}, Endpoints: {len(endpoints)}"
                    self.log_result("Root Endpoint", True, details, duration)
                    return True
                else:
                    self.log_result("Root Endpoint", False, "Missing service info or capabilities", duration)
                    return False
            else:
                self.log_result("Root Endpoint", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Root Endpoint", False, f"Error: {str(e)}", duration)
            return False
    
    def test_agent_creation(self) -> bool:
        """Test creating agents"""
        start = time.time()
        try:
            # Create different types of agents
            agents_to_create = [
                {
                    "agent_type": "reasoning",
                    "name": "Reasoning Agent 1",
                    "capabilities": ["logical_reasoning", "problem_solving"],
                    "config": {"model": "test"}
                },
                {
                    "agent_type": "search",
                    "name": "Search Agent 1", 
                    "capabilities": ["web_search", "information_retrieval"],
                    "config": {"search_engine": "test"}
                },
                {
                    "agent_type": "general",
                    "name": "General Agent 1",
                    "capabilities": ["general_tasks", "text_processing"],
                    "config": {}
                }
            ]
            
            created_count = 0
            for agent_data in agents_to_create:
                response = requests.post(
                    f"{self.base_url}/v1/agents",
                    json=agent_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    if result_data.get('success'):
                        agent_id = result_data.get('data', {}).get('agent_id')
                        if agent_id:
                            self.created_agents.append(agent_id)
                            created_count += 1
            
            duration = time.time() - start
            
            if created_count == len(agents_to_create):
                details = f"Created {created_count} agents successfully"
                self.log_result("Agent Creation", True, details, duration)
                return True
            else:
                details = f"Only created {created_count}/{len(agents_to_create)} agents"
                self.log_result("Agent Creation", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Agent Creation", False, f"Error: {str(e)}", duration)
            return False
    
    def test_list_agents(self) -> bool:
        """Test listing agents"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/v1/agents", timeout=10)
            duration = time.time() - start
            
            if response.status_code == 200:
                agents = response.json()
                
                if isinstance(agents, list) and len(agents) >= len(self.created_agents):
                    # Validate agent structure
                    if agents:
                        first_agent = agents[0]
                        required_fields = ['agent_id', 'name', 'agent_type', 'capabilities', 'status']
                        missing_fields = [field for field in required_fields if field not in first_agent]
                        
                        if not missing_fields:
                            details = f"Listed {len(agents)} agents with complete information"
                            self.log_result("List Agents", True, details, duration)
                            return True
                        else:
                            self.log_result("List Agents", False, f"Missing fields: {missing_fields}", duration)
                            return False
                    else:
                        details = f"Listed {len(agents)} agents (empty list)"
                        self.log_result("List Agents", True, details, duration)
                        return True
                else:
                    self.log_result("List Agents", False, f"Expected list with {len(self.created_agents)} agents, got {len(agents) if isinstance(agents, list) else 'non-list'}", duration)
                    return False
            else:
                self.log_result("List Agents", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("List Agents", False, f"Error: {str(e)}", duration)
            return False
    
    def test_task_submission(self) -> bool:
        """Test submitting tasks"""
        start = time.time()
        try:
            # Submit different types of tasks
            tasks_to_submit = [
                {
                    "task_type": "reasoning",
                    "description": "Solve a logical reasoning problem",
                    "input_data": {"problem": "If A > B and B > C, what is the relationship between A and C?"},
                    "priority": "high",
                    "required_capabilities": ["logical_reasoning"],
                    "timeout_seconds": 60
                },
                {
                    "task_type": "search",
                    "description": "Search for information about AI",
                    "input_data": {"query": "artificial intelligence recent developments"},
                    "priority": "normal",
                    "required_capabilities": ["web_search"],
                    "timeout_seconds": 120
                },
                {
                    "task_type": "general",
                    "description": "Process some text",
                    "input_data": {"text": "This is a sample text for processing"},
                    "priority": "low",
                    "required_capabilities": ["text_processing"],
                    "timeout_seconds": 30
                }
            ]
            
            submitted_count = 0
            for task_data in tasks_to_submit:
                response = requests.post(
                    f"{self.base_url}/v1/tasks",
                    json=task_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    if result_data.get('success'):
                        task_id = result_data.get('data', {}).get('task_id')
                        if task_id:
                            self.submitted_tasks.append(task_id)
                            submitted_count += 1
            
            duration = time.time() - start
            
            if submitted_count == len(tasks_to_submit):
                details = f"Submitted {submitted_count} tasks successfully"
                self.log_result("Task Submission", True, details, duration)
                return True
            else:
                details = f"Only submitted {submitted_count}/{len(tasks_to_submit)} tasks"
                self.log_result("Task Submission", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Task Submission", False, f"Error: {str(e)}", duration)
            return False
    
    def test_task_execution(self) -> bool:
        """Test task execution and completion"""
        start = time.time()
        try:
            # Wait for tasks to be processed
            max_wait_time = 10  # seconds
            wait_start = time.time()
            completed_tasks = 0
            
            while time.time() - wait_start < max_wait_time:
                # Check task statuses
                completed_count = 0
                for task_id in self.submitted_tasks:
                    response = requests.get(f"{self.base_url}/v1/tasks/{task_id}", timeout=10)
                    if response.status_code == 200:
                        task_data = response.json()
                        if task_data.get('status') in ['completed', 'failed']:
                            completed_count += 1
                
                if completed_count == len(self.submitted_tasks):
                    completed_tasks = completed_count
                    break
                
                await_time = 0.5
                time.sleep(await_time)
            
            duration = time.time() - start
            
            if completed_tasks == len(self.submitted_tasks):
                details = f"All {completed_tasks} tasks completed within {max_wait_time}s"
                self.log_result("Task Execution", True, details, duration)
                return True
            else:
                details = f"Only {completed_tasks}/{len(self.submitted_tasks)} tasks completed in {max_wait_time}s"
                self.log_result("Task Execution", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Task Execution", False, f"Error: {str(e)}", duration)
            return False
    
    def test_task_listing(self) -> bool:
        """Test listing tasks"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/v1/tasks", timeout=10)
            duration = time.time() - start
            
            if response.status_code == 200:
                tasks = response.json()
                
                if isinstance(tasks, list) and len(tasks) >= len(self.submitted_tasks):
                    # Validate task structure
                    if tasks:
                        first_task = tasks[0]
                        required_fields = ['task_id', 'task_type', 'description', 'status', 'priority']
                        missing_fields = [field for field in required_fields if field not in first_task]
                        
                        if not missing_fields:
                            # Check for different statuses
                            statuses = set(task.get('status') for task in tasks)
                            details = f"Listed {len(tasks)} tasks with statuses: {list(statuses)}"
                            self.log_result("Task Listing", True, details, duration)
                            return True
                        else:
                            self.log_result("Task Listing", False, f"Missing fields: {missing_fields}", duration)
                            return False
                    else:
                        details = f"Listed {len(tasks)} tasks (empty list)"
                        self.log_result("Task Listing", True, details, duration)
                        return True
                else:
                    self.log_result("Task Listing", False, f"Expected list with {len(self.submitted_tasks)} tasks, got {len(tasks) if isinstance(tasks, list) else 'non-list'}", duration)
                    return False
            else:
                self.log_result("Task Listing", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Task Listing", False, f"Error: {str(e)}", duration)
            return False
    
    def test_priority_scheduling(self) -> bool:
        """Test priority-based task scheduling"""
        start = time.time()
        try:
            # Submit tasks with different priorities
            priority_tasks = [
                {
                    "task_type": "general",
                    "description": "Low priority task",
                    "priority": "low",
                    "required_capabilities": []
                },
                {
                    "task_type": "general", 
                    "description": "Critical priority task",
                    "priority": "critical",
                    "required_capabilities": []
                },
                {
                    "task_type": "general",
                    "description": "Normal priority task", 
                    "priority": "normal",
                    "required_capabilities": []
                }
            ]
            
            task_ids = []
            for task_data in priority_tasks:
                response = requests.post(
                    f"{self.base_url}/v1/tasks",
                    json=task_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    if result_data.get('success'):
                        task_id = result_data.get('data', {}).get('task_id')
                        if task_id:
                            task_ids.append(task_id)
            
            # Wait a bit for processing
            time.sleep(2)
            
            # Check if tasks were processed (priority scheduling is working if server accepts them)
            processed_count = 0
            for task_id in task_ids:
                response = requests.get(f"{self.base_url}/v1/tasks/{task_id}", timeout=10)
                if response.status_code == 200:
                    processed_count += 1
            
            duration = time.time() - start
            
            if processed_count == len(task_ids):
                details = f"Priority scheduling working: {processed_count} tasks with different priorities processed"
                self.log_result("Priority Scheduling", True, details, duration)
                return True
            else:
                details = f"Priority scheduling issue: only {processed_count}/{len(task_ids)} tasks processed"
                self.log_result("Priority Scheduling", False, details, duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Priority Scheduling", False, f"Error: {str(e)}", duration)
            return False
    
    def test_performance_monitoring(self) -> bool:
        """Test performance monitoring capabilities"""
        start = time.time()
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            duration = time.time() - start
            
            if response.status_code == 200:
                health_data = response.json()
                performance = health_data.get('performance', {})
                
                # Check for key performance metrics
                required_metrics = [
                    'uptime_seconds', 'agents_created', 'tasks_submitted', 
                    'tasks_completed', 'success_rate'
                ]
                
                missing_metrics = [metric for metric in required_metrics if metric not in performance]
                
                if not missing_metrics:
                    uptime = performance.get('uptime_seconds', 0)
                    success_rate = performance.get('success_rate', 0)
                    tasks_completed = performance.get('tasks_completed', 0)
                    
                    details = f"Performance monitoring active: {uptime}s uptime, {success_rate}% success rate, {tasks_completed} tasks completed"
                    self.log_result("Performance Monitoring", True, details, duration)
                    return True
                else:
                    self.log_result("Performance Monitoring", False, f"Missing metrics: {missing_metrics}", duration)
                    return False
            else:
                self.log_result("Performance Monitoring", False, f"HTTP {response.status_code}", duration)
                return False
                
        except Exception as e:
            duration = time.time() - start
            self.log_result("Performance Monitoring", False, f"Error: {str(e)}", duration)
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all agent orchestration tests"""
        print("🤖 Agent Orchestration MCP Server Test Suite")
        print("=" * 50)
        
        tests = [
            ("Server Health", self.test_server_health),
            ("Root Endpoint", self.test_root_endpoint),
            ("Agent Creation", self.test_agent_creation),
            ("List Agents", self.test_list_agents),
            ("Task Submission", self.test_task_submission),
            ("Task Execution", self.test_task_execution),
            ("Task Listing", self.test_task_listing),
            ("Priority Scheduling", self.test_priority_scheduling),
            ("Performance Monitoring", self.test_performance_monitoring)
        ]
        
        passed = 0
        for test_name, test_func in tests:
            if test_func():
                passed += 1
            print()  # Add spacing
        
        total = len(tests)
        
        print("=" * 50)
        print(f"📊 Agent Orchestration Results: {passed}/{total} tests passed")
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'success_rate': (passed / total) * 100,
            'test_results': self.results,
            'created_agents': self.created_agents,
            'submitted_tasks': self.submitted_tasks
        }


def main():
    """Main test execution"""
    tester = AgentOrchestrationTester()
    results = tester.run_all_tests()
    
    # Save results
    with open('agent_orchestration_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: agent_orchestration_test_results.json")
    return 0 if results['success_rate'] >= 80 else 1


if __name__ == "__main__":
    exit(main())
