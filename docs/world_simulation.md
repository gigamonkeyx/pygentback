# PyGent Factory World Simulation System

## Overview

The PyGent Factory World Simulation System is a comprehensive multi-agent environment that implements emergent behavior detection, evolutionary optimization, and autonomous agent coordination. The system follows the RIPER-Ω protocol for structured development and maintains observer supervision throughout all operations.

## Architecture

### Core Components

1. **Simulation Environment** (`src/core/sim_env.py`)
   - Resource management with mathematical decay
   - Agent lifecycle management
   - Environment sensing and adaptation
   - State persistence and recovery

2. **Agent Population Manager**
   - 10 specialized agent types with defined roles
   - Trait randomization using genetic algorithms
   - Performance tracking and optimization

3. **Evolution System**
   - Two-phase evolution (Explore/Exploit)
   - Fitness evaluation with bloat penalties
   - Convergence detection and adaptive stopping

4. **Emergent Behavior Monitor**
   - Spontaneous cooperation detection
   - Resource optimization pattern analysis
   - Tool sharing network formation
   - Adaptive trigger mechanisms

5. **RIPER-Ω Integration**
   - Mode-locked workflow execution
   - Confidence threshold validation
   - Context7 MCP specification syncing

## Mathematical Foundations

### Resource Decay Model

Resources decay over time according to the exponential decay formula:

```
R(t) = R₀ * e^(-λt)
```

Where:
- `R(t)` = Available resources at time t
- `R₀` = Initial resource amount
- `λ` = Decay rate (default: 0.05)
- `t` = Time elapsed

Implementation:
```python
decay_amount = available * decay_rate * time_delta
available = max(0.0, available - decay_amount)
```

### Fitness Function

Agent fitness is calculated using the comprehensive formula:

```
fitness = (environment_coverage * efficiency) - bloat_penalty
```

Where:
- `environment_coverage` = |agent_tools ∩ available_tools| / |available_tools|
- `efficiency` = Agent performance score (0.0 to 1.0)
- `bloat_penalty` = (total_capabilities * penalty_rate)

### Convergence Detection

Evolution converges when fitness improvement over the last 3 generations is below threshold:

```
improvement = max(fitness[-3:]) - min(fitness[-3:])
converged = improvement < threshold (default: 0.05)
```

## Agent Specifications

### Role Distribution

| Role | Count | Primary Function | Key Capabilities |
|------|-------|------------------|------------------|
| Explorer | 2 | Environment scanning, tool discovery | `environment_scanning`, `tool_discovery`, `resource_mapping` |
| Builder | 2 | Module integration, system construction | `module_integration`, `system_construction`, `architecture_design` |
| Harvester | 2 | Resource gathering, optimization | `resource_gathering`, `data_extraction`, `efficiency_optimization` |
| Defender | 1 | Guideline enforcement, safety monitoring | `guideline_enforcement`, `safety_monitoring`, `threat_detection` |
| Communicator | 1 | A2A coordination, message routing | `a2a_coordination`, `message_routing`, `protocol_management` |
| Adapter | 2 | Gap filling, dynamic role assignment | `gap_filling`, `dynamic_assignment`, `flexible_response` |

### Trait System

Each agent has randomized traits within ±10% of base values:

- **Curiosity** (0.6-0.9): Exploration tendency
- **Efficiency** (0.8-0.9): Task completion optimization
- **Precision** (0.8-0.9): Accuracy in operations
- **Adaptability** (0.8-0.9): Response to environmental changes
- **Collaboration Preference** (0.3-0.9): Tendency to cooperate

## API Documentation

### Monitoring Endpoints

#### GET /sim/status
Returns current simulation status and basic metrics.

**Response:**
```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "environment": {
    "id": "sim_env_123",
    "status": "active",
    "uptime_seconds": 3600,
    "cycle_count": 120
  },
  "resources": {
    "compute": {"available": 800.0, "total": 1000.0, "utilization": 0.2},
    "memory": {"available": 6000.0, "total": 8192.0, "utilization": 0.27}
  },
  "population": {
    "population_size": 10,
    "role_distribution": {"explorer": 2, "builder": 2, "harvester": 2},
    "average_performance": 0.75
  }
}
```

#### GET /sim/metrics
Returns detailed performance metrics and statistics.

#### GET /sim/agents
Returns agent population state and performance data.

#### GET /sim/behaviors
Returns emergent behavior detection results and patterns.

#### GET /sim/evolution
Returns evolution system status and history.

### Control Endpoints

#### POST /sim/control/start
Starts or resumes world simulation.

#### POST /sim/control/stop
Gracefully stops world simulation.

## Emergent Behavior Patterns

### Spontaneous Cooperation
Detected when:
- Multiple collaboration events occur within 5-minute window
- Success rate exceeds 60% threshold
- Involves 3+ unique agents

**Pattern Structure:**
```json
{
  "type": "spontaneous_cooperation",
  "participants": ["agent_1", "agent_2", "agent_3"],
  "success_rate": 0.85,
  "collaboration_count": 5,
  "significance": "high"
}
```

### Resource Optimization
Detected when:
- Resource sharing events involve 2+ sharers and 2+ receivers
- Optimization efficiency exceeds 20% threshold
- Total shared resources > 0

### Tool Sharing Networks
Detected when:
- Connected components of 3+ agents form
- Tool sharing interactions create network density > 0.3
- Network exhibits emergent coordination properties

## RIPER-Ω Protocol Integration

### Mode Workflow

1. **RESEARCH Mode**
   - Environment sensing and analysis
   - Context7 MCP specification syncing
   - Population assessment
   - Resource trend analysis

2. **PLAN Mode**
   - Evolution strategy development
   - Parameter optimization
   - Fitness target setting
   - Risk assessment

3. **EXECUTE Mode**
   - Agent generation and evolution cycles
   - Emergent behavior monitoring
   - Real-time adaptation
   - Performance tracking

4. **REVIEW Mode**
   - Performance analysis and validation
   - Convergence assessment
   - System health evaluation
   - Recommendation generation

### Confidence Thresholds

Each mode requires 70% confidence to proceed:
- **Research**: Environment completeness + population assessment + context sync
- **Plan**: Strategy completeness + parameter validity + target realism
- **Execute**: Evolution success + fitness progression + behavior detection
- **Review**: Performance improvement + system stability + recommendation quality

## Extension Points

### Custom Agent Types

Add new agent types by extending the agent specification system:

```python
custom_agent_spec = {
    "type": "custom_type",
    "role": "custom_role",
    "capabilities": ["custom_capability_1", "custom_capability_2"],
    "mcp_tools": ["custom_tool_1", "custom_tool_2"],
    "traits": {"custom_trait": 0.8}
}
```

### Custom Fitness Functions

Implement alternative fitness evaluation:

```python
async def custom_fitness_evaluation(agent_data, env_state):
    # Custom fitness logic
    coverage = calculate_custom_coverage(agent_data, env_state)
    efficiency = get_custom_efficiency(agent_data)
    penalty = apply_custom_penalty(agent_data)
    return (coverage * efficiency) - penalty
```

### Custom Behavior Detectors

Add new emergent behavior patterns:

```python
async def detect_custom_behavior(self, interaction_history):
    # Custom behavior detection logic
    if meets_custom_criteria(interaction_history):
        return {
            "type": "custom_behavior",
            "pattern_data": extract_pattern_data(interaction_history),
            "significance": calculate_significance()
        }
    return None
```

### Custom Evolution Operators

Implement specialized mutation/crossover operators:

```python
async def custom_mutation_operator(agent_data, environment_needs):
    # Custom mutation logic based on specific requirements
    if should_apply_custom_mutation(agent_data, environment_needs):
        return {
            "type": "custom_mutation",
            "modification": generate_custom_modification(),
            "reason": "custom_optimization_strategy"
        }
    return None
```

## Configuration

### Environment Variables

- `WORLD_SIM_ENABLED`: Enable world simulation system
- `WORLD_SIM_MONITORING_PORT`: Monitoring API port (default: 8090)
- `WORLD_SIM_MAX_AGENTS`: Maximum agent population (default: 50)
- `WORLD_SIM_EVOLUTION_GENERATIONS`: Evolution generations per cycle (default: 10)
- `WORLD_SIM_RESOURCE_DECAY_RATE`: Resource decay rate (default: 0.05)
- `WORLD_SIM_FITNESS_THRESHOLD`: Convergence threshold (default: 0.05)

### Docker Deployment

The world simulation system is fully containerized and can be deployed using:

```bash
docker-compose up -d
```

Monitoring endpoints will be available at `http://localhost:8090/sim/`

## Performance Optimization

### Resource Management
- Implement async throttling in evolution loops
- Use connection pooling for database operations
- Cache frequently accessed environment states
- Implement graceful degradation under resource pressure

### Scalability Considerations
- Agent population scales linearly with available compute resources
- Evolution cycles can be parallelized across multiple cores
- NetworkX graph operations scale with O(n²) for alliance detection
- Redis persistence provides horizontal scaling capabilities

## Testing

Comprehensive test suite covers:
- Unit tests for all core components (80%+ coverage)
- Integration tests for multi-phase workflows
- Performance benchmarks for scalability validation
- End-to-end simulation runs with validation

Run tests with:
```bash
pytest tests/world_sim/ -v --cov=src/core/sim_env
```

## Troubleshooting

### Common Issues

1. **Evolution Stagnation**
   - Increase mutation rate or population diversity
   - Adjust fitness function parameters
   - Check for resource constraints

2. **Emergent Behavior Not Detected**
   - Verify agent interaction frequency
   - Check threshold parameters
   - Ensure sufficient population size

3. **Performance Degradation**
   - Monitor resource utilization
   - Check for memory leaks in long-running simulations
   - Optimize NetworkX graph operations

### Monitoring and Alerts

Use the monitoring endpoints to track:
- Resource utilization trends
- Evolution convergence rates
- Emergent behavior frequency
- System performance metrics

Set up alerts for:
- Resource utilization > 90%
- Evolution stagnation > 5 generations
- System errors or crashes
- Performance degradation > 50%
