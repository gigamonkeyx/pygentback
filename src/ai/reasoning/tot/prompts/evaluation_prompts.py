"""
Evaluation Prompts for Tree of Thoughts

Specialized prompts for evaluating thoughts in different contexts and domains.
"""

# Value evaluation prompts
VALUE_EVALUATION_PROMPTS = {
    "general": """
# Thought Evaluation

Problem: {problem}
Thought to evaluate: {thought_content}

Evaluate this thought on the following criteria (score 0.0-1.0):

1. **Logical Consistency** (0.0-1.0)
   - Is the reasoning sound and coherent?
   - Are there any logical contradictions?

2. **Solution Progress** (0.0-1.0)
   - How much progress does this make toward solving the problem?
   - Does it provide actionable steps?

3. **Clarity and Completeness** (0.0-1.0)
   - Is the thought clearly articulated?
   - Does it provide sufficient detail?

4. **Innovation and Creativity** (0.0-1.0)
   - Does it offer novel approaches or insights?
   - How original is the thinking?

Provide an overall score from 0.0 to 1.0 where:
- 0.0-0.3: Poor quality, major issues
- 0.4-0.6: Average quality, some problems
- 0.7-0.8: Good quality, minor issues  
- 0.9-1.0: Excellent quality, highly valuable

Overall Score: """,

    "coding": """
# Coding Thought Evaluation

Problem: {problem}
Coding approach to evaluate: {thought_content}

Evaluate this coding approach on the following criteria (score 0.0-1.0):

1. **Algorithmic Correctness** (0.0-1.0)
   - Is the approach algorithmically sound?
   - Will it solve the problem correctly?

2. **Implementation Quality** (0.0-1.0)
   - Is the code structure well-designed?
   - Does it follow best practices?

3. **Efficiency** (0.0-1.0)
   - Is the time/space complexity reasonable?
   - Are there obvious optimizations?

4. **Maintainability** (0.0-1.0)
   - Is the code readable and well-organized?
   - Would it be easy to maintain and extend?

Provide an overall score from 0.0 to 1.0:

Overall Score: """,

    "mathematical": """
# Mathematical Reasoning Evaluation

Problem: {problem}
Mathematical approach: {thought_content}

Evaluate this mathematical reasoning:

1. **Mathematical Correctness** (0.0-1.0)
   - Are the mathematical steps valid?
   - Is the logic mathematically sound?

2. **Problem Progress** (0.0-1.0)
   - Does this advance toward the solution?
   - Are the steps productive?

3. **Clarity of Reasoning** (0.0-1.0)
   - Are the mathematical steps clear?
   - Is the reasoning easy to follow?

Overall Score: """
}

# Comparative evaluation prompts for voting
COMPARATIVE_EVALUATION_PROMPTS = {
    "general": """
# Thought Comparison

Problem: {problem}

Compare these two approaches and determine which is better:

**Approach A:**
{thought_a}

**Approach B:**
{thought_b}

Evaluation criteria:
- Quality of reasoning and logic
- Progress toward solving the problem
- Clarity and actionability
- Innovation and practical value

Which approach is better for solving this problem? 
Respond with 'A' or 'B' and provide a brief explanation.

Winner: """,

    "coding": """
# Coding Approach Comparison

Problem: {problem}

Compare these two coding approaches:

**Approach A:**
{thought_a}

**Approach B:**
{thought_b}

Consider:
- Algorithmic correctness and efficiency
- Code quality and maintainability
- Implementation feasibility
- Best practices adherence

Which coding approach is superior?
Respond with 'A' or 'B' and explain why.

Winner: """,

    "creative": """
# Creative Solution Comparison

Problem: {problem}

Compare these two creative approaches:

**Option A:**
{thought_a}

**Option B:**
{thought_b}

Evaluate based on:
- Creativity and originality
- Practical feasibility
- Potential impact and effectiveness
- Innovation and uniqueness

Which creative solution is more promising?
Respond with 'A' or 'B' and justify your choice.

Winner: """
}

# Context-specific evaluation templates
CONTEXT_EVALUATION_TEMPLATES = {
    "debugging": """
# Debugging Approach Evaluation

Problem: {problem}
Current state: {current_state}
Debugging approach: {thought_content}

Evaluate this debugging strategy:

1. **Diagnostic Accuracy** - Will this help identify the root cause?
2. **Efficiency** - Is this a time-effective debugging approach?
3. **Systematic Approach** - Does it follow good debugging practices?
4. **Completeness** - Does it cover the likely problem areas?

Score (0.0-1.0): """,

    "optimization": """
# Optimization Strategy Evaluation

Problem: {problem}
Optimization approach: {thought_content}

Evaluate this optimization strategy:

1. **Performance Impact** - Will this significantly improve performance?
2. **Implementation Complexity** - Is the optimization practical to implement?
3. **Trade-offs** - Are the trade-offs well-considered?
4. **Maintainability** - Will this make the code harder to maintain?

Score (0.0-1.0): """,

    "architecture": """
# Architecture Design Evaluation

Problem: {problem}
Architectural approach: {thought_content}

Evaluate this architectural design:

1. **Scalability** - Will this design scale effectively?
2. **Maintainability** - Is it easy to understand and modify?
3. **Flexibility** - Can it adapt to changing requirements?
4. **Best Practices** - Does it follow established patterns?

Score (0.0-1.0): """
}

def build_evaluation_prompt(
    evaluation_type: str,
    prompt_category: str,
    problem: str,
    thought_content: str,
    **kwargs
) -> str:
    """
    Build an evaluation prompt based on type and context
    
    Args:
        evaluation_type: 'value' or 'comparative'
        prompt_category: Category like 'coding', 'general', etc.
        problem: Problem description
        thought_content: Content to evaluate
        **kwargs: Additional template variables
        
    Returns:
        Formatted evaluation prompt
    """
    if evaluation_type == 'value':
        templates = VALUE_EVALUATION_PROMPTS
    elif evaluation_type == 'comparative':
        templates = COMPARATIVE_EVALUATION_PROMPTS
    elif evaluation_type == 'context':
        templates = CONTEXT_EVALUATION_TEMPLATES
    else:
        templates = VALUE_EVALUATION_PROMPTS
    
    # Select template
    if prompt_category in templates:
        template = templates[prompt_category]
    else:
        # Fallback to general template
        template = templates.get('general', list(templates.values())[0])
    
    # Format template
    return template.format(
        problem=problem,
        thought_content=thought_content,
        **kwargs
    )

def build_comparative_prompt(
    prompt_category: str,
    problem: str,
    thought_a: str,
    thought_b: str,
    **kwargs
) -> str:
    """
    Build a comparative evaluation prompt
    
    Args:
        prompt_category: Category like 'coding', 'general', etc.
        problem: Problem description
        thought_a: First thought to compare
        thought_b: Second thought to compare
        **kwargs: Additional template variables
        
    Returns:
        Formatted comparative prompt
    """
    return build_evaluation_prompt(
        'comparative',
        prompt_category,
        problem,
        '',  # Not used in comparative
        thought_a=thought_a,
        thought_b=thought_b,
        **kwargs
    )

# Specialized prompts for different domains
DOMAIN_PROMPTS = {
    "web_development": """
# Web Development Approach Evaluation

Problem: {problem}
Web development approach: {thought_content}

Evaluate considering:
- Frontend/backend architecture
- User experience implications
- Security considerations
- Performance and scalability
- Modern web standards compliance

Score (0.0-1.0): """,

    "data_science": """
# Data Science Approach Evaluation

Problem: {problem}
Data science approach: {thought_content}

Evaluate considering:
- Data preprocessing and cleaning
- Algorithm selection appropriateness
- Statistical validity
- Interpretability and explainability
- Scalability to larger datasets

Score (0.0-1.0): """,

    "machine_learning": """
# Machine Learning Solution Evaluation

Problem: {problem}
ML approach: {thought_content}

Evaluate considering:
- Model architecture appropriateness
- Training strategy and data requirements
- Evaluation metrics and validation
- Deployment and inference considerations
- Ethical and bias considerations

Score (0.0-1.0): """
}

def get_domain_prompt(domain: str, problem: str, thought_content: str) -> str:
    """Get domain-specific evaluation prompt"""
    if domain in DOMAIN_PROMPTS:
        return DOMAIN_PROMPTS[domain].format(
            problem=problem,
            thought_content=thought_content
        )
    else:
        # Fall back to general coding prompt
        return VALUE_EVALUATION_PROMPTS['coding'].format(
            problem=problem,
            thought_content=thought_content
        )
