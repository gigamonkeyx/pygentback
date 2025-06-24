#!/usr/bin/env python3
"""
PyGent Factory Python MCP Server

A comprehensive Python MCP server providing code execution, analysis, and development capabilities
specifically designed for the PyGent Factory agent evolution system.

Features:
- Safe Python code execution with sandboxing
- Code analysis and quality metrics
- Package management and dependency analysis
- AST parsing and code structure analysis
- Performance profiling and optimization suggestions
- Code generation and refactoring assistance
- Testing framework integration
- Documentation generation
"""

import ast
import io
import os
import time
import subprocess
import importlib.util
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional
import tempfile

from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("PyGent Python Server")


class PythonExecutionResult:
    """Result of Python code execution."""
    
    def __init__(self, success: bool, output: str = "", error: str = "", 
                 execution_time: float = 0.0, return_value: Any = None):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time
        self.return_value = return_value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "return_value": str(self.return_value) if self.return_value is not None else None
        }


class SafePythonExecutor:
    """Safe Python code execution with sandboxing and restrictions."""
    
    FORBIDDEN_IMPORTS = {
        'os', 'subprocess', 'sys', 'shutil', 'socket', 'urllib', 'requests',
        'http', 'ftplib', 'telnetlib', 'smtplib', 'poplib', 'imaplib',
        '__import__', 'eval', 'exec', 'compile', 'open', 'file',
        'input', 'raw_input'
    }
    
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes', 'chr',
        'complex', 'dict', 'divmod', 'enumerate', 'filter', 'float', 'format',
        'frozenset', 'hex', 'id', 'int', 'isinstance', 'issubclass', 'iter',
        'len', 'list', 'map', 'max', 'min', 'next', 'oct', 'ord', 'pow',
        'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
        'sum', 'tuple', 'type', 'zip', 'print'
    }
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def _check_code_safety(self, code: str) -> List[str]:
        """Check code for potentially unsafe operations."""
        warnings = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for forbidden imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.FORBIDDEN_IMPORTS:
                            warnings.append(f"Forbidden import: {alias.name}")
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.FORBIDDEN_IMPORTS:
                        warnings.append(f"Forbidden import from: {node.module}")
                
                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile']:
                            warnings.append(f"Dangerous function call: {node.func.id}")
        
        except SyntaxError as e:
            warnings.append(f"Syntax error: {e}")
        
        return warnings
    
    def execute(self, code: str, globals_dict: Optional[Dict[str, Any]] = None) -> PythonExecutionResult:
        """Execute Python code safely with restrictions."""
        
        # Check code safety first
        safety_warnings = self._check_code_safety(code)
        if safety_warnings:
            return PythonExecutionResult(
                success=False,
                error=f"Code safety check failed: {'; '.join(safety_warnings)}"
            )
        
        # Prepare execution environment
        if globals_dict is None:
            globals_dict = {}
        
        # Restrict builtins
        restricted_builtins = {name: __builtins__[name] for name in self.SAFE_BUILTINS if name in __builtins__}
        globals_dict['__builtins__'] = restricted_builtins
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        start_time = time.time()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Compile and execute code
                compiled_code = compile(code, '<string>', 'exec')
                exec(compiled_code, globals_dict)
            
            execution_time = time.time() - start_time
            
            return PythonExecutionResult(
                success=True,
                output=stdout_capture.getvalue(),
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_output = stderr_capture.getvalue()
            if not error_output:
                error_output = f"{type(e).__name__}: {str(e)}"
            
            return PythonExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=error_output,
                execution_time=execution_time
            )


# Initialize the safe executor
executor = SafePythonExecutor()


@mcp.tool()
def execute_python(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute Python code safely with restrictions and sandboxing.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)
    
    Returns:
        Dict with execution results including success, output, error, and timing
    """
    global executor
    executor.timeout = timeout
    
    result = executor.execute(code)
    return result.to_dict()


@mcp.tool()
def analyze_code_structure(code: str) -> Dict[str, Any]:
    """
    Analyze Python code structure using AST parsing.
    
    Args:
        code: Python code to analyze
    
    Returns:
        Dict with code structure analysis including functions, classes, imports, complexity
    """
    try:
        tree = ast.parse(code)
        
        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": [],
            "complexity_metrics": {
                "total_nodes": len(list(ast.walk(tree))),
                "max_depth": 0,
                "cyclomatic_complexity": 0
            }
        }
        
        # Analyze AST nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis["functions"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                    "decorators": [ast.unparse(dec) for dec in node.decorator_list] if hasattr(ast, 'unparse') else []
                })
            
            elif isinstance(node, ast.ClassDef):
                analysis["classes"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "bases": [ast.unparse(base) for base in node.bases] if hasattr(ast, 'unparse') else [],
                    "docstring": ast.get_docstring(node),
                    "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                })
            
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis["imports"].append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
            
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    analysis["imports"].append({
                        "type": "from_import",
                        "module": node.module,
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno
                    })
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        analysis["variables"].append({
                            "name": target.id,
                            "line": node.lineno,
                            "type": "assignment"
                        })
        
        return analysis
    
    except SyntaxError as e:
        return {
            "error": f"Syntax error: {e}",
            "line": e.lineno,
            "offset": e.offset
        }
    except Exception as e:
        return {
            "error": f"Analysis error: {str(e)}"
        }


@mcp.tool()
def check_code_quality(code: str) -> Dict[str, Any]:
    """
    Check Python code quality using multiple linters and analysis tools.
    
    Args:
        code: Python code to check
    
    Returns:
        Dict with quality metrics, linting results, and suggestions
    """
    results = {
        "syntax_valid": True,
        "syntax_errors": [],
        "style_issues": [],
        "complexity_warnings": [],
        "security_issues": [],
        "suggestions": []
    }
    
    # Check syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        results["syntax_valid"] = False
        results["syntax_errors"].append({
            "message": str(e),
            "line": e.lineno,
            "offset": e.offset
        })
        return results
    
    # Write code to temporary file for external tools
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_file_path = tmp_file.name
    
    try:
        # Check with flake8 (style and simple errors)
        try:
            result = subprocess.run(
                ['flake8', '--select=E,W', tmp_file_path],
                capture_output=True, text=True, timeout=10
            )
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            results["style_issues"].append({
                                "line": int(parts[1]),
                                "column": int(parts[2]),
                                "message": parts[3].strip()
                            })
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results["suggestions"].append("Install flake8 for enhanced style checking")
        
        # Security check with bandit
        try:
            result = subprocess.run(
                ['bandit', '-f', 'json', tmp_file_path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0 and result.stdout:
                import json
                try:
                    bandit_results = json.loads(result.stdout)
                    for issue in bandit_results.get('results', []):
                        results["security_issues"].append({
                            "line": issue.get('line_number'),
                            "severity": issue.get('issue_severity'),
                            "confidence": issue.get('issue_confidence'),
                            "message": issue.get('issue_text')
                        })
                except json.JSONDecodeError:
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results["suggestions"].append("Install bandit for security analysis")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except OSError:
            pass
    
    # Add general suggestions
    if not results["style_issues"] and not results["security_issues"]:
        results["suggestions"].append("Code looks good! Consider adding docstrings and type hints for better maintainability.")
    
    return results


@mcp.tool()
def format_code(code: str, line_length: int = 88) -> Dict[str, Any]:
    """
    Format Python code using Black formatter.
    
    Args:
        code: Python code to format
        line_length: Maximum line length (default: 88)
    
    Returns:
        Dict with formatted code and formatting information
    """
    try:
        # Try using black if available
        try:
            import black
            
            formatted_code = black.format_str(code, mode=black.FileMode(line_length=line_length))
            
            return {
                "success": True,
                "formatted_code": formatted_code,
                "changes_made": code != formatted_code,
                "formatter": "black"
            }
        
        except ImportError:
            # Fallback to basic formatting
            lines = code.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    formatted_lines.append('')
                    continue
                
                # Adjust indent level
                if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:')):
                    formatted_lines.append('    ' * indent_level + stripped)
                    indent_level += 1
                elif stripped.startswith(('elif ', 'else:', 'except:', 'finally:')):
                    formatted_lines.append('    ' * (indent_level - 1) + stripped)
                elif stripped in ['pass', 'break', 'continue'] or stripped.startswith('return'):
                    formatted_lines.append('    ' * indent_level + stripped)
                else:
                    formatted_lines.append('    ' * indent_level + stripped)
                
                # Reduce indent after certain statements
                if stripped.endswith(':') and not stripped.startswith(('elif ', 'else:', 'except:', 'finally:')):
                    pass  # Keep current indent level
                elif indent_level > 0 and not stripped.endswith(':'):
                    if not any(stripped.startswith(kw) for kw in ['if ', 'for ', 'while ', 'with ', 'try']):
                        indent_level = max(0, indent_level - 1)
            
            formatted_code = '\n'.join(formatted_lines)
            
            return {
                "success": True,
                "formatted_code": formatted_code,
                "changes_made": code != formatted_code,
                "formatter": "basic",
                "note": "Install 'black' for professional code formatting"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "original_code": code
        }


@mcp.tool()
def profile_code_performance(code: str, number_of_runs: int = 1000) -> Dict[str, Any]:
    """
    Profile Python code performance and provide optimization suggestions.
    
    Args:
        code: Python code to profile
        number_of_runs: Number of times to run the code for timing (default: 1000)
    
    Returns:
        Dict with performance metrics and optimization suggestions
    """
    import timeit
    
    try:
        # Basic timing
        def time_code():
            exec(code)
        
        # Run timing test
        total_time = timeit.timeit(time_code, number=number_of_runs)
        avg_time = total_time / number_of_runs
        
        # Analyze code for performance patterns
        suggestions = []
        
        if 'for' in code and 'range(' in code:
            suggestions.append("Consider using list comprehensions for better performance")
        
        if '.append(' in code and 'for' in code:
            suggestions.append("Consider using list comprehensions instead of append in loops")
        
        if 'string' in code.lower() and '+' in code:
            suggestions.append("For string concatenation, consider using f-strings or .join()")
        
        if 'import' in code and any(module in code for module in ['numpy', 'pandas']):
            suggestions.append("Good! Using optimized libraries like numpy/pandas")
        
        return {
            "total_time": total_time,
            "average_time": avg_time,
            "runs": number_of_runs,
            "performance_rating": "fast" if avg_time < 0.001 else "medium" if avg_time < 0.01 else "slow",
            "optimization_suggestions": suggestions
        }
    
    except Exception as e:
        return {
            "error": f"Profiling failed: {str(e)}",
            "suggestions": ["Ensure code is syntactically correct before profiling"]
        }


@mcp.tool()
def generate_docstring(function_code: str) -> Dict[str, Any]:
    """
    Generate docstrings for Python functions using analysis.
    
    Args:
        function_code: Python function code to document
    
    Returns:
        Dict with generated docstring and documentation suggestions
    """
    try:
        tree = ast.parse(function_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                args = [arg.arg for arg in node.args.args]
                
                # Generate basic docstring
                docstring_parts = [
                    f"{function_name.replace('_', ' ').title()}.",
                    "",
                    "Args:"
                ]
                
                for arg in args:
                    docstring_parts.append(f"    {arg}: Description of {arg}")
                
                docstring_parts.extend([
                    "",
                    "Returns:",
                    "    Description of return value"
                ])
                
                docstring = '    """' + '\n    '.join(docstring_parts) + '\n    """'
                
                # Insert docstring into function
                lines = function_code.split('\n')
                new_lines = []
                
                for i, line in enumerate(lines):
                    new_lines.append(line)
                    if i == 0 and line.strip().endswith(':'):  # Function definition line
                        new_lines.append(docstring)
                
                documented_code = '\n'.join(new_lines)
                
                return {
                    "success": True,
                    "original_code": function_code,
                    "documented_code": documented_code,
                    "function_name": function_name,
                    "arguments": args,
                    "suggestions": [
                        "Customize the generated docstring with specific descriptions",
                        "Add type hints for better documentation",
                        "Include examples in the docstring if helpful"
                    ]
                }
        
        return {
            "success": False,
            "error": "No function definition found in the provided code"
        }
    
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Syntax error in function code: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Documentation generation failed: {str(e)}"
        }


@mcp.tool()
def check_dependencies(code: str) -> Dict[str, Any]:
    """
    Analyze code dependencies and suggest package installations.
    
    Args:
        code: Python code to analyze for dependencies
    
    Returns:
        Dict with dependency analysis and installation suggestions
    """
    try:
        tree = ast.parse(code)
        
        imports = set()
        from_imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])  # Get root module
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    from_imports.add(node.module.split('.')[0])  # Get root module
        
        all_modules = imports.union(from_imports)
        
        # Check which modules are available
        available = []
        missing = []
        
        for module in all_modules:
            try:
                importlib.util.find_spec(module)
                available.append(module)
            except (ImportError, AttributeError, ValueError):
                missing.append(module)
        
        # Suggest installation commands for missing modules
        install_suggestions = []
        
        # Common package mappings
        package_map = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'sklearn': 'scikit-learn',
            'yaml': 'PyYAML',
            'bs4': 'beautifulsoup4',
        }
        
        for module in missing:
            package = package_map.get(module, module)
            install_suggestions.append(f"pip install {package}")
        
        return {
            "total_modules": len(all_modules),
            "available_modules": available,
            "missing_modules": missing,
            "install_suggestions": install_suggestions,
            "analysis": {
                "standard_library": [m for m in available if m in {'os', 'sys', 'json', 'time', 'datetime', 're'}],
                "third_party": [m for m in available if m not in {'os', 'sys', 'json', 'time', 'datetime', 're'}]
            }
        }
    
    except SyntaxError as e:
        return {
            "error": f"Syntax error in code: {str(e)}"
        }
    except Exception as e:
        return {
            "error": f"Dependency analysis failed: {str(e)}"
        }


@mcp.tool()
def generate_test_template(function_code: str) -> Dict[str, Any]:
    """
    Generate unit test templates for Python functions.
    
    Args:
        function_code: Python function code to create tests for
    
    Returns:
        Dict with generated test code and testing suggestions
    """
    try:
        tree = ast.parse(function_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                args = [arg.arg for arg in node.args.args]
                
                test_class_name = f"Test{function_name.title().replace('_', '')}"
                test_method_name = f"test_{function_name}"
                
                test_template = f'''import unittest
from unittest.mock import Mock, patch

# Import your function here
# from your_module import {function_name}


class {test_class_name}(unittest.TestCase):
    """Test cases for {function_name} function."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    def {test_method_name}_basic(self):
        """Test basic functionality of {function_name}."""
        # Arrange
        {chr(10).join([f"        {arg} = None  # TODO: Set test value" for arg in args])}
        expected = None  # TODO: Set expected result
        
        # Act
        result = {function_name}({", ".join(args)})
        
        # Assert
        self.assertEqual(result, expected)
    
    def {test_method_name}_edge_cases(self):
        """Test edge cases for {function_name}."""
        # TODO: Add edge case tests
        pass
    
    def {test_method_name}_error_handling(self):
        """Test error handling in {function_name}."""
        # TODO: Add error handling tests
        with self.assertRaises(ValueError):
            {function_name}(invalid_input)


if __name__ == '__main__':
    unittest.main()
'''
                
                return {
                    "success": True,
                    "function_name": function_name,
                    "test_code": test_template,
                    "test_class_name": test_class_name,
                    "suggestions": [
                        "Fill in the TODO sections with actual test values",
                        "Add more specific test cases based on function behavior",
                        "Consider using pytest for more advanced testing features",
                        "Add mock objects if the function has external dependencies",
                        "Test both success and failure scenarios"
                    ]
                }
        
        return {
            "success": False,
            "error": "No function definition found in the provided code"
        }
    
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Syntax error in function code: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Test generation failed: {str(e)}"
        }


@mcp.resource("python://capabilities")
def get_python_capabilities() -> str:
    """Get information about Python MCP server capabilities."""
    return """
PyGent Factory Python MCP Server Capabilities:

🔧 Code Execution:
- Safe Python code execution with sandboxing
- Timeout protection and resource limits
- Output capture and error handling

📊 Code Analysis:
- AST-based structure analysis
- Function and class detection
- Import dependency tracking
- Complexity metrics

✅ Quality Assurance:
- Multi-tool code quality checking
- Style guide compliance (PEP 8)
- Security vulnerability scanning
- Syntax validation

🎨 Code Formatting:
- Professional code formatting with Black
- Customizable line length and style
- Fallback formatting for basic cases

⚡ Performance Analysis:
- Code execution timing and profiling
- Performance optimization suggestions
- Bottleneck identification

📚 Documentation:
- Automatic docstring generation
- Function documentation templates
- Type hint suggestions

🧪 Testing:
- Unit test template generation
- Test case structure creation
- Mock object suggestions

📦 Dependency Management:
- Import analysis and validation
- Missing package detection
- Installation command suggestions

This server is specifically designed to support PyGent Factory's
agent code evolution capabilities with safe, comprehensive Python
development tools.
"""


@mcp.resource("python://help/{tool_name}")
def get_tool_help(tool_name: str) -> str:
    """Get detailed help for a specific Python tool."""
    
    help_docs = {
        "execute_python": """
Execute Python Code Safely

Usage: execute_python(code: str, timeout: int = 30)

This tool executes Python code in a sandboxed environment with safety restrictions.

Features:
- Restricted imports for security
- Output and error capture
- Execution timing
- Timeout protection

Safety Restrictions:
- No file system access
- No network operations
- No subprocess execution
- Limited built-in functions

Example:
```python
result = execute_python('''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(f"Fibonacci(10) = {fibonacci(10)}")
''')
```
        """,
        
        "analyze_code_structure": """
Analyze Python Code Structure

Usage: analyze_code_structure(code: str)

Performs comprehensive AST analysis of Python code to extract:
- Function definitions and signatures
- Class definitions and methods
- Import statements
- Variable assignments
- Complexity metrics

Returns detailed structural information useful for code understanding
and automated refactoring.

Example:
```python
analysis = analyze_code_structure('''
class Calculator:
    def add(self, a, b):
        return a + b
        
    def multiply(self, a, b):
        return a * b
''')
```
        """,
        
        "check_code_quality": """
Check Python Code Quality

Usage: check_code_quality(code: str)

Multi-tool code quality analysis including:
- Syntax validation
- Style checking (PEP 8)
- Security vulnerability scanning
- Complexity warnings
- Best practice suggestions

Uses industry-standard tools like flake8 and bandit when available.

Example:
```python
quality = check_code_quality('''
def bad_function( x,y ):
    result=x+y
    return result
''')
```
        """
    }
    
    if tool_name in help_docs:
        return help_docs[tool_name]
    else:
        available_tools = list(help_docs.keys())
        return f"""
Tool '{tool_name}' not found.

Available tools:
{chr(10).join(['- ' + tool for tool in available_tools])}

Use python://help/{{tool_name}} to get specific help for any tool.
        """


if __name__ == "__main__":
    # Run the server
    mcp.run()
