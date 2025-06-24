"""
Simple test to verify pytest is working.
"""

def test_simple_math():
    """Test basic math operations."""
    assert 2 + 2 == 4
    assert 3 * 3 == 9
    assert 10 / 2 == 5


def test_string_operations():
    """Test string operations."""
    text = "PyGent Factory"
    assert "PyGent" in text
    assert text.startswith("PyGent")
    assert text.endswith("Factory")


def test_list_operations():
    """Test list operations."""
    items = [1, 2, 3, 4, 5]
    assert len(items) == 5
    assert 3 in items
    assert items[0] == 1
    assert items[-1] == 5
