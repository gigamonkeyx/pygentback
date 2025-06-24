import random
import json
from typing import List, Tuple, Dict, Any
from enum import Enum


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1) 
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class SnakeGame:
    """Snake game with 3 lives and aggressive food-seeking behavior."""
    
    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.reset_game()
    
    def reset_game(self):
        """Reset game to initial state."""
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = Direction.RIGHT
        self.food = self._generate_food()
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.won = False
    
    def _generate_food(self) -> Tuple[int, int]:
        """Generate food at random position not occupied by snake."""
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food
    
    def move(self) -> bool:
        """Move snake one step. Returns True if move was successful."""
        if self.game_over:
            return False
        
        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        
        # Check collision with walls
        if (new_head[0] < 0 or new_head[0] >= self.width or 
            new_head[1] < 0 or new_head[1] >= self.height):
            return self._handle_collision()
        
        # Check collision with self
        if new_head in self.snake:
            return self._handle_collision()
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        if new_head == self.food:
            self.score += 10
            self.food = self._generate_food()
            # Snake grows (don't remove tail)
        else:
            # Remove tail (snake doesn't grow)
            self.snake.pop()
        
        return True
    
    def _handle_collision(self) -> bool:
        """Handle collision - lose a life."""
        self.lives -= 1
        if self.lives <= 0:
            self.game_over = True
            return False
        else:
            # Reset snake position but keep score
            self.snake = [(self.width // 2, self.height // 2)]
            self.direction = Direction.RIGHT
            return True
    
    def change_direction(self, new_direction: Direction):
        """Change snake direction if valid."""
        # Can't reverse directly
        current_dx, current_dy = self.direction.value
        new_dx, new_dy = new_direction.value
        
        if (current_dx, current_dy) != (-new_dx, -new_dy):
            self.direction = new_direction
    
    def get_state(self) -> Dict[str, Any]:
        """Get current game state."""
        return {
            "snake": self.snake,
            "food": self.food,
            "score": self.score,
            "lives": self.lives,
            "game_over": self.game_over,
            "width": self.width,
            "height": self.height,
            "direction": self.direction.name
        }
    
    def get_food_direction(self) -> Direction:
        """Calculate direction towards food for AI."""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Simple aggressive seeking - go towards food
        dx = food_x - head_x
        dy = food_y - head_y
        
        # Prioritize the larger difference
        if abs(dx) > abs(dy):
            return Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            return Direction.DOWN if dy > 0 else Direction.UP
