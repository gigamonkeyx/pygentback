from snake_game import SnakeGame, Direction
import random


class SnakeAI:
    """Aggressive food-seeking AI for snake game."""
    
    def __init__(self, game: SnakeGame):
        self.game = game
        self.aggression_level = 0.8  # How aggressively to seek food
    
    def get_next_move(self) -> Direction:
        """Calculate next move with aggressive food-seeking behavior."""
        
        # Get current state
        head = self.game.snake[0]
        food = self.game.food
        
        # Calculate all possible moves
        possible_moves = []
        for direction in Direction:
            if self._is_safe_move(direction):
                distance_to_food = self._calculate_food_distance(direction)
                possible_moves.append((direction, distance_to_food))
        
        if not possible_moves:
            # No safe moves, try any direction
            return random.choice(list(Direction))
        
        # Aggressive behavior: always choose move that gets closest to food
        if random.random() < self.aggression_level:
            # Sort by distance to food (ascending)
            possible_moves.sort(key=lambda x: x[1])
            return possible_moves[0][0]
        else:
            # Occasionally make random move to avoid getting stuck
            return random.choice([move[0] for move in possible_moves])
    
    def _is_safe_move(self, direction: Direction) -> bool:
        """Check if a move is safe (won't cause collision)."""
        head_x, head_y = self.game.snake[0]
        dx, dy = direction.value
        new_pos = (head_x + dx, head_y + dy)
        
        # Check walls
        if (new_pos[0] < 0 or new_pos[0] >= self.game.width or
            new_pos[1] < 0 or new_pos[1] >= self.game.height):
            return False
        
        # Check self collision (except tail which will move)
        if new_pos in self.game.snake[:-1]:
            return False
        
        return True
    
    def _calculate_food_distance(self, direction: Direction) -> float:
        """Calculate Manhattan distance to food after making this move."""
        head_x, head_y = self.game.snake[0]
        dx, dy = direction.value
        new_x, new_y = head_x + dx, head_y + dy
        
        food_x, food_y = self.game.food
        return abs(new_x - food_x) + abs(new_y - food_y)
