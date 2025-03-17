# Checkers Game

A simple implementation of the Checkers (Draughts) game in Python using Pygame. The game features both 2-player mode and a simple bot opponent.

## Features

- 2-player mode with timer (5 minutes per player)
- Bot mode with a simple AI opponent
- King pieces (when a piece reaches the opposite end)
- Valid move highlighting
- Forced capture rules
- Timer display for each player

## Requirements

- Python 3.x
- Pygame 2.5.2

## Installation

1. Make sure you have Python installed on your system
2. Install the required package:

```bash
pip install -r requirements.txt
```

## How to Play

1. Run the game:

```bash
python checker_game.py
```

2. Select game mode:

   - Press '1' for 2-player mode
   - Press '2' to play against the bot

3. Game Rules:
   - Red moves first
   - Click on your piece to select it
   - Green circles show valid moves
   - Pieces move diagonally forward
   - Kings can move both forward and backward
   - If a capture is available, it must be taken
   - Each player has 5 minutes total time

## Improving the Bot

The current bot implementation (in the `bot_move` method of the `Game` class) uses a simple random strategy. Here are some ways you can improve it:

1. Add Move Evaluation:

   - Assign scores to different moves based on:
     - Distance to becoming a king
     - Number of pieces captured
     - Board position (center control)
     - Piece protection

2. Implement Minimax Algorithm:

   - Add depth-first search to look ahead several moves
   - Use alpha-beta pruning for better performance
   - Consider move ordering to improve pruning

3. Add Position Evaluation:

   - Count piece advantage
   - Value kings more than regular pieces
   - Consider piece positioning
   - Evaluate pawn structure

4. Implement Opening Book:
   - Add common opening moves
   - Store and use successful move sequences

Example structure for an improved bot:

```python
def evaluate_position(self, board):
    # Add position evaluation logic
    pass

def minimax(self, depth, board, alpha, beta, maximizing_player):
    # Add minimax implementation
    pass

def get_best_move(self):
    # Use minimax to find the best move
    pass
```

## Code Structure

- `Piece` class: Handles individual checker pieces
- `Board` class: Manages the game board and valid moves
- `Game` class: Controls game flow and user interaction
- `main()`: Handles game initialization and main loop

Feel free to modify and improve the code!
