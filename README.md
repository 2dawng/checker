# Checkers Game

A modern implementation of the classic Checkers (Draughts) game using Python and Pygame. Features both a two-player mode and a sophisticated AI opponent.

## Features

- Two game modes:
  - Two-player mode with timer (33.3 minutes per player)
  - AI mode with adaptive difficulty for single-player gameplay
- Modern chess.com style board design
- Forced capture rules with multi-capture support
- King pieces with special movement rules
- Move highlighting system:
  - Cyan indicators for pieces that can capture
  - Green indicators for valid moves
  - Pulsing animation for selected pieces
- Detailed move logging with algebraic notation
- Game state tracking and display
- Home button for easy navigation
- Animation for piece movement
- Draw detection:
  - Position repetition (5 times)
  - 50 moves without captures
  - 50 moves without non-king piece movement

## Requirements

- Python 3.x
- Pygame 2.5.2

## Installation

1. Ensure Python is installed on your system
2. Install required packages:

```bash
pip install -r requirements.txt
```

## How to Play

1. Run the game:

```bash
python checker_game.py
```

2. Select game mode from the menu:
   - Click "2 Players" for two-player mode
   - Click "vs Bot" to play against the computer

### Game Rules

1. Basic Movement:

   - Regular pieces move diagonally forward only
   - Kings can move diagonally in any direction
   - Pieces capture by jumping over opponent pieces

2. Capture Rules:

   - Captures are mandatory when available
   - Multiple captures must be completed with the same piece
   - The game shows available captures with cyan indicators

3. King Promotion:

   - Pieces become kings when reaching the opposite end
   - Kings are marked with a gold crown
   - Kings can move and capture in any diagonal direction

4. Game End Conditions:
   - Capturing all opponent pieces
   - No legal moves available (stalemate)
   - Time runs out (in two-player mode)
   - Draw conditions:
     - Position repetition: Same position occurs 5 times
     - 50-move rule: 50 moves without captures
     - 50-move rule: 50 moves without moving a non-king piece

### Controls

- Left-click to:
  - Select a piece
  - Move to a valid square
  - Deselect a piece by clicking elsewhere
- Home button to return to mode selection

## Game Interface

- Main board with 8x8 squares
- Side panel showing:
  - Move log in algebraic notation
  - Timer display (two-player mode)
  - Home button
- Visual indicators:
  - Cyan circles: Pieces that can capture
  - Green circles: Valid move destinations
  - Gold highlight: Selected piece
  - Crown symbol: King pieces

## Move Logging

The game automatically logs all moves in algebraic notation:

- Format: `[turn number]. [white's move] [black's move]`
- Example: `1. E3-F4 B6-C5`
- Captures are marked with 'x': `2. F4-D6x1`
- Multiple captures: `F4-H6x1 D6-B4x1`
- King promotions are marked with 'K': `3. A7-B8K`

The move log is displayed in the side panel during gameplay and also saved to a file for later review.

## AI Opponent

The game features an intelligent AI opponent using the minimax algorithm with alpha-beta pruning:

- Adaptive search depth based on game phase
  - Early game: 3-ply depth
  - Mid game: 4-ply depth
  - End game: 5-ply depth for more precise calculations
- Position evaluation considers:
  - Material balance (pieces and kings)
  - Board control and piece positioning
  - Forward progress for regular pieces
  - King mobility
- The AI enforces optimal play by prioritizing captures and considering sequential multi-captures

## File Structure

- `checker_game.py`: Main game logic and UI
- `board.py`: Board state and move validation
- `piece.py`: Piece movement and properties
- `game_logic.py`: AI algorithms, move logging, and game mechanics
- `button.py`: UI button implementation
- `constants.py`: Game constants and colors

## Development

The game is built with a modular structure for easy extension:

- Board logic is separated from UI
- Move validation is handled by the Board class
- Game state management in Game class
- AI logic in separate functions for easy modification
- Constants are centralized for easy modification

Feel free to modify and improve the code!
