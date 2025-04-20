# NNUE Training Module
import numpy as np
import os
import time
import random
import pickle
from tqdm import tqdm
import glob
import re
from constants import *
from board import Board
from piece import Piece
from nnue import CheckersNNUE, checkers_nnue
from game_logic import nnue_search

class NNUETrainer:
    """
    Trainer for the NNUE model using self-play and optionally existing game data
    """
    def __init__(self, games_per_iteration=20, iterations=5, learning_rate=0.01, batch_size=32):
        self.games_per_iteration = games_per_iteration
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.game_positions = []
        self.game_results = []
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # Check if we should create a new model or use the existing one
        if hasattr(checkers_nnue, 'weights1'):
            self.model = checkers_nnue
            print("Using existing NNUE model")
        else:
            self.model = CheckersNNUE()
            print("Created new NNUE model")
            
    def parse_game_logs(self, max_games=50):
        """
        Parse existing game logs to extract positions and outcomes for training
        """
        print("Parsing game logs for training data...")
        log_files = glob.glob(os.path.join('logs', 'bot', '*.txt'))
        
        # Sort by date (newest first) and limit to max_games
        log_files.sort(reverse=True)
        log_files = log_files[:max_games]
        
        positions = []
        results = []
        
        for log_file in tqdm(log_files, desc="Processing log files"):
            try:
                # Read the log file
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Determine the game outcome
                if "Black wins" in content:
                    result = -1  # Player (Black) won
                elif "White wins" in content:
                    result = 1   # Bot (White) won
                else:
                    result = 0   # Draw or incomplete game
                
                # Parse the moves to reconstruct the game
                board = Board()
                move_pattern = r'([a-h][1-8])-([a-h][1-8])(?:x(\d+))?(?:K)?'
                moves = re.findall(move_pattern, content)
                
                # Replay the game
                current_color = PLAYER1_COLOR  # Black starts
                for move in moves:
                    from_sq, to_sq, captures = move
                    
                    # Convert algebraic notation to board coordinates
                    from_col = ord(from_sq[0]) - ord('a')
                    from_row = int(from_sq[1]) - 1
                    to_col = ord(to_sq[0]) - ord('a')
                    to_row = int(to_sq[1]) - 1
                    
                    # Find the piece
                    piece = board.get_piece(from_row, from_col)
                    if piece == 0 or piece.color != current_color:
                        continue  # Skip invalid moves
                    
                    # Check if move is valid
                    valid_moves = board.get_valid_moves(piece)
                    if (to_row, to_col) not in valid_moves:
                        continue  # Skip invalid moves
                        
                    # Store the position before the move
                    positions.append(board.copy())
                    results.append(result)
                    
                    # Make the move
                    board.move(piece, to_row, to_col)
                    
                    # Handle captures
                    if captures and captures != '':
                        num_captures = int(captures)
                        captured_pieces = valid_moves.get((to_row, to_col), [])
                        if captured_pieces:
                            board.remove(captured_pieces[:num_captures])
                    
                    # Switch turn
                    current_color = PLAYER2_COLOR if current_color == PLAYER1_COLOR else PLAYER1_COLOR
                    
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
                continue
        
        print(f"Extracted {len(positions)} positions from {len(log_files)} game logs")
        self.game_positions.extend(positions)
        self.game_results.extend(results)
        
    def play_self_play_games(self):
        """
        Play games between the NNUE model against itself to generate training data
        """
        print(f"Playing {self.games_per_iteration} self-play games...")
        
        for game_idx in tqdm(range(self.games_per_iteration), desc="Self-play games"):
            board = Board()
            positions = []
            current_color = PLAYER1_COLOR  # Black starts
            moves_without_progress = 0
            
            # Play until game over or draw
            while True:
                # Check for game over
                winner = board.check_winner()
                if winner is not None:
                    result = 1 if winner == PLAYER2_COLOR else -1
                    break
                    
                # Check for draw conditions
                if moves_without_progress >= 40:  # 40 moves without progress = draw
                    result = 0
                    break
                
                # Save the current position
                positions.append(board.copy())
                
                # Get the best move using NNUE search
                search_depth = 4  # Use a lower depth for faster training
                _, best_move = nnue_search(
                    board, search_depth, float('-inf'), float('inf'), 
                    current_color == PLAYER2_COLOR, None
                )
                
                if best_move is None:
                    # No legal moves
                    result = -1 if current_color == PLAYER2_COLOR else 1
                    break
                
                piece, move = best_move
                row, col = move
                
                # Get valid moves and check for captures
                valid_moves = board.get_valid_moves(piece)
                captured_pieces = valid_moves.get(move, [])
                
                # Make the move
                board.move(piece, row, col)
                
                # Handle captures
                if captured_pieces:
                    board.remove(captured_pieces)
                    moves_without_progress = 0
                else:
                    moves_without_progress += 1
                    
                # Check for additional captures
                next_moves = board.get_valid_moves(piece)
                additional_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                
                while additional_captures:
                    # Find best next capture
                    _, follow_move = nnue_search(
                        board, 2, float('-inf'), float('inf'),
                        current_color == PLAYER2_COLOR, None
                    )
                    
                    if not follow_move:
                        break
                        
                    follow_piece, follow_pos = follow_move
                    
                    # Save the position before the follow-up move
                    positions.append(board.copy())
                    
                    # Get captures for this move
                    next_captures = next_moves.get(follow_pos, [])
                    
                    # Make the move
                    board.move(follow_piece, follow_pos[0], follow_pos[1])
                    
                    # Remove captured pieces
                    if next_captures:
                        board.remove(next_captures)
                        moves_without_progress = 0
                    
                    # Check for more captures
                    next_moves = board.get_valid_moves(follow_piece)
                    additional_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                
                # Switch turn if no additional captures
                if not additional_captures:
                    current_color = PLAYER2_COLOR if current_color == PLAYER1_COLOR else PLAYER1_COLOR
            
            # Add positions with the game result
            self.game_positions.extend(positions)
            self.game_results.extend([result] * len(positions))
            
    def train_iteration(self):
        """
        Train the NNUE model on collected positions for one iteration
        """
        if not self.game_positions:
            print("No training data available. Skipping training.")
            return
            
        print(f"Training NNUE on {len(self.game_positions)} positions...")
        
        # Combine positions and results into pairs
        training_data = list(zip(self.game_positions, self.game_results))
        
        # Shuffle the data
        random.shuffle(training_data)
        
        # Split into training and validation sets (80/20)
        split_idx = int(len(training_data) * 0.8)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Train the model
        self.model.train_from_positions(
            [pos for pos, _ in train_data],
            [res for _, res in train_data],
            learning_rate=self.learning_rate,
            epochs=3,
            batch_size=self.batch_size
        )
        
        # Validate the model
        val_loss = self._validate_model(val_data)
        print(f"Validation loss: {val_loss:.6f}")
        
        # Save the model
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.model.save_weights(f"models/checkers_nnue_{timestamp}.pkl")
        
    def _validate_model(self, validation_data):
        """
        Calculate the validation loss on a set of positions
        """
        total_loss = 0
        
        for board, target in validation_data:
            features = self.model.board_to_features(board)
            z1 = np.dot(features, self.model.weights1) + self.model.bias1
            a1 = np.maximum(0, z1)  # ReLU
            z2 = np.dot(a1, self.model.weights2) + self.model.bias2
            prediction = z2[0, 0]
            
            # MSE loss
            loss = 0.5 * (prediction - target) ** 2
            total_loss += loss
            
        return total_loss / len(validation_data) if validation_data else 0
        
    def train(self):
        """
        Run the complete training process
        """
        print("Starting NNUE training process...")
        
        # Parse existing game logs
        self.parse_game_logs()
        
        # Run training iterations
        for iteration in range(self.iterations):
            print(f"\nIteration {iteration+1}/{self.iterations}")
            
            # Play self-play games
            self.play_self_play_games()
            
            # Train on collected data
            self.train_iteration()
            
        # Save the final model
        self.model.save_weights("models/checkers_nnue_final.pkl")
        print("\nTraining complete! Final model saved to models/checkers_nnue_final.pkl")
        
        # Update the global NNUE instance
        global checkers_nnue
        checkers_nnue = self.model


def main():
    print("NNUE Training for Checkers")
    print("=========================")
    
    # Get training parameters
    games_per_iteration = int(input("Enter number of self-play games per iteration [20]: ") or "20")
    iterations = int(input("Enter number of training iterations [5]: ") or "5")
    learning_rate = float(input("Enter learning rate [0.01]: ") or "0.01")
    
    # Create trainer
    trainer = NNUETrainer(
        games_per_iteration=games_per_iteration,
        iterations=iterations,
        learning_rate=learning_rate
    )
    
    # Start training
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print("You can now use the trained model in your game!")


if __name__ == "__main__":
    main()
