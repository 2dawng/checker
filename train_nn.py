# Checkers neural network training module
import numpy as np
import time
import random
import pickle
from tqdm import tqdm
import os
from constants import *
from board import Board
from piece import Piece
from neural_net import CheckersNN
from game_logic import minimax_alpha_beta

class TrainingEnvironment:
    """Environment for training the neural network through self-play"""
    
    def __init__(self, games_per_generation=100, iterations=10):
        self.games_per_generation = games_per_generation
        self.iterations = iterations
        self.game_history = []
        self.model_history = []
        self.current_model = CheckersNN()
        self.best_model = self.current_model
        
        # Create models directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
    
    def play_game(self, max_moves=200):
        """Play a complete game between two neural network instances"""
        board = Board()
        moves_history = []
        positions_history = []
        current_turn = PLAYER1_COLOR
        moves_without_progress = 0
        
        # Play until game is over or max moves reached
        for _ in range(max_moves):
            # Check if game is over
            winner = board.check_winner()
            if winner is not None:
                break
            
            # Detect stalemate
            if board.check_stalemate(current_turn):
                winner = "DRAW"
                break
            
            # Get best move using minimax with neural network evaluation
            depth = 3  # Use a fixed depth for training games
            _, best_move = minimax_alpha_beta(
                board, depth, float('-inf'), float('inf'), 
                current_turn == PLAYER2_COLOR, None
            )
            
            if best_move is None:
                # No legal moves available
                winner = PLAYER2_COLOR if current_turn == PLAYER1_COLOR else PLAYER1_COLOR
                break
            
            # Store board position before the move
            position_data = self.encode_position(board, current_turn)
            positions_history.append(position_data)
            
            # Make the move
            piece, move = best_move
            row, col = move
            
            # Get captures
            moves = board.get_valid_moves(piece)
            captured_pieces = moves.get(move, [])
            
            # Record move
            moves_history.append({
                'piece': (piece.row, piece.col),
                'move': move,
                'captures': [(p.row, p.col) for p in captured_pieces],
                'turn': current_turn
            })
            
            # Make the move
            original_position = (piece.row, piece.col)
            board.move(piece, row, col)
            
            # Check if captures
            has_captures = False
            if captured_pieces:
                board.remove(captured_pieces)
                has_captures = True
                moves_without_progress = 0
            else:
                moves_without_progress += 1
                
            # Check for additional captures
            additional_captures = False
            if has_captures:
                next_moves = board.get_valid_moves(piece)
                additional_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                
                # Handle multi-jumps
                while additional_captures:
                    # Get next capture
                    _, follow_move = minimax_alpha_beta(
                        board, 2, float('-inf'), float('inf'),
                        current_turn == PLAYER2_COLOR, None
                    )
                    
                    if not follow_move:
                        break
                        
                    follow_piece, follow_pos = follow_move
                    
                    # Get captures
                    next_moves = board.get_valid_moves(follow_piece)
                    additional_captures_list = next_moves.get(follow_pos, [])
                    
                    # Store position before multi-jump
                    position_data = self.encode_position(board, current_turn)
                    positions_history.append(position_data)
                    
                    # Record move
                    moves_history.append({
                        'piece': (follow_piece.row, follow_piece.col),
                        'move': follow_pos,
                        'captures': [(p.row, p.col) for p in additional_captures_list],
                        'turn': current_turn
                    })
                    
                    # Make the move
                    board.move(follow_piece, follow_pos[0], follow_pos[1])
                    
                    # Remove captured pieces
                    if additional_captures_list:
                        board.remove(additional_captures_list)
                        moves_without_progress = 0
                    
                    # Check for more captures
                    next_moves = board.get_valid_moves(follow_piece)
                    additional_captures = any(len(next_moves[m]) > 0 for m in next_moves)
            
            # Check for draw conditions
            if moves_without_progress >= 40:  # 40 half-moves = 20 full moves
                winner = "DRAW"
                break
                
            # Switch turn if no additional captures
            if not additional_captures:
                current_turn = PLAYER2_COLOR if current_turn == PLAYER1_COLOR else PLAYER1_COLOR
        
        # Determine final result
        result = 0  # Draw
        if winner == PLAYER1_COLOR:
            result = -1  # Player 1 wins
        elif winner == PLAYER2_COLOR:
            result = 1  # Player 2 wins
            
        # Return game history with final result
        return {
            'moves': moves_history,
            'positions': positions_history,
            'result': result
        }
    
    def encode_position(self, board, current_turn):
        """Encode board position for learning"""
        # Create a vector representation of the board
        board_vector = np.zeros(64)
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                idx = row * BOARD_SIZE + col
                piece = board.get_piece(row, col)
                
                if piece != 0:
                    if piece.color == PLAYER1_COLOR:  # Black
                        board_vector[idx] = -1 if not piece.king else -2
                    else:  # White
                        board_vector[idx] = 1 if not piece.king else 2
        
        # Add turn information
        turn_value = 1 if current_turn == PLAYER2_COLOR else -1
        
        # Return the encoded position
        return {
            'board': board_vector,
            'turn': turn_value
        }
    
    def train_generation(self):
        """Train one generation of the neural network"""
        print(f"Starting generation training with {self.games_per_generation} games...")
        
        # Play games and collect training data
        training_data = []
        
        for i in tqdm(range(self.games_per_generation), desc="Playing games"):
            game_data = self.play_game()
            self.game_history.append(game_data)
            
            # Prepare training data
            result = game_data['result']
            positions = game_data['positions']
            
            for pos_data in positions:
                training_data.append((pos_data, result))
        
        # Train the neural network on collected data
        self._train_network(training_data)
        
        # Save model after each generation
        self._save_model()
        
        return self.current_model
    
    def _train_network(self, training_data):
        """Train the neural network on collected game data"""
        print(f"Training network on {len(training_data)} positions...")
        
        # Training hyperparameters
        learning_rate = 0.01
        batch_size = 64
        epochs = 5
        
        # Shuffle the training data
        random.shuffle(training_data)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            # Process in batches
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                batch_loss = self._train_batch(batch, learning_rate)
                total_loss += batch_loss
            
            avg_loss = total_loss / (len(training_data) // batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def _train_batch(self, batch, learning_rate):
        """Train on a single batch of data"""
        total_loss = 0
        gradients_w1 = np.zeros_like(self.current_model.weights1)
        gradients_b1 = np.zeros_like(self.current_model.bias1)
        gradients_w2 = np.zeros_like(self.current_model.weights2)
        gradients_b2 = np.zeros_like(self.current_model.bias2)
        
        # Accumulate gradients for the batch
        for pos_data, target in batch:
            board_vector = pos_data['board']
            turn = pos_data['turn']
            
            # Forward pass
            x = board_vector.reshape(1, -1)
            
            # First layer
            z1 = np.dot(x, self.current_model.weights1) + self.current_model.bias1
            a1 = np.maximum(0, z1)  # ReLU
            
            # Output layer
            z2 = np.dot(a1, self.current_model.weights2) + self.current_model.bias2
            prediction = z2[0, 0]
            
            # Calculate loss
            loss = 0.5 * (prediction - target) ** 2
            total_loss += loss
            
            # Backpropagation
            # Output layer gradients
            dz2 = prediction - target
            dw2 = np.dot(a1.T, dz2.reshape(1, 1))
            db2 = dz2
            
            # Hidden layer gradients
            da1 = np.dot(dz2.reshape(1, 1), self.current_model.weights2.T)
            dz1 = da1.copy()
            dz1[z1 <= 0] = 0  # ReLU derivative
            dw1 = np.dot(x.T, dz1)
            db1 = dz1
            
            # Accumulate gradients
            gradients_w2 += dw2
            gradients_b2 += db2
            gradients_w1 += dw1
            gradients_b1 += db1[0]
        
        # Apply gradients
        batch_size = len(batch)
        self.current_model.weights2 -= learning_rate * gradients_w2 / batch_size
        self.current_model.bias2 -= learning_rate * gradients_b2 / batch_size
        self.current_model.weights1 -= learning_rate * gradients_w1 / batch_size
        self.current_model.bias1 -= learning_rate * gradients_b1 / batch_size
        
        return total_loss / batch_size
    
    def _save_model(self):
        """Save the current model to disk"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_data = {
            'weights1': self.current_model.weights1,
            'bias1': self.current_model.bias1,
            'weights2': self.current_model.weights2,
            'bias2': self.current_model.bias2,
        }
        
        filename = os.path.join('models', f'checkers_model_{timestamp}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
        
        # Keep a reference to this model
        self.model_history.append(filename)
    
    def load_model(self, filename):
        """Load a previously saved model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Update current model
        self.current_model.weights1 = model_data['weights1']
        self.current_model.bias1 = model_data['bias1']
        self.current_model.weights2 = model_data['weights2']
        self.current_model.bias2 = model_data['bias2']
        
        print(f"Model loaded from {filename}")
        return self.current_model
    
    def train(self):
        """Run the full training process"""
        print(f"Starting training process with {self.iterations} iterations")
        
        for i in range(self.iterations):
            print(f"\nIteration {i+1}/{self.iterations}")
            self.train_generation()
        
        print("\nTraining complete!")
        return self.current_model


def main():
    """Main training entry point"""
    print("Checkers Neural Network Training")
    print("-------------------------------")
    
    # Training parameters
    games_per_generation = 50  # Smaller value for faster testing
    iterations = 5
    
    # Create training environment
    trainer = TrainingEnvironment(games_per_generation, iterations)
    
    # Start training
    start_time = time.time()
    final_model = trainer.train()
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Final model saved in models directory")
    
    # Save the final model
    final_model_data = {
        'weights1': final_model.weights1,
        'bias1': final_model.bias1,
        'weights2': final_model.weights2,
        'bias2': final_model.bias2,
    }
    
    final_filename = os.path.join('models', 'checkers_model_final.pkl')
    with open(final_filename, 'wb') as f:
        pickle.dump(final_model_data, f)
    
    print(f"Final model saved to {final_filename}")
    print("\nYou can now use this model in your game by updating neural_net.py")
    print("to load weights from this file.")


if __name__ == "__main__":
    main()
