"""
Neural network for checkers evaluation
This module provides a simple neural network implementation for evaluating checker board positions
"""

import numpy as np
import os
import pickle
from constants import *

class CheckersNN:
    """
    A simple neural network for checkers position evaluation
    """
    def __init__(self, model_path=None):
        # Define network architecture: input -> hidden -> output
        self.input_size = 64  # 8x8 board flattened
        self.hidden_size = 32
        self.output_size = 1  # Single value output (position score)
        
        # Initialize weights with pre-trained values 
        # Weights are initialized with values that emphasize key checker strategies
        np.random.seed(42)  # For reproducibility
        
        # Initialize weights with small random values
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.bias2 = np.zeros((1, self.output_size))
        
        # Apply domain knowledge to weights
        self._initialize_with_domain_knowledge()
        
        # If a model path is provided, load the weights from it
        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)
    
    def _initialize_with_domain_knowledge(self):
        """Apply checkers domain knowledge to weights initialization"""
        # Enhance importance of center squares
        center_indices = [27, 28, 35, 36]  # Center squares (in 8x8 board, flattened)
        for idx in center_indices:
            self.weights1[idx, :] *= 1.5
        
        # Enhance importance of back row for king safety
        back_row_p1 = [56, 58, 60, 62]  # Player 1's back row
        back_row_p2 = [1, 3, 5, 7]      # Player 2's back row
        
        for idx in back_row_p1 + back_row_p2:
            self.weights1[idx, :] *= 1.3
        
        # Enhance importance of edges (edge control)
        edge_indices = [8, 16, 24, 32, 40, 48, 23, 31, 39, 47, 55, 63]
        for idx in edge_indices:
            self.weights1[idx, :] *= 0.7  # Actually discourage edge positions slightly
            
    def load_weights(self, model_path):
        """Load neural network weights from a saved model file"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            # Update weights and biases
            self.weights1 = model_data['weights1']
            self.bias1 = model_data['bias1']
            self.weights2 = model_data['weights2']
            self.bias2 = model_data['bias2']
            
            print(f"Successfully loaded weights from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
            
    def board_to_input(self, board):
        """Convert a board position to neural network input"""
        input_vector = np.zeros(self.input_size)
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                idx = row * BOARD_SIZE + col
                piece = board.get_piece(row, col)
                
                if piece != 0:
                    if piece.color == PLAYER1_COLOR:  # Black (player)
                        input_vector[idx] = -1 if not piece.king else -2
                    else:  # White (bot)
                        input_vector[idx] = 1 if not piece.king else 2
        
        return input_vector.reshape(1, -1)  # Reshape to batch size of 1
    
    def forward(self, x):
        """Forward pass through the network"""
        # First layer
        self.layer1 = self._relu(np.dot(x, self.weights1) + self.bias1)
        # Output layer
        output = np.dot(self.layer1, self.weights2) + self.bias2
        return output[0, 0]  # Return scalar value
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def evaluate(self, board):
        """Evaluate a board position using the neural network"""
        # Convert board to input format
        x = self.board_to_input(board)
        
        # Get raw neural network output
        nn_output = self.forward(x)
        
        # Add some traditional evaluation components for stability
        material_score = self._calculate_material_score(board)
        positional_score = self._calculate_positional_score(board)
        
        # Combine neural network output with traditional evaluation
        # This helps stabilize the evaluation while still benefiting from NN pattern recognition
        final_score = 0.6 * nn_output + 0.3 * material_score + 0.1 * positional_score
        
        return final_score
    
    def _calculate_material_score(self, board):
        """Calculate material score (piece count and type)"""
        score = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board.get_piece(row, col)
                if piece != 0:
                    value = 5 if piece.king else 1
                    if piece.color == PLAYER2_COLOR:  # Bot pieces
                        score += value
                    else:  # Player pieces
                        score -= value
        return score
    
    def _calculate_positional_score(self, board):
        """Calculate positional score based on piece positioning"""
        score = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board.get_piece(row, col)
                if piece != 0:
                    # Center control bonus
                    center_dist = abs(col - 3.5) + abs(row - 3.5)
                    center_bonus = 0.15 * (4 - center_dist / 2)
                    
                    # Advancement bonus for non-kings
                    if not piece.king:
                        if piece.color == PLAYER2_COLOR:  # Bot pieces
                            advance_bonus = 0.1 * row
                        else:  # Player pieces
                            advance_bonus = 0.1 * (BOARD_SIZE - 1 - row)
                    else:
                        advance_bonus = 0
                    
                    # Calculate total position bonus
                    position_bonus = center_bonus + advance_bonus
                    
                    # Apply to score
                    if piece.color == PLAYER2_COLOR:  # Bot pieces
                        score += position_bonus
                    else:  # Player pieces
                        score -= position_bonus
        
        return score

# Create a singleton instance for use throughout the application
# By default, use the untrained model. After training, you can update this path
model_path = None
if os.path.exists('models/checkers_model_final.pkl'):
    model_path = 'models/checkers_model_final.pkl'

checkers_nn = CheckersNN(model_path)