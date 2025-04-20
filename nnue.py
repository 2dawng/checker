import torch
import numpy as np
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NNUE hyperparameters
input_size = 64  # 8x8 board with piece information
hidden_size = 256  # Hidden layer size

# Initialize weights randomly at first
weights1 = None
bias1 = None
weights2 = None
bias2 = None

def initialize_weights():
    """Initialize weights with random values"""
    global weights1, bias1, weights2, bias2
    weights1 = torch.randn(input_size, hidden_size) / np.sqrt(input_size)
    bias1 = torch.zeros(hidden_size)
    weights2 = torch.randn(hidden_size, 1) / np.sqrt(hidden_size)
    bias2 = torch.zeros(1)

# Initialize weights
initialize_weights()

def load_model():
    """Load the latest NNUE model from the models directory"""
    global weights1, bias1, weights2, bias2
    
    try:
        # Look for NNUE model files
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        # Check if models directory exists
        if not os.path.exists(model_dir):
            logger.warning(f"Models directory not found at {model_dir}")
            return False
            
        # Look for the final model first
        final_model_path = os.path.join(model_dir, 'checkers_nnue_final.pkl')
        if os.path.exists(final_model_path):
            logger.info(f"Loading final NNUE model from {final_model_path}")
            try:
                # In PyTorch 2.6, weights_only=True is default, we need to set it to False
                model_state = torch.load(final_model_path, weights_only=False)
                
                weights1 = model_state['weights1']
                bias1 = model_state['bias1']
                weights2 = model_state['weights2']
                bias2 = model_state['bias2']
                
                logger.info(f"NNUE model loaded successfully: {weights1.shape}")
                return True
            except Exception as e:
                logger.error(f"Error loading NNUE model: {str(e)}")
                # Try loading with old method for backward compatibility
                try:
                    model_state = torch.load(final_model_path)
                    
                    weights1 = model_state['weights1']
                    bias1 = model_state['bias1']
                    weights2 = model_state['weights2']
                    bias2 = model_state['bias2']
                    
                    logger.info(f"NNUE model loaded successfully with legacy method: {weights1.shape}")
                    return True
                except Exception as e2:
                    logger.error(f"Error loading NNUE model with legacy method: {str(e2)}")
                    return False
        
        # If final model not found, look for the latest trained model
        trained_models = glob.glob(os.path.join(model_dir, 'checkers_nnue_trained_*.pkl'))
        if trained_models:
            # Sort by timestamp (newest first)
            trained_models.sort(reverse=True)
            latest_model = trained_models[0]
            
            logger.info(f"Loading latest NNUE model from {latest_model}")
            try:
                # In PyTorch 2.6, weights_only=True is default, we need to set it to False
                model_state = torch.load(latest_model, weights_only=False)
                
                weights1 = model_state['weights1']
                bias1 = model_state['bias1']
                weights2 = model_state['weights2']
                bias2 = model_state['bias2']
                
                logger.info(f"NNUE model loaded successfully: {weights1.shape}")
                return True
            except Exception as e:
                # Try loading with old method for backward compatibility
                try:
                    model_state = torch.load(latest_model)
                    
                    weights1 = model_state['weights1']
                    bias1 = model_state['bias1']
                    weights2 = model_state['weights2']
                    bias2 = model_state['bias2']
                    
                    logger.info(f"NNUE model loaded successfully with legacy method: {weights1.shape}")
                    return True
                except Exception as e2:
                    logger.error(f"Error loading NNUE model with legacy method: {str(e2)}")
                    return False
            
        # If trained models not found, look for any NNUE model
        nnue_models = glob.glob(os.path.join(model_dir, 'checkers_nnue_*.pkl'))
        if nnue_models:
            # Sort by timestamp (newest first)
            nnue_models.sort(reverse=True)
            latest_model = nnue_models[0]
            
            logger.info(f"Loading NNUE model from {latest_model}")
            try:
                # In PyTorch 2.6, weights_only=True is default, we need to set it to False
                model_state = torch.load(latest_model, weights_only=False)
                
                weights1 = model_state['weights1']
                bias1 = model_state['bias1']
                weights2 = model_state['weights2']
                bias2 = model_state['bias2']
                
                logger.info(f"NNUE model loaded successfully: {weights1.shape}")
                return True
            except Exception as e:
                # Try loading with old method for backward compatibility
                try:
                    model_state = torch.load(latest_model)
                    
                    weights1 = model_state['weights1']
                    bias1 = model_state['bias1']
                    weights2 = model_state['weights2']
                    bias2 = model_state['bias2']
                    
                    logger.info(f"NNUE model loaded successfully with legacy method: {weights1.shape}")
                    return True
                except Exception as e2:
                    logger.error(f"Error loading NNUE model with legacy method: {str(e2)}")
                    return False
            
        logger.warning("No NNUE model files found")
        return False
    except Exception as e:
        logger.error(f"Error loading NNUE model: {str(e)}")
        return False

# Try to load the model when the module is imported
model_loaded = load_model()
logger.info(f"NNUE model loaded: {model_loaded}")

# Constants for the board representation
EMPTY = 0
BLACK = 1  # Player 1 (typically black pieces)
WHITE = 2  # Player 2 (typically white pieces)
KING_FACTOR = 2  # Multiplier for king pieces

def board_to_features(board):
    """Convert a board object to a feature vector for the NNUE"""
    features = torch.zeros(input_size)
    
    # Iterate through the board and set feature values
    for row in range(8):
        for col in range(8):
            piece = board.get_piece(row, col)
            index = row * 8 + col
            
            if piece != 0:
                if piece.color == BLACK:
                    features[index] = 1 * (KING_FACTOR if piece.king else 1)
                else:  # WHITE
                    features[index] = -1 * (KING_FACTOR if piece.king else 1)
    
    return features

def incremental_update(board, old_features, move):
    """Update the feature vector incrementally based on a move"""
    new_features = old_features.clone()
    
    # Extract move information
    piece, (new_row, new_col) = move
    old_row, old_col = piece.row, piece.col  # These should be the old positions
    
    # Clear the old position and set the new position
    old_index = old_row * 8 + old_col
    new_index = new_row * 8 + new_col
    
    # Reset old position to empty
    new_features[old_index] = 0
    
    # Set new position with the piece
    if piece.color == BLACK:
        new_features[new_index] = 1 * (KING_FACTOR if piece.king else 1)
    else:  # WHITE
        new_features[new_index] = -1 * (KING_FACTOR if piece.king else 1)
    
    return new_features

def evaluate(board, old_features=None, move=None):
    """Evaluate a board position using the NNUE"""
    # Convert board to features if not provided
    if old_features is None or move is None:
        features = board_to_features(board)
    else:
        # Use incremental update for efficiency
        features = incremental_update(board, old_features, move)
    
    # First layer
    hidden = torch.relu(torch.matmul(features, weights1) + bias1)
    
    # Output layer (single value)
    output = torch.matmul(hidden, weights2) + bias2
    
    return output.item()

# Load the model again just in case it wasn't loaded during import
if not model_loaded:
    model_loaded = load_model()
    logger.info(f"NNUE model loaded on second attempt: {model_loaded}")

# Create a module-level object for the game_logic module to import
class CheckersNNUE:
    def __init__(self):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights1 = weights1
        self.bias1 = bias1
        self.weights2 = weights2
        self.bias2 = bias2
    
    def evaluate(self, board, old_features=None, move=None):
        return evaluate(board, old_features, move)
    
    def board_to_features(self, board):
        return board_to_features(board)
    
    def incremental_update(self, board, old_features, move):
        return incremental_update(board, old_features, move)

# Create an instance for importing
checkers_nnue = CheckersNNUE()
print(f"NNUE model loaded: {model_loaded}")
print(f"NNUE model input size: {input_size}, hidden size: {hidden_size}")
