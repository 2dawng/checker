import pygame

# Board Constants
BOARD_SIZE = 8
SQUARE_SIZE = 80  # Reduced board size
BOARD_WIDTH = BOARD_SIZE * SQUARE_SIZE
SIDE_PANEL_WIDTH = 200
WINDOW_SIZE = (BOARD_WIDTH + SIDE_PANEL_WIDTH, BOARD_WIDTH)

# Colors (chess.com style)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_SQUARE = (238, 238, 210)  # Light square color
DARK_SQUARE = (118, 150, 86)    # Dark square color
PLAYER1_COLOR = (0, 0, 0)       # Black pieces
PLAYER2_COLOR = (255, 255, 255)  # White pieces
CROWN_COLOR = (255, 215, 0)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER_COLOR = (100, 160, 210)
PANEL_COLOR = (40, 40, 40)
LAST_MOVE_COLOR = (230, 230, 250)  # Lavender for last move indicator
CAPTURE_COLOR = (219, 112, 147)    # Light-mid red for captured pieces indicator

WOOD_TEXTURE = (189, 154, 122)
BOARD_BORDER = (78, 53, 36)
BACKGROUND_COLOR = (28, 28, 28)
NEON_HIGHLIGHT = (0, 255, 255)
BUTTON_GRADIENT = [(40, 40, 40), (60, 60, 60)]
