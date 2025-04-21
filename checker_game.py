import pygame
import sys
import time
import math
import os
from typing import List, Tuple, Optional
from constants import *
from button import Button
from board import Board
from game_logic import log_move, draw_move_log, draw_valid_moves, bot_move

# Initialize Pygame
pygame.init()

class Game:
    def __init__(self, win, vs_bot=True):
        self.win = win
        self.vs_bot = vs_bot
        self.board = Board()
        self.selected_piece = None
        self.turn = PLAYER1_COLOR
        self.valid_moves = {}
        self.game_over = False
        self.winner = None
        self.start_time = time.time()
        self.player1_time = 2000  # 33.3 minutes in seconds
        self.player2_time = 2000
        self.last_time_update = time.time()
        self.animation_start_time = time.time()
        self.move_number = 0  # Start at 0 to get correct numbering
        self.move_log = []
        self.log_file = None
        self.setup_log_file()
        self.home_button = Button(BOARD_WIDTH + 20, 20, 160, 40, "Home")
        self.bot_thinking = False
        # Add last move animation tracking
        self.last_moved_piece = None
        self.last_move_start = None  # Track the starting position of the last moved piece
        self.last_captured_pieces = []  # Track pieces captured in the last move
        self.pulse_speed = 3.0  # Speed of the pulse animation
        self.current_turn_number = 1  # Initialize turn number to 1
        self.turn_started = False  # Track if turn has started
        self.capturing_piece = None  # Track the piece that just made a capture
        self.last_capture_color = None  # Track color of last capturing piece
        
        # For draw detection
        self.position_history = []  # Track board positions for repetition detection
        self.moves_without_capture = 0  # Track moves without capture
        self.moves_without_pawn_move = 0  # Track moves without non-king piece movement

    def setup_log_file(self):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Create a timestamp for the filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode_dir = 'bot' if self.vs_bot else 'pvp'
        filename = os.path.join('logs', f"{mode_dir}/{timestamp}.txt")

        # Create mode subdirectory if it doesn't exist
        os.makedirs(os.path.join('logs', mode_dir), exist_ok=True)

        self.log_file = open(filename, 'w')

        # Write header
        self.log_file.write(
            f"Checkers Game Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(
            f"Mode: {'Bot' if self.vs_bot else 'Two Player'}\n")
        self.log_file.write("-" * 50 + "\n\n")

    def check_winner(self):
        # First check for piece elimination
        player1_pieces = 0
        player2_pieces = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board.get_piece(row, col)
                if piece != 0:
                    if piece.color == PLAYER1_COLOR:
                        player1_pieces += 1
                    else:
                        player2_pieces += 1

        if player1_pieces == 0:
            self.game_over = True
            return 'Player 2 WINS!'
        elif player2_pieces == 0:
            self.game_over = True
            return 'Player 1 WINS!'

        # Then check for stalemate only if both players still have pieces
        if self.board.check_stalemate(self.turn):
            return 'Stalemate - Draw'

        return None

    def check_draw(self):
        """
        Check if the game is a draw based on:
        1. Position repetition (5 times)
        2. 50 moves without captures
        3. 50 moves without non-king piece movement
        """
        # Check for position repetition (5 times)
        current_state = self.encode_board_state()
        position_count = self.position_history.count(current_state) + 1  # +1 for current position
        
        if position_count >= 5:
            self.game_over = True
            return "Position Repetition - Draw"
            
        # Check for 50 moves without captures
        if self.moves_without_capture >= 100:  # 100 half-moves = 50 full moves
            self.game_over = True
            return "No Captures in 50 Moves - Draw"
            
        # Check for 50 moves without pawn movement
        if self.moves_without_pawn_move >= 100:  # 100 half-moves = 50 full moves
            self.game_over = True
            return "No Pawn Moves in 50 Moves - Draw"
            
        return None

    def update(self):
        # Draw board and pieces
        self.board.draw(self.win)
        
        # Draw last move's starting square with light lavender color (reduced opacity)
        if self.last_move_start and not self.game_over:
            start_row, start_col = self.last_move_start
            start_rect = pygame.Rect(start_col * SQUARE_SIZE, start_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            # Draw very light lavender overlay (reduced opacity to 80)
            lavender_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            lavender_surface.fill((*LAST_MOVE_COLOR, 80))  # Reduced alpha for subtle effect
            self.win.blit(lavender_surface, start_rect)
        
        # Draw last captured pieces with light red color (reduced opacity)
        if self.last_captured_pieces and not self.game_over:
            for piece in self.last_captured_pieces:
                captured_rect = pygame.Rect(piece.col * SQUARE_SIZE, piece.row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                # Draw very light red overlay (reduced opacity to 80)
                capture_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                capture_surface.fill((*CAPTURE_COLOR, 80))  # Reduced alpha for subtle effect
                self.win.blit(capture_surface, captured_rect)

        # Draw last move indicator (gold circle around destination square)
        if self.last_moved_piece and not self.game_over:
            x = self.last_moved_piece.col * SQUARE_SIZE + SQUARE_SIZE // 2
            y = self.last_moved_piece.row * SQUARE_SIZE + SQUARE_SIZE // 2
            pygame.draw.circle(self.win, CROWN_COLOR,
                               (x, y), SQUARE_SIZE//2 - 8, 3)

        # Draw selected piece highlight with pulsing animation
        if self.selected_piece and (self.turn == PLAYER1_COLOR or not self.vs_bot):
            current_time = time.time()
            elapsed = current_time - self.animation_start_time
            # Calculate pulse size using sine wave (continuous)
            pulse = abs(math.sin(elapsed * self.pulse_speed)) * \
                5  # Pulse between 0 and 5 pixels
            x = self.selected_piece.col * SQUARE_SIZE + SQUARE_SIZE // 2
            y = self.selected_piece.row * SQUARE_SIZE + SQUARE_SIZE // 2
            pygame.draw.circle(self.win, CROWN_COLOR, (x, y),
                               SQUARE_SIZE//2 - 4 + pulse, 3)

        # Only draw valid moves for human player
        if self.turn == PLAYER1_COLOR or not self.vs_bot:
            draw_valid_moves(self)

        # Draw side panel
        pygame.draw.rect(self.win, PANEL_COLOR, (BOARD_WIDTH,
                         0, SIDE_PANEL_WIDTH, WINDOW_SIZE[1]))

        # Draw timers
        self.draw_timer()

        # Draw move log
        draw_move_log(self)

        # Draw home button
        self.home_button.draw(self.win)

        pygame.display.update()

    def draw_timer(self):
        if not self.vs_bot:  # Only draw timer in 2-player mode
            font = pygame.font.Font(None, 36)

            # Timer background colors
            P1_BG = (50, 50, 50)  # Dark gray for Player 1
            P2_BG = (50, 50, 50)  # Dark gray for Player 2
            BORDER_COLOR = (100, 100, 100)  # Light gray for borders

            # Timer text colors
            P1_TEXT = (255, 255, 255)  # White for Player 1
            P2_TEXT = (255, 255, 255)  # White for Player 2

            # Timer box dimensions
            BOX_WIDTH = 160
            BOX_HEIGHT = 60
            BOX_PADDING = 10

            # Player 2 timer box (moved up)
            p2_box = pygame.Rect(BOARD_WIDTH + 20, 70, BOX_WIDTH, BOX_HEIGHT)
            pygame.draw.rect(self.win, P2_BG, p2_box)
            pygame.draw.rect(self.win, BORDER_COLOR, p2_box, 2)

            # Player 2 timer
            p2_timer = font.render(f"{int(self.player2_time)}s", True, P2_TEXT)
            self.win.blit(p2_timer, (p2_box.x + BOX_PADDING,
                          p2_box.y + BOX_HEIGHT//2 - 10))

            # Player 1 timer box (below Player 2)
            p1_box = pygame.Rect(BOARD_WIDTH + 20, 140, BOX_WIDTH, BOX_HEIGHT)
            pygame.draw.rect(self.win, P1_BG, p1_box)
            pygame.draw.rect(self.win, BORDER_COLOR, p1_box, 2)

            # Player 1 timer
            p1_timer = font.render(f"{int(self.player1_time)}s", True, P1_TEXT)
            self.win.blit(p1_timer, (p1_box.x + BOX_PADDING,
                          p1_box.y + BOX_HEIGHT//2 - 10))

    def update_timer(self):
        if not self.vs_bot and not self.game_over:  # Only update timer in 2-player mode
            current_time = time.time()
            elapsed = current_time - self.last_time_update
            self.last_time_update = current_time

            if self.turn == PLAYER1_COLOR:
                self.player1_time -= elapsed
            else:
                self.player2_time -= elapsed

            if self.player1_time <= 0:
                self.game_over = True
                return 'Player 2'
            elif self.player2_time <= 0:
                self.game_over = True
                return 'Player 1'
        return None

    def select(self, row: int, col: int) -> bool:
        # Only allow selection if it's human's turn
        if self.vs_bot and self.turn == PLAYER2_COLOR:
            return False

        piece = self.board.get_piece(row, col)

        # If captures are available
        if self.board.has_captures_available(self.turn):
            # If there's a piece that just made a capture and has more captures
            if self.capturing_piece:
                # Allow deselecting the current piece by clicking elsewhere
                if self.selected_piece and (row != self.selected_piece.row or col != self.selected_piece.col):
                    # Check if clicking on a valid capture move
                    if (row, col) in self.valid_moves and len(self.valid_moves[(row, col)]) > 0:
                        return self._move(row, col)
                    # If not clicking on a valid move, deselect the piece
                    self.selected_piece = None
                    self.valid_moves = {}
                    return False

                # Allow selecting the capturing piece
                if piece == self.capturing_piece:
                    self.selected_piece = piece
                    self.valid_moves = self.board.get_valid_moves(piece)
                    self.animation_start_time = time.time()
                    return True
                return False

            pieces_with_captures = self.board.get_pieces_with_captures(self.turn)

            # If clicking on a piece that can capture
            if piece != 0 and piece.color == self.turn and piece in pieces_with_captures:
                self.selected_piece = piece
                self.valid_moves = self.board.get_valid_moves(piece)
                self.animation_start_time = time.time()
                return True

            # If a piece is selected and clicking on a valid capture destination
            if self.selected_piece and self.selected_piece in pieces_with_captures:
                if (row, col) in self.valid_moves and len(self.valid_moves[(row, col)]) > 0:
                    return self._move(row, col)

            # If clicking elsewhere, deselect the piece
            self.selected_piece = None
            self.valid_moves = {}
            return False

        # No captures available - normal move logic
        if self.selected_piece:
            # Try to make a move
            if (row, col) in self.valid_moves:
                return self._move(row, col)
            # Allow selecting a different piece
            elif piece != 0 and piece.color == self.turn:
                self.selected_piece = piece
                self.valid_moves = self.board.get_valid_moves(piece)
                self.animation_start_time = time.time()
                return True
            else:
                self.selected_piece = None
                self.valid_moves = {}
                return False
        elif piece != 0 and piece.color == self.turn:
            self.selected_piece = piece
            self.valid_moves = self.board.get_valid_moves(piece)
            self.animation_start_time = time.time()
            return True

        return False

    def _move(self, row: int, col: int) -> bool:
        if self.selected_piece and (row, col) in self.valid_moves:
            # Store the piece's previous state for logging
            was_king = self.selected_piece.king
            # Update was_king before making the move
            self.selected_piece.was_king = was_king

            # Log the move before making it to get correct starting position
            captured_pieces = self.valid_moves[(row, col)]
            if captured_pieces:
                log_move(self, self.selected_piece, row, col, captured_pieces)
            else:
                log_move(self, self.selected_piece, row, col)

            # Store original starting position before any moves in this sequence
            # Only update if this is not a continuation of a multi-capture
            if not self.capturing_piece:
                self.last_move_start = (self.selected_piece.row, self.selected_piece.col)
            
            # Track captured pieces
            self.last_captured_pieces = captured_pieces.copy() if captured_pieces else []

            # Update draw detection variables
            # Record board position before the move
            current_state = self.encode_board_state()
            self.position_history.append(current_state)
            
            # If we captured pieces, reset the no-capture counter
            if captured_pieces:
                self.moves_without_capture = 0
            else:
                self.moves_without_capture += 1
                
            # If we moved a non-king piece, reset the no-pawn-move counter
            if not self.selected_piece.king:
                self.moves_without_pawn_move = 0
            else:
                self.moves_without_pawn_move += 1

            # Make the move
            self.board.move(self.selected_piece, row, col)
            if captured_pieces:
                self.board.remove(captured_pieces)
                self.last_captured = captured_pieces

            # Update last move for animation
            self.last_moved_piece = self.selected_piece
            self.animation_start_time = time.time()

            # Check for additional captures
            if captured_pieces:  # Only check for more captures if we just made a capture
                next_moves = self.board.get_valid_moves(self.selected_piece)
                has_more_captures = any(
                    len(next_moves[m]) > 0 for m in next_moves)

                if has_more_captures:
                    # Keep the same piece selected and update valid moves
                    self.valid_moves = next_moves
                    self.capturing_piece = self.selected_piece  # Track the capturing piece
                    return True  # Don't change turn yet, there are more captures

            # If no more captures, clear selection and change turn
            self.selected_piece = None
            self.valid_moves = {}
            self.capturing_piece = None  # Clear the capturing piece
            self.change_turn()
            return True
        return False

    def change_turn(self):
        self.valid_moves = {}
        self.selected_piece = None
        self.turn = PLAYER2_COLOR if self.turn == PLAYER1_COLOR else PLAYER1_COLOR

        # Auto-select if there's only one piece that can capture
        if self.board.has_captures_available(self.turn):
            pieces_with_captures = self.board.get_pieces_with_captures(self.turn)
            if len(pieces_with_captures) == 1 and (self.turn == PLAYER1_COLOR or not self.vs_bot):
                self.selected_piece = pieces_with_captures[0]
                self.valid_moves = self.board.get_valid_moves(
                    self.selected_piece)

        # Only increment turn number after Player 2's move
        if self.turn == PLAYER1_COLOR:
            self.current_turn_number += 1

    def encode_board_state(self):
        """
        Encode the current board state into a string representation for position repetition detection.
        """
        state = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board.get_piece(row, col)
                if piece != 0:
                    # Encode as: position + color + king status
                    # e.g. "23B1" means row 2, col 3, Black, King
                    king_status = "1" if piece.king else "0"
                    color_code = "B" if piece.color == PLAYER1_COLOR else "W"
                    state.append(f"{row}{col}{color_code}{king_status}")
        
        # Also encode whose turn it is
        turn_code = "B" if self.turn == PLAYER1_COLOR else "W"
        state.append(f"TURN{turn_code}")
        
        # Sort for consistent representation regardless of piece ordering
        state.sort()
        return "|".join(state)

    def __del__(self):
        if self.log_file:
            self.log_file.close()


def main():
    win = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('Checkers')

    while True:  # Main game loop
        # Create mode selection buttons
        two_player_btn = Button(
            WINDOW_SIZE[0]//4, WINDOW_SIZE[1]//2 - 60, 200, 50, "2 Players")
        vs_bot_btn = Button(
            WINDOW_SIZE[0]//4, WINDOW_SIZE[1]//2 + 10, 200, 50, "vs Bot")

        # Mode selection screen
        waiting_for_mode = True
        while waiting_for_mode:
            win.fill(PANEL_COLOR)
            two_player_btn.draw(win)
            vs_bot_btn.draw(win)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if two_player_btn.handle_event(event):
                    waiting_for_mode = False
                    vs_bot = False
                elif vs_bot_btn.handle_event(event):
                    waiting_for_mode = False
                    vs_bot = True

        # Game loop
        while True:  # Keep playing in the same mode until home button is clicked
            game = Game(win, vs_bot)
            game_running = True
            go_to_home = False

            while game_running:  # Individual game loop
                # Check for piece-based winner first
                piece_winner = game.check_winner()
                if piece_winner:
                    # Create result box
                    font = pygame.font.Font(None, 72)
                    text = font.render(f"{piece_winner}", True, WHITE)
                    text_rect = text.get_rect()

                    # Box dimensions
                    box_width = text_rect.width + 100
                    box_height = text_rect.height + 60
                    box_x = BOARD_WIDTH//2 - box_width//2
                    box_y = BOARD_WIDTH//2 - box_height//2

                    # Draw box with border
                    box = pygame.Rect(box_x, box_y, box_width, box_height)
                    pygame.draw.rect(win, (50, 50, 50), box)  # Dark background
                    pygame.draw.rect(win, (100, 100, 100),
                                     box, 3)  # Light border

                    # Draw text centered in box
                    text_x = box_x + (box_width - text_rect.width)//2
                    text_y = box_y + (box_height - text_rect.height)//2
                    win.blit(text, (text_x, text_y))

                    # Create Play Again button
                    play_again_btn = Button(
                        BOARD_WIDTH//2 - 100, box_y + box_height + 20, 200, 50, "Play Again")
                    play_again_btn.draw(win)
                    game.home_button.draw(win)  # Draw home button
                    pygame.display.update()

                    # Wait for button click or quit
                    waiting_for_input = True
                    while waiting_for_input:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit()
                            if event.type == pygame.MOUSEBUTTONDOWN:
                                if play_again_btn.handle_event(event):
                                    game_running = False  # End current game to start new one
                                    waiting_for_input = False
                                elif game.home_button.handle_event(event):
                                    game_running = False  # End current game
                                    waiting_for_input = False
                                    go_to_home = True  # Return to home screen
                            if event.type == pygame.MOUSEMOTION:
                                play_again_btn.handle_event(event)
                                game.home_button.handle_event(
                                    event)  # Handle home button hover
                    continue

                # Check for draw conditions
                draw_result = game.check_draw()
                if draw_result:
                    # Create result box
                    font = pygame.font.Font(None, 72)
                    text = font.render(f"{draw_result}", True, WHITE)
                    text_rect = text.get_rect()

                    # Box dimensions
                    box_width = text_rect.width + 100
                    box_height = text_rect.height + 60
                    box_x = BOARD_WIDTH//2 - box_width//2
                    box_y = BOARD_WIDTH//2 - box_height//2

                    # Draw box with border
                    box = pygame.Rect(box_x, box_y, box_width, box_height)
                    pygame.draw.rect(win, (50, 50, 50), box)  # Dark background
                    pygame.draw.rect(win, (100, 100, 100),
                                     box, 3)  # Light border

                    # Draw text centered in box
                    text_x = box_x + (box_width - text_rect.width)//2
                    text_y = box_y + (box_height - text_rect.height)//2
                    win.blit(text, (text_x, text_y))

                    # Create Play Again button
                    play_again_btn = Button(
                        BOARD_WIDTH//2 - 100, box_y + box_height + 20, 200, 50, "Play Again")
                    play_again_btn.draw(win)
                    game.home_button.draw(win)  # Draw home button
                    pygame.display.update()

                    # Wait for button click or quit
                    waiting_for_input = True
                    while waiting_for_input:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit()
                            if event.type == pygame.MOUSEBUTTONDOWN:
                                if play_again_btn.handle_event(event):
                                    game_running = False  # End current game to start new one
                                    waiting_for_input = False
                                elif game.home_button.handle_event(event):
                                    game_running = False  # End current game
                                    waiting_for_input = False
                                    go_to_home = True  # Return to home screen
                            if event.type == pygame.MOUSEMOTION:
                                play_again_btn.handle_event(event)
                                game.home_button.handle_event(
                                    event)  # Handle home button hover
                    continue

                # Then check for timer-based winner (only in 2-player mode)
                if not game.vs_bot:
                    timer_winner = game.update_timer()
                    if timer_winner:
                        # Create result box
                        font = pygame.font.Font(None, 72)
                        text = font.render(
                            f"{timer_winner}", True, WHITE)
                        text_rect = text.get_rect()

                        # Box dimensions
                        box_width = text_rect.width + 100
                        box_height = text_rect.height + 60
                        box_x = BOARD_WIDTH//2 - box_width//2
                        box_y = BOARD_WIDTH//2 - box_height//2

                        # Draw box with border
                        box = pygame.Rect(box_x, box_y, box_width, box_height)
                        # Dark background
                        pygame.draw.rect(win, (50, 50, 50), box)
                        pygame.draw.rect(win, (100, 100, 100),
                                         box, 3)  # Light border

                        # Draw text centered in box
                        text_x = box_x + (box_width - text_rect.width)//2
                        text_y = box_y + (box_height - text_rect.height)//2
                        win.blit(text, (text_x, text_y))

                        # Create Play Again button
                        play_again_btn = Button(
                            BOARD_WIDTH//2 - 100, box_y + box_height + 20, 200, 50, "Play Again")
                        play_again_btn.draw(win)
                        game.home_button.draw(win)  # Draw home button
                        pygame.display.update()

                        # Wait for button click or quit
                        waiting_for_input = True
                        while waiting_for_input:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit()
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    if play_again_btn.handle_event(event):
                                        game_running = False  # End current game to start new one
                                        waiting_for_input = False
                                if event.type == pygame.MOUSEMOTION:
                                    play_again_btn.handle_event(event)
                                    game.home_button.handle_event(
                                        event)  # Handle home button hover
                        continue

                if game.turn == PLAYER2_COLOR and game.vs_bot and not game.game_over:
                    bot_move(game)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        if pos[0] < BOARD_WIDTH:  # Click on board
                            row = pos[1] // SQUARE_SIZE
                            col = pos[0] // SQUARE_SIZE
                            if not game.game_over:
                                game.select(row, col)
                        else:  # Click on side panel
                            if game.home_button.handle_event(event):
                                game_running = False
                                go_to_home = True  # Set flag to return to home screen

                    # Update button hover states
                    if event.type == pygame.MOUSEMOTION:
                        game.home_button.handle_event(event)

                game.update()

            # If home button was clicked, break out to mode selection
            if go_to_home:
                break  # Break out of the same-mode loop to return to mode selection


if __name__ == "__main__":
    main()
