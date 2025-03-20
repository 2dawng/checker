import pygame
import sys
import time
import math
from typing import List, Tuple, Optional
from constants import *
from button import Button
from board import Board
import os

# Initialize Pygame
pygame.init()

# Constants
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
        self.pulse_speed = 3.0  # Speed of the pulse animation
        self.current_turn_number = 1  # Initialize turn number to 1
        self.turn_started = False  # Track if turn has started
        self.capturing_piece = None  # Track the piece that just made a capture

    def setup_log_file(self):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # Create a timestamp for the filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join('logs', f"checkers_game_{timestamp}.log")
        self.log_file = open(filename, 'w')
        # Write header
        self.log_file.write(
            f"Checkers Game Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(
            f"Mode: {'Bot' if self.vs_bot else 'Two Player'}\n")
        self.log_file.write("-" * 50 + "\n\n")

    def log_move(self, piece, row, col, captured_pieces=None):
        def get_square_name(row, col):
            # Always use uppercase letters now
            col_letter = chr(ord('a') + col).upper()
            row_number = BOARD_SIZE - row
            return f"{col_letter}{row_number}"

        start_pos = get_square_name(piece.row, piece.col)
        end_pos = get_square_name(row, col)

        # For file logging
        if piece.color == PLAYER1_COLOR:
            # Check if this is a sequential capture
            if captured_pieces and self.move_number > 0:
                # Check if the last move was also a capture
                last_was_capture = (len(self.move_log) > 0 and
                                    len(self.move_log[-1]) > 0 and
                                    'x' in self.move_log[-1][-1])
                if last_was_capture:
                    # Sequential capture - use 1 tab
                    file_text = f"\n\t{start_pos}-{end_pos}"
                else:
                    # New capture sequence - start new turn
                    file_text = f"\n{self.current_turn_number}.\t{
                        start_pos}-{end_pos}"
            else:
                # Normal move - start new turn
                file_text = f"{self.current_turn_number}.\t{
                    start_pos}-{end_pos}" if not self.turn_started else f"\n{self.current_turn_number}.\t{start_pos}-{end_pos}"
                self.turn_started = True

            if captured_pieces:
                file_text += f"x{len(captured_pieces)}"
            if piece.king and not piece.was_king:
                file_text += "K"
            self.log_file.write(file_text)

            # Only increment turn number for non-sequential moves
            if not (captured_pieces and last_was_capture):
                self.current_turn_number += 1

        else:
            # Black's moves
            if captured_pieces and self.move_number > 0:
                # Check if the last move was also a capture
                last_was_capture = (len(self.move_log) > 0 and
                                    len(self.move_log[-1]) > 0 and
                                    'x' in self.move_log[-1][-1])
                if last_was_capture:
                    # Sequential capture - use 4 tabs
                    file_text = f"\n\t\t\t\t{start_pos}-{end_pos}"
                else:
                    # New capture sequence - add to current line
                    file_text = f"\t\t{start_pos}-{end_pos}"
            else:
                # Normal move - add to current line
                file_text = f"\t\t{start_pos}-{end_pos}"

            if captured_pieces:
                file_text += f"x{len(captured_pieces)}"
            if piece.king and not piece.was_king:
                file_text += "K"

            # Write the move
            self.log_file.write(file_text)

            # Only add newline if this is not part of a capture sequence
            if not (captured_pieces and last_was_capture):
                self.log_file.write("\n")

        # For in-game display (use uppercase and single line)
        move_text = f"{start_pos}-{end_pos}"
        if captured_pieces:
            move_text += f"x{len(captured_pieces)}"
        if piece.king and not piece.was_king:
            move_text += "K"

        if piece.color == PLAYER1_COLOR:
            # Only add turn number if not a sequential capture
            if not (captured_pieces and last_was_capture):
                # Calculate display turn number based on existing moves
                display_turn = len(self.move_log) + 1
                display_text = f"{display_turn}. {move_text}"
            else:
                display_text = f" {move_text}"
        else:
            display_text = f" {move_text}"

        # Store complete turn information
        if piece.color == PLAYER1_COLOR and not (captured_pieces and last_was_capture):
            self.move_log.append([display_text])  # Start new turn
        else:
            if self.move_log:  # Add to current turn
                self.move_log[-1].append(display_text)

        self.log_file.flush()
        self.move_number += 1

    def get_all_moves(self):
        moves = {}
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board.get_piece(row, col)
                if piece != 0 and piece.color == self.turn:
                    piece_moves = self.board.get_valid_moves(piece)
                    if piece_moves:
                        moves[piece] = piece_moves
        return moves

    def has_captures_available(self):
        all_moves = self.get_all_moves()
        for piece in all_moves:
            for move in all_moves[piece]:
                if len(all_moves[piece][move]) > 0:  # If there are pieces to capture
                    return True
        return False

    def get_pieces_with_captures(self):
        pieces_with_captures = []
        all_moves = self.get_all_moves()
        for piece in all_moves:
            for move in all_moves[piece]:
                if len(all_moves[piece][move]) > 0:
                    pieces_with_captures.append(piece)
                    break
        return pieces_with_captures

    def check_stalemate(self):
        # Get all possible moves for current player
        all_moves = self.get_all_moves()

        # If there are no possible moves, it's a stalemate
        if not all_moves:
            self.game_over = True
            return True
        return False

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
            return 'Player 2'
        elif player2_pieces == 0:
            self.game_over = True
            return 'Player 1'

        # Then check for stalemate only if both players still have pieces
        if self.check_stalemate():
            return 'Stalemate - Draw'

        return None

    def update(self):
        # Draw board and pieces
        self.board.draw(self.win)

        # Draw last move indicator (static circle)
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
            self.draw_valid_moves()

        # Draw side panel
        pygame.draw.rect(self.win, PANEL_COLOR, (BOARD_WIDTH,
                         0, SIDE_PANEL_WIDTH, WINDOW_SIZE[1]))

        # Draw timers
        self.draw_timer()

        # Draw move log
        self.draw_move_log()

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
        if self.has_captures_available():
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

            pieces_with_captures = self.get_pieces_with_captures()

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
                self.log_move(self.selected_piece, row, col, captured_pieces)
            else:
                self.log_move(self.selected_piece, row, col)

            # Make the move
            self.board.move(self.selected_piece, row, col)
            if captured_pieces:
                self.board.remove(captured_pieces)
                self.last_captured = captured_pieces

            # Update last move for animation
            self.last_moved_piece = self.selected_piece
            self.animation_start_time = time.time()

            # Check for king promotion
            if self.selected_piece.king and not was_king:
                self.print_debug_info("King Promotion")

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
        if self.has_captures_available():
            pieces_with_captures = self.get_pieces_with_captures()
            if len(pieces_with_captures) == 1 and (self.turn == PLAYER1_COLOR or not self.vs_bot):
                self.selected_piece = pieces_with_captures[0]
                self.valid_moves = self.board.get_valid_moves(
                    self.selected_piece)
                self.print_debug_info("Auto-selected Single Capture Piece")

        # Print debug when turn changes
        self.print_debug_info("Turn Changed")

    def draw_valid_moves(self):
        # If captures are available
        if self.has_captures_available():
            # If there's a capturing piece, only highlight that piece and its moves
            if self.capturing_piece:
                piece_x = self.capturing_piece.col * SQUARE_SIZE + SQUARE_SIZE//2
                piece_y = self.capturing_piece.row * SQUARE_SIZE + SQUARE_SIZE//2
                pygame.draw.circle(self.win, (0, 255, 255),
                                   (piece_x, piece_y), 10)

                if self.selected_piece == self.capturing_piece:
                    moves = self.valid_moves
                    for move in moves:
                        if len(moves[move]) > 0:  # Only show capture moves
                            row, col = move
                            pygame.draw.circle(self.win, (0, 255, 0),
                                               (col * SQUARE_SIZE + SQUARE_SIZE//2,
                                                row * SQUARE_SIZE + SQUARE_SIZE//2), 20)
            else:
                pieces_with_captures = self.get_pieces_with_captures()
                # Draw indicators on all pieces that can capture
                for piece in pieces_with_captures:
                    piece_x = piece.col * SQUARE_SIZE + SQUARE_SIZE//2
                    piece_y = piece.row * SQUARE_SIZE + SQUARE_SIZE//2
                    pygame.draw.circle(
                        self.win, (0, 255, 255), (piece_x, piece_y), 10)

                # If a piece is selected, show only its capture destinations
                if self.selected_piece and self.selected_piece in pieces_with_captures:
                    moves = self.valid_moves
                    for move in moves:
                        if len(moves[move]) > 0:  # Only show capture moves
                            row, col = move
                            pygame.draw.circle(self.win, (0, 255, 0),
                                               (col * SQUARE_SIZE + SQUARE_SIZE//2,
                                                row * SQUARE_SIZE + SQUARE_SIZE//2), 20)

        # If no captures and a piece is selected, show its regular moves
        elif self.selected_piece:
            for move in self.valid_moves:
                row, col = move
                pygame.draw.circle(self.win, (0, 255, 0),
                                   (col * SQUARE_SIZE + SQUARE_SIZE//2,
                                    row * SQUARE_SIZE + SQUARE_SIZE//2), 15)

    def bot_move(self):
        if not self.game_over and not self.bot_thinking:
            self.bot_thinking = True
            import random
            import time

            pygame.time.wait(500)

            all_moves = self.get_all_moves()
            pieces = []

            # Check for captures first
            for piece in all_moves:
                moves = all_moves[piece]
                has_capture = any(len(moves[move]) > 0 for move in moves)
                if has_capture:
                    pieces.append((piece, moves))

            # If no captures, get all valid moves
            if not pieces:
                for piece in all_moves:
                    moves = all_moves[piece]
                    if moves:
                        pieces.append((piece, moves))

            if pieces:
                piece, moves = random.choice(pieces)
                self.print_debug_info("Bot Selected Piece")

                # If captures available, only choose from capture moves
                if any(len(moves[move]) > 0 for move in moves):
                    capture_moves = {k: v for k, v in moves.items() if v}
                    move = random.choice(list(capture_moves.keys()))
                else:
                    move = random.choice(list(moves.keys()))

                # Store the piece's previous state for logging
                was_king = piece.king
                piece.was_king = was_king

                # Log the move before making it to get correct starting position
                skipped = moves[move]
                if skipped:
                    self.log_move(piece, move[0], move[1], skipped)
                else:
                    self.log_move(piece, move[0], move[1])

                # Make the move
                self.board.move(piece, move[0], move[1])
                # Start animation for the bot's moved piece
                self.last_moved_piece = piece
                self.animation_start_time = time.time()

                if skipped:
                    self.board.remove(skipped)
                    # Check for additional captures
                    next_moves = self.board.get_valid_moves(piece)
                    while any(len(next_moves[m]) > 0 for m in next_moves):
                        self.print_debug_info("Bot Multi-Capture")
                        capture_moves = {k: v for k,
                                         v in next_moves.items() if v}
                        move = random.choice(list(capture_moves.keys()))
                        # Log the additional capture before making it
                        self.log_move(piece, move[0],
                                      move[1], next_moves[move])
                        self.board.move(piece, move[0], move[1])
                        # Update animation for multi-capture moves
                        self.last_moved_piece = piece
                        self.animation_start_time = time.time()
                        self.board.remove(next_moves[move])
                        next_moves = self.board.get_valid_moves(piece)

                self.change_turn()
                self.print_debug_info("Bot Move Complete")

            self.bot_thinking = False

    def print_debug_info(self, action):
        # Disabled terminal logging
        pass

    def draw_move_log(self):
        # Draw move log title
        font = pygame.font.Font(None, 36)
        title = font.render("Move Log", True, WHITE)
        self.win.blit(title, (BOARD_WIDTH + 20, 220))

        # Draw moves with smaller font
        font = pygame.font.Font(None, 20)  # Reduced font size
        y_pos = 270
        max_moves = 20  # Show more moves since they take less space

        # Calculate start index for display
        total_turns = len(self.move_log)
        start_idx = max(0, total_turns - max_moves)

        # Display moves
        for turn_moves in self.move_log[start_idx:]:
            # Combine all moves in the turn
            line = "".join(turn_moves)
            text = font.render(line, True, WHITE)
            self.win.blit(text, (BOARD_WIDTH + 20, y_pos))
            y_pos += 20  # Space between lines

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
                    text = font.render(f"{piece_winner} WINS!", True, WHITE)
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
                    continue

                # Then check for timer-based winner (only in 2-player mode)
                if not game.vs_bot:
                    timer_winner = game.update_timer()
                    if timer_winner:
                        # Create result box
                        font = pygame.font.Font(None, 72)
                        text = font.render(
                            f"{timer_winner} WINS!", True, WHITE)
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
                        continue

                if game.turn == PLAYER2_COLOR and game.vs_bot and not game.game_over:
                    game.bot_move()

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
