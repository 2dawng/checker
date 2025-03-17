import pygame
import sys
import time
import math
from typing import List, Tuple, Optional
from constants import *
from button import Button
from board import Board

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
    def __init__(self, win):
        self.win = win
        self.board = Board()
        self.selected_piece = None
        self.turn = PLAYER1_COLOR
        self.valid_moves = {}
        self.game_over = False
        self.vs_bot = False
        # 5 minutes per player
        self.timer = {PLAYER1_COLOR: 300, PLAYER2_COLOR: 300}
        self.last_time = time.time()
        self.paused = False
        self.pause_button = Button(BOARD_WIDTH + 20, 20, 160, 40, "Pause")
        self.resume_button = Button(BOARD_WIDTH + 20, 70, 160, 40, "Resume")
        self.bot_thinking = False
        self.debug_font = pygame.font.Font(None, 24)
        self.frame_count = 0
        # Add last move animation tracking
        self.last_moved_piece = None
        self.animation_start_time = 0
        self.pulse_speed = 3.0  # Speed of the pulse animation

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

        # Draw last move animation
        if self.last_moved_piece and not self.paused:
            current_time = time.time()
            elapsed = current_time - self.animation_start_time
            # Calculate pulse size using sine wave (continuous)
            pulse = abs(math.sin(elapsed * self.pulse_speed)) * \
                5  # Pulse between 0 and 5 pixels
            x = self.last_moved_piece.col * SQUARE_SIZE + SQUARE_SIZE // 2
            y = self.last_moved_piece.row * SQUARE_SIZE + SQUARE_SIZE // 2
            # Draw pulsing circle
            pygame.draw.circle(self.win, CROWN_COLOR, (x, y),
                               SQUARE_SIZE//2 - 4 + pulse, 3)

        # Draw selected piece highlight (only for human player)
        if self.selected_piece and (self.turn == PLAYER1_COLOR or not self.vs_bot):
            x = self.selected_piece.col * SQUARE_SIZE + SQUARE_SIZE // 2
            y = self.selected_piece.row * SQUARE_SIZE + SQUARE_SIZE // 2
            pygame.draw.circle(self.win, CROWN_COLOR,
                               (x, y), SQUARE_SIZE//2 - 8, 3)

        # Only draw valid moves for human player
        if self.turn == PLAYER1_COLOR or not self.vs_bot:
            self.draw_valid_moves()

        # Draw side panel
        pygame.draw.rect(self.win, PANEL_COLOR, (BOARD_WIDTH,
                         0, SIDE_PANEL_WIDTH, WINDOW_SIZE[1]))

        # Draw timers
        self.draw_timer()

        # Draw pause/resume button
        if self.paused:
            self.resume_button.draw(self.win)
        else:
            self.pause_button.draw(self.win)

        # Draw debug information
        self.draw_debug_info()

        pygame.display.update()

    def draw_timer(self):
        if not self.vs_bot:  # Only draw timer in 2-player mode
            font = pygame.font.Font(None, 36)
            # Player 1 timer
            p1_text = font.render("Player 1", True, PLAYER1_COLOR)
            p1_timer = font.render(
                f"{int(self.timer[PLAYER1_COLOR])}s", True, PLAYER1_COLOR)
            self.win.blit(p1_text, (BOARD_WIDTH + 20, 150))
            self.win.blit(p1_timer, (BOARD_WIDTH + 20, 180))

            # Player 2 timer
            p2_text = font.render("Player 2", True, PLAYER2_COLOR)
            p2_timer = font.render(
                f"{int(self.timer[PLAYER2_COLOR])}s", True, PLAYER2_COLOR)
            self.win.blit(p2_text, (BOARD_WIDTH + 20, 250))
            self.win.blit(p2_timer, (BOARD_WIDTH + 20, 280))

    def update_timer(self):
        if not self.vs_bot and not self.paused:  # Only update timer in 2-player mode
            current_time = time.time()
            elapsed = current_time - self.last_time
            self.last_time = current_time

            if self.turn == PLAYER1_COLOR:
                self.timer[PLAYER1_COLOR] -= elapsed
            else:
                self.timer[PLAYER2_COLOR] -= elapsed

            if self.timer[PLAYER1_COLOR] <= 0:
                self.game_over = True
                return 'Player 2'
            elif self.timer[PLAYER2_COLOR] <= 0:
                self.game_over = True
                return 'Player 1'
        return None

    def select(self, row: int, col: int) -> bool:
        # Only allow selection if it's human's turn
        if self.vs_bot and self.turn == PLAYER2_COLOR:
            return False

        if self.selected_piece:
            # If captures are available, only allow making capture moves
            if self.has_captures_available():
                piece_moves = self.board.get_valid_moves(self.selected_piece)
                has_capture = any(
                    len(piece_moves[move]) > 0 for move in piece_moves)
                if has_capture:
                    # Only try to make a move, don't allow deselection
                    return self._move(row, col)

            # Normal behavior for non-capture situations
            result = self._move(row, col)
            if not result:
                # Only allow deselection if no captures are available
                if not self.has_captures_available():
                    self.selected_piece = None
                    self.valid_moves = {}
                    self.select(row, col)
            return result

        piece = self.board.get_piece(row, col)
        if piece != 0 and piece.color == self.turn:
            # If captures are available, only allow selecting pieces that can capture
            if self.has_captures_available():
                pieces_with_captures = self.get_pieces_with_captures()
                if piece not in pieces_with_captures:
                    # If a piece with captures is already selected, keep it selected
                    if self.selected_piece in pieces_with_captures:
                        return False
                    return False

            self.selected_piece = piece
            self.valid_moves = self.board.get_valid_moves(piece)
            # Print debug when piece is selected
            self.print_debug_info("Piece Selected")
            return True

        return False

    def _move(self, row: int, col: int) -> bool:
        piece = self.board.get_piece(row, col)
        if self.selected_piece and piece == 0 and (row, col) in self.valid_moves:
            # Check if there are captures available
            if self.has_captures_available():
                # Only allow capture moves
                if len(self.valid_moves[(row, col)]) == 0:
                    return False

            # Make the move
            self.board.move(self.selected_piece, row, col)
            # Start animation for the moved piece
            self.last_moved_piece = self.selected_piece
            self.animation_start_time = time.time()

            skipped = self.valid_moves[(row, col)]
            if skipped:
                self.board.remove(skipped)
                # Check for additional captures
                next_moves = self.board.get_valid_moves(self.selected_piece)
                has_more_captures = any(
                    len(next_moves[move]) > 0 for move in next_moves)
                if has_more_captures:
                    self.valid_moves = next_moves
                    self.print_debug_info("Multi-Capture Available")
                    # Keep the same piece selected and show all capture options
                    return True

            # Only change turn if no more captures are available
            self.change_turn()
            self.print_debug_info("Move Complete")
            return True
        return False

    def change_turn(self):
        self.valid_moves = {}
        self.selected_piece = None
        self.turn = PLAYER2_COLOR if self.turn == PLAYER1_COLOR else PLAYER1_COLOR

        # Auto-select a piece with captures if available (for human players only)
        if (self.turn == PLAYER1_COLOR or (not self.vs_bot)):
            if self.has_captures_available():
                pieces_with_captures = self.get_pieces_with_captures()
                if pieces_with_captures:
                    # Select the first piece with captures
                    self.selected_piece = pieces_with_captures[0]
                    self.valid_moves = self.board.get_valid_moves(
                        self.selected_piece)

        # Print debug when turn changes
        self.print_debug_info("Turn Changed")

    def draw_valid_moves(self):
        # If captures are available, only show capture moves
        if self.has_captures_available():
            for move in self.valid_moves:
                # Only draw if it's a capture move
                if len(self.valid_moves[move]) > 0:
                    row, col = move
                    # Draw a larger circle for capture moves to make them more visible
                    pygame.draw.circle(self.win, (0, 255, 0),
                                       (col * SQUARE_SIZE + SQUARE_SIZE//2,
                                        row * SQUARE_SIZE + SQUARE_SIZE//2), 20)
        else:
            # Show all valid moves if no captures are available
            for move in self.valid_moves:
                row, col = move
                pygame.draw.circle(self.win, (0, 255, 0),
                                   (col * SQUARE_SIZE + SQUARE_SIZE//2,
                                    row * SQUARE_SIZE + SQUARE_SIZE//2), 15)

    def bot_move(self):
        if not self.paused and not self.bot_thinking:
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

                # Make the move
                self.board.move(piece, move[0], move[1])
                # Start animation for the bot's moved piece
                self.last_moved_piece = piece
                self.animation_start_time = time.time()

                skipped = moves[move]
                if skipped:
                    self.board.remove(skipped)
                    # Check for additional captures
                    next_moves = self.board.get_valid_moves(piece)
                    while any(len(next_moves[m]) > 0 for m in next_moves):
                        self.print_debug_info("Bot Multi-Capture")
                        capture_moves = {k: v for k,
                                         v in next_moves.items() if v}
                        move = random.choice(list(capture_moves.keys()))
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
        # Skip frame updates
        if action == "Frame Update":
            return

        # Count pieces
        p1_pieces = p2_pieces = 0
        p1_kings = p2_kings = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board.get_piece(row, col)
                if piece != 0:
                    if piece.color == PLAYER1_COLOR:
                        p1_pieces += 1
                        if piece.king:
                            p1_kings += 1
                    else:
                        p2_pieces += 1
                        if piece.king:
                            p2_kings += 1

        # Get current state info
        all_moves = self.get_all_moves()
        captures_available = self.has_captures_available()
        pieces_with_captures = self.get_pieces_with_captures()
        total_moves = sum(len(moves) for moves in all_moves.values())
        is_stalemate = self.check_stalemate()

        # Copyable debug format
        copyable_debug = [
            f"ACTION: {action}",
            f"T:{1 if self.turn == PLAYER1_COLOR else 2}",
            f"M:{1 if self.vs_bot else 2}",
            f"S:{1 if self.paused else 0}",
            f"P1:{p1_pieces},{p1_kings}",
            f"P2:{p2_pieces},{p2_kings}",
            f"C:{1 if captures_available else 0}",
            f"CP:{len(pieces_with_captures)}",
            f"TM:{total_moves}",
            f"ST:{1 if is_stalemate else 0}"
        ]
        if self.selected_piece:
            copyable_debug.extend([
                f"SP:{self.selected_piece.row},{self.selected_piece.col}",
                f"K:{1 if self.selected_piece.king else 0}",
                f"VM:{len(self.valid_moves)}",
                f"CM:{[m for m, s in self.valid_moves.items() if s]}"
            ])

        # Print copyable debug to console
        print(" | ".join(copyable_debug))

    def draw_debug_info(self):
        # Count pieces
        p1_pieces = p2_pieces = 0
        p1_kings = p2_kings = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board.get_piece(row, col)
                if piece != 0:
                    if piece.color == PLAYER1_COLOR:
                        p1_pieces += 1
                        if piece.king:
                            p1_kings += 1
                    else:
                        p2_pieces += 1
                        if piece.king:
                            p2_kings += 1

        # Get current state info
        all_moves = self.get_all_moves()
        captures_available = self.has_captures_available()
        pieces_with_captures = self.get_pieces_with_captures()
        total_moves = sum(len(moves) for moves in all_moves.values())
        is_stalemate = self.check_stalemate()

        # Copyable debug format
        copyable_debug = [
            "DEBUG:",
            f"T:{1 if self.turn == PLAYER1_COLOR else 2}",
            f"M:{1 if self.vs_bot else 2}",
            f"S:{1 if self.paused else 0}",
            f"P1:{p1_pieces},{p1_kings}",
            f"P2:{p2_pieces},{p2_kings}",
            f"C:{1 if captures_available else 0}",
            f"CP:{len(pieces_with_captures)}",
            f"TM:{total_moves}",
            f"ST:{1 if is_stalemate else 0}"
        ]
        if self.selected_piece:
            copyable_debug.extend([
                f"SP:{self.selected_piece.row},{self.selected_piece.col}",
                f"K:{1 if self.selected_piece.king else 0}",
                f"VM:{len(self.valid_moves)}",
                f"CM:{[m for m, s in self.valid_moves.items() if s]}"
            ])

        # Create copyable text but don't print it
        copyable_text = " | ".join(copyable_debug)

        # User-friendly debug lines
        debug_lines = [
            "=== DEBUG INFO ===",
            f"Current Turn: {'Black' if self.turn ==
                             PLAYER1_COLOR else 'White'}",
            f"Game Mode: {'vs Bot' if self.vs_bot else '2 Players'}",
            f"Game State: {'Paused' if self.paused else 'Running'}",
            "",
            "=== PIECES ===",
            f"Black: {p1_pieces} (Kings: {p1_kings})",
            f"White: {p2_pieces} (Kings: {p2_kings})",
            "",
            "=== MOVES ===",
            f"Captures Available: {captures_available}",
            f"Pieces with Captures: {len(pieces_with_captures)}",
            f"Total Valid Moves: {total_moves}",
            f"Stalemate: {is_stalemate}",
            "",
        ]

        if self.selected_piece:
            debug_lines.extend([
                "=== SELECTED PIECE ===",
                f"Position: ({self.selected_piece.row}, {
                    self.selected_piece.col})",
                f"Is King: {self.selected_piece.king}",
                f"Valid Moves: {len(self.valid_moves)}",
                "Capture Moves: " +
                str([move for move, skipped in self.valid_moves.items() if skipped]),
            ])

        # Draw copyable debug text at the top
        y_offset = 100 if self.vs_bot else 100
        text = self.debug_font.render(copyable_text, True, WHITE)
        self.win.blit(text, (BOARD_WIDTH + 10, y_offset))

        # Draw user-friendly debug text
        y_offset = 150 if self.vs_bot else 350
        for line in debug_lines:
            text = self.debug_font.render(line, True, WHITE)
            self.win.blit(text, (BOARD_WIDTH + 10, y_offset))
            y_offset += 20


def main():
    win = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('Checkers')

    # Create mode selection buttons
    two_player_btn = Button(
        WINDOW_SIZE[0]//4, WINDOW_SIZE[1]//2 - 60, 200, 50, "2 Players")
    vs_bot_btn = Button(WINDOW_SIZE[0]//4,
                        WINDOW_SIZE[1]//2 + 10, 200, 50, "vs Bot")

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

    game = Game(win)
    game.vs_bot = vs_bot

    while True:
        # Check for piece-based winner first
        piece_winner = game.check_winner()
        if piece_winner:
            font = pygame.font.Font(None, 72)
            text = font.render(f"{piece_winner} WINS!", True, WHITE)
            win.blit(text, (BOARD_WIDTH//4, BOARD_WIDTH//2))
            pygame.display.update()
            pygame.time.wait(3000)
            break

        # Then check for timer-based winner (only in 2-player mode)
        if not game.vs_bot:
            timer_winner = game.update_timer()
            if timer_winner:
                font = pygame.font.Font(None, 72)
                text = font.render(f"{timer_winner} WINS!", True, WHITE)
                win.blit(text, (BOARD_WIDTH//4, BOARD_WIDTH//2))
                pygame.display.update()
                pygame.time.wait(3000)
                break

        if game.turn == PLAYER2_COLOR and game.vs_bot and not game.paused:
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
                    if not game.paused:
                        game.select(row, col)
                else:  # Click on side panel
                    if game.paused:
                        if game.resume_button.handle_event(event):
                            game.paused = False
                            game.last_time = time.time()
                    else:
                        if game.pause_button.handle_event(event):
                            game.paused = True

            # Update button hover states
            if event.type == pygame.MOUSEMOTION:
                if game.paused:
                    game.resume_button.handle_event(event)
                else:
                    game.pause_button.handle_event(event)

        game.update()


if __name__ == "__main__":
    main()
