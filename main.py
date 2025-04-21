import pygame
import sys
import time
import math
import os
import traceback
from typing import List, Tuple, Optional
from constants import *
from button import Button
from board import Board
from game_logic_new_version import log_move, draw_move_log, draw_valid_moves, bot_move

pygame.init()
pygame.font.init()

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
        self.player1_time = 2000
        self.player2_time = 2000
        self.last_time_update = time.time()
        self.animation_start_time = time.time()  # Initialize with a valid float
        self.move_number = 0
        self.move_log = []
        self.log_file = None
        self.setup_log_file()
        self.home_button = Button(BOARD_WIDTH + 20, 20, 160, 40, "Home")
        self.bot_thinking = False
        self.last_moved_piece = None
        self.last_move_start = None
        self.last_captured_pieces = []
        self.pulse_speed = 3.0
        self.current_turn_number = 1
        self.turn_started = False
        self.capturing_piece = None
        self.last_capture_color = None
        self.position_history = []
        self.moves_without_capture = 0
        self.moves_without_pawn_move = 0
        self.result_alpha = 0
        self.debug_board_state = False  # Flag to control board state logging

    def setup_log_file(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode_dir = 'bot' if self.vs_bot else 'pvp'
        filename = os.path.join('logs', f"{mode_dir}/{timestamp}.txt")
        os.makedirs(os.path.join('logs', mode_dir), exist_ok=True)
        self.log_file = open(filename, 'w')
        self.log_file.write(
            f"Checkers Game Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(
            f"Mode: {'Bot' if self.vs_bot else 'Two Player'}\n")
        self.log_file.write("-" * 50 + "\n\n")

    def check_winner(self):
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
        if self.board.check_stalemate(self.turn):
            return 'Stalemate - Draw'
        return None

    def check_draw(self):
        current_state = self.encode_board_state()
        position_count = self.position_history.count(current_state) + 1
        if position_count >= 5:
            self.game_over = True
            return "Position Repetition - Draw"
        if self.moves_without_capture >= 100:
            self.game_over = True
            return "No Captures in 50 Moves - Draw"
        if self.moves_without_pawn_move >= 100:
            self.game_over = True
            return "No Pawn Moves in 50 Moves - Draw"
        return None

    def update(self):
        try:
            self.win.fill(BACKGROUND_COLOR)
            border_size = 10
            pygame.draw.rect(self.win, BOARD_BORDER, (0, 0, BOARD_WIDTH + border_size * 2, BOARD_WIDTH + border_size * 2))
            pygame.draw.rect(self.win, WOOD_TEXTURE, (border_size, border_size, BOARD_WIDTH, BOARD_WIDTH))
            
            # Draw the board
            self.board.draw(self.win)

            # Only log board state when a move is made (controlled by debug_board_state)
            if self.debug_board_state:
                print("Board drawn")
                for row in range(BOARD_SIZE):
                    for col in range(BOARD_SIZE):
                        piece = self.board.get_piece(row, col)
                        if piece != 0:
                            print(f"Piece at ({row}, {col}): color={piece.color}, king={piece.king}, x={piece.x}, y={piece.y}")
                self.debug_board_state = False  # Reset flag

            if self.last_move_start and not self.game_over:
                start_row, start_col = self.last_move_start
                start_rect = pygame.Rect(start_col * SQUARE_SIZE, start_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                lavender_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                lavender_surface.fill((*LAST_MOVE_COLOR, 80))
                self.win.blit(lavender_surface, start_rect)

            if self.last_captured_pieces and not self.game_over:
                for piece in self.last_captured_pieces:
                    captured_rect = pygame.Rect(piece.col * SQUARE_SIZE, piece.row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                    capture_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    capture_surface.fill((*CAPTURE_COLOR, 80))
                    self.win.blit(capture_surface, captured_rect)

            if self.selected_piece and (self.turn == PLAYER1_COLOR or not self.vs_bot):
                current_time = time.time()
                if self.animation_start_time is None:
                    self.animation_start_time = current_time
                elapsed = current_time - self.animation_start_time
                pulse = abs(math.sin(elapsed * self.pulse_speed)) * 5
                x = self.selected_piece.col * SQUARE_SIZE + SQUARE_SIZE // 2
                y = self.selected_piece.row * SQUARE_SIZE + SQUARE_SIZE // 2
                pygame.draw.circle(self.win, NEON_HIGHLIGHT, (x, y), SQUARE_SIZE // 2 - 4 + pulse, 3)

            if self.turn == PLAYER1_COLOR or not self.vs_bot:
                draw_valid_moves(self)

            pygame.draw.rect(self.win, PANEL_COLOR, (BOARD_WIDTH, 0, SIDE_PANEL_WIDTH, WINDOW_SIZE[1]))
            self.draw_timer()
            pygame.draw.rect(self.win, (60, 60, 60), (BOARD_WIDTH + 10, 210, 180, 400), border_radius=10)
            draw_move_log(self)
            self.home_button.draw(self.win)
            pygame.display.update()
        except Exception as e:
            print(f"Error in update: {e}")
            traceback.print_exc()

    def draw_timer(self):
        if not self.vs_bot:
            font = pygame.font.Font(None, 36)
            P1_BG = (50, 50, 50)
            P2_BG = (50, 50, 50)
            BORDER_COLOR = (100, 100, 100)
            P1_TEXT = WHITE
            P2_TEXT = WHITE
            BOX_WIDTH = 160
            BOX_HEIGHT = 60
            BOX_PADDING = 10
            p2_box = pygame.Rect(BOARD_WIDTH + 20, 70, BOX_WIDTH, BOX_HEIGHT)
            pygame.draw.rect(self.win, P2_BG, p2_box, border_radius=5)
            pygame.draw.rect(self.win, BORDER_COLOR, p2_box, 2)
            p2_timer = font.render(f"{int(self.player2_time)}s", True, P2_TEXT)
            self.win.blit(p2_timer, (p2_box.x + BOX_PADDING, p2_box.y + BOX_HEIGHT // 2 - 10))
            p1_box = pygame.Rect(BOARD_WIDTH + 20, 140, BOX_WIDTH, BOX_HEIGHT)
            pygame.draw.rect(self.win, P1_BG, p1_box, border_radius=5)
            pygame.draw.rect(self.win, BORDER_COLOR, p1_box, 2)
            p1_timer = font.render(f"{int(self.player1_time)}s", True, P1_TEXT)
            self.win.blit(p1_timer, (p1_box.x + BOX_PADDING, p1_box.y + BOX_HEIGHT // 2 - 10))

    def update_timer(self):
        if not self.vs_bot and not self.game_over:
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
        try:
            if self.vs_bot and self.turn == PLAYER2_COLOR:
                return False
            piece = self.board.get_piece(row, col)
            if not piece and not self.selected_piece:
                return False
            if self.board.has_captures_available(self.turn):
                if self.capturing_piece:
                    if self.selected_piece and (row != self.selected_piece.row or col != self.selected_piece.col):
                        if (row, col) in self.valid_moves and len(self.valid_moves[(row, col)]) > 0:
                            return self._move(row, col)
                        self.selected_piece = None
                        self.valid_moves = {}
                        return False
                    if piece == self.capturing_piece:
                        self.selected_piece = piece
                        self.valid_moves = self.board.get_valid_moves(piece)
                        print(f"Valid moves for selected piece at ({piece.row}, {piece.col}): {self.valid_moves}")
                        self.animation_start_time = time.time()
                        return True
                    return False
                pieces_with_captures = self.board.get_pieces_with_captures(self.turn)
                if piece != 0 and piece.color == self.turn and piece in pieces_with_captures:
                    self.selected_piece = piece
                    self.valid_moves = self.board.get_valid_moves(piece)
                    print(f"Valid moves for selected piece at ({piece.row}, {piece.col}): {self.valid_moves}")
                    self.animation_start_time = time.time()
                    return True
                if self.selected_piece and self.selected_piece in pieces_with_captures:
                    if (row, col) in self.valid_moves and len(self.valid_moves[(row, col)]) > 0:
                        return self._move(row, col)
                self.selected_piece = None
                self.valid_moves = {}
                return False
            if self.selected_piece:
                if (row, col) in self.valid_moves:
                    return self._move(row, col)
                elif piece != 0 and piece.color == self.turn:
                    self.selected_piece = piece
                    self.valid_moves = self.board.get_valid_moves(piece)
                    print(f"Valid moves for selected piece at ({piece.row}, {piece.col}): {self.valid_moves}")
                    self.animation_start_time = time.time()
                    return True
                else:
                    self.selected_piece = None
                    self.valid_moves = {}
                    return False
            elif piece != 0 and piece.color == self.turn:
                self.selected_piece = piece
                self.valid_moves = self.board.get_valid_moves(piece)
                print(f"Valid moves for selected piece at ({piece.row}, {piece.col}): {self.valid_moves}")
                self.animation_start_time = time.time()
                return True
            return False
        except Exception as e:
            print(f"Error in select: {e}")
            traceback.print_exc()
            return False

    def _move(self, row: int, col: int) -> bool:
        try:
            if self.selected_piece and (row, col) in self.valid_moves:
                print(f"Attempting move to ({row}, {col}), valid moves: {self.valid_moves}")
                was_king = self.selected_piece.king
                self.selected_piece.was_king = was_king
                captured_pieces = self.valid_moves[(row, col)]
                print(f"Captured pieces for move to ({row}, {col}): {[f'({p.row}, {p.col})' for p in captured_pieces] if captured_pieces else 'None'}")
                if not self.capturing_piece:
                    self.last_move_start = (self.selected_piece.row, self.selected_piece.col)
                self.last_captured_pieces = captured_pieces.copy() if captured_pieces else []
                current_state = self.encode_board_state()
                self.position_history.append(current_state)
                if captured_pieces:
                    self.moves_without_capture = 0
                else:
                    self.moves_without_capture += 1
                if not self.selected_piece.king:
                    self.moves_without_pawn_move = 0
                else:
                    self.moves_without_pawn_move += 1
                if captured_pieces:
                    log_move(self, self.selected_piece, row, col, captured_pieces)
                else:
                    log_move(self, self.selected_piece, row, col)
                print(f"Moving piece from ({self.selected_piece.row}, {self.selected_piece.col}) to ({row}, {col}), captured: {len(captured_pieces)}")
                self.board.move(self.selected_piece, row, col)
                print(f"Piece after move: ({self.selected_piece.row}, {self.selected_piece.col}), x={self.selected_piece.x}, y={self.selected_piece.y}")
                if captured_pieces:
                    print(f"Removing captured pieces: {[f'({p.row}, {p.col})' for p in captured_pieces]}")
                    self.board.remove(captured_pieces)
                self.last_moved_piece = self.selected_piece
                self.animation_start_time = time.time()
                self.debug_board_state = True  # Trigger board state logging in update
                if captured_pieces:
                    next_moves = self.board.get_valid_moves(self.selected_piece)
                    has_more_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                    if has_more_captures:
                        self.valid_moves = next_moves
                        self.capturing_piece = self.selected_piece
                        return True
                self.selected_piece = None
                self.valid_moves = {}
                self.capturing_piece = None
                self.change_turn()
                return True
            print(f"Move to ({row}, {col}) not in valid moves: {self.valid_moves}")
            return False
        except Exception as e:
            print(f"Error in _move: {e}")
            traceback.print_exc()
            return False

    def change_turn(self):
        try:
            self.valid_moves = {}
            self.selected_piece = None
            self.turn = PLAYER2_COLOR if self.turn == PLAYER1_COLOR else PLAYER1_COLOR
            if self.board.has_captures_available(self.turn):
                pieces_with_captures = self.board.get_pieces_with_captures(self.turn)
                if len(pieces_with_captures) == 1 and (self.turn == PLAYER1_COLOR or not self.vs_bot):
                    self.selected_piece = pieces_with_captures[0]
                    self.valid_moves = self.board.get_valid_moves(self.selected_piece)
            if self.turn == PLAYER1_COLOR:
                self.current_turn_number += 1
        except Exception as e:
            print(f"Error in change_turn: {e}")
            traceback.print_exc()

    def encode_board_state(self):
        try:
            state = []
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    piece = self.board.get_piece(row, col)
                    if piece != 0:
                        king_status = "1" if piece.king else "0"
                        color_code = "B" if piece.color == PLAYER1_COLOR else "W"
                        state.append(f"{row}{col}{color_code}{king_status}")
            turn_code = "B" if self.turn == PLAYER1_COLOR else "W"
            state.append(f"TURN{turn_code}")
            state.sort()
            return "|".join(state)
        except Exception as e:
            print(f"Error in encode_board_state: {e}")
            traceback.print_exc()
            return ""

    def __del__(self):
        if self.log_file:
            self.log_file.close()

def main():
    win = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('Checkers')
    
    background = pygame.image.load('background.jpg')
    background = pygame.transform.scale(background, WINDOW_SIZE)

    while True:
        button_width = 200
        button_height = 50
        spacing = 70
        x_pos = (WINDOW_SIZE[0] - button_width) // 2
        y_start = (WINDOW_SIZE[1] - (button_height * 2 + spacing)) // 2
        two_player_btn = Button(x_pos, y_start, button_width, button_height, "2 Players")
        vs_bot_btn = Button(x_pos, y_start + button_height + spacing, button_width, button_height, "VS Bot")

        waiting_for_mode = True
        while waiting_for_mode:
            try:
                win.blit(background, (0, 0))
                font = pygame.font.Font(None, 48)
                title = font.render("Checkers", True, WHITE)
                title_rect = title.get_rect(center=(WINDOW_SIZE[0] // 2, y_start - 50))
                win.blit(title, title_rect)
                two_player_btn.draw(win)
                vs_bot_btn.draw(win)
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if two_player_btn.handle_event(event):
                            waiting_for_mode = False
                            vs_bot = False
                        elif vs_bot_btn.handle_event(event):
                            waiting_for_mode = False
                            vs_bot = True
                    if event.type == pygame.MOUSEMOTION:
                        two_player_btn.handle_event(event)
                        vs_bot_btn.handle_event(event)
            except Exception as e:
                print(f"Error in mode selection loop: {e}")
                traceback.print_exc()

        while True:
            game = Game(win, vs_bot)
            game_running = True
            go_to_home = False

            while game_running:
                try:
                    piece_winner = game.check_winner()
                    draw_result = game.check_draw()
                    timer_winner = None if game.vs_bot else game.update_timer()

                    if piece_winner or draw_result or timer_winner:
                        result_text = piece_winner or draw_result or timer_winner
                        font = pygame.font.Font(None, 72)
                        text = font.render(result_text, True, WHITE)
                        text_rect = text.get_rect()
                        box_width = text_rect.width + 100
                        box_height = text_rect.height + 60
                        box_x = BOARD_WIDTH // 2 - box_width // 2
                        box_y = BOARD_WIDTH // 2 - box_height // 2
                        # Set alpha to 255 immediately for full opacity
                        game.result_alpha = 255
                        box_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
                        box_surface.fill((50, 50, 50, game.result_alpha))
                        text.set_alpha(game.result_alpha)
                        game.update()
                        win.blit(box_surface, (box_x, box_y))
                        pygame.draw.rect(win, (100, 100, 100, game.result_alpha), (box_x, box_y, box_width, box_height), 3)
                        text_x = box_x + (box_width - text_rect.width) // 2
                        text_y = box_y + (box_height - text_rect.height) // 2
                        win.blit(text, (text_x, text_y))
                        play_again_btn = Button(BOARD_WIDTH // 2 - 100, box_y + box_height + 20, 200, 50, "Play Again")
                        play_again_btn.draw(win)
                        game.home_button.draw(win)
                        pygame.display.update()
                        waiting_for_input = True
                        while waiting_for_input:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit()
                                if event.type == pygame.MOUSEBUTTONDOWN:
                                    if play_again_btn.handle_event(event):
                                        game_running = False
                                        waiting_for_input = False
                                    elif game.home_button.handle_event(event):
                                        game_running = False
                                        waiting_for_input = False
                                        go_to_home = True
                                if event.type == pygame.MOUSEMOTION:
                                    play_again_btn.handle_event(event)
                                    game.home_button.handle_event(event)
                        continue

                    if game.turn == PLAYER2_COLOR and game.vs_bot and not game.game_over and not game.bot_thinking:
                        bot_move(game)

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            pos = pygame.mouse.get_pos()
                            if pos[0] < BOARD_WIDTH and not game.game_over:
                                row = pos[1] // SQUARE_SIZE
                                col = pos[0] // SQUARE_SIZE
                                game.select(row, col)
                            elif game.home_button.handle_event(event):
                                game_running = False
                                go_to_home = True
                        if event.type == pygame.MOUSEMOTION:
                            game.home_button.handle_event(event)

                    game.update()

                except Exception as e:
                    print(f"Error in game loop: {e}")
                    traceback.print_exc()
                    game_running = False  # Stop the game loop to prevent further crashes

            if go_to_home:
                break

if __name__ == "__main__":
    main()