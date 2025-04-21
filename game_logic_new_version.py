import pygame
import time
import random
import math
import traceback
from typing import List, Tuple, Optional
from constants import *
from board import Board
from piece import Piece

def log_move(game, piece, row, col, captured_pieces=None):
    """Record moves in algebraic notation for both file logging and in-game display"""
    try:
        start_pos = game.board.get_square_name(piece.row, piece.col)
        end_pos = game.board.get_square_name(row, col)

        last_was_capture = (len(game.move_log) > 0 and
                            len(game.move_log[-1]) > 0 and
                            'x' in game.move_log[-1][-1] and
                            game.last_capture_color == piece.color)

        if piece.color == PLAYER1_COLOR:
            if captured_pieces and game.move_number > 0 and last_was_capture:
                file_text = f"\n\t{start_pos}-{end_pos}"
            else:
                file_text = f"{game.current_turn_number}.\t{start_pos}-{end_pos}" if game.move_number == 0 else f"\n{game.current_turn_number}.\t{start_pos}-{end_pos}"

            if captured_pieces:
                file_text += f"x{len(captured_pieces)}"
                game.last_capture_color = piece.color
            else:
                game.last_capture_color = None
            if piece.king and not piece.was_king:
                file_text += "K"
            game.log_file.write(file_text)

        else:
            if captured_pieces and game.move_number > 0 and last_was_capture:
                file_text = f"\n\t\t\t\t{start_pos}-{end_pos}"
            else:
                file_text = f"\t\t{start_pos}-{end_pos}"

            if captured_pieces:
                file_text += f"x{len(captured_pieces)}"
                game.last_capture_color = piece.color
            else:
                game.last_capture_color = None
            if piece.king and not piece.was_king:
                file_text += "K"
            game.log_file.write(file_text)

        move_text = f"{start_pos}-{end_pos}"
        if captured_pieces:
            move_text += f"x{len(captured_pieces)}"
        if piece.king and not piece.was_king:
            move_text += "K"

        if piece.color == PLAYER1_COLOR:
            if not (captured_pieces and last_was_capture):
                display_text = f"{game.current_turn_number}. {move_text}"
            else:
                display_text = f" {move_text}"
        else:
            display_text = f" {move_text}"

        if piece.color == PLAYER1_COLOR and not (captured_pieces and last_was_capture):
            game.move_log.append([display_text])
        else:
            if game.move_log:
                game.move_log[-1].append(display_text)

        game.log_file.flush()
        game.move_number += 1
    except Exception as e:
        print(f"Error in log_move: {e}")
        traceback.print_exc()

def draw_move_log(game):
    """Draw the move log in the side panel"""
    try:
        font = pygame.font.Font(None, 36)
        title = font.render("Move Log", True, WHITE)
        game.win.blit(title, (BOARD_WIDTH + 20, 220))

        font = pygame.font.Font(None, 20)
        y_pos = 270
        max_moves = 20

        total_turns = len(game.move_log)
        start_idx = max(0, total_turns - max_moves)

        for turn_moves in game.move_log[start_idx:]:
            line = "".join(turn_moves)
            text = font.render(line, True, WHITE)
            game.win.blit(text, (BOARD_WIDTH + 20, y_pos))
            y_pos += 20
    except Exception as e:
        print(f"Error in draw_move_log: {e}")
        traceback.print_exc()

def draw_valid_moves(game):
    """Draw visual indicators for valid moves and pieces that can capture"""
    try:
        if game.board.has_captures_available(game.turn):
            if game.capturing_piece:
                piece_x = game.capturing_piece.col * SQUARE_SIZE + SQUARE_SIZE // 2
                piece_y = game.capturing_piece.row * SQUARE_SIZE + SQUARE_SIZE // 2
                pygame.draw.circle(game.win, (0, 255, 255), (piece_x, piece_y), 10)

                if game.selected_piece == game.capturing_piece:
                    moves = game.valid_moves
                    for move in moves:
                        if len(moves[move]) > 0:
                            row, col = move
                            pygame.draw.circle(game.win, (0, 255, 0),
                                               (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                                row * SQUARE_SIZE + SQUARE_SIZE // 2), 20)
            else:
                pieces_with_captures = game.board.get_pieces_with_captures(game.turn)
                for piece in pieces_with_captures:
                    piece_x = piece.col * SQUARE_SIZE + SQUARE_SIZE // 2
                    piece_y = piece.row * SQUARE_SIZE + SQUARE_SIZE // 2
                    pygame.draw.circle(game.win, (0, 255, 255), (piece_x, piece_y), 10)

                if game.selected_piece and game.selected_piece in pieces_with_captures:
                    moves = game.valid_moves
                    for move in moves:
                        if len(moves[move]) > 0:
                            row, col = move
                            pygame.draw.circle(game.win, (0, 255, 0),
                                               (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                                row * SQUARE_SIZE + SQUARE_SIZE // 2), 20)
        elif game.selected_piece:
            for move in game.valid_moves:
                row, col = move
                pygame.draw.circle(game.win, (0, 255, 0),
                                   (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                    row * SQUARE_SIZE + SQUARE_SIZE // 2), 15)
    except Exception as e:
        print(f"Error in draw_valid_moves: {e}")
        traceback.print_exc()

def evaluate_position(board):
    """Evaluate the current board position from Player2's perspective (white/bot)"""
    try:
        score = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board.get_piece(row, col)
                if piece != 0:
                    value = 5 if piece.king else 1
                    center_bonus = 0.1 * (4 - abs(col - 3.5))
                    if not piece.king:
                        if piece.color == PLAYER2_COLOR:
                            progress_bonus = 0.2 * row
                        else:
                            progress_bonus = 0.2 * (BOARD_SIZE - 1 - row)
                    else:
                        progress_bonus = 0
                    edge_penalty = 0
                    if piece.king and (col == 0 or col == BOARD_SIZE - 1):
                        edge_penalty = 0.3
                    piece_value = value + center_bonus + progress_bonus - edge_penalty
                    if piece.color == PLAYER2_COLOR:
                        score += piece_value
                    else:
                        score -= piece_value
        return score
    except Exception as e:
        print(f"Error in evaluate_position: {e}")
        traceback.print_exc()
        return 0

def minimax_alpha_beta(board, depth, alpha, beta, maximizing_player, game):
    """Minimax algorithm with alpha-beta pruning for checker AI"""
    try:
        if depth == 0 or board.check_winner() is not None:
            return evaluate_position(board), None

        current_color = PLAYER2_COLOR if maximizing_player else PLAYER1_COLOR
        has_captures = board.has_captures_available(current_color)
        print(f"Checking captures for {current_color}: {has_captures}")

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            all_moves = board.get_all_moves(current_color)

            if has_captures:
                pieces_with_captures = []
                for piece in all_moves:
                    for move in all_moves[piece]:
                        if all_moves[piece][move]:
                            pieces_with_captures.append(piece)
                            break
                move_sources = {p: all_moves[p] for p in pieces_with_captures}
                print(f"Pieces with captures: {[f'({p.row}, {p.col})' for p in pieces_with_captures]}")
            else:
                move_sources = all_moves
                print("No captures available, considering all moves")

            for piece in move_sources:
                for move, captures in move_sources[piece].items():
                    if has_captures and not captures:
                        continue
                    temp_board = board.copy()
                    temp_piece = temp_board.get_piece(piece.row, piece.col)
                    if not temp_piece:
                        continue
                    temp_board.move(temp_piece, move[0], move[1])
                    if captures:
                        pieces_to_remove = [temp_board.get_piece(captured.row, captured.col) for captured in captures if temp_board.get_piece(captured.row, captured.col) != 0]
                        if pieces_to_remove:
                            temp_board.remove(pieces_to_remove)
                    additional_captures = False
                    if captures:
                        next_moves = temp_board.get_valid_moves(temp_piece)
                        additional_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                    if additional_captures:
                        eval_val, _ = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, maximizing_player, game)
                    else:
                        eval_val, _ = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, False, game)
                    if eval_val > max_eval:
                        max_eval = eval_val
                        best_move = (piece, move)
                    alpha = max(alpha, eval_val)
                    if beta <= alpha:
                        break
            return max_eval, best_move

        else:
            min_eval = float('inf')
            best_move = None
            all_moves = board.get_all_moves(current_color)

            if has_captures:
                pieces_with_captures = []
                for piece in all_moves:
                    for move in all_moves[piece]:
                        if all_moves[piece][move]:
                            pieces_with_captures.append(piece)
                            break
                move_sources = {p: all_moves[p] for p in pieces_with_captures}
                print(f"Pieces with captures: {[f'({p.row}, {p.col})' for p in pieces_with_captures]}")
            else:
                move_sources = all_moves
                print("No captures available, considering all moves")

            for piece in move_sources:
                for move, captures in move_sources[piece].items():
                    if has_captures and not captures:
                        continue
                    temp_board = board.copy()
                    temp_piece = temp_board.get_piece(piece.row, piece.col)
                    if not temp_piece:
                        continue
                    temp_board.move(temp_piece, move[0], move[1])
                    if captures:
                        pieces_to_remove = [temp_board.get_piece(captured.row, captured.col) for captured in captures if temp_board.get_piece(captured.row, captured.col) != 0]
                        if pieces_to_remove:
                            temp_board.remove(pieces_to_remove)
                    additional_captures = False
                    if captures:
                        next_moves = temp_board.get_valid_moves(temp_piece)
                        additional_captures = any(len(next_moves[m]) > 0 for m in next_moves)
                    if additional_captures:
                        eval_val, _ = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, maximizing_player, game)
                    else:
                        eval_val, _ = minimax_alpha_beta(temp_board, depth - 1, alpha, beta, True, game)
                    if eval_val < min_eval:
                        min_eval = eval_val
                        best_move = (piece, move)
                    beta = min(beta, eval_val)
                    if beta <= alpha:
                        break
            return min_eval, best_move
    except Exception as e:
        print(f"Error in minimax_alpha_beta: {e}")
        traceback.print_exc()
        return 0, None

def find_best_move(game, depth=3):
    """Find the best move for the bot using minimax"""
    try:
        if game.game_over:
            return None
        score, best_move = minimax_alpha_beta(
            game.board, depth, float('-inf'), float('inf'), True, game
        )
        return best_move
    except Exception as e:
        print(f"Error in find_best_move: {e}")
        traceback.print_exc()
        return None

def bot_move(game):
    """Improved AI opponent move selection using minimax with alpha-beta pruning"""
    try:
        if game.game_over or game.bot_thinking:
            print("Bot move skipped: game over or bot thinking")
            return

        game.bot_thinking = True
        total_pieces = sum(1 for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if game.board.get_piece(row, col) != 0)

        if total_pieces > 16:
            depth = 3
        elif total_pieces > 8:
            depth = 4
        else:
            depth = 5

        print(f"Bot calculating move with depth {depth}, total pieces: {total_pieces}")
        best_move = find_best_move(game, depth)

        if not best_move:
            game.bot_thinking = False
            print("No valid move found for bot, checking winner...")
            winner = game.board.check_winner()
            if winner:
                game.game_over = True
                game.winner = winner
            return

        piece, move = best_move
        if not piece or not hasattr(piece, 'row') or not hasattr(piece, 'col'):
            print("Invalid piece in best_move")
            game.bot_thinking = False
            return

        row, col = move
        was_king = piece.king
        piece.was_king = was_king
        moves = game.board.get_valid_moves(piece)
        captured_pieces = moves.get(move, [])
        original_start = (piece.row, piece.col)
        game.last_move_start = original_start
        game.last_captured_pieces = captured_pieces.copy() if captured_pieces else []

        current_state = game.encode_board_state()
        game.position_history.append(current_state)
        if captured_pieces:
            game.moves_without_capture = 0
        else:
            game.moves_without_capture += 1
        if not piece.king:
            game.moves_without_pawn_move = 0
        else:
            game.moves_without_pawn_move += 1

        print(f"Bot moving piece from {original_start} to ({row}, {col}), captured: {len(captured_pieces)}")
        if captured_pieces:
            log_move(game, piece, row, col, captured_pieces)
        else:
            log_move(game, piece, row, col)

        game.board.move(piece, row, col)
        print(f"Bot piece after move: ({piece.row}, {piece.col}), x={piece.x}, y={piece.y}")
        game.last_moved_piece = piece
        game.animation_start_time = time.time()

        if captured_pieces:
            print(f"Bot removing captured pieces: {[f'({p.row}, {p.col})' for p in captured_pieces]}")
            game.board.remove(captured_pieces)
            next_moves = game.board.get_valid_moves(piece)
            while any(len(next_moves[m]) > 0 for m in next_moves):
                print("Bot checking for additional captures...")
                follow_up_move = find_best_move(game, 2)
                if not follow_up_move:
                    print("No follow-up move found")
                    break
                follow_piece, follow_pos = follow_up_move
                if not follow_piece or not hasattr(follow_piece, 'row') or not hasattr(follow_piece, 'col'):
                    print("Invalid follow_piece in follow_up_move")
                    break
                current_state = game.encode_board_state()
                game.position_history.append(current_state)
                game.moves_without_capture = 0
                if not follow_piece.king:
                    game.moves_without_pawn_move = 0
                additional_captures = next_moves.get(follow_pos, [])
                if additional_captures:
                    game.last_captured_pieces.extend(additional_captures)
                print(f"Bot making additional capture from ({follow_piece.row}, {follow_piece.col}) to {follow_pos}")
                log_move(game, follow_piece, follow_pos[0], follow_pos[1], additional_captures)
                game.board.move(follow_piece, follow_pos[0], follow_pos[1])
                print(f"Bot piece after additional move: ({follow_piece.row}, {follow_piece.col}), x={follow_piece.x}, y={follow_piece.y}")
                game.last_moved_piece = follow_piece
                game.animation_start_time = time.time()
                game.board.remove(additional_captures)
                next_moves = game.board.get_valid_moves(follow_piece)

        game.change_turn()
        game.bot_thinking = False
        print("Bot move completed")
        game.debug_board_state = True  # Trigger board state logging in update
    except Exception as e:
        print(f"Error in bot_move: {e}")
        traceback.print_exc()
        game.bot_thinking = False