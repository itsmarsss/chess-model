from .utils import chess_manager, GameContext
from chess import Move
import torch
import torch.nn as nn
import numpy as np
import chess
import os
import random
from .model.neural_network import ChessPolicyNet, ResidualBlock
from .preprocessing.preprocess_data import fen_to_tensor

# ---------- Load Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "final.pt")
print("Looking for model at:", model_path)
print("Model exists:", os.path.exists(model_path))

model = ChessPolicyNet()
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

def move_from_policy(model, board: chess.Board, device: torch.device):
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        return None, None
    
    # Convert board to tensor
    
    board = board.fen()[:-3]
    tensor = fen_to_tensor(board).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    print ("test2")
    
    # Score each legal move
    move_scores = {}
    for move in legal_moves:
        from_x, from_y = square_to_coords(chess.square_name(move.from_square))
        from_sq = from_x + from_y
        to_x, to_y = square_to_coords(chess.square_name(move.to_square))
        to_sq = to_x + to_y
        promo_char = None
        if move.promotion == chess.QUEEN: promo_char = "q"
        elif move.promotion == chess.ROOK: promo_char = "r"
        elif move.promotion == chess.KNIGHT: promo_char = "n"
        #elif move.promotion == chess.BISHOP: promo_char = "b"
        
        print("test3")

        plane = move_to_plane(from_sq, to_sq, promo_char)
        move_scores[move]=probs[plane]
        

    selected_move = max(move_scores.items(), key=lambda x: x[1])[0]
    return selected_move, move_scores

# ---------- Chess Bot Entrypoint ----------
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    try:
        legal_moves = list(ctx.board.generate_legal_moves())
        move, move_probs = move_from_policy(model, ctx.board, device)
        if move is None:
           ctx.logProbabilities({})
           raise ValueError("No legal moves available")
        
        ctx.logProbabilities({m: float(p) for m, p in zip(legal_moves, move_probs)})
        return move

    except Exception as e:
        print("Error in test_func:", e)
        ctx.logProbabilities({})
        return random.choice(list(ctx.board.generate_legal_moves()))

@chess_manager.reset
def reset_func(ctx: GameContext):
    print("New game started")