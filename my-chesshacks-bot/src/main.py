from .utils import chess_manager, GameContext
from chess import Move
import torch
import torch.nn as nn
import numpy as np
import chess
import os

# ---------- Model Definition (must match training code) ----------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.relu(out)
        return out

class ChessEvalCNN(nn.Module):
    def __init__(self, num_blocks=6, num_channels=128, value_hidden=128):
        super().__init__()
        # Initial convolution
        self.conv_input = nn.Sequential(
            nn.Conv2d(13, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        
        # Residual tower
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        
        # Value head - more compact
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=1),  # Reduced from 32 to 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, value_hidden),  # 1024 -> value_hidden
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(value_hidden, 1)  # value_hidden -> 1
        )

    def forward(self, x):
        x = self.conv_input(x)
        x = self.residual_blocks(x)
        x = self.value_head(x)
        return x.squeeze(-1)

# ---------- Helper Functions ----------
PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# NEW: basic material values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,  # arbitrary large
}


def fen_to_tensor(fen: str) -> torch.Tensor:
    """Convert FEN string to tensor representation."""
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    board = chess.Board(fen)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        base = 0 if piece.color == chess.WHITE else 6
        p_index = PIECE_TO_INDEX[piece.piece_type] + base

        file = chess.square_file(square)
        rank = chess.square_rank(square)
        planes[p_index, 7 - rank, file] = 1.0

    stm_plane = 12
    planes[stm_plane, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return torch.from_numpy(planes)

def evaluate_position(model: ChessEvalCNN, board: chess.Board, device: torch.device, cp_scale: float = 1000.0) -> float:
    """Evaluate a chess position using the model."""
    fen = board.fen()
    tensor = fen_to_tensor(fen).unsqueeze(0).to(device)  # Add batch dimension
    
    with torch.no_grad():
        eval_score = model(tensor).item()
    
    # Convert back from normalized score to centipawns
    return eval_score * cp_scale

# ---------- NEW: Tactical Heuristic ----------
def tactical_heuristic(board_before: chess.Board,
                       board_after: chess.Board,
                       move: chess.Move) -> float:
    """
    Strong tactical heuristic in centipawns from the side-to-move's perspective
    (board_before.turn).

    Penalizes:
      - Losing material by SEE
      - Leaving/moving a piece where it can be taken by a *lower-value* piece
    Rewards:
      - Checkmate
      - Check
    """
    score = 0.0

    side_to_move = board_before.turn
    enemy_side = not side_to_move

    # ---------- 1) Static Exchange Evaluation (SEE) ----------
    see_value = 0
    if hasattr(board_before, "see"):
        try:
            see_value = board_before.see(move)
        except Exception:
            see_value = 0
    # Small weight from SEE itself
    score += 0.5 * see_value

    # ---------- 2) Hanging / dominated piece detection ----------
    to_sq = move.to_square
    piece_after = board_after.piece_at(to_sq)

    if piece_after is not None and piece_after.color == side_to_move:
        moved_val = PIECE_VALUES.get(piece_after.piece_type, 0)

        # All enemy attackers on the destination square
        enemy_attackers = board_after.attackers(enemy_side, to_sq)

        if enemy_attackers:
            attacker_values = []
            for sq in enemy_attackers:
                p = board_after.piece_at(sq)
                if p is None:
                    continue
                attacker_values.append(PIECE_VALUES.get(p.piece_type, 0))

            if attacker_values:
                min_attacker_val = min(attacker_values)

                # If a LOWER-VALUE piece can capture ours, treat as a serious blunder.
                if min_attacker_val < moved_val:
                    # Difference in material value
                    hanging_loss = moved_val - min_attacker_val

                    # Very big penalty (scaled); this is the key change.
                    score -= 6.0 * hanging_loss  # e.g. queen hanging to knight ⇒ ~3500cp penalty

                # Optional: mild penalty if equal-value piece can capture us
                elif min_attacker_val == moved_val:
                    score -= 2.0 * moved_val  # still quite bad to leave a free trade

    # ---------- 3) Checks and mates ----------
    if board_after.is_checkmate():
        # Huge bonus to force mate
        score += 100000.0
    else:
        if board_after.is_check():
            score += 80.0  # slightly stronger check bonus

        # Slight penalty for immediate non-winning game over (stalemate, repetition, etc.)
        if board_after.is_game_over():
            score -= 50.0

    return float(score)

# ---------- Load Model Once ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "chess_eval_cnn.pt")

# Add debug output
print(f"Script directory: {script_dir}")
print(f"Looking for model at: {model_path}")
print(f"Model file exists: {os.path.exists(model_path)}")

model = None
cp_scale = 1000.0

try:
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config to get model parameters
    config = checkpoint.get("config", {})
    num_blocks = config.get("num_blocks", 6)  # Updated default
    num_channels = config.get("num_channels", 128)  # Updated default
    value_hidden = config.get("value_hidden", 128)  # New parameter
    cp_scale = config.get("cp_scale", 1000.0)
    
    # Initialize model with correct architecture
    model = ChessEvalCNN(num_blocks=num_blocks, num_channels=num_channels, value_hidden=value_hidden).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded successfully from {model_path} on {device}")
    print(f"Model architecture: {num_blocks} blocks, {num_channels} channels, {value_hidden} value_hidden")
except FileNotFoundError:
    print(f"WARNING: Model file not found at {model_path}. Using random moves as fallback.")
    model = None
except Exception as e:
    print(f"WARNING: Error loading model: {e}. Using random moves as fallback.")
    model = None

# ---------- Chess Bot Entry Point ----------
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Evaluate all legal moves and choose the one with the best NN + heuristic evaluation.
    From the current player's perspective, higher score is better.
    """
    legal_moves = list(ctx.board.generate_legal_moves())
    
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")
    
    if model is None:
        # Fallback to random if model not loaded
        print("Model not available, using random move")
        move_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        ctx.logProbabilities(move_probs)
        return legal_moves[0]
    
    move_scores = {}
    debug_info = {}

    # Weight for how much we trust the heuristic vs the NN eval
    heuristic_weight = 1

    for move in legal_moves:
        # Make the move on a copy of the board
        board_copy = ctx.board.copy()
        board_copy.push(move)
        
        # NN evaluation of the resulting position in centipawns
        nn_eval = evaluate_position(model, board_copy, device, cp_scale)

        # Make it "from current player's perspective":
        # positive = good for the side to move in ctx.board
        if ctx.board.turn == chess.BLACK:
            nn_eval = -nn_eval

        # Heuristic tactical score from current player's perspective
        heuristic_eval = tactical_heuristic(ctx.board, board_copy, move)

        # Combine NN eval and heuristic
        combined_score = nn_eval + heuristic_weight * heuristic_eval

        move_scores[move] = combined_score
        debug_info[move] = (nn_eval, heuristic_eval, combined_score)
    
    # Find the best move (highest combined score from current player's perspective)
    best_move = max(move_scores.keys(), key=lambda m: move_scores[m])
    best_nn, best_heur, best_total = debug_info[best_move]
    
    print(
        f"Best move: {best_move} | NN: {best_nn:.2f} cp | "
        f"Heuristic: {best_heur:.2f} cp | Combined: {best_total:.2f} cp"
    )
    
    # Convert evaluations to probabilities using softmax over combined scores
    temperature = 0.1  # Lower = more deterministic, higher = more random
    scores_array = np.array([move_scores[m] for m in legal_moves], dtype=np.float64)
    scores_array = scores_array / temperature
    
    # Subtract max for numerical stability
    scores_array = scores_array - np.max(scores_array)
    exp_scores = np.exp(scores_array)
    probs = exp_scores / np.sum(exp_scores)
    
    move_probs = {move: float(prob) for move, prob in zip(legal_moves, probs)}
    ctx.logProbabilities(move_probs)
    
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Called when a new game begins."""
    print("New game started")
    # Clear any caches if needed
    pass
