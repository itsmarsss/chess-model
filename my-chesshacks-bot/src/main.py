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
    def __init__(self, num_blocks=6, num_channels=128, value_head_channels=32, value_hidden=256):
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
        
        # Value head - improved capacity
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, value_head_channels, kernel_size=1),
            nn.BatchNorm2d(value_head_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_head_channels * 8 * 8, value_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(value_hidden, 1)
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
    num_blocks = config.get("num_blocks", 6)
    num_channels = config.get("num_channels", 128)
    value_head_channels = config.get("value_head_channels", 32)
    value_hidden = config.get("value_hidden", 256)
    cp_scale = config.get("cp_scale", 1000.0)
    
    # Initialize model with correct architecture
    model = ChessEvalCNN(
        num_blocks=num_blocks, 
        num_channels=num_channels, 
        value_head_channels=value_head_channels,
        value_hidden=value_hidden
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded successfully from {model_path} on {device}")
    print(f"Model architecture: {num_blocks} blocks, {num_channels} channels, {value_head_channels} value_head_channels, {value_hidden} value_hidden")
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
    Evaluate all legal moves and choose the one with the best evaluation.
    From the current player's perspective, higher eval is better.
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
    
    # Evaluate each legal move
    move_evals = {}
    
    for move in legal_moves:
        # Make the move on a copy of the board
        board_copy = ctx.board.copy()
        board_copy.push(move)
        
        # Evaluate the resulting position
        eval_score = evaluate_position(model, board_copy, device, cp_scale)
        
        # If it's black's turn, we want lower eval (negative is better for black)
        # If it's white's turn, we want higher eval (positive is better for white)
        if ctx.board.turn == chess.BLACK:
            eval_score = -eval_score
        
        move_evals[move] = eval_score
    
    # Find the best move (highest eval from current player's perspective)
    best_move = max(move_evals.keys(), key=lambda m: move_evals[m])
    best_eval = move_evals[best_move]
    
    print(f"Best move: {best_move} with eval: {best_eval:.2f} cp")
    
    # Convert evaluations to probabilities using softmax
    # Use temperature to control exploration vs exploitation
    temperature = 0.1  # Lower = more deterministic, higher = more random
    eval_array = np.array([move_evals[m] for m in legal_moves])
    eval_array = eval_array / temperature
    
    # Subtract max for numerical stability
    eval_array = eval_array - np.max(eval_array)
    exp_evals = np.exp(eval_array)
    probs = exp_evals / np.sum(exp_evals)
    
    move_probs = {move: float(prob) for move, prob in zip(legal_moves, probs)}
    ctx.logProbabilities(move_probs)
    
    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """Called when a new game begins."""
    print("New game started")
    # Clear any caches if needed
    pass