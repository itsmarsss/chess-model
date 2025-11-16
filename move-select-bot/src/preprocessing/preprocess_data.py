""" 
This module will preprocess data loaded from Lichess/chess-position-evaluations.

The format of the data is:
{
  "fen": "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -",
  "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4",
  "depth": 36,
  "knodes": 206765,
  "cp": 311,
  "mate": None
}

We need to:
- Convert the fen board position into a 8x8xchannels tensor (what is channels?)
    - 12 binary planes, one per piece type (white/black, pawn/knight/bishop/rook/queen/king + side to move) - could also add castling rights, move count, etc
- Convert the line of best position into a target distribution over moves
    - Map position in line to index (ie. e2e4 -> 1433, as initial_pos * 64 + end_pos, add extra )
    - Use soft target distribution - +1 for first move, +0.8 for second, etc. 0 for all others
- Decode function
    
"""

import torch
from datasets import load_dataset
from .encode_move import line_to_policy, POLICY_SIZE

piece_to_plane = {
  "p": 6, "P": 0,
  "n": 7, "N": 1,
  "b": 8, "B": 2,
  "r": 9, "R": 3,
  "q": 10, "Q": 4,
  "k": 11, "K": 5,
}

def fen_to_tensor(fen:str) -> torch.Tensor:
  """ FEN notation: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
  ie. rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
  where black pieces are lowercase, white are uppercase, and numbers are empty squares
  we also have active color, castling, en passant, and halfmove/fullmove
  """
  """rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"""
  board, side, castling, ep = fen.split()
  
  x = torch.zeros((12, 8, 8), dtype=torch.float32) # make tensors - 12x8x8 of floats
  
  rows = board.split("/")
  # ie. rnbqkbnr, pppppppp, 8, 8, ...
  
  for rank_idx, rank in enumerate(rows): 
    file_idx = 0
    for symbol in rank:
      if symbol.isdigit(): #for numbers - skip
        file_idx += int(symbol)
      else:
        plane = piece_to_plane[symbol]
        x[plane, rank_idx, file_idx] = 1
        file_idx += 1
        
  side_plane = torch.ones((8,8), dtype=torch.float32) if side == "b" else torch.zeros((8,8))    
  x = torch.cat([x, side_plane.unsqueeze(0)], dim=0)
  return x

def line_to_dist(line: str):
  return line_to_policy(line)

def encode_targets(fen: str, line:str):
  board_tensor = fen_to_tensor(fen)
  policy_tensor = line_to_dist(line)
  return board_tensor, policy_tensor

def dataset_loader(split="train", max_examples=None, cache_path=None):
    """
    Loads Lichess/chess-position-evaluations dataset, preprocesses, and caches tensors.

    Args:
        split: 'train', 'test', etc.
        max_examples: int, optional - limit number of games for fast dev
        cache_path: str, optional - if provided, saves preprocessed tensors for reuse

    Returns:
        list of tuples: [(board_tensor, policy_tensor), ...]
    """
    print(f"Loading dataset {split} from HuggingFace...")
    dset = load_dataset("Lichess/chess-position-evaluations", split=split)

    if max_examples:
        dset = dset.select(range(max_examples))

    data = []
    for idx, ex in enumerate(dset):
        fen = ex["fen"]
        line = ex["line"]

        try:
            x, y = encode_targets(fen, line)
            data.append((x, y))
        except Exception as e:
            # Skip problematic positions
            print(f"Skipping example {idx}: {e}")
            continue

        if idx % 1000 == 0:
            print(f"Processed {idx} examples...")

    if cache_path:
        print(f"Caching preprocessed dataset to {cache_path}")
        torch.save(data, cache_path)

    return data


# -------------------------
# Test
# -------------------------
if __name__ == "__main__":
    example_line = "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
    print("Policy tensor shape:", line_to_dist(example_line).shape)
    
    # Quick local test: load 100 examples
    dataset = dataset_loader(max_examples=100)
    print(f"Loaded {len(dataset)} preprocessed examples")
    print("Board tensor shape:", dataset[0][0].shape)
    print("Policy tensor shape:", dataset[0][1].shape)