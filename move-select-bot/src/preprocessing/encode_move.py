import torch

# ========================
# 1. CONSTANTS
# ========================

POLICY_PLANES = 73
POLICY_SIZE = 64 * POLICY_PLANES

# Sliding directions: N, S, E, W, NE, NW, SE, SW
SLIDE_DIRS = [
    (-1, 0),  # N
    (1, 0),   # S
    (0, 1),   # E
    (0, -1),  # W
    (-1, 1),  # NE
    (-1, -1), # NW
    (1, 1),   # SE
    (1, -1),  # SW
]

# Knight move offsets
KNIGHT_DIRS = [
    (-2, 1), (-1, 2), (1, 2), (2, 1),
    (2, -1), (1, -2), (-1, -2), (-2, -1)
]

# Promotion offsets (from white perspective)
PROMO_DIRS = [
    (-1, 0),  # forward
    (-1, -1), # capture-left
    (-1, 1),  # capture-right
]
PROMO_PIECES = ["q", "r", "n"]  # queen, rook, knight


# ========================
# 2. UTILITY FUNCTIONS
# ========================

def square_to_coords(sq: str):
    """
    Convert algebraic square like 'e2' → (rank, file) 0..7
    Rank 0 = row 8, File 0 = column a
    """
    file = ord(sq[0]) - ord('a')
    rank = 8 - int(sq[1])
    return rank, file


def parse_uci_move(move: str):
    """
    Parse UCI like 'e2e4' or 'e7e8q'
    Returns: (from_sq, to_sq, promo)
    """
    if len(move) == 4:
        return move[:2], move[2:], None
    elif len(move) == 5:
        return move[:2], move[2:4], move[4].lower()
    else:
        raise ValueError(f"Invalid UCI move: {move}")


# ========================
# 3. MOVE TO PLANE
# ========================

def move_to_plane(from_sq, to_sq, promo=None):
    """
    Compute plane index (0..72) for a move
    from_sq, to_sq: (rank, file)
    promo: 'q','r','n' or None
    """
    r1, c1 = from_sq
    r2, c2 = to_sq
    dr = r2 - r1
    dc = c2 - c1

    # ---- 1) Promotions ----
    if promo is not None:
        if promo not in PROMO_PIECES:
            raise ValueError(f"Unsupported promotion piece: {promo}")
        piece_idx = PROMO_PIECES.index(promo)

        # Determine direction
        if dc == 0:
            dir_idx = 0  # straight
        elif dc == -1:
            dir_idx = 1  # capture-left
        elif dc == 1:
            dir_idx = 2  # capture-right
        else:
            raise ValueError(f"Illegal promotion move: dr={dr}, dc={dc}")
        return 64 + dir_idx * 3 + piece_idx

    # ---- 2) Knight moves ----
    for idx, (kr, kc) in enumerate(KNIGHT_DIRS):
        if (dr, dc) == (kr, kc):
            return 56 + idx

    # ---- 3) Sliding moves ----
    for dir_idx, (rr, cc) in enumerate(SLIDE_DIRS):
        for dist in range(1, 8):
            if (dr, dc) == (rr * dist, cc * dist):
                return dir_idx * 7 + (dist - 1)

    raise ValueError(f"Move cannot be encoded: dr={dr}, dc={dc}, promo={promo}")
#TODO: fix bishop promos

# ========================
# 4. MOVE TO INDEX (0..4671)
# ========================

def move_to_index(move: str):
    from_sq_str, to_sq_str, promo = parse_uci_move(move)
    from_sq = square_to_coords(from_sq_str)
    to_sq = square_to_coords(to_sq_str)
    plane = move_to_plane(from_sq, to_sq, promo)
    square_index = from_sq[0] * 8 + from_sq[1]
    return square_index * POLICY_PLANES + plane


# ========================
# 5. LINE TO POLICY VECTOR
# ========================

def line_to_policy(line: str):
    """
    Convert a space-separated move line into a torch tensor of shape (4672,)
    Scores: 1/(i+1) for moves in line
    """
    dist = torch.zeros(POLICY_SIZE, dtype=torch.float32)
    moves = line.strip().split()
    for i, move in enumerate(moves):
        idx = move_to_index(move)
        score = 1.0 / (i + 1)
        dist[idx] = score
    return dist


# ========================
# 6. TEST
# ========================

if __name__ == "__main__":
    line = "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
    policy = line_to_policy(line)
    print(policy.shape)           # torch.Size([4672])
    print((policy > 0).sum())     # 10 nonzero moves
