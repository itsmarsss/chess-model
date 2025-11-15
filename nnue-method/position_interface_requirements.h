/*
 * Position Interface Requirements for NNUE Integration
 * 
 * This file documents the minimum interface your Position class must implement
 * to use the ported NNUE evaluation function.
 * 
 * You can either:
 * 1. Implement these methods in your existing Position class
 * 2. Create an adapter class that wraps your Position and implements this interface
 */

#ifndef POSITION_INTERFACE_REQUIREMENTS_H
#define POSITION_INTERFACE_REQUIREMENTS_H

#include "types.h"

namespace Stockfish {

// Forward declaration - your Position class should be in this namespace
class Position {
public:
    // ============================================
    // REQUIRED METHODS FOR NNUE EVALUATION
    // ============================================
    
    // Get the piece at a given square
    Piece piece_on(Square s) const;
    
    // Get all pieces on the board
    Bitboard pieces() const;
    
    // Get all pieces of a given color
    Bitboard pieces(Color c) const;
    
    // Get pieces of specific type and color (template method)
    template<typename... PieceTypes>
    Bitboard pieces(Color c, PieceTypes... pts) const;
    
    // Get the side to move
    Color side_to_move() const;
    
    // Count pieces of a specific type (template method)
    template<PieceType Pt>
    int count() const;
    
    // Count pieces of a specific type for a color (template method)
    template<PieceType Pt>
    int count(Color c) const;
    
    // Get the square of a piece type for a color (template method)
    // Returns the square where the piece is located
    template<PieceType Pt>
    Square square(Color c) const;
    
    // Get non-pawn material value for a color
    // This should return the sum of material values (excluding pawns)
    Value non_pawn_material(Color c) const;
    
    // Get the 50-move rule counter
    int rule50_count() const;
    
    // ============================================
    // REQUIRED FOR INCREMENTAL UPDATES (Optional but Recommended)
    // ============================================
    
    // These are needed if you want to use incremental accumulator updates
    // instead of full refreshes on every move
    
    // Make a move (for incremental updates)
    // void do_move(Move m, StateInfo& newSt, const TranspositionTable* tt);
    // void do_move(Move m, StateInfo& newSt, bool givesCheck, 
    //              DirtyPiece& dp, DirtyThreats& dts, const TranspositionTable* tt);
    
    // Undo a move
    // void undo_move(Move m);
    
    // ============================================
    // HELPER METHODS (Used by features)
    // ============================================
    
    // These are used internally by the feature extraction code
    
    // Get piece array (used by some features)
    const std::array<Piece, SQUARE_NB>& piece_array() const;
    
    // Check if square is empty
    bool empty(Square s) const;
};

} // namespace Stockfish

#endif // POSITION_INTERFACE_REQUIREMENTS_H

