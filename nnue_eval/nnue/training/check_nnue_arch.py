import sys
import struct

def read_nnue_header(filepath):
    """Read NNUE file header to determine architecture."""
    try:
        with open(filepath, 'rb') as f:
            # Read version
            version = struct.unpack('<I', f.read(4))[0]
            print(f"Version: 0x{version:08X}")
            
            # Read hash
            hash_value = struct.unpack('<I', f.read(4))[0]
            print(f"Hash: 0x{hash_value:08X}")
            
            # Read description length
            desc_len = struct.unpack('<I', f.read(4))[0]
            print(f"Description length: {desc_len}")
            
            # Read description
            description = f.read(desc_len).decode('utf-8', errors='ignore')
            print(f"Description: {description}")
            
            # Try to determine architecture from file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            print(f"\nFile size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
            # Rough size estimates for different architectures:
            # L1=512:  ~20-40 MB
            # L1=1024: ~80-120 MB  
            # L1=2048: ~200-300 MB
            # L1=3072: ~400-500 MB (your training config)
            
            if file_size < 50_000_000:
                print("→ Likely L1=512 or smaller")
            elif 50_000_000 <= file_size < 150_000_000:
                print("→ Likely L1=1024 (COMPATIBLE with your Stockfish)")
            elif 150_000_000 <= file_size < 350_000_000:
                print("→ Likely L1=2048")
            else:
                print("→ Likely L1=3072 or larger (matches your training config)")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python check_nnue_arch.py <nnue_file>")
        sys.exit(1)
    
    read_nnue_header(sys.argv[1])
