"""
Modal deployment script for NNUE training with Hugging Face datasets.

Usage:
    modal run modal_train.py --dataset-name linrock/test80-2024
"""

import modal
from modal import Image, App, Volume, Secret
import os
from pathlib import Path

# Define the Modal image with all dependencies
# Mount training code directly into the image
training_dir = Path(__file__).parent

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        "cmake",
        "git",
        "wget",
        "libboost-all-dev",  # For C++ data loader
        "zstd",  # For decompressing .zst files
    )
    .uv_pip_install(
        "psutil",
        "asciimatics",
        "GPUtil",
        "python-chess==0.31.4",
        "matplotlib",
        "tensorboard",
        "numba",
        "numpy<2.0",
        "requests",
        "lightning",
        "datasets",
        "huggingface-hub",
        "pybind11",  # For C++ bindings
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .add_local_dir(
        training_dir,
        remote_path="/root/training",
        ignore=["__pycache__", "*.pyc", ".git", "*.log"],
        copy=True,  # Copy files into image so we can build
    )
    .run_commands(
        # Compile C++ data loader during image build
        "cd /root/training && "
        "mkdir -p build && "
        "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./ && "
        "cmake --build build --target install && "
        "echo 'Library search:' && "
        "find . -name '*training_data_loader*.so' 2>/dev/null | head -5 && "
        "ls -la data_loader/*.so 2>/dev/null || echo 'No .so in data_loader/' && "
        "ls -la *.so 2>/dev/null | head -3 || echo 'No .so in root'"
    )
)

app = App("nnue-training", image=image)

# Volume for persisting checkpoints and models
checkpoint_volume = Volume.from_name("nnue-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",  # Use A10G GPU, can change to A100 or T4
    volumes={"/checkpoints": checkpoint_volume},
    timeout=86400,  # 24 hours
    # Optional: Add Secret for private Hugging Face datasets
    # secrets=[Secret.from_name("huggingface")],
)
def train_nnue(
    dataset_name: str = "linrock/test80-2024",
    batch_size: int = 16384,
    max_epochs: int = 10,
    features: str = "HalfKAv2_hm^",
    learning_rate: float = 8.75e-4,
    num_workers: int = 4,
    epoch_size: int = 1638400,
    validation_size: int = 16384,
    lr: float = None,  # Override learning rate
    test_mode: bool = False,
    test_mode_file_count: int = 1,  # Number of files to use in test mode
    max_files: int = None,  # Limit number of files to download (None = all)
    specific_file: str = None,  # Download a specific file by name
    # High-impact parameters
    start_lambda: float = 1.0,
    end_lambda: float = 1.0,
    gamma: float = 0.992,
    pow_exp: float = 2.5,
    qp_asymmetry: float = 0.0,
    in_offset: float = 270.0,
    out_offset: float = 270.0,
    in_scaling: float = 340.0,
    out_scaling: float = 380.0,
):
    """
    Train NNUE network using Hugging Face dataset.
    
    Args:
        dataset_name: Hugging Face dataset name (e.g., "linrock/test80-2024")
        batch_size: Training batch size
        max_epochs: Maximum number of epochs
        features: Feature set to use (e.g., "HalfKAv2_hm^")
        learning_rate: Learning rate
        num_workers: Number of data loading workers
        epoch_size: Size of each epoch
        validation_size: Validation set size
        test_mode: If True, runs minimal training for quick verification
        test_mode_file_count: Number of files to use when test_mode is True
    """
    import subprocess
    import sys
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import tempfile
    import shutil
    
    # Setup working directory
    work_dir = Path("/root/nnue_training")
    work_dir.mkdir(exist_ok=True)
    os.chdir(work_dir)
    
    print(f"Working directory: {work_dir}")
    print(f"Loading dataset: {dataset_name}")
    
    binpack_files = []
    
    # Strategy 1: Try to list files directly from the repository
    print("Attempting to list files from repository...")
    try:
        from huggingface_hub import list_repo_files
        repo_files = list_repo_files(dataset_name, repo_type="dataset")
        print(f"Found {len(repo_files)} files in repository:")
        for f in repo_files[:10]:  # Show first 10
            print(f"  - {f}")
        
        # Download .binpack files (both compressed .zst and uncompressed)
        # Skip .tar.zst files - we only want .binpack.zst files
        compressed_files = []
        
        # Determine max files: test_mode -> test_mode_file_count, specific_file -> 1, max_files param, or None (all)
        if test_mode:
            effective_max_files = test_mode_file_count
        elif specific_file:
            effective_max_files = 1
        elif max_files is None:
            effective_max_files = None  # None means download all files
        else:
            effective_max_files = max_files
        
        # Filter to only .binpack.zst files (not .tar.zst)
        binpack_zst_files = [f for f in repo_files if f.endswith(".binpack.zst") and not f.endswith(".tar.zst")]
        uncompressed_files = [f for f in repo_files if f.endswith(".binpack") and not f.endswith(".zst")]
        
        print(f"Found {len(binpack_zst_files)} compressed .binpack.zst files")
        print(f"Found {len(uncompressed_files)} uncompressed .binpack files")
        
        # If specific file requested, filter to that file
        if specific_file:
            print(f"Looking for specific file: {specific_file}")
            matching_files = [f for f in binpack_zst_files if specific_file in f]
            if matching_files:
                binpack_zst_files = matching_files
                print(f"  ✓ Found matching file: {binpack_zst_files[0]}")
            else:
                print(f"  ⚠️  Specific file not found, will use first available file")
        
        if effective_max_files:
            print(f"Limiting download to {effective_max_files} file(s) for faster processing")
        
        file_count = 0
        # Process compressed files first
        for file in binpack_zst_files:
            if effective_max_files and file_count >= effective_max_files:
                print(f"Reached file limit ({effective_max_files}), stopping download")
                break
                
            print(f"Downloading compressed file: {file}...")
            print(f"  File size: ~7-8 GB (will decompress to larger size)")
            try:
                compressed_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=file,
                    repo_type="dataset"
                )
                compressed_files.append((compressed_path, file))
                file_count += 1
                file_size_mb = Path(compressed_path).stat().st_size / (1024 * 1024)
                print(f"  ✓ Downloaded to: {compressed_path} ({file_size_mb:.1f} MB)")
            except Exception as e:
                print(f"  ✗ Failed to download {file}: {e}")
        
        # Process uncompressed files if any
        for file in uncompressed_files:
            if effective_max_files and file_count >= effective_max_files:
                break
                
            print(f"Downloading uncompressed file: {file}...")
            try:
                local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=file,
                    repo_type="dataset"
                )
                binpack_files.append(local_path)
                file_count += 1
                print(f"  ✓ Downloaded to: {local_path}")
            except Exception as e:
                print(f"  ✗ Failed to download {file}: {e}")
        
        # Decompress .zst files
        if compressed_files:
            print(f"\nDecompressing {len(compressed_files)} .zst file(s)...")
            print("This may take a few minutes for large files...")
            for compressed_path, original_name in compressed_files:
                # Output path without .zst extension
                output_path = compressed_path.replace(".zst", "")
                compressed_size_mb = Path(compressed_path).stat().st_size / (1024 * 1024)
                print(f"Decompressing {Path(original_name).name}...")
                print(f"  Compressed size: {compressed_size_mb:.1f} MB")
                print(f"  Output: {Path(output_path).name}")
                
                # Use zstd to decompress
                # zstd -d decompresses, -f forces overwrite, -o specifies output
                import time
                start_time = time.time()
                result = subprocess.run(
                    ["zstd", "-d", "-f", compressed_path, "-o", output_path],
                    check=False,  # Don't raise on error, check manually
                    capture_output=True,
                    text=True
                )
                decompress_time = time.time() - start_time
                
                if result.returncode != 0:
                    print(f"  ✗ zstd error: {result.stderr}")
                    continue
                
                if Path(output_path).exists():
                    binpack_files.append(output_path)
                    decompressed_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                    print(f"  ✓ Decompressed in {decompress_time:.1f}s")
                    print(f"  Decompressed size: {decompressed_size_mb:.1f} MB")
                    # Remove compressed file to save space
                    Path(compressed_path).unlink()
                    print(f"  ✓ Ready: {output_path}")
                else:
                    print(f"  ✗ Decompression failed - output file not found")
                    
    except Exception as e:
        print(f"Error listing repository files: {e}")
        import traceback
        traceback.print_exc()
        print("Trying alternative method...")
    
    # Strategy 2: If no files found, try loading as a dataset
    if not binpack_files:
        print("Attempting to load as Hugging Face dataset...")
        try:
            dataset = load_dataset(dataset_name, split="train")
            print(f"Dataset loaded. Features: {dataset.features}")
            print(f"Dataset size: {len(dataset)}")
            
            # Check first item to see structure
            if len(dataset) > 0:
                first_item = dataset[0]
                print(f"First item keys: {list(first_item.keys())}")
                print(f"First item sample: {str(first_item)[:200]}")
            
            # Try to find file references in the dataset
            for i, item in enumerate(dataset):
                # Look for any field that might contain a file path
                for key, value in item.items():
                    if isinstance(value, str) and (".binpack" in value.lower() or "pack" in key.lower()):
                        print(f"Found potential file reference in field '{key}': {value}")
                        # Try to download it
                        try:
                            local_path = hf_hub_download(
                                repo_id=dataset_name,
                                filename=value,
                                repo_type="dataset"
                            )
                            binpack_files.append(local_path)
                            print(f"  ✓ Downloaded: {local_path}")
                        except Exception as e:
                            print(f"  ✗ Failed to download {value}: {e}")
                
                # Limit to first 100 items to avoid iterating through huge datasets
                if i >= 100:
                    break
        except Exception as e:
            print(f"Error loading as dataset: {e}")
            import traceback
            traceback.print_exc()
    
    # Strategy 3: Try common file names if still no files
    if not binpack_files:
        print("Trying common file names...")
        common_names = ["data.binpack", "train.binpack", "dataset.binpack", "data/train.binpack"]
        for name in common_names:
            try:
                local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=name,
                    repo_type="dataset"
                )
                binpack_files.append(local_path)
                print(f"  ✓ Found: {name}")
                break
            except Exception:
                continue
    
    if not binpack_files:
        error_msg = (
            f"No binpack files found in dataset '{dataset_name}'!\n"
            f"Please verify:\n"
            f"  1. The dataset exists and is accessible\n"
            f"  2. The dataset contains .binpack files\n"
            f"  3. You have permission to access the dataset\n"
            f"\nYou can check the dataset at: https://huggingface.co/datasets/{dataset_name}"
        )
        raise ValueError(error_msg)
    
    print(f"Found {len(binpack_files)} binpack files")
    for f in binpack_files:
        print(f"  - {f}")
    
    # Training code is mounted at /root/training
    training_dir = Path("/root/training")
    os.chdir(training_dir)
    print(f"Training directory: {training_dir}")
    print(f"Files in training dir: {list(training_dir.iterdir())[:10]}")
    
    # Install cupy if missing (large package, install at runtime to avoid image build hangs)
    try:
        import cupy
        print("✓ CuPy already installed")
    except ImportError:
        print("⚠️  CuPy not found, installing... (this may take a few minutes)")
        import subprocess
        result = subprocess.run(
            ["pip", "install", "--no-cache-dir", "cupy-cuda12x"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        if result.returncode == 0:
            print("✓ CuPy installed successfully")
        else:
            print(f"⚠️  CuPy installation failed: {result.stderr}")
            print("Trying alternative: cupy-cuda11x...")
            result2 = subprocess.run(
                ["pip", "install", "--no-cache-dir", "cupy-cuda11x"],
                capture_output=True,
                text=True,
                timeout=600
            )
            if result2.returncode != 0:
                raise RuntimeError(f"Failed to install CuPy: {result2.stderr}")
    
    # Check if C++ data loader shared library exists
    print("Checking C++ data loader...")
    # Look for the compiled shared library (matches _native.py search pattern)
    # _native.py searches for "./*training_data_loader.*" in current directory
    so_files = list(Path(".").glob("*training_data_loader*.so"))
    so_files.extend(Path("data_loader").glob("*training_data_loader*.so"))
    so_files.extend(Path(".").glob("libtraining_data_loader.so"))
    
    if not so_files:
        print("⚠️  C++ data loader shared library not found. Compiling...")
        try:
            build_dir = Path("build")
            build_dir.mkdir(exist_ok=True)
            
            # Configure with CMake
            print("  Running cmake...")
            cmake_result = subprocess.run(
                ["cmake", "-S", ".", "-B", str(build_dir), 
                 "-DCMAKE_BUILD_TYPE=Release", 
                 "-DCMAKE_INSTALL_PREFIX=./"],
                cwd=training_dir,
                check=False,
                capture_output=True,
                text=True
            )
            
            if cmake_result.returncode != 0:
                print(f"  ✗ CMake configuration failed: {cmake_result.stderr}")
                raise RuntimeError("CMake configuration failed")
            
            # Build and install
            print("  Building library...")
            build_result = subprocess.run(
                ["cmake", "--build", str(build_dir), "--target", "install"],
                cwd=training_dir,
                check=False,
                capture_output=True,
                text=True
            )
            
            if build_result.returncode != 0:
                print(f"  ✗ Build failed: {build_result.stderr}")
                raise RuntimeError("Build failed")
            
            # Check again for the library (matches _native.py search pattern)
            so_files = list(Path(".").glob("*training_data_loader*.so"))
            so_files.extend(Path("data_loader").glob("*training_data_loader*.so"))
            so_files.extend(Path(".").glob("libtraining_data_loader.so"))
            
            if so_files:
                print(f"  ✓ Compiled successfully: {so_files[0]}")
            else:
                print("  ⚠️  Build completed but library not found in expected location")
                print("  Searching for library files...")
                all_so = list(Path(".").rglob("*.so"))
                for f in all_so[:5]:
                    print(f"    Found: {f}")
        except Exception as e:
            print(f"  ✗ Compilation failed: {e}")
            import traceback
            traceback.print_exc()
            print("  ⚠️  Training may fail without the C++ data loader")
    else:
        print(f"✓ Found C++ data loader: {so_files[0]}")
    
    # Prepare training command
    # train.py expects datasets as positional arguments
    cmd = ["python", "train.py"]
    
    # Add training datasets (all binpack files)
    # train.py expects datasets as positional arguments, not as a list
    for dataset in binpack_files:
        cmd.append(str(dataset))
    
    # Add validation dataset (use first file if only one, or separate if multiple)
    # train.py expects --validation-data as a list, but we'll skip it for single file
    # (train.py will use the same data for validation if no validation data is specified)
    if len(binpack_files) > 1:
        cmd.extend(["--validation-data", str(binpack_files[-1])])  # Use last file for validation
    
    # Add other arguments
    cmd.extend([
        "--batch-size", str(batch_size),
        "--max_epochs", str(max_epochs),
        "--features", features,
        "--num-workers", str(num_workers),
        "--lr", str(lr if lr is not None else learning_rate),
        "--default_root_dir", "/checkpoints",
        "--epoch-size", str(epoch_size),
        "--validation-size", str(validation_size),
        # High-impact parameters
        "--start-lambda", str(start_lambda),
        "--end-lambda", str(end_lambda),
        "--gamma", str(gamma),
        "--pow-exp", str(pow_exp),
        "--qp-asymmetry", str(qp_asymmetry),
        "--in-offset", str(in_offset),
        "--out-offset", str(out_offset),
        "--in-scaling", str(in_scaling),
        "--out-scaling", str(out_scaling),
    ])
    
    if test_mode:
        print("🧪 TEST MODE: Using minimal settings for quick verification")
        # In test mode, don't save networks frequently, just verify it runs
        cmd.extend([
            "--network-save-period", "999",  # Don't save during test
        ])
    else:
        cmd.extend([
            "--network-save-period", "10",  # Save every 10 epochs
        ])
    
    print(f"Running training command: {' '.join(cmd)}")
    
    # Run training
    result = subprocess.run(cmd, cwd=training_dir)
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with return code {result.returncode}")
    
    # Save checkpoints to volume
    checkpoint_dir = Path("/checkpoints")
    
    # PyTorch Lightning saves to default_root_dir/lightning_logs/version_X/checkpoints/
    logdir = Path("/checkpoints/lightning_logs")
    if logdir.exists():
        # Copy all lightning logs (includes checkpoints)
        for version_dir in logdir.glob("version_*"):
            checkpoint_path = version_dir / "checkpoints"
            if checkpoint_path.exists():
                print(f"Found checkpoints in: {checkpoint_path}")
                # List all checkpoint files
                ckpt_files = list(checkpoint_path.glob("*.ckpt"))
                for ckpt in ckpt_files:
                    print(f"  - {ckpt.name} ({ckpt.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Also check for checkpoints in training_dir (if default_root_dir wasn't used)
    if Path(training_dir / "checkpoints").exists():
        shutil.copytree(
            training_dir / "checkpoints",
            checkpoint_dir / "checkpoints",
            dirs_exist_ok=True
        )
        print(f"Saved checkpoints to volume")
    
    # Check for .nnue files (if serialize.py was run)
    nnue_files = list(Path("/checkpoints").rglob("*.nnue"))
    if nnue_files:
        print(f"\nFound {len(nnue_files)} .nnue file(s):")
        for nnue in nnue_files:
            print(f"  - {nnue} ({nnue.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("\n⚠️  No .nnue files found. To convert .ckpt to .nnue, run:")
        print("   python serialize.py <checkpoint.ckpt> <output.nnue>")
        print(f"\nCheckpoint files are in: /checkpoints/lightning_logs/version_*/checkpoints/")
    
    return "Training completed successfully!"


@app.local_entrypoint()
def main(
    dataset_name: str = "linrock/test80-2024",
    batch_size: int = 32768,  # Larger batch = better gradients
    max_epochs: int = 15,  # More epochs for better convergence
    features: str = "HalfKAv2_hm^",
    learning_rate: float = 8.75e-4,
    num_workers: int = 4,
    epoch_size: int = 1638400,
    validation_size: int = 16384,
    test_mode: bool = False,
    test_mode_file_count: int = 1,  # Number of files to use in test mode
    file_count: int = 1,  # Number of files to use in normal mode (0 = all files)
    specific_file: str = None,  # Download specific file by name
    # Lambda scheduling: blend eval scores and game outcomes
    start_lambda: float = 1.0,  # Start with pure eval scores
    end_lambda: float = 0.5,  # End with 50% eval / 50% game results
    gamma: float = 0.995,  # Slower LR decay for longer training
    # Loss function improvements
    pow_exp: float = 2.6,  # Penalize large errors more
    qp_asymmetry: float = 0.15,  # Penalize overconfidence
    # Sigmoid parameters (run perf_sigmoid_fitter.py to optimize these)
    in_offset: float = 270.0,
    out_offset: float = 270.0,
    in_scaling: float = 340.0,
    out_scaling: float = 380.0,
):
    """
    Local entrypoint to run training on Modal.
    
    Args:
        test_mode: If True, runs a quick 1-minute test to verify setup
        test_mode_file_count: Number of files to use when test_mode is True (default: 1)
        file_count: Number of files to use in normal mode (default: 1, use 0 for all files)
        specific_file: Download a specific file by name
        start_lambda: Lambda at first epoch (1.0 = pure evals, 0.0 = pure game results)
        end_lambda: Lambda at last epoch (linear interpolation)
        gamma: LR decay multiplier per epoch (0.995 = slower decay)
        pow_exp: Loss exponent (higher = penalize large errors more)
        qp_asymmetry: Penalty when prediction > actual (0 = symmetric)
        in_offset, out_offset, in_scaling, out_scaling: Sigmoid conversion params
            Run 'python perf_sigmoid_fitter.py' to optimize for your data
    
    Example:
        # Quick test (1 minute, 1 file)
        modal run modal_train.py --test-mode
        
        # Optimized training (high impact improvements)
        modal run modal_train.py --file-count 5 --max-epochs 15
        
        # Custom lambda schedule
        modal run modal_train.py --start-lambda 1.0 --end-lambda 0.3
        
        # Optimize sigmoid params first (recommended):
        # python perf_sigmoid_fitter.py
        # Then use the suggested values:
        modal run modal_train.py --in-offset 250 --out-offset 260 --in-scaling 350 --out-scaling 400
    """
    if test_mode:
        print(f"🧪 Running in TEST MODE - quick 1-minute verification with {test_mode_file_count} file(s)")
        print("This will use minimal settings to verify the setup works")
        result = train_nnue.remote(
            dataset_name=dataset_name,
            batch_size=4096,  # Smaller batch
            max_epochs=1,  # Just 1 epoch
            features=features,
            learning_rate=learning_rate,
            num_workers=2,  # Fewer workers
            epoch_size=10000,  # Very small epoch (should complete in ~1 minute)
            validation_size=1000,  # Small validation
            lr=learning_rate,
            test_mode=True,
            test_mode_file_count=test_mode_file_count,
            max_files=None,  # Let test_mode_file_count control it
            specific_file=specific_file,
        )
    else:
        # Convert file_count: 0 means all files, otherwise use the specified count
        effective_max_files = None if file_count == 0 else file_count
        
        print(f"🚀 Starting optimized training:")
        print(f"  • Batch size: {batch_size} (larger = better gradients)")
        print(f"  • Max epochs: {max_epochs}")
        print(f"  • Lambda schedule: {start_lambda} → {end_lambda} (eval/result blending)")
        print(f"  • Gamma: {gamma} (LR decay per epoch)")
        print(f"  • Loss: pow={pow_exp}, asymmetry={qp_asymmetry}")
        print(f"  • Sigmoid: in_offset={in_offset}, out_offset={out_offset}")
        print(f"  • Files: {file_count if file_count > 0 else 'all'}")
        
        result = train_nnue.remote(
            dataset_name=dataset_name,
            batch_size=batch_size,
            max_epochs=max_epochs,
            features=features,
            learning_rate=learning_rate,
            num_workers=num_workers,
            epoch_size=epoch_size,
            validation_size=validation_size,
            lr=learning_rate,
            test_mode=False,
            test_mode_file_count=1,  # Not used when test_mode=False
            max_files=effective_max_files,
            specific_file=specific_file,
            start_lambda=start_lambda,
            end_lambda=end_lambda,
            gamma=gamma,
            pow_exp=pow_exp,
            qp_asymmetry=qp_asymmetry,
            in_offset=in_offset,
            out_offset=out_offset,
            in_scaling=in_scaling,
            out_scaling=out_scaling,
        )
    print(result)

