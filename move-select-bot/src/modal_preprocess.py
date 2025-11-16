# modal_preprocess.py
import modal
import torch
import math

app = modal.App("chess-preprocess")

volume = modal.Volume.from_name("lichess-preproc-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("datasets", "torch")
    .add_local_python_source("preprocessing") 
)

@app.function(
    image=image,
    timeout=3600 * 6,
    volumes={"/mnt/data": volume},
)
def preprocess_all():
    import sys
    sys.path.insert(0, "/root/src")

    from preprocessing.preprocess_data import dataset_loader

    cache_path = "/mnt/data/lichess_preprocessed.pt"
    print(f"Preprocessing ALL data → {cache_path}")

    # Load full dataset
    data = dataset_loader(
        split="train",
        max_examples=None,      
        cache_path=cache_path,  
    )

    print(f"Done! Total samples: {len(data)}")

    vol.commit()
    print("Committed volume.")
    return len(data)
