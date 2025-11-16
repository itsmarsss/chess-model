import modal
from preprocessing.preprocess_data import dataset_loader

app = modal.App(
    "chess_preprocess",
    volumes={"/mnt/data": modal.Volume.from_name("lichess-preproc-vol", create_if_missing=True)},
)

# Build an image that *includes your local Python package*
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "datasets", "python-chess")
    .add_local_python_source("preprocessing")  # <--- add your local module
)

@app.function(image=image, timeout=3600)
def preprocess_and_cache(max_examples=100000):
    cache_path = "/mnt/data/lichess_preprocessed.pt"
    print(f"Preprocessing {max_examples} examples, saving to {cache_path}")
    data = dataset_loader(split="train", max_examples=max_examples, cache_path=cache_path)
    print(f"Preprocessed {len(data)} examples")
    # commit the volume
    vol = modal.Volume.from_name("lichess-preproc-vol")
    vol.commit()
    print("Committed volume")
    return len(data)
