from huggingface_hub import snapshot_download
import os

def download_models():
    """Download BGE models locally for offline use."""
    
    # Download base model
    if not os.path.exists("bge-base-en-v1.5"):
        print("Downloading BGE base model...")
        snapshot_download(
            repo_id="BAAI/bge-base-en-v1.5", 
            local_dir="bge-base-en-v1.5"
        )
        print("✓ BGE base model downloaded")
    else:
        print("✓ BGE base model already exists")
    
    # Download large model
    if not os.path.exists("bge-large-en-v1.5"):
        print("Downloading BGE large model...")
        snapshot_download(
            repo_id="BAAI/bge-large-en-v1.5", 
            local_dir="bge-large-en-v1.5"
        )
        print("✓ BGE large model downloaded")
    else:
        print("✓ BGE large model already exists")
    
    # Download reranker model
    if not os.path.exists("bge-reranker-base"):
        print("Downloading BGE reranker model...")
        snapshot_download(
            repo_id="BAAI/bge-reranker-base", 
            local_dir="bge-reranker-base"
        )
        print("✓ BGE reranker model downloaded")
    else:
        print("✓ BGE reranker model already exists")
    
    print("\n🎉 All models downloaded successfully!")
    print("You can now run the application offline.")

if __name__ == "__main__":
    download_models()