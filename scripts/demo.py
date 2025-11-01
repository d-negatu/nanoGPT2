#!/usr/bin/env python3
"""
Demo script to generate sample images of nanoGPT2 in action.
This script can be used to create GIF animations or static images.
"""

import argparse
import torch
from pathlib import Path
from src.model import GPT
from src.config import get_config

def generate_demo_text(model, prompt="", max_tokens=50, device="cpu"):
    """
    Generate demo text from the model.
    
    Args:
        model: GPT model instance
        prompt: Starting prompt text
        max_tokens: Maximum tokens to generate
        device: Device to use for computation
    
    Returns:
        Generated text string
    """
    model.eval()
    with torch.no_grad():
        # Create context tensor from prompt
        context = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long, device=device)
        
        # Generate samples
        samples = model.generate(context, max_new_tokens=max_tokens)
        
        # Decode to text
        text = ''.join([chr(token.item()) for token in samples[0]])
    return text

def main():
    parser = argparse.ArgumentParser(description="Generate demo outputs for nanoGPT2")
    parser.add_argument("--model-path", type=str, default="checkpoints/model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="demo_outputs",
                        help="Output directory for demo files")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of samples to generate")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens per sample")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get("config", get_config())
    
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    
    print(f"Loaded model from {args.model_path}")
    print(f"Generating {args.num_samples} samples...")
    
    # Generate samples
    for i in range(args.num_samples):
        text = generate_demo_text(model, prompt="", max_tokens=args.max_tokens, device=device)
        
        # Save to file
        output_file = output_dir / f"sample_{i+1}.txt"
        with open(output_file, "w") as f:
            f.write(text)
        print(f"Saved sample {i+1} to {output_file}")
    
    print(f"Demo generation complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
