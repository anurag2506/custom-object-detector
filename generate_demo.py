"""
Generate demo images and GIF for the report.
Run this after training to create visualization assets.
"""

import os
import torch
import glob
import random
from inference import load_model, detect_image, create_gif, benchmark_speed
import config


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    weights_path = "./model/best.pth"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found. Train the model first.")
        return

    model = load_model(weights_path, device)
    print("Model loaded successfully")

    os.makedirs("assets", exist_ok=True)

    # Generate sample detections from validation set
    val_images = glob.glob("data/street_objects/images/validation/*.jpg")
    if val_images:
        print("\nGenerating sample detection images...")
        random.seed(42)
        sample_images = random.sample(val_images, min(10, len(val_images)))

        for i, img_path in enumerate(sample_images):
            output_path = f"assets/detection_example_{i + 1}.jpg"
            detect_image(
                model, img_path, device, conf_thresh=0.3, output_path=output_path
            )

        # Create GIF from samples
        print("\nCreating demo GIF...")
        create_gif(
            model,
            "data/street_objects/images/validation",
            device,
            conf_thresh=0.3,
            output_path="assets/detection_demo.gif",
            max_frames=30,
        )
    else:
        print("No validation images found. Run prepare_data.py first.")

    # Run benchmark
    print("\nRunning speed benchmark...")
    avg_time, std_time = benchmark_speed(model, device, img_size=416, num_runs=100)

    # Print summary
    print("\n" + "=" * 50)
    print("DEMO GENERATION COMPLETE")
    print("=" * 50)
    print(f"\nAssets created in ./assets/:")
    for f in glob.glob("assets/*"):
        print(f"  - {f}")
    print(f"\nPerformance:")
    print(f"  - Inference: {avg_time:.1f}ms ({1000 / avg_time:.1f} FPS)")
    print(f"\nNext steps:")
    print("  1. Push to GitHub: git push origin main")
    print("  2. Update REPORT.md with your GitHub URL")
    print("  3. Add the generated assets to your repo")


if __name__ == "__main__":
    main()
