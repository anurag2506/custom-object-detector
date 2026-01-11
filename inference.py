import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import argparse

import config
from src import FasterRCNN


COLORS = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
}


def load_model(weights_path, device):
    model = FasterRCNN(config.NUM_CLASSES, config.BACKBONE)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(img, img_size=416):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    orig_w, orig_h = img_pil.size
    img_resized = img_pil.resize((img_size, img_size), Image.BILINEAR)
    img_tensor = TF.to_tensor(img_resized)
    img_tensor = TF.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img_tensor, orig_w, orig_h


def postprocess_detections(outputs, orig_w, orig_h, img_size=416, conf_thresh=0.5):
    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    mask = scores >= conf_thresh
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]

    scale_x = orig_w / img_size
    scale_y = orig_h / img_size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    return boxes, labels, scores


def draw_detections(img, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        class_name = config.CLASSES[label]
        text = f"{class_name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(
            img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return img


@torch.no_grad()
def detect_image(model, image_path, device, conf_thresh=0.5, output_path=None):
    img = cv2.imread(image_path)
    img_tensor, orig_w, orig_h = preprocess_image(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    outputs = model(img_tensor)[0]
    boxes, labels, scores = postprocess_detections(
        outputs, orig_w, orig_h, conf_thresh=conf_thresh
    )

    result_img = draw_detections(img.copy(), boxes, labels, scores)

    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Saved: {output_path}")

    return result_img, boxes, labels, scores


@torch.no_grad()
def detect_video(model, video_path, device, conf_thresh=0.5, output_path=None):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    inference_times = []
    pbar = tqdm(total=total_frames, desc="Processing video")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor, orig_w, orig_h = preprocess_image(frame)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

        if device == "cuda":
            start_time.record()

        outputs = model(img_tensor)[0]

        if device == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            inference_times.append(start_time.elapsed_time(end_time))

        boxes, labels, scores = postprocess_detections(
            outputs, orig_w, orig_h, conf_thresh=conf_thresh
        )
        result_frame = draw_detections(frame, boxes, labels, scores)

        if output_path:
            out.write(result_frame)

        pbar.update(1)

    cap.release()
    if output_path:
        out.release()
        print(f"Saved: {output_path}")

    if inference_times:
        avg_time = np.mean(inference_times)
        print(f"Average inference time: {avg_time:.2f}ms ({1000 / avg_time:.1f} FPS)")

    return inference_times


@torch.no_grad()
def create_gif(
    model,
    image_folder,
    device,
    conf_thresh=0.5,
    output_path="detection.gif",
    max_frames=50,
):
    import glob
    from PIL import Image as PILImage

    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))[:max_frames]
    frames = []

    for img_path in tqdm(image_paths, desc="Creating GIF"):
        result_img, _, _, _ = detect_image(model, img_path, device, conf_thresh)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        frames.append(PILImage.fromarray(result_img_rgb))

    if frames:
        frames[0].save(
            output_path, save_all=True, append_images=frames[1:], duration=100, loop=0
        )
        print(f"Saved GIF: {output_path}")


def benchmark_speed(model, device, img_size=416, num_runs=100):
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    # Warmup
    for _ in range(10):
        _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in tqdm(range(num_runs), desc="Benchmarking"):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            import time

            start = time.time()
            _ = model(dummy_input)
            times.append((time.time() - start) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\nBenchmark Results ({device}):")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  FPS: {1000 / avg_time:.1f}")
    return avg_time, std_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection Inference")
    parser.add_argument(
        "--weights", type=str, default="output/best.pth", help="Model weights path"
    )
    parser.add_argument("--source", type=str, help="Image or video path")
    parser.add_argument(
        "--output", type=str, default="output/result", help="Output path"
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--benchmark", action="store_true", help="Run speed benchmark")
    parser.add_argument("--gif", type=str, help="Create GIF from image folder")
    args = parser.parse_args()

    device = config.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.weights, device)
    print("Model loaded successfully")

    if args.benchmark:
        benchmark_speed(model, device)
    elif args.gif:
        create_gif(model, args.gif, device, args.conf, args.output + ".gif")
    elif args.source:
        if args.source.endswith((".mp4", ".avi", ".mov")):
            detect_video(model, args.source, device, args.conf, args.output + ".mp4")
        else:
            detect_image(model, args.source, device, args.conf, args.output + ".jpg")
    else:
        print("Please provide --source, --benchmark, or --gif")
