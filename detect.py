import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

import config
from src import FasterRCNN


# colors for each class
COLORS = {
    'Person': (255, 0, 0),
    'Car': (0, 255, 0),
    'Truck': (0, 0, 255),
    'Bicycle': (255, 255, 0),
    'Traffic light': (255, 0, 255)
}


def load_model(checkpoint):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    model = FasterRCNN(config.NUM_CLASSES, config.BACKBONE)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def preprocess(img_path, device):
    img = Image.open(img_path).convert('RGB')
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return tensor.to(device), img


def draw_boxes(img, boxes, labels, scores, thresh=0.5):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for box, label, score in zip(boxes, labels, scores):
        if score < thresh:
            continue

        x1, y1, x2, y2 = map(int, box)
        cls_name = config.CLASSES[label]
        color = COLORS.get(cls_name, (0, 255, 0))
        color = (color[2], color[1], color[0])  # RGB to BGR

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{cls_name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def detect_image(model, device, img_path, output_path=None, thresh=0.5):
    tensor, img = preprocess(img_path, device)

    with torch.no_grad():
        outputs = model([tensor])[0]

    boxes = outputs['boxes'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()

    # filter by threshold
    keep = scores >= thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    print(f"\nDetected {len(boxes)} objects:")
    for box, label, score in zip(boxes, labels, scores):
        print(f"  {config.CLASSES[label]}: {score:.3f}")

    result = draw_boxes(img, boxes, labels, scores, thresh)

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"\nSaved: {output_path}")
    else:
        cv2.imshow("Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_video(model, device, source, output_path=None, thresh=0.5):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)

    if output_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # preprocess
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model([tensor])[0]

        boxes = outputs['boxes'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()

        keep = scores >= thresh
        result = draw_boxes(img, boxes[keep], labels[keep], scores[keep], thresh)

        cv2.imshow("Detection", result)
        if output_path:
            writer.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='output/best.pth')
    parser.add_argument('--image', help='Path to image')
    parser.add_argument('--video', help='Video source (0 for webcam or file path)')
    parser.add_argument('--output', help='Output path')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    model, device = load_model(args.checkpoint)

    if args.image:
        detect_image(model, device, args.image, args.output, args.threshold)
    elif args.video:
        detect_video(model, device, args.video, args.output, args.threshold)
    else:
        print("Provide --image or --video")


if __name__ == '__main__':
    main()
