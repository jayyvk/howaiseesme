# How AI Sees You

A real-time visualization of how AI vision models perceive you. CLIP ViT-B/16 runs entirely in your browser - no server, no data leaves your device.

## What it does

Your webcam feed is processed by OpenAI's CLIP model every 1.5 seconds, producing a 512-dimensional embedding vector - the actual numbers that represent "you" to the AI. The fullscreen display shows these numbers overlaid on your silhouette, with your body highlighted as a colored heatmap.

Type any description in the search bar - "a happy person", "someone wearing glasses", "a cat" - and see how closely CLIP thinks you match, updating live. Add multiple descriptions to compare them as relative percentages.

## Tech

- **Next.js** + React
- **Transformers.js 3.8** (HuggingFace) — CLIP inference via ONNX Runtime Web
- **Web Worker** — ML runs off main thread
- **MediaPipe Selfie Segmentation** — person detection for silhouette heatmap (falls back to brightness detection)


## Architecture

```
Browser (100% client-side)
├── Main Thread (React)
│   ├── Webcam capture + preview
│   ├── Silhouette detection (MediaPipe / brightness fallback)
│   ├── 512-number fullscreen grid with heatmap overlay
│   └── Interactive text similarity search bar
└── Web Worker
    ├── CLIP ViT-B/16 Image Encoder (~40MB quantized)
    ├── CLIP Text Encoder (on-demand, cached)
    └── Cosine similarity + softmax scoring
```
