import { AutoProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoTokenizer, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

let imageProcessor = null;
let visionModel = null;
let textModel = null;
let tokenizer = null;
let lastImageEmbedding = null;

// Cache encoded text vectors to avoid re-encoding
const textCache = new Map();

const MODEL_ID = 'Xenova/clip-vit-base-patch16';

// CLIP's learned temperature: logit_scale = ln(100) ≈ 4.605
// This stretches the ~0.15-0.35 cosine range into ~0.0-1.0 after softmax
const LOGIT_SCALE = 100.0;

async function encodeText(text) {
  if (textCache.has(text)) return textCache.get(text);
  const inputs = tokenizer([text], { padding: true, truncation: true });
  const output = await textModel(inputs);
  const vec = Array.from(output.text_embeds.data);
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  const normalized = vec.map(v => v / norm);
  textCache.set(text, normalized);
  // Keep cache reasonable
  if (textCache.size > 50) {
    const firstKey = textCache.keys().next().value;
    textCache.delete(firstKey);
  }
  return normalized;
}

function cosine(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === 'load') {
    try {
      self.postMessage({ type: 'progress', data: { stage: 'Loading image processor...', pct: 5 } });
      imageProcessor = await AutoProcessor.from_pretrained(MODEL_ID);

      self.postMessage({ type: 'progress', data: { stage: 'Loading vision model...', pct: 25 } });
      visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, { quantized: true });

      self.postMessage({ type: 'progress', data: { stage: 'Loading text encoder...', pct: 55 } });
      tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
      textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID, { quantized: true });

      self.postMessage({ type: 'progress', data: { stage: 'Ready', pct: 100 } });
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'error', data: err.message + '\n' + err.stack });
    }
  }

  if (type === 'infer') {
    try {
      const { imageData, width, height } = data;
      const image = new RawImage(new Uint8ClampedArray(imageData), width, height, 4);
      const processed = await imageProcessor(image);
      const output = await visionModel(processed);

      const imgEmb = Array.from(output.image_embeds.data);
      const norm = Math.sqrt(imgEmb.reduce((s, v) => s + v * v, 0));
      const normalized = imgEmb.map(v => v / norm);
      lastImageEmbedding = normalized;

      self.postMessage({ type: 'result', data: { embedding: normalized } });
    } catch (err) {
      self.postMessage({ type: 'error', data: err.message + '\n' + err.stack });
    }
  }

  // Encode text and score against current image
  if (type === 'encode_text') {
    try {
      const { text, id } = data;
      const textVec = await encodeText(text);
      let rawScore = 0;
      if (lastImageEmbedding) {
        rawScore = cosine(lastImageEmbedding, textVec);
      }
      // Apply CLIP logit scale
      const scaledScore = rawScore * LOGIT_SCALE;
      self.postMessage({
        type: 'text_result',
        data: { text, rawScore, scaledScore, id }
      });
    } catch (err) {
      self.postMessage({ type: 'error', data: err.message + '\n' + err.stack });
    }
  }

  // Score multiple texts at once (for batch re-scoring on new frame)
  if (type === 'score_batch') {
    try {
      const { texts } = data;
      if (!lastImageEmbedding) return;
      const results = [];
      for (const text of texts) {
        const textVec = await encodeText(text); // uses cache
        const rawScore = cosine(lastImageEmbedding, textVec);
        results.push({ text, rawScore, scaledScore: rawScore * LOGIT_SCALE });
      }
      self.postMessage({ type: 'batch_result', data: { results } });
    } catch (err) {
      self.postMessage({ type: 'error', data: err.message + '\n' + err.stack });
    }
  }
};
