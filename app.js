const MODEL_PATH = 'model/model.json';
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const undoBtn = document.getElementById('undoBtn');
const saveBtn = document.getElementById('saveBtn');
const modelStatus = document.getElementById('modelStatus');
const topDigit = document.getElementById('topDigit');
const barsContainer = document.getElementById('bars');
const confidenceText = document.getElementById('confidenceText');

let model = null;
let drawing = false;
let history = [];
let lastPoint = { x: 0, y: 0 };

const CANVAS_SIZE = 280;
const LINE_WIDTH = 22;
const THRESHOLD = 15;

function clearCanvas() {
  ctx.fillStyle = '#08111f';
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
}

function pushHistory() {
  history.push(canvas.toDataURL());
  if (history.length > 20) history.shift();
}

function restoreHistory() {
  if (!history.length) {
    clearCanvas();
    updateUI([]);
    return;
  }
  const last = history.pop();
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.drawImage(img, 0, 0);
  };
  img.src = last;
}

function setupCanvas() {
  clearCanvas();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'white';
  ctx.lineWidth = LINE_WIDTH;
}

function getCanvasCoords(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  };
}

canvas.addEventListener('pointerdown', (event) => {
  drawing = true;
  const { x, y } = getCanvasCoords(event);
  lastPoint = { x, y };
  ctx.beginPath();
  ctx.moveTo(x, y);
});

canvas.addEventListener('pointermove', (event) => {
  if (!drawing) return;
  const { x, y } = getCanvasCoords(event);
  ctx.lineTo(x, y);
  ctx.stroke();
  lastPoint = { x, y };
});

canvas.addEventListener('pointerup', async () => {
  if (!drawing) return;
  drawing = false;
  pushHistory();
  await runPrediction();
});

canvas.addEventListener('pointercancel', () => {
  drawing = false;
});

clearBtn.addEventListener('click', () => {
  clearCanvas();
  history = [];
  updateUI([]);
});

undoBtn.addEventListener('click', restoreHistory);

saveBtn.addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = 'digit.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
});

function buildBars() {
  barsContainer.innerHTML = '';
  for (let i = 0; i < 10; i += 1) {
    const row = document.createElement('div');
    row.className = 'flex items-center gap-3';

    const label = document.createElement('div');
    label.className = 'w-6 text-slate-300';
    label.textContent = i;

    const wrapper = document.createElement('div');
    wrapper.className = 'bar flex-1';

    const fill = document.createElement('div');
    fill.className = 'bar-inner';
    fill.dataset.index = i;
    wrapper.appendChild(fill);

    const percent = document.createElement('div');
    percent.className = 'w-14 text-slate-400 text-xs text-right';
    percent.textContent = '0%';

    row.appendChild(label);
    row.appendChild(wrapper);
    row.appendChild(percent);
    barsContainer.appendChild(row);
  }
}

function updateUI(probabilities) {
  if (!probabilities || probabilities.length !== 10) {
    topDigit.textContent = '—';
    confidenceText.textContent = 'Confidence: —';
    document.querySelectorAll('.bar-inner').forEach((bar) => { bar.style.width = '0%'; });
    document.querySelectorAll('#bars .text-slate-400').forEach((el) => { el.textContent = '0%'; });
    return;
  }
  const maxValue = Math.max(...probabilities);
  const predicted = probabilities.indexOf(maxValue);
  topDigit.textContent = String(predicted);
  confidenceText.textContent = `Confidence: ${(maxValue * 100).toFixed(1)}%`;

  document.querySelectorAll('.bar-inner').forEach((bar) => {
    const idx = Number(bar.dataset.index);
    const width = Math.round(probabilities[idx] * 100);
    bar.style.width = `${width}%`;
    const percent = bar.parentElement.nextElementSibling;
    if (percent) percent.textContent = `${width}%`;
  });
}

function preprocessCanvas() {
  const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const pixels = imageData.data;
  const gray = new Uint8ClampedArray(CANVAS_SIZE * CANVAS_SIZE);

  for (let i = 0, j = 0; i < pixels.length; i += 4, j += 1) {
    gray[j] = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
  }

  let minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = 0, maxY = 0;
  let found = false;
  for (let y = 0; y < CANVAS_SIZE; y += 1) {
    for (let x = 0; x < CANVAS_SIZE; x += 1) {
      const value = gray[y * CANVAS_SIZE + x];
      if (value > THRESHOLD) {
        found = true;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (!found) {
    return new Float32Array(28 * 28);
  }

  const padding = 8;
  minX = Math.max(0, minX - padding);
  minY = Math.max(0, minY - padding);
  maxX = Math.min(CANVAS_SIZE - 1, maxX + padding);
  maxY = Math.min(CANVAS_SIZE - 1, maxY + padding);

  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;

  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = cropWidth;
  cropCanvas.height = cropHeight;
  const cropCtx = cropCanvas.getContext('2d');
  cropCtx.drawImage(canvas, minX, minY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

  const scaledCanvas = document.createElement('canvas');
  scaledCanvas.width = 20;
  scaledCanvas.height = 20;
  const scaledCtx = scaledCanvas.getContext('2d');
  scaledCtx.imageSmoothingEnabled = true;
  scaledCtx.drawImage(cropCanvas, 0, 0, 20, 20);

  const scaledData = scaledCtx.getImageData(0, 0, 20, 20).data;
  let sum = 0;
  let centerX = 0;
  let centerY = 0;

  for (let y = 0; y < 20; y += 1) {
    for (let x = 0; x < 20; x += 1) {
      const idx = (y * 20 + x) * 4;
      const value = (scaledData[idx] + scaledData[idx + 1] + scaledData[idx + 2]) / 3;
      const weight = value;
      sum += weight;
      centerX += x * weight;
      centerY += y * weight;
    }
  }

  if (sum > 0) {
    centerX /= sum;
    centerY /= sum;
  } else {
    centerX = 9.5;
    centerY = 9.5;
  }

  const finalCanvas = document.createElement('canvas');
  finalCanvas.width = 28;
  finalCanvas.height = 28;
  const finalCtx = finalCanvas.getContext('2d');
  finalCtx.fillStyle = 'black';
  finalCtx.fillRect(0, 0, 28, 28);

  const offsetX = Math.round(14 - centerX);
  const offsetY = Math.round(14 - centerY);
  const drawX = Math.min(Math.max(offsetX, 0), 8);
  const drawY = Math.min(Math.max(offsetY, 0), 8);
  finalCtx.drawImage(scaledCanvas, drawX, drawY);

  const finalData = finalCtx.getImageData(0, 0, 28, 28).data;
  const normalized = new Float32Array(28 * 28);
  for (let i = 0, j = 0; i < finalData.length; i += 4, j += 1) {
    normalized[j] = finalData[i] / 255.0;
  }

  return normalized;
}

async function runPrediction() {
  if (!model) return;
  const input = preprocessCanvas();
  const tensor = tf.tensor(input, [1, 28, 28, 1]);
  const prediction = model.predict(tensor);
  const probabilities = await prediction.data();
  tensor.dispose();
  prediction.dispose();
  updateUI(Array.from(probabilities));
}

async function loadModel() {
  try {
    modelStatus.textContent = 'Loading model...';
    model = await tf.loadLayersModel(MODEL_PATH);
    await model.predict(tf.zeros([1, 28, 28, 1])).data();
    modelStatus.textContent = 'Model loaded';
  } catch (error) {
    console.error(error);
    modelStatus.textContent = 'Model loading failed';
  }
}

setupCanvas();
buildBars();
loadModel();
