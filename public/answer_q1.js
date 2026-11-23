// --- Canvas drawing (for Wacom / mouse) ---
const canvas = document.getElementById("answerCanvas");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
}
window.addEventListener("load", resizeCanvas);
window.addEventListener("resize", resizeCanvas);

let drawing = false;

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top,
  };
}

canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  drawing = true;
  const { x, y } = getPos(e);
  ctx.beginPath();
  ctx.moveTo(x, y);
});

canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const { x, y } = getPos(e);
  ctx.lineTo(x, y);
  ctx.stroke();
});

canvas.addEventListener("pointerup", (e) => {
  drawing = false;
  canvas.releasePointerCapture(e.pointerId);
});

// --- Buttons that talk to FastAPI backend ---
const userInput = document.getElementById("userId");
const qInput = document.getElementById("questionId");
const startBtn = document.getElementById("startBtn");
const endBtn = document.getElementById("endBtn");
const statusSpan = document.getElementById("status");
const logDiv = document.getElementById("log");

function log(msg) {
  const now = new Date().toLocaleTimeString();
  logDiv.textContent += `[${now}] ${msg}\n`;
}

startBtn.onclick = async () => {
  const body = {
    userNumber: userInput.value,
    qNumber: qInput.value,
  };
  const res = await fetch("http://127.0.0.1:8000/api/init", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  statusSpan.textContent = " logging...";
  log(`Start: status ${res.status}, response: ${text}`);
};

endBtn.onclick = async () => {
  const res = await fetch("http://127.0.0.1:8000/api/end", {
    method: "POST",
  });
  const text = await res.text();
  statusSpan.textContent = " stopped.";
  log(`End: status ${res.status}, response: ${text}`);
};