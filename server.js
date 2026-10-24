const Fastify = require('fastify');
const cors = require('@fastify/cors');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const cheerio = require('cheerio');
const crypto = require('crypto');

const WEIGHTS_FILE = path.join(__dirname, 'weights.json');
const NAMES_FILE = path.join(__dirname, 'user-names.json');
const HISTORY_FILE = path.join(__dirname, 'chat-history.json');

function loadHistory() {
  if (fs.existsSync(HISTORY_FILE)) {
    return JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf-8'));
  }
  return {};
}

function appendHistory(sessionId, userMessage, botReply) {
  const history = loadHistory();
  if (!history[sessionId]) history[sessionId] = [];
  history[sessionId].push({ r: 'user', t: userMessage });
  history[sessionId].push({ r: 'bot', t: botReply });
  fs.writeFileSync(HISTORY_FILE, JSON.stringify(history));
}


function loadNames() {
  if (fs.existsSync(NAMES_FILE)) {
    return JSON.parse(fs.readFileSync(NAMES_FILE, 'utf-8'));
  }
  return {};
}

function saveName(userId, name) {
  const names = loadNames();
  names[userId] = name;
  fs.writeFileSync(NAMES_FILE, JSON.stringify(names));
}

function getName(userId) {
  const names = loadNames();
  return names[userId] || null;
}

// ambil nama dari kalimat
function extractName(text) {
  const lower = text.toLowerCase();
  const patterns = [
    /nama saya ([a-z]+(?: [a-z]+)?)/,
    /namaku ([a-z]+(?: [a-z]+)?)/,
    /aku dipanggil ([a-z]+(?: [a-z]+)?)/,
    /panggil aku ([a-z]+(?: [a-z]+)?)/,
  ];
  const stopWords = ['saya', 'adalah', 'dan', 'yang', 'itu', 'ini'];

  for (const p of patterns) {
    const match = lower.match(p);
    if (match) {
      const words = match[1].split(' ').filter(w => !stopWords.includes(w));
      return words.slice(0, 2).join(' ');
    }
  }
  return null;
}


function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
function sigmoidDerivative(x) {
  return x * (1 - x);
}

class MLP {
  constructor(inputSize, hiddenSize, outputSize) {
    this.lr = 0.3;
    this.w1 = Array.from({ length: inputSize }, () =>
      Array.from({ length: hiddenSize }, () => Math.random() * 2 - 1)
    );
    this.b1 = Array(hiddenSize).fill(0).map(() => Math.random() * 2 - 1);
    this.w2 = Array.from({ length: hiddenSize }, () =>
      Array.from({ length: outputSize }, () => Math.random() * 2 - 1)
    );
    this.b2 = Array(outputSize).fill(0).map(() => Math.random() * 2 - 1);
  }

  forward(input) {
    this.hidden = this.w1[0].map((_, j) => {
      const sum = input.reduce((acc, val, i) => acc + val * this.w1[i][j], this.b1[j]);
      return sigmoid(sum);
    });
    this.output = this.w2[0].map((_, j) => {
      const sum = this.hidden.reduce((acc, val, i) => acc + val * this.w2[i][j], this.b2[j]);
      return sigmoid(sum);
    });
    return this.output;
  }

  train(input, target) {
    const output = this.forward(input);
    const outputError = output.map((o, i) => target[i] - o);
    const outputDelta = outputError.map((e, i) => e * sigmoidDerivative(output[i]));

    const hiddenError = this.hidden.map((_, i) =>
      outputDelta.reduce((acc, d, j) => acc + d * this.w2[i][j], 0)
    );
    const hiddenDelta = hiddenError.map((e, i) => e * sigmoidDerivative(this.hidden[i]));

    for (let i = 0; i < this.w2.length; i++) {
      for (let j = 0; j < this.w2[i].length; j++) {
        this.w2[i][j] += this.lr * outputDelta[j] * this.hidden[i];
      }
    }
    this.b2 = this.b2.map((b, j) => b + this.lr * outputDelta[j]);

    for (let i = 0; i < this.w1.length; i++) {
      for (let j = 0; j < this.w1[i].length; j++) {
        this.w1[i][j] += this.lr * hiddenDelta[j] * input[i];
      }
    }
    this.b1 = this.b1.map((b, j) => b + this.lr * hiddenDelta[j]);
  }

  toJSON() {
    return { w1: this.w1, b1: this.b1, w2: this.w2, b2: this.b2 };
  }

  loadJSON(data) {
    this.w1 = data.w1;
    this.b1 = data.b1;
    this.w2 = data.w2;
    this.b2 = data.b2;
  }
}

// list vocab
const vocab = [
  "halo", "hai", "hello", "hi", "hey", "pagi", "siang", "sore", "malam", "selamat",
  "apa", "kabar", "kamu", "siapa", "nama", "namamu", "kenalan",
  "makasih", "terima", "kasih", "thanks", "thank", "you",
  "bye", "dadah", "sampai", "jumpa", "pergi", "pamit",
  "cuaca", "hari", "ini", "makan", "minum", "tidur",
  "jam", "waktu", "sekarang", "berapa",
  "tolong", "bantu", "bantuan", "bisa", "gimana", "caranya",
  "keren", "mantap", "bagus", "hebat", "pintar",
  "cari", "cariin", "carikan", "itu", "info",
  "namaku", "aku", "panggil",
  "cuaca", "hujan", "panas", "dingin", "suhu", "kota", "di",
];

function textToVector(text) {
  const words = text.toLowerCase().split(" ");
  return vocab.map(v => (words.includes(v) ? 1 : 0));
}

// urutan index
const intents = [
  "salam", "tanya_nama", "terima_kasih", "pamitan",
  "tanya_waktu", "minta_bantuan", "tanya_kabar", "pujian",
  "cari_info", "kenalan", "tanya_namaku", "tanya_cuaca", "tidak_dikenali",
];

const intentReplies = {
  salam: "Halo! Saya Pythagon, model 0.1.",
  tanya_nama: "Nama saya Pythagon.",
  terima_kasih: "Sama-sama!",
  pamitan: "Sampai jumpa lagi!",
  tanya_waktu: "Maaf, saya belum bisa cek waktu sekarang, fitur ini masih dalam pengembangan.",
  minta_bantuan: "Tentu, coba jelaskan apa yang kamu butuhkan.",
  tanya_kabar: "Saya baik, terima kasih sudah bertanya!",
  pujian: "Terima kasih atas pujiannya!",
  cari_info: null,
  kenalan: null,
  tanya_namaku: null,
  tanya_cuaca: null,
  tidak_dikenali: "Maaf, saya belum paham maksud kamu.",
};

function intentToTarget(intentName) {
  return intents.map(i => (i === intentName ? 1 : 0));
}

function outputToIntent(output) {
  let maxIdx = 0;
  for (let i = 1; i < output.length; i++) {
    if (output[i] > output[maxIdx]) maxIdx = i;
  }
  return { intent: intents[maxIdx], confidence: output[maxIdx] };
}

// wallahua'lam saya serahkan kepada claude saja 
const mlp = new MLP(vocab.length, 10, intents.length);

// Data awal supaya model langsung punya dasar tiap intent
const seedData = [
  { text: "halo", intent: "salam" },
  { text: "hai", intent: "salam" },
  { text: "hello", intent: "salam" },
  { text: "hi", intent: "salam" },
  { text: "selamat pagi", intent: "salam" },
  { text: "selamat siang", intent: "salam" },
  { text: "selamat malam", intent: "salam" },
  { text: "siapa kamu", intent: "tanya_nama" },
  { text: "siapa nama kamu", intent: "tanya_nama" },
  { text: "namamu siapa", intent: "tanya_nama" },
  { text: "makasih", intent: "terima_kasih" },
  { text: "terima kasih", intent: "terima_kasih" },
  { text: "thanks", intent: "terima_kasih" },
  { text: "thank you", intent: "terima_kasih" },
  { text: "bye", intent: "pamitan" },
  { text: "dadah", intent: "pamitan" },
  { text: "sampai jumpa", intent: "pamitan" },
  { text: "piye kabare", intent: "tanya_kabar" },
  { text: "kabar kamu gimana", intent: "tanya_kabar" },
  { text: "gimana kabar kamu", intent: "tanya_kabar" },
  { text: "jam berapa sekarang", intent: "tanya_waktu" },
  { text: "sekarang jam berapa", intent: "tanya_waktu" },
  { text: "jam berapa ini", intent: "tanya_waktu" },
  { text: "tolong bantu aku", intent: "minta_bantuan" },
  { text: "bisa bantu tidak", intent: "minta_bantuan" },
  { text: "gimana caranya", intent: "minta_bantuan" },
  { text: "kamu keren", intent: "pujian" },
  { text: "kamu pintar", intent: "pujian" },
  { text: "mantap kamu", intent: "pujian" },
  { text: "kamu hebat", intent: "pujian" },
  { text: "cuaca hari ini", intent: "tanya_cuaca" },
  { text: "makan apa", intent: "tidak_dikenali" },
  { text: "cari apa itu fastify", intent: "cari_info" },
  { text: "cariin informasi tentang node js", intent: "cari_info" },
  { text: "carikan info cuaca jakarta", intent: "cari_info" },
  { text: "apa itu neural network", intent: "cari_info" },
  { text: "nama saya budi", intent: "kenalan" },
  { text: "namaku andi", intent: "kenalan" },
  { text: "aku dipanggil rina", intent: "kenalan" },
  { text: "panggil aku joko", intent: "kenalan" },
  { text: "siapa namaku", intent: "tanya_namaku" },
  { text: "namaku siapa", intent: "tanya_namaku" },
  { text: "kamu ingat nama aku tidak", intent: "tanya_namaku" },
  { text: "cuaca di jakarta", intent: "tanya_cuaca" },
  { text: "cuaca hari ini di bandung", intent: "tanya_cuaca" },
  { text: "apakah hujan di surabaya", intent: "tanya_cuaca" },
  { text: "suhu di jakarta berapa", intent: "tanya_cuaca" },
];

if (fs.existsSync(WEIGHTS_FILE)) {
  const saved = JSON.parse(fs.readFileSync(WEIGHTS_FILE, 'utf-8'));
  mlp.loadJSON(saved);
  console.log('Bobot dimuat dari weights.json');
} else {
  console.log('Model baru, training dengan seed data...');
  for (let epoch = 0; epoch < 3000; epoch++) {
    seedData.forEach(({ text, intent }) => {
      mlp.train(textToVector(text), intentToTarget(intent));
    });
  }
  fs.writeFileSync(WEIGHTS_FILE, JSON.stringify(mlp.toJSON()));
  console.log('Seed training selesai, weights.json dibuat');
}

function saveWeights() {
  fs.writeFileSync(WEIGHTS_FILE, JSON.stringify(mlp.toJSON()));
}

// ---- Browsing sederhana via DuckDuckGo HTML (tanpa API key) ----
async function searchWeb(query) {
  try {
    const response = await axios.get('https://html.duckduckgo.com/html/', {
      params: { q: query },
      headers: { 'User-Agent': 'Mozilla/5.0' },
      timeout: 8000,
    });

    const $ = cheerio.load(response.data);
    const results = [];

    $('.result__body').each((i, el) => {
      if (i >= 3) return; // ambil 3 hasil teratas saja
      const title = $(el).find('.result__title').text().trim();
      const snippet = $(el).find('.result__snippet').text().trim();
      if (title || snippet) {
        results.push({ title, snippet });
      }
    });

    return results;
  } catch (err) {
    return null; // gagal browsing (timeout/error jaringan)
  }
}

// Ambil query dari kalimat, buang kata pemicu seperti "cari", "apa itu", dll
function extractQuery(text) {
  return text
    .toLowerCase()
    .replace(/\b(cari|cariin|carikan|info|tentang|apa itu)\b/g, '')
    .trim();
}


// Ambil nama kota dari kalimat, misal "cuaca di jakarta" -> "jakarta"
function extractCity(text) {
  const lower = text.toLowerCase();
  const match = lower.match(/di ([a-z]+)/);
  if (match) return match[1];
  return 'jakarta'; // default kalau kota tidak disebut
}

// ---- Kalkulator matematika sederhana & aman ----
// Hanya izinkan angka, operator +-*/, tanda kurung, titik desimal, dan spasi
const MATH_PATTERN = /^[0-9+\-*/().\s]+$/;

function extractMathExpression(text) {
  // Cari kemungkinan ekspresi matematika di dalam kalimat
  const match = text.match(/[0-9+\-*/().\s]{3,}/);
  if (!match) return null;
  const expr = match[0].trim();
  // Wajib ada minimal 1 operator, biar tidak nangkep angka polos tanpa operasi
  if (!/[+\-*/]/.test(expr)) return null;
  return expr;
}

function safeCalculate(expr) {
  if (!MATH_PATTERN.test(expr)) return null;
  try {
    const result = Function(`"use strict"; return (${expr})`)();
    if (typeof result !== 'number' || !isFinite(result)) return null;
    return result;
  } catch (err) {
    return null;
  }
}

async function getWeather(city) {
  try {
    const response = await axios.get(`https://wttr.in/${encodeURIComponent(city)}`, {
      params: { format: '%l:+%C,+%t+(terasa+%f),+kelembapan+%h,+angin+%w' },
      timeout: 8000,
    });
    return response.data.trim();
  } catch (err) {
    return null;
  }
}

const fastify = Fastify({ logger: true });

const { TRAIN_API_KEY } = require('./config');

function checkApiKey(request, reply, done) {
  const key = request.headers['x-api-key'];
  if (key !== TRAIN_API_KEY) {
    reply.code(401).send({ error: 'Unauthorized: x-api-key tidak valid' });
    return;
  }
  done();
}

fastify.get('/status', async () => {
  return { status: 'ok', vocabSize: vocab.length, intents };
});

fastify.post('/session', async () => {
  const sessionId = crypto.randomUUID();
  return { sessionId };
});

fastify.get('/history/:sessionId', async (request) => {
  const history = loadHistory();
  return history[request.params.sessionId] || [];
});

fastify.addHook('onSend', async (request, reply, payload) => {
  if (request.method === 'POST' && request.url === '/chat') {
    try {
      const body = JSON.parse(payload);
      const sessionId = request.body.userId;
      if (sessionId && body.reply) {
        appendHistory(sessionId, request.body.message, body.reply);
      }
    } catch (err) {
      // abaikan kalau payload bukan JSON valid
    }
  }
  return payload;
});
//tolong fix bagian ini 

fastify.post('/chat', async (request, reply) => {
  const { message, userId } = request.body;
  if (!message) {
    return reply.code(400).send({ error: 'message wajib diisi' });
  }

  // Cek dulu apakah ini ekspresi matematika, sebelum masuk classifier intent
  const mathExpr = extractMathExpression(message);
  if (mathExpr) {
    const result = safeCalculate(mathExpr);
    if (result !== null) {
      return { message, intent: 'hitung_matematika', expression: mathExpr, reply: `${mathExpr.trim()} = ${result}` };
    }
  }

  const vector = textToVector(message);
  const output = mlp.forward(vector);
  const { intent, confidence } = outputToIntent(output);

  if (intent === 'kenalan') {
    const name = extractName(message);
    if (!name || !userId) {
      return { message, intent, confidence, reply: 'Maaf, saya belum bisa mengenali nama kamu dari kalimat itu.' };
    }
    saveName(userId, name);
    return { message, intent, confidence, reply: `Senang berkenalan, ${name}! Saya akan mengingat namamu.` };
  }

  if (intent === 'tanya_namaku') {
    const name = userId ? getName(userId) : null;
    const reply_ = name ? `Namamu ${name}, benar?` : 'Maaf, saya belum tahu namamu. Coba kenalan dulu, misal: "nama saya Budi".';
    return { message, intent, confidence, reply: reply_ };
  }

  if (intent === 'tanya_cuaca') {
    const city = extractCity(message);
    const weather = await getWeather(city);
    if (!weather) {
      return { message, intent, confidence, reply: 'Maaf, saya gagal mengambil data cuaca saat ini.' };
    }
    return { message, intent, confidence, city, reply: weather };
  }

  if (intent === 'cari_info') {
    const query = extractQuery(message);
    const results = await searchWeb(query);

    if (!results) {
      return { message, intent, confidence, reply: 'Maaf, saya gagal mengambil informasi dari internet saat ini.' };
    }
    if (results.length === 0) {
      return { message, intent, confidence, reply: `Tidak ditemukan hasil untuk "${query}".` };
    }

    const summary = results.map((r, i) => `${i + 1}. ${r.title}: ${r.snippet}`).join('\n');
    return { message, intent, confidence, query, reply: summary };
  }

  return {
    message,
    intent,
    confidence,
    reply: intentReplies[intent],
  };
});

fastify.post('/train', { preHandler: checkApiKey }, async (request, reply) => {
  const { text, intent } = request.body;
  if (!text || !intents.includes(intent)) {
    return reply.code(400).send({ error: `text wajib diisi, intent harus salah satu dari: ${intents.join(', ')}` });
  }
  const vector = textToVector(text);
  mlp.train(vector, intentToTarget(intent));
  saveWeights();

  const output = mlp.forward(vector);
  const result = outputToIntent(output);
  return { text, intent, newResult: result, saved: true };
});

const start = async () => {
  try {
    await fastify.register(cors, { origin: true });
    await fastify.listen({ port: process.env.PORT || 3000, host: '0.0.0.0' });
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();
