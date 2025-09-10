import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import fetch from "node-fetch";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// ---------------- CONFIG ----------------
const MODEL = "deepseek/deepseek-chat";   // DeepSeek V3.1 free model
const OPENROUTER_API_KEY = "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";
const OPENROUTER_BASE = "https://openrouter.ai/api/v1";

// ---------------- SIMPLE RAG over knowledge.json ----------------
const stopwords = new Set(["a","an","and","are","as","at","be","by","for","from","has","he","in","is","it","its","of","on","that","the","to","was","were","will","with","you","your","i","we","our","us"]);

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/).filter(Boolean).filter(w => !stopwords.has(w));
}
function buildTf(tokens) {
  const map = new Map();
  for (const t of tokens) map.set(t, (map.get(t) || 0) + 1);
  return Object.fromEntries([...map.entries()].map(([k,v]) => [k, v / tokens.length]));
}
function cosine(a,b) {
  let dot=0, na=0, nb=0;
  for (const [k,v] of Object.entries(a)) {
    if (k in b) dot += v * b[k];
    na += v*v;
  }
  for (const v of Object.values(b)) nb += v*v;
  return Math.sqrt(na) && Math.sqrt(nb) ? dot / (Math.sqrt(na)*Math.sqrt(nb)) : 0;
}

const kbPath = path.join(__dirname, "data", "knowledge.json");
let KB = JSON.parse(fs.readFileSync(kbPath, "utf-8"));

function buildIndex(kb) {
  const docs = kb.items.map((it,i) => {
    const text = `${it.q}\n${it.a}`;
    const tokens = tokenize(text);
    return { q: it.q, a: it.a, tokens, tf: buildTf(tokens) };
  });
  const df = new Map();
  for (const d of docs) {
    for (const t of new Set(d.tokens)) df.set(t, (df.get(t)||0)+1);
  }
  const N = docs.length;
  const idf = new Map();
  for (const [t,c] of df.entries()) idf.set(t, Math.log((N+1)/(c+1))+1);
  for (const d of docs) {
    const out = {};
    for (const [t,v] of Object.entries(d.tf)) out[t] = v * (idf.get(t)||0);
    d.tfidf = out;
  }
  return { docs, idf };
}
let INDEX = buildIndex(KB);

function topK(query, k=3) {
  const qTokens = tokenize(query);
  const qTf = buildTf(qTokens);
  const qVec = {};
  for (const [t,v] of Object.entries(qTf)) qVec[t] = v * (INDEX.idf.get(t)||0);
  return INDEX.docs
    .map(d => ({ d, sim: cosine(qVec, d.tfidf) }))
    .sort((a,b) => b.sim - a.sim)
    .slice(0,k);
}

// ---------------- CALL OPENROUTER ----------------
async function callDeepSeek(messages) {
  const resp = await fetch(`${OPENROUTER_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENROUTER_API_KEY}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "VS Chatbot DeepSeek"
    },
    body: JSON.stringify({ model: MODEL, messages, temperature: 0.3 })
  });
  if (!resp.ok) throw new Error(await resp.text());
  const data = await resp.json();
  return data?.choices?.[0]?.message?.content?.trim() || "No response.";
}

// ---------------- ROUTES ----------------
app.post("/api/chat", async (req,res) => {
  const { message } = req.body || {};
  if (!message) return res.json({ reply: "Say something 🙂" });

  const ctx = topK(message,3).filter(x => x.sim > (KB.threshold ?? 0.12));
  const contextText = ctx.map(x => `Q: ${x.d.q}\nA: ${x.d.a}`).join("\n\n");

  const system = `You are a helpful assistant. Prefer using CONTEXT if relevant.`;
  const user = `Question: ${message}\n\nCONTEXT:\n${contextText || "(none)"}`;

  try {
    const reply = await callDeepSeek([
      { role:"system", content: system },
      { role:"user", content: user }
    ]);
    res.json({ reply, context_used: ctx.map(x => ({q:x.d.q, score:x.sim.toFixed(3)})) });
  } catch (e) {
    console.error(e);
    res.status(500).json({ reply:"Error calling DeepSeek API", error:String(e) });
  }
});

app.post("/api/reload", (req,res) => {
  try {
    KB = JSON.parse(fs.readFileSync(kbPath,"utf-8"));
    INDEX = buildIndex(KB);
    res.json({ ok:true, count: INDEX.docs.length });
  } catch(e) {
    res.status(500).json({ ok:false, error:String(e) });
  }
});

const PORT = 3000;
app.listen(PORT, () => console.log(`🚀 DeepSeek Chatbot running at http://localhost:${PORT}`));
