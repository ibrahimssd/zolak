// Zolak front-end (GitHub Pages) -> chatbot backend (Gradio app) via @gradio/client.
import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client@1.19.1/dist/index.min.js";

const CONFIG = {
  // Gradio backend: a Hugging Face Space id ("user/space") or a full https URL. Empty = not deployed yet.
  backend: "ibrahimssd11/zolak",
  repo: "https://github.com/ibrahimssd/zolak",
};
const SOURCES_SEP = "\n\n---\n";
const TOPICS = [
  { ico: "✈️", title: "الفيزا والسفارة", hint: "مواعيد، موافقة أمنية، أوراق", q: "الموافقة الأمنية للفيزا بتاخد قدر شنو؟", c: "var(--nile)" },
  { ico: "🏦", title: "الحساب المغلق", hint: "Sperrkonto، تحويل القروش", q: "الحساب المغلق Sperrkonto بفتحو كيف؟", c: "var(--sun)" },
  { ico: "🪪", title: "الإقامة", hint: "Ausländerbehörde، تجديد، Fiktion", q: "كيف أجدد الإقامة في Ausländerbehörde وشنو الأوراق المطلوبة؟", c: "var(--indigo)" },
  { ico: "🎓", title: "الجامعة والقبول", hint: "uni-assist، ماستر، معادلة", q: "كيف أقدم ماستر في ألمانيا عن طريق uni-assist؟", c: "var(--palm)" },
  { ico: "💼", title: "الشغل", hint: "Werkstudent، Minijob، Ausbildung", q: "ممكن اشتغل وانا طالب؟ كم ساعة مسموح؟", c: "var(--clay)" },
  { ico: "🏠", title: "السكن والتسجيل", hint: "WG، سكن طلاب، Anmeldung", q: "Anmeldung دي شنو وبعملها وين؟", c: "var(--blue-nile)" },
  { ico: "🛂", title: "الجواز السوداني", hint: "تجديد، السفارة السودانية", q: "جوازي السوداني قرب يخلص وانا في ألمانيا، اجدده وين؟", c: "var(--sun)" },
  { ico: "🩺", title: "التأمين الصحي", hint: "TK، AOK، تأمين طلاب", q: "التأمين الصحي للطلاب ياتو أحسن؟", c: "var(--nile)" },
];

// The Gradio client sends credentialed requests. Hugging Face's edge proxy answers the CORS preflight without
// Access-Control-Allow-Credentials, so browsers block them. The public Space needs no cookies: drop credentials
// for Hugging Face hosts only.
const nativeFetch = window.fetch.bind(window);
window.fetch = (input, init = {}) => {
  const url = typeof input === "string" ? input : input?.url || String(input);
  if (/\.hf\.space\/|huggingface\.co\//.test(url)) init = { ...init, credentials: "omit" };
  return nativeFetch(input, init);
};

const $ = (id) => document.getElementById(id);
const chat = $("chat"), form = $("ask"), q = $("q"), send = $("send"), statusEl = $("status"), notice = $("notice");
let client = null;
let history = [];   // [{role, content}] — last turns go to the backend
let busy = false;

$("src-link").href = CONFIG.repo;

// ---------- helpers
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
// Minimal, safe formatting: escape first, then lists / line breaks / https links.
function render(text) {
  const lines = escapeHtml(text.trim()).split("\n");
  let html = "", inList = false;
  for (const line of lines) {
    const m = line.match(/^\s*(?:[-•*]|\d+[.)])\s+(.*)$/);
    if (m) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${m[1]}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    html += line ? `${line}<br>` : "<br>";
  }
  if (inList) html += "</ul>";
  return html.replace(/(https:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener nofollow">$1</a>');
}
const setStatus = (t) => { statusEl.hidden = !t; statusEl.textContent = t || ""; };
const showNotice = (t) => { notice.hidden = !t; notice.textContent = t || ""; };
const scrollChat = () => { chat.scrollTop = chat.scrollHeight; };

function addMessage(role, text = "") {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (text) el.textContent = text;
  chat.appendChild(el);
  scrollChat();
  return el;
}
function fillBot(el, full, done) {
  const [body, sources] = full.split(SOURCES_SEP);
  el.innerHTML = render(body || "") + (sources ? `<div class="sources">${escapeHtml(sources)}</div>` : "");
  el.classList.toggle("cursor", !done);
}
function addFeedback(el, question, answer) {
  const bar = document.createElement("div");
  bar.className = "actions";
  for (const [rating, label] of [["up", "👍 مفيد"], ["down", "👎 ما مفيد"]]) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label;
    b.onclick = async () => {
      bar.querySelectorAll("button").forEach((x) => (x.disabled = true));
      b.classList.add("chosen");
      try { await client.predict("/feedback", [question, answer, rating]); } catch (e) { console.warn(e); }
    };
    bar.appendChild(b);
  }
  el.appendChild(bar);
}

// ---------- backend connection (free backends sleep when idle)
const NOT_READY = "🚧 زولك لسه في مرحلة التجهيز وقريب حيكون جاهز للأسئلة. ارجع لينا بعد شوية 🙏";
async function connect() {
  if (client) return client;
  if (!CONFIG.backend) throw new Error("not-deployed");
  const slow = setTimeout(() => setStatus("⏳ الخادم كان نايم وبيصحى هسي… ممكن ياخد دقيقة أو اتنين"), 4000);
  try {
    client = await Client.connect(CONFIG.backend, { events: ["data", "status"] });
    showNotice("");
    return client;
  } finally { clearTimeout(slow); setStatus(""); }
}

// ---------- ask
async function ask(question) {
  question = (question || "").trim();
  if (busy || !question) return;
  busy = true; send.disabled = true;
  addMessage("user", question);
  const botEl = addMessage("bot");
  botEl.classList.add("cursor");
  let full = "";
  try {
    const c = await connect();
    const job = c.submit("/ask", [question, history.slice(-4)]);
    for await (const msg of job) {
      if (msg.type === "status") {
        if (msg.stage === "pending" && typeof msg.position === "number" && msg.position > 0) {
          setStatus(`في ${msg.position} قبلك في الصف… لحظة`);
        } else if (msg.stage === "error") {
          throw new Error(msg.message || "error");
        } else { setStatus(""); }
      } else if (msg.type === "data") {
        full = String(msg.data?.[0] ?? "");
        fillBot(botEl, full, false);
        scrollChat();
      }
    }
    fillBot(botEl, full, true);
    const answer = full.split(SOURCES_SEP)[0];
    history.push({ role: "user", content: question }, { role: "assistant", content: answer });
    addFeedback(botEl, question, answer);
  } catch (err) {
    if (String(err?.message) !== "not-deployed") console.error(err);
    botEl.classList.remove("cursor");
    if (String(err?.message) === "not-deployed") {
      botEl.textContent = NOT_READY;
    } else {
      botEl.classList.add("error");
      botEl.textContent = "معليش، حصلت مشكلة في الاتصال. جرب تاني بعد شوية.";
    }
  } finally {
    setStatus(""); busy = false; send.disabled = false;
  }
}

// ---------- UI wiring
for (const t of TOPICS) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "topic"; b.style.setProperty("--c", t.c);
  b.innerHTML = `<span class="ico" aria-hidden="true">${t.ico}</span><strong>${escapeHtml(t.title)}</strong><small>${escapeHtml(t.hint)}</small>`;
  b.onclick = () => { document.querySelector(".chat-card").scrollIntoView({ block: "start" }); ask(t.q); };
  $("topics").appendChild(b);
}
form.addEventListener("submit", (e) => { e.preventDefault(); const v = q.value; q.value = ""; autosize(); ask(v); });
q.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
});
function autosize() { q.style.height = "auto"; q.style.height = Math.min(q.scrollHeight, 160) + "px"; }
q.addEventListener("input", autosize);
$("new-chat").onclick = () => {
  history = [];
  chat.querySelectorAll(".msg:not(.intro)").forEach((m) => m.remove());
};

if (!CONFIG.backend) showNotice(NOT_READY);
else connect().catch(() => showNotice(NOT_READY));   // start waking the backend early
