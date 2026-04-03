//! Embedded chat UI served from the root path.

use axum::response::Html;

/// Returns the embedded chat UI HTML page.
pub async fn chat_ui_handler() -> Html<&'static str> {
    Html(CHAT_UI_HTML)
}

const CHAT_UI_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Candle-vLLM Chat</title>
<style>
  :root { --bg: #0d1117; --surface: #161b22; --border: #30363d; --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff; --user-bg: #1f2937; --assist-bg: #161b22; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }
  header { padding: 12px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; background: var(--surface); }
  header h1 { font-size: 16px; font-weight: 600; }
  header .model { font-size: 12px; color: var(--muted); background: var(--bg); padding: 4px 8px; border-radius: 4px; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .msg { max-width: 85%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; font-size: 14px; }
  .msg.user { align-self: flex-end; background: var(--user-bg); border: 1px solid var(--border); }
  .msg.assistant { align-self: flex-start; background: var(--assist-bg); border: 1px solid var(--border); }
  .msg .role { font-size: 11px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  .msg.thinking { opacity: 0.6; font-style: italic; }
  #input-area { padding: 16px 20px; border-top: 1px solid var(--border); background: var(--surface); display: flex; gap: 10px; }
  #input { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; color: var(--text); font-size: 14px; outline: none; resize: none; min-height: 44px; max-height: 120px; font-family: inherit; }
  #input:focus { border-color: var(--accent); }
  #send { background: var(--accent); color: #fff; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; cursor: pointer; font-weight: 500; }
  #send:hover { opacity: 0.9; }
  #send:disabled { opacity: 0.4; cursor: not-allowed; }
  .stats { font-size: 11px; color: var(--muted); margin-top: 6px; }
</style>
</head>
<body>
<header>
  <h1>Candle-vLLM</h1>
  <span class="model" id="model-name">loading...</span>
</header>
<div id="chat"></div>
<div id="input-area">
  <textarea id="input" rows="1" placeholder="Type a message..." autofocus></textarea>
  <button id="send">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const modelEl = document.getElementById('model-name');
let modelName = '';

// Auto-resize textarea
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
});

// Fetch model name
fetch('/v1/models').then(r => r.json()).then(d => {
  const def = d.data?.find(m => m.default) || d.data?.[0];
  modelName = def?.id || 'default';
  modelEl.textContent = modelName;
});

function addMsg(role, content, extra) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const roleLabel = document.createElement('div');
  roleLabel.className = 'role';
  roleLabel.textContent = role;
  div.appendChild(roleLabel);
  const body = document.createElement('div');
  body.textContent = content;
  div.appendChild(body);
  if (extra) {
    const stats = document.createElement('div');
    stats.className = 'stats';
    stats.textContent = extra;
    div.appendChild(stats);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return body;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';
  sendBtn.disabled = true;

  addMsg('user', text);
  const bodyEl = addMsg('assistant', '');

  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelName,
        messages: [{ role: 'user', content: text }],
        max_tokens: 1024,
        stream: false
      })
    });
    const data = await res.json();
    if (data.choices?.[0]) {
      const msg = data.choices[0].message;
      bodyEl.textContent = msg.content || '';
      const u = data.usage;
      if (u) {
        const parent = bodyEl.parentElement;
        const stats = document.createElement('div');
        stats.className = 'stats';
        stats.textContent = `${u.prompt_tokens} prompt + ${u.completion_tokens} completion tokens | ${u.completion_time_costs}ms`;
        parent.appendChild(stats);
      }
    } else {
      bodyEl.textContent = data.message || 'Error: no response';
    }
  } catch (e) {
    bodyEl.textContent = 'Error: ' + e.message;
  }
  sendBtn.disabled = false;
  input.focus();
  chat.scrollTop = chat.scrollHeight;
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
</script>
</body>
</html>
"##;
