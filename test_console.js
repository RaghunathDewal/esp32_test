// Run: node test_console.js (offline playback/URL checks, no browser needed).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const html = fs.readFileSync(`${__dirname}/static/console.html`, 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const sockets = [];
const contexts = [];
const document = {
  createElement: () => ({}),
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, {
      style: {}, classList: { remove() {} }, appendChild() {},
      addEventListener(event, handler) { this[event] = handler; },
    });
    return elements.get(id);
  },
};
class AudioContext {
  constructor() {
    this.currentTime = 0;
    this.buffers = [];
    contexts.push(this);
  }
  createBuffer(channels, length, rate) {
    const buffer = { channels, length, rate, duration: length / rate, copyToChannel() {} };
    this.buffers.push(buffer);
    return buffer;
  }
  createBufferSource() { return { connect() {}, start() {}, stop() {} }; }
  close() { this.closed = true; }
}
const context = vm.createContext({
  document, URL,
  window: { location: { protocol: 'http:', host: 'localhost:8000' }, AudioContext },
  WebSocket: class {
    constructor(url) { this.url = url; sockets.push(this); }
  },
});
vm.runInContext(script, context);
assert.equal(elements.get('urlField').value, 'ws://localhost:8000/ws?output_rate=16000');

for (const [query, rate] of [
  ['?output_rate=16000', 8000],
  ['?output_rate=24000', 24000],
  ['', 24000],
  ['?output_rate=8000', 24000],
  ['?output_rate=invalid', 24000],
  ['?output_rate=0x3e80', 24000],
  ['?output_rate=16000', 8000], // Reconnect after using a different rate.
]) {
  const url = 'ws://localhost:8000/ws' + query;
  elements.get('urlField').value = url;
  elements.get('connectBtn').click();
  const socket = sockets.at(-1);
  assert.equal(socket.url, url, 'The URL parameter must not be rewritten');
  socket.onopen();
  assert.equal(elements.get('rxLabel').textContent, `${rate / 1000} kHz`);
  socket.onmessage({ data: new Int16Array(rate).buffer });
  const playback = contexts.at(-1);
  assert.equal(playback.buffers[0].rate, rate);
  assert.equal(playback.buffers[0].duration, 1);
  socket.onclose({ code: 1000 });
  assert.equal(playback.closed, true);
}
console.log('OK: console preserves URLs, matches backend rates, and schedules one second of PCM correctly');
