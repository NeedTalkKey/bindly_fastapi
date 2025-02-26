import { config } from "./config.js";

// tokenizeClient.js
export async function getTokenChunks(text, max_length = 300, overlap = 50) {
    const tokenizeUrl = 'http://127.0.0.1:8000/tokenize';
    // const tokenizeUrl = `${config.hosting.host_ip}:${config.hosting.back_port}/tokenize`;
    const response = await fetch(tokenizeUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, max_length, overlap }),
    });
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`토큰화 모델 에러: ${response.status} - ${errText}`);
    }
    const result = await response.json();
    return result.chunks;
  }
  