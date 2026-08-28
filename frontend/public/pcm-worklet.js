/**
 * AudioWorklet that converts microphone audio to the PCM16 chunks the Gemini
 * Live API expects.
 *
 * This runs on the audio rendering thread, which is why it is a worklet and
 * not a ScriptProcessorNode: capture must not stutter while the main thread
 * is busy drawing detection boxes or handling WebSocket traffic. Glitchy
 * microphone audio is far more damaging here than a dropped video frame,
 * because it corrupts the user's actual words.
 */
class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // ~128 samples arrive per render quantum; batching to 1024 keeps the
    // WebSocket message rate reasonable (about 16/sec at 16kHz) without
    // adding meaningful latency.
    this.buffer = new Float32Array(1024);
    this.offset = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i += 1) {
      this.buffer[this.offset] = channel[i];
      this.offset += 1;
      if (this.offset === this.buffer.length) {
        // Float32 [-1,1] -> little-endian PCM16, the Live API's input format.
        const pcm = new Int16Array(this.buffer.length);
        for (let j = 0; j < this.buffer.length; j += 1) {
          const clamped = Math.max(-1, Math.min(1, this.buffer[j]));
          pcm[j] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer]);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PCMCaptureProcessor);
