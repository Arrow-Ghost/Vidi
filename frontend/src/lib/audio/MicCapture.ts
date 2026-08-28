/**
 * Microphone capture -> PCM16 16kHz -> base64, ready for the Live API.
 *
 * The AudioContext is explicitly created at 16000Hz so the browser resamples
 * in native code rather than us doing it in JS. Echo cancellation and noise
 * suppression are ON: the assistant's own voice comes out of the user's
 * speaker, and without cancellation the model hears itself and interrupts
 * its own turn.
 */

export type MicState = 'idle' | 'starting' | 'listening' | 'muted' | 'error';

export interface MicCaptureOptions {
  /** Called with each base64 PCM16 chunk (~64ms of audio at 16kHz). */
  onChunk: (base64: string) => void;
  onStateChange?: (state: MicState, detail?: string) => void;
  /** Rough input level 0..1, for the UI meter. */
  onLevel?: (level: number) => void;
}

const TARGET_SAMPLE_RATE = 16000;

function toBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  // Chunked to avoid blowing the argument limit on large buffers.
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

export class MicCapture {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private state: MicState = 'idle';
  private muted = false;

  constructor(private readonly options: MicCaptureOptions) {}

  getState(): MicState {
    return this.state;
  }

  async start(): Promise<void> {
    if (this.state === 'listening') return;
    this.setState('starting');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });

      this.context = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      await this.context.audioWorklet.addModule('/pcm-worklet.js');

      this.source = this.context.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.context, 'pcm-capture');

      this.node.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        if (this.muted) return;
        const buffer = event.data;
        if (this.options.onLevel) {
          const samples = new Int16Array(buffer);
          let peak = 0;
          for (let i = 0; i < samples.length; i += 8) {
            const value = Math.abs(samples[i]);
            if (value > peak) peak = value;
          }
          this.options.onLevel(peak / 32768);
        }
        this.options.onChunk(toBase64(buffer));
      };

      this.source.connect(this.node);
      // The worklet produces no output; connecting to the destination would
      // feed the microphone straight back into the speaker.
      this.setState('listening');

      for (const track of this.stream.getAudioTracks()) {
        track.addEventListener('ended', () => this.setState('error', 'microphone-disconnected'));
      }
    } catch (err) {
      this.setState('error', err instanceof Error ? err.message : 'unknown-error');
      throw err;
    }
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    if (this.state === 'listening' || this.state === 'muted') {
      this.setState(muted ? 'muted' : 'listening');
    }
  }

  async stop(): Promise<void> {
    this.node?.port.close();
    this.node?.disconnect();
    this.source?.disconnect();
    if (this.stream) for (const track of this.stream.getTracks()) track.stop();
    if (this.context && this.context.state !== 'closed') await this.context.close();
    this.node = null;
    this.source = null;
    this.stream = null;
    this.context = null;
    this.setState('idle');
  }

  private setState(state: MicState, detail?: string): void {
    this.state = state;
    this.options.onStateChange?.(state, detail);
  }
}
