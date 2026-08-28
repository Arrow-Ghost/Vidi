/**
 * Gapless playback of the assistant's streamed PCM16 24kHz audio.
 *
 * Two requirements drive this design:
 *
 * 1. NO GAPS. Chunks arrive over a WebSocket at irregular intervals, so each
 *    buffer is scheduled against a running cursor rather than "now" — playing
 *    each chunk on arrival produces audible clicks between them.
 * 2. INSTANT INTERRUPTION. When the user barges in, the assistant must stop
 *    talking immediately. Anything already scheduled has to be torn down, not
 *    allowed to finish, or the assistant talks over the person it is meant to
 *    be listening to.
 */

const OUTPUT_SAMPLE_RATE = 24000;
// Small lead so the first chunk does not start in the past on a busy machine.
const SCHEDULING_LEAD_S = 0.06;

export type PlayerState = 'idle' | 'speaking';

export interface AudioPlayerOptions {
  onStateChange?: (state: PlayerState) => void;
}

export class AudioPlayer {
  private context: AudioContext | null = null;
  private cursor = 0;
  private sources = new Set<AudioBufferSourceNode>();
  private state: PlayerState = 'idle';

  constructor(private readonly options: AudioPlayerOptions = {}) {}

  getState(): PlayerState {
    return this.state;
  }

  /** Must be called from a user gesture — browsers block audio otherwise. */
  async resume(): Promise<void> {
    if (!this.context) {
      this.context = new AudioContext({ sampleRate: OUTPUT_SAMPLE_RATE });
    }
    if (this.context.state === 'suspended') await this.context.resume();
  }

  enqueue(base64Pcm: string): void {
    if (!this.context) return;

    const binary = atob(base64Pcm);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    const pcm = new Int16Array(bytes.buffer);
    if (pcm.length === 0) return;

    const buffer = this.context.createBuffer(1, pcm.length, OUTPUT_SAMPLE_RATE);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i += 1) channel[i] = pcm[i] / 32768;

    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);

    const now = this.context.currentTime;
    // If the stream stalled, the cursor is in the past — restart from now.
    if (this.cursor < now + SCHEDULING_LEAD_S) this.cursor = now + SCHEDULING_LEAD_S;
    source.start(this.cursor);
    this.cursor += buffer.duration;

    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      if (this.sources.size === 0) this.setState('idle');
    };
    this.setState('speaking');
  }

  /**
   * Barge-in. Stops everything currently scheduled and drops the queue.
   * Called when the model reports the user interrupted it.
   */
  interrupt(): void {
    for (const source of this.sources) {
      try {
        source.onended = null;
        source.stop();
      } catch {
        // Already finished — nothing to stop.
      }
    }
    this.sources.clear();
    this.cursor = this.context ? this.context.currentTime : 0;
    this.setState('idle');
  }

  async close(): Promise<void> {
    this.interrupt();
    if (this.context && this.context.state !== 'closed') await this.context.close();
    this.context = null;
  }

  private setState(state: PlayerState): void {
    if (this.state === state) return;
    this.state = state;
    this.options.onStateChange?.(state);
  }
}
