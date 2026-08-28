import type { CameraSourceOptions, CameraState } from './types';

/**
 * Owns the browser camera permission/stream lifecycle ONLY — start, stop,
 * pause, resume, and disconnect detection. Deliberately does not know about
 * frame sampling or transport (see FrameSampler, and the capture loop in
 * index.astro); keeping this single-purpose is what lets those be
 * unit-tested without a real camera at all.
 *
 * Requires a real browser (navigator.mediaDevices) — there is no meaningful
 * way to unit-test getUserMedia itself outside one; see DEVELOPMENT.md for
 * the manual verification steps run in an actual browser session.
 */
export class CameraSource {
  private stream: MediaStream | null = null;
  private state: CameraState = 'idle';

  constructor(private readonly options: CameraSourceOptions = {}) {}

  getState(): CameraState {
    return this.state;
  }

  async start(): Promise<MediaStream> {
    this.setState('starting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: this.options.width ?? 640,
          height: this.options.height ?? 480,
          facingMode: this.options.facingMode ?? 'environment',
        },
        audio: false,
      });
      this.stream = stream;
      for (const track of stream.getVideoTracks()) {
        // Fires on physical disconnect, OS-level revocation, etc.
        track.addEventListener('ended', () => {
          this.setState('error', 'camera-disconnected');
        });
      }
      this.setState('running');
      return stream;
    } catch (err) {
      this.setState('error', err instanceof Error ? err.message : 'unknown-error');
      throw err;
    }
  }

  pause(): void {
    if (!this.stream) return;
    for (const track of this.stream.getVideoTracks()) track.enabled = false;
    this.setState('paused');
  }

  resume(): void {
    if (!this.stream) return;
    for (const track of this.stream.getVideoTracks()) track.enabled = true;
    this.setState('running');
  }

  stop(): void {
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
    }
    this.stream = null;
    this.setState('stopped');
  }

  private setState(state: CameraState, detail?: string): void {
    this.state = state;
    this.options.onStateChange?.(state, detail);
  }
}
