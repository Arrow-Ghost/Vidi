import type { FrameSamplerConfig } from './types';

/**
 * Pure decision logic for Segment 2's "don't send every frame" requirement.
 *
 * Deliberately has no dependency on MediaStream/canvas/WebSocket — every
 * decision here is a function of (current time, internal counters), which
 * is what makes it fully unit-testable without a browser or a real camera
 * (see FrameSampler.test.ts). CameraSource and the capture loop in
 * index.astro are what touch real browser APIs; this class is the one that
 * decides whether they should bother.
 */
export class FrameSampler {
  // null = "no frame sent yet". A numeric sentinel like 0 would be
  // indistinguishable from a real timestamp of 0 and cause a spurious
  // "too soon" rejection on the very first frame — this bit a unit test
  // during development, which is exactly why it's a null, not a 0.
  private lastSentAt: number | null = null;
  private inFlight = 0;
  private seq = 0;
  private readonly minIntervalMs: number;

  constructor(private readonly config: FrameSamplerConfig) {
    if (config.targetFps <= 0) throw new Error('targetFps must be > 0');
    if (config.maxInFlight <= 0) throw new Error('maxInFlight must be > 0');
    this.minIntervalMs = 1000 / config.targetFps;
  }

  /**
   * Call once per raw camera frame (e.g. every rAF/requestVideoFrameCallback
   * tick). Returns whether this frame should actually be sent.
   */
  shouldSample(nowMs: number): boolean {
    if (this.inFlight >= this.config.maxInFlight) {
      // Backpressure: the network/backend hasn't acked recent frames yet.
      // Drop this frame rather than queueing it — there is no frame buffer.
      return false;
    }
    if (this.lastSentAt !== null && nowMs - this.lastSentAt < this.minIntervalMs) {
      // Too soon relative to targetFps — this is a stale/redundant frame.
      return false;
    }
    return true;
  }

  /** Call immediately after deciding to send a frame. Returns its sequence number. */
  markSent(nowMs: number): number {
    this.lastSentAt = nowMs;
    this.inFlight += 1;
    this.seq += 1;
    return this.seq;
  }

  /** Call when a frame_ack arrives from the backend for a previously sent frame. */
  markAcked(): void {
    this.inFlight = Math.max(0, this.inFlight - 1);
  }

  reset(): void {
    this.lastSentAt = null;
    this.inFlight = 0;
    this.seq = 0;
  }
}
