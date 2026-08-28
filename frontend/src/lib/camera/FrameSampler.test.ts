import { describe, expect, it } from 'vitest';
import { FrameSampler } from './FrameSampler';

describe('FrameSampler', () => {
  it('drops frames arriving faster than targetFps', () => {
    const sampler = new FrameSampler({ targetFps: 5, maxInFlight: 4 }); // 200ms interval
    expect(sampler.shouldSample(0)).toBe(true);
    sampler.markSent(0);
    expect(sampler.shouldSample(50)).toBe(false);
    expect(sampler.shouldSample(199)).toBe(false);
    expect(sampler.shouldSample(200)).toBe(true);
  });

  it('applies backpressure once maxInFlight unacked frames are outstanding', () => {
    const sampler = new FrameSampler({ targetFps: 30, maxInFlight: 2 });
    let t = 0;
    expect(sampler.shouldSample(t)).toBe(true);
    sampler.markSent(t);
    t += 1000;
    expect(sampler.shouldSample(t)).toBe(true);
    sampler.markSent(t);
    t += 1000;
    expect(sampler.shouldSample(t)).toBe(false); // 2 in flight, cap is 2
    sampler.markAcked();
    expect(sampler.shouldSample(t)).toBe(true);
  });

  it('never lets inFlight underflow from extra acks', () => {
    const sampler = new FrameSampler({ targetFps: 10, maxInFlight: 1 });
    sampler.markAcked();
    sampler.markAcked();
    expect(sampler.shouldSample(0)).toBe(true);
  });

  it('reset clears sequence and in-flight counters', () => {
    const sampler = new FrameSampler({ targetFps: 10, maxInFlight: 1 });
    sampler.markSent(0);
    expect(sampler.shouldSample(0)).toBe(false); // in-flight cap hit
    sampler.reset();
    expect(sampler.shouldSample(0)).toBe(true);
  });

  it('rejects invalid config', () => {
    expect(() => new FrameSampler({ targetFps: 0, maxInFlight: 1 })).toThrow();
    expect(() => new FrameSampler({ targetFps: 5, maxInFlight: 0 })).toThrow();
  });
});
