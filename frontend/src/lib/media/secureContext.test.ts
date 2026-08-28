import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  MediaUnavailableError,
  describeMediaError,
  isLocalhost,
  requireMediaDevices,
} from './secureContext';

/**
 * These tests exist because of a real failure on a phone: opening the app over
 * http:// produced
 *
 *     Cannot read properties of undefined (reading 'getUserMedia')
 *
 * which gives a user no idea that the fix is simply to use the https:// URL.
 */

function setEnvironment(opts: {
  secure: boolean;
  hostname: string;
  mediaDevices?: unknown;
  port?: string;
}) {
  vi.stubGlobal('window', {
    isSecureContext: opts.secure,
    location: {
      hostname: opts.hostname,
      port: opts.port ?? '4321',
      pathname: '/',
      protocol: opts.secure ? 'https:' : 'http:',
    },
  });
  vi.stubGlobal('navigator', { mediaDevices: opts.mediaDevices });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('requireMediaDevices', () => {
  it('explains the http problem and gives the https address to use', () => {
    setEnvironment({ secure: false, hostname: '10.225.189.2' });

    let caught: unknown;
    try {
      requireMediaDevices();
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(MediaUnavailableError);
    const err = caught as MediaUnavailableError;
    // The message must name the cause and the cure, not a property access.
    expect(err.message).not.toMatch(/undefined/i);
    expect(err.hint).toMatch(/https/i);
    expect(err.suggestedUrl).toBe('https://10.225.189.2:4321/');
  });

  it('allows localhost over http, which really is a secure context', () => {
    const fake = { getUserMedia: () => Promise.resolve({}) };
    setEnvironment({ secure: false, hostname: 'localhost', mediaDevices: fake });
    expect(requireMediaDevices()).toBe(fake);
  });

  it('reports an unsupported browser separately from an insecure origin', () => {
    setEnvironment({ secure: true, hostname: 'example.com', mediaDevices: undefined });
    expect(() => requireMediaDevices()).toThrow(/will not give access/i);
    try {
      requireMediaDevices();
    } catch (error) {
      // A webview inside another app is the usual cause; say so.
      expect((error as MediaUnavailableError).hint).toMatch(/in-app browser/i);
    }
  });

  it('returns the real mediaDevices when capture is genuinely possible', () => {
    const fake = { getUserMedia: () => Promise.resolve({}) };
    setEnvironment({ secure: true, hostname: '10.225.189.2', mediaDevices: fake });
    expect(requireMediaDevices()).toBe(fake);
  });
});

describe('isLocalhost', () => {
  it.each(['localhost', '127.0.0.1', '::1'])('treats %s as local', (host) => {
    setEnvironment({ secure: false, hostname: host });
    expect(isLocalhost()).toBe(true);
  });

  it('does not treat a LAN address as local', () => {
    setEnvironment({ secure: false, hostname: '10.225.189.2' });
    expect(isLocalhost()).toBe(false);
  });
});

describe('describeMediaError', () => {
  it('turns a denied permission into an instruction', () => {
    const error = new DOMException('denied', 'NotAllowedError');
    expect(describeMediaError(error)).toMatch(/Allow camera and microphone/i);
  });

  it('explains a camera already in use', () => {
    const error = new DOMException('busy', 'NotReadableError');
    expect(describeMediaError(error)).toMatch(/already in use/i);
  });

  it('includes the https address for an insecure origin', () => {
    setEnvironment({ secure: false, hostname: '10.225.189.2' });
    try {
      requireMediaDevices();
    } catch (error) {
      const spoken = describeMediaError(error);
      expect(spoken).toContain('https://10.225.189.2:4321/');
      // It has to be sayable out loud, not a stack trace.
      expect(spoken).not.toMatch(/undefined|TypeError/);
    }
  });
});
