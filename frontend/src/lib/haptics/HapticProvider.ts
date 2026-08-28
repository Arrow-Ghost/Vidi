/**
 * Haptic providers — Parts 38, 40, 64.
 *
 * The backend decides WHICH pattern to play and whether it is allowed to play
 * at all. This file only delivers it, so the rest of the system does not care
 * what hardware is attached.
 *
 * HONESTY ABOUT CAPABILITY (Part 40). Each provider declares what it can
 * genuinely do, and `supportsDirection` is reported back to the backend so
 * nothing ever claims a left/right buzz on a single-motor device:
 *
 *   MockHapticProvider     always available. Delivers to the dashboard only —
 *                          every decision is visible, nothing is faked as a
 *                          physical buzz that did not happen.
 *   BrowserHapticProvider  navigator.vibrate(). Real on Android Chrome; absent
 *                          on desktop browsers. ONE motor, so rhythm can carry
 *                          urgency but never a side.
 *   GamepadHapticProvider  dual-rumble via the Gamepad API. This one genuinely
 *                          can express left vs right.
 */

export interface HapticDispatch {
  event: string;
  name: string;
  pulses: number[];
  priority: string;
  directional: boolean;
  duration_ms: number;
  intensity: number;
  note?: string;
}

export interface HapticProvider {
  readonly id: string;
  readonly label: string;
  readonly supportsDirection: boolean;
  isAvailable(): boolean;
  connect(): Promise<boolean>;
  disconnect(): void;
  vibrate(dispatch: HapticDispatch): Promise<boolean>;
  stop(): void;
}

/**
 * Always-available fallback. It does NOT pretend to vibrate — it reports the
 * dispatch so the dashboard can show exactly what would have been felt.
 */
export class MockHapticProvider implements HapticProvider {
  readonly id = 'mock';
  readonly label = 'Mock (dashboard only — no physical vibration)';
  readonly supportsDirection = false;
  private connected = false;

  constructor(private readonly onDispatch?: (d: HapticDispatch) => void) {}

  isAvailable(): boolean {
    return true;
  }

  async connect(): Promise<boolean> {
    this.connected = true;
    return true;
  }

  disconnect(): void {
    this.connected = false;
  }

  async vibrate(dispatch: HapticDispatch): Promise<boolean> {
    if (!this.connected) return false;
    this.onDispatch?.(dispatch);
    return true;
  }

  stop(): void {}
}

/**
 * navigator.vibrate(). Real on Android Chrome, unavailable on desktop.
 *
 * Note `supportsDirection = false`: this is one motor. The turn patterns still
 * differ in rhythm so they are distinguishable from each other, but the SIDE is
 * carried by the spoken instruction, never by the buzz alone.
 */
export class BrowserHapticProvider implements HapticProvider {
  readonly id = 'browser-vibrate';
  readonly label = 'Device vibration (navigator.vibrate)';
  readonly supportsDirection = false;
  private connected = false;

  isAvailable(): boolean {
    return typeof navigator !== 'undefined' && typeof navigator.vibrate === 'function';
  }

  async connect(): Promise<boolean> {
    if (!this.isAvailable()) return false;
    // Deliberately NO probe call here. Browsers block navigator.vibrate()
    // until the user has tapped the page, so probing at load time always
    // fails, logs a console error, and would make us report the provider as
    // unavailable on a device where it works perfectly after one tap.
    // Selection is therefore driven from a real user gesture instead.
    this.connected = true;
    return true;
  }

  disconnect(): void {
    this.stop();
    this.connected = false;
  }

  async vibrate(dispatch: HapticDispatch): Promise<boolean> {
    if (!this.connected || !this.isAvailable()) return false;
    try {
      return navigator.vibrate(dispatch.pulses);
    } catch {
      return false;
    }
  }

  stop(): void {
    try {
      navigator.vibrate?.(0);
    } catch {
      /* already stopped */
    }
  }
}

/**
 * Dual-motor rumble. The only provider here that can honestly express a side,
 * so it is the only one reporting supportsDirection = true.
 */
export class GamepadHapticProvider implements HapticProvider {
  readonly id = 'gamepad';
  readonly label = 'Game controller rumble';
  readonly supportsDirection = true;
  private index: number | null = null;

  isAvailable(): boolean {
    if (typeof navigator === 'undefined' || !navigator.getGamepads) return false;
    return Array.from(navigator.getGamepads() ?? []).some(
      (pad) => pad && (pad as any).vibrationActuator,
    );
  }

  async connect(): Promise<boolean> {
    const pads = Array.from(navigator.getGamepads?.() ?? []);
    const found = pads.findIndex((pad) => pad && (pad as any).vibrationActuator);
    if (found < 0) return false;
    this.index = found;
    return true;
  }

  disconnect(): void {
    this.stop();
    this.index = null;
  }

  async vibrate(dispatch: HapticDispatch): Promise<boolean> {
    if (this.index === null) return false;
    const pad = navigator.getGamepads?.()[this.index] as any;
    const actuator = pad?.vibrationActuator;
    if (!actuator?.playEffect) return false;

    // Bias the motors for a directional pattern: strong-magnitude on the side
    // being indicated. Only meaningful because this device really has two.
    const left = dispatch.event.endsWith('LEFT');
    const right = dispatch.event.endsWith('RIGHT');
    const strength = Math.min(1, 0.6 * dispatch.intensity);
    try {
      await actuator.playEffect('dual-rumble', {
        duration: dispatch.duration_ms,
        strongMagnitude: right ? strength : left ? 0.15 : strength,
        weakMagnitude: left ? strength : right ? 0.15 : strength,
      });
      return true;
    } catch {
      return false;
    }
  }

  stop(): void {
    const pad = this.index !== null ? (navigator.getGamepads?.()[this.index] as any) : null;
    pad?.vibrationActuator?.reset?.();
  }
}

/**
 * Picks the best provider actually present, in descending capability order.
 * Never returns null — the mock always works, so the haptic channel is always
 * observable even with no hardware at all.
 */
export async function selectHapticProvider(
  onMockDispatch?: (d: HapticDispatch) => void,
): Promise<HapticProvider> {
  const candidates: HapticProvider[] = [
    new GamepadHapticProvider(),
    new BrowserHapticProvider(),
  ];
  for (const provider of candidates) {
    if (provider.isAvailable() && (await provider.connect())) return provider;
  }
  const mock = new MockHapticProvider(onMockDispatch);
  await mock.connect();
  return mock;
}
