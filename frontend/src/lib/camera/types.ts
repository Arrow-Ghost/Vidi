export type CameraState = 'idle' | 'starting' | 'running' | 'paused' | 'stopped' | 'error';

export interface CameraSourceOptions {
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
  onStateChange?: (state: CameraState, detail?: string) => void;
}

export interface FrameSamplerConfig {
  /** Frames per second actually sent to the backend — deliberately far below raw camera FPS. */
  targetFps: number;
  /** Max frames sent but not yet acknowledged before new frames are dropped instead of queued. */
  maxInFlight: number;
}
