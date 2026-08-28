/**
 * Typed client for the backend's single multiplexed WebSocket.
 *
 * One connection carries camera frames up, microphone audio up, and both
 * perception updates and assistant audio down. Keeping it to one socket is
 * what guarantees the assistant's answer refers to the scene the user was
 * actually looking at when they asked.
 */

export interface DetectionBox {
  class: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface PerceivedEntity {
  id: string;
  type: string;
  direction: string;
  direction_confidence: number;
  distance_range: string;
  distance_confidence: number;
  confidence: number;
  is_person: boolean;
}

export interface PerceptionStats {
  frames_processed: number;
  frames_failed: number;
  p50_ms: number | null;
  p95_ms: number | null;
  effective_fps: number | null;
}

export interface SignReading {
  text: string;
  direction: string;
  confidence: number;
  navigational: boolean;
  warning: boolean;
  seconds_since_seen?: number;
}

export interface EnrollmentRecord {
  enrollment_id: string;
  kind: 'person' | 'belonging';
  name: string;
  object_class: string | null;
  sample_count: number;
}

export interface CrowdState {
  size: string;
  density: string;
  motion: string;
  flow: string;
  person_count: number;
  count_is_approximate: boolean;
  approaching: boolean;
  departing: boolean;
  congested: boolean;
  flow_direction: string | null;
  centre_direction: string | null;
  confidence: number;
}

export interface FriendState {
  name: string;
  state: string;
  direction_phrase: string;
  identity_confidence: number;
  is_confident: boolean;
  confirmations: number;
  in_crowd: boolean;
  movement: string | null;
}

export type ServerMessage =
  | { type: 'health'; status: string; vision_available: boolean; vision_backend: string | null;
      voice_available: boolean; memory_available?: boolean; semantic_search?: boolean;
      deep_reading?: boolean; face_recognition?: boolean; belonging_recognition?: boolean;
      tools: string[]; timestamp: string }
  | { type: 'frame_ack'; seq?: number; dropped?: boolean; frames_dropped?: number;
      vision_available?: boolean }
  | { type: 'perception'; seq?: number; latency_ms: number;
      // depth_ms is present only on frames where the depth model actually ran;
      // depth_age_s reports how stale the reused map is on the frames between.
      depth_ms?: number | null; depth_age_s?: number | null;
      image: { width: number; height: number }; entities: PerceivedEntity[];
      boxes: DetectionBox[]; events: { event: string; id: string; class: string }[];
      stats: PerceptionStats; frames_dropped: number;
      crowd?: CrowdState | null; friends?: FriendState[];
      faces_seen?: number; face_ms?: number | null }
  | { type: 'tool_call'; tool: string; args: Record<string, unknown> }
  | { type: 'tool_result'; tool: string; result: Record<string, unknown> }
  | { type: 'user_transcript'; text: string }
  | { type: 'agent_transcript'; text: string }
  | { type: 'agent_audio'; data: string; sample_rate: number }
  | { type: 'agent_turn_complete' }
  | { type: 'voice_ready'; model: string }
  | { type: 'voice_stopped' }
  | { type: 'voice_interrupted' }
  | { type: 'voice_error'; message: string }
  | { type: 'voice_disconnected'; reason: string }
  | { type: 'signs'; ocr_ms: number | null; regions: SignReading[]; tracked: SignReading[] }
  | { type: 'proactive_announcement'; signs: SignReading[] }
  | { type: 'enrollments'; records: EnrollmentRecord[]; faces_available: boolean;
      belongings_available: boolean }
  | { type: 'enroll_result'; ok: boolean; message: string; name?: string;
      kind?: string; sample_count?: number }
  | { type: 'enrollment_deleted'; name: string; deleted_count: number }
  | { type: 'profile'; name: string | null; personality: Record<string, number>;
      dominant_traits: string[]; communication: Record<string, string>;
      preferences: Record<string, unknown>; behaviour: Record<string, unknown>;
      attention: Record<string, unknown>; haptics: Record<string, unknown>;
      recommendations: Record<string, unknown> }
  | { type: 'attention_decision'; speak: boolean; haptic: boolean; score: number;
      threshold: number; reason: string; event: Record<string, any> }
  | { type: 'haptic'; event: string; name: string; pulses: number[]; priority: string;
      directional: boolean; duration_ms: number; intensity: number; note?: string }
  | { type: 'haptic_status'; enabled: boolean; provider: string; connected: boolean;
      supports_direction: boolean; fired: number; suppressed: number }
  | { type: 'quiet_mode'; enabled: boolean }
  | { type: 'memory_cleared'; deleted_count: number }
  | { type: 'pong' }
  | { type: 'error'; message: string; scope?: string };

export type ConnectionState = 'connecting' | 'open' | 'closed' | 'error';

export interface AgentClientHandlers {
  onMessage: (message: ServerMessage) => void;
  onStateChange?: (state: ConnectionState) => void;
}

export class AgentClient {
  private socket: WebSocket | null = null;
  private shouldReconnect = true;
  private reconnectDelayMs = 1000;

  constructor(
    private readonly url: string,
    private readonly handlers: AgentClientHandlers,
  ) {}

  connect(): void {
    this.handlers.onStateChange?.('connecting');
    const socket = new WebSocket(this.url);
    this.socket = socket;

    socket.addEventListener('open', () => {
      this.reconnectDelayMs = 1000;
      this.handlers.onStateChange?.('open');
    });

    socket.addEventListener('message', (event) => {
      try {
        this.handlers.onMessage(JSON.parse(event.data) as ServerMessage);
      } catch {
        // A malformed frame must not tear down the UI.
      }
    });

    socket.addEventListener('error', () => this.handlers.onStateChange?.('error'));

    socket.addEventListener('close', () => {
      this.handlers.onStateChange?.('closed');
      if (!this.shouldReconnect) return;
      // Backing off matters here: a backend that is down should not be hit
      // with a reconnect every 100ms while the user is still on the page.
      window.setTimeout(() => this.connect(), this.reconnectDelayMs);
      this.reconnectDelayMs = Math.min(this.reconnectDelayMs * 2, 15000);
    });
  }

  get isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  send(message: Record<string, unknown>): void {
    if (!this.isOpen) return;
    this.socket!.send(JSON.stringify(message));
  }

  sendFrame(seq: number, dataUrl: string): void {
    this.send({ type: 'frame', seq, capturedAt: performance.now(), data: dataUrl });
  }

  sendAudio(base64: string): void {
    this.send({ type: 'audio', data: base64 });
  }

  sendText(text: string): void {
    this.send({ type: 'text', text });
  }

  startVoice(): void {
    this.send({ type: 'start_voice' });
  }

  stopVoice(): void {
    this.send({ type: 'stop_voice' });
  }

  /** Dashboard-only access to read-only tools (backend enforces an allowlist). */
  invokeTool(tool: string, args: Record<string, unknown> = {}): void {
    this.send({ type: 'invoke_tool', tool, args });
  }

  /** Dashboard-only profile edit (a deliberate human gesture, not a tool). */
  setProfile(patch: Record<string, unknown>): void {
    this.send({ type: 'set_profile', ...patch });
  }

  close(): void {
    this.shouldReconnect = false;
    this.socket?.close();
    this.socket = null;
  }
}
