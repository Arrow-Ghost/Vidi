# Crowd Intelligence, Friend Recognition and Haptics

Parts 19–32, 38–44, 55, 57, 58 of the spec.

---

# Crowd detection

Part 23 is explicit that **counting people is not crowd detection**. The engine
analyses density, distribution, movement, flow, congestion and approach, and
emits state transitions rather than a running commentary.

## Velocity comes first

Crowd flow, approach detection and friend movement all need motion, so
`ObjectTracker` now carries exponentially-smoothed velocity per track:

```
vx, vy        normalized frame units / second
scale_rate    apparent size change / second  (growing = approaching)
```

Smoothed because raw frame-to-frame deltas at 6 FPS from a hand-held camera
are far too jittery to drive "the crowd is moving left".

## Classification

| Dimension | Values |
|---|---|
| Size | no_crowd · small_group · moderate · large · dense |
| Density | low · medium · high (from frame coverage, not count) |
| Motion | static · slow_moving · moving · fast_moving · mixed_flow |
| Flow | none · unidirectional · bidirectional · converging · diverging · chaotic |

All thresholds live in `CrowdConfig` and are meant to be tuned per deployment
(Part 24) — a corridor camera and an open plaza do not share a definition of
"dense".

## Measured behaviour

Five scenarios, 12 synthetic people, 8 frames each:

| Scenario | Result |
|---|---|
| All drifting right | `large_crowd · moving · unidirectional · "toward your right"` |
| Nobody moving | `static`, *"They appear mostly stationary"* |
| Two opposing streams | `bidirectional`, *"People are moving in both directions"* |
| All growing in size | `dense_crowd · approaching=True`, *"getting closer"* |
| Two people only | `no_crowd` — correctly not a crowd |

Transitions emitted: `CROWD_APPEARED`, `CROWD_STARTED_MOVING`,
`CROWD_APPROACHING`, `CROWD_DENSITY_INCREASED/DECREASED`, `CROWD_STOPPED`,
`CROWD_DIRECTION_CHANGED`, `CROWD_DEPARTING`, `CROWD_CONGESTION`,
`CROWD_DISPERSED`.

Debounced by `confirm_frames` — at 6 FPS one missed detection can flip
"moderate" to "small group" and back, and announcing that churn would be
maddening.

## What it refuses to claim

- **Counts are approximate.** Detection misses occluded people precisely when
  crowds form. Every payload carries `count_is_approximate: true` and the agent
  says "roughly".
- **Directions are relative** — left/right/ahead. Compass bearings would need
  orientation or GPS this system does not have.
- **No intent, ever** (Part 55). "The crowd is moving toward your left" is
  allowed. "The crowd is going to the event" is not; no field carries a
  destination or purpose.
- **Never "safe" or "clear."** Only what is or is not currently detected.

---

# Friend recognition

## The problem this solves

Face matching was single-frame. One bad frame — motion blur, an odd angle, a
partially occluded stranger — could put a name on the wrong person. For a blind
user that is not a glitch; they would walk up to a stranger believing a friend
is there.

## Multi-frame confirmation (Part 58)

A name is never spoken from one observation. Identity confidence blends three
independent signals:

```
similarity   how far above SFace's same-identity threshold the match sits
consistency  how many recent observations agreed
quality      how large and usable the face was
```

| Frame | State | Confidence | Event |
|---|---|---:|---|
| 1 | candidate | 0.558 | — |
| 2 | candidate | 0.658 | — |
| 3 | **confirmed** | 0.758 | `FRIEND_DETECTED` |
| 4 | confirmed | 0.758 | — |

A single spurious frame produces **no event and no name**.

## Confidence bands (Part 57)

| Combined confidence | Behaviour |
|---:|---|
| ≥ 0.90 | *"Ravi is directly ahead of you."* |
| ≥ 0.75 | *"I think that may be Ravi."* |
| < 0.75 | *"There is someone here, but I am not confident enough to say who it is."* |

The floor is enforced inside `describe()` as well as at the confirmation gate —
a `describe()` that can name a 0.48-confidence candidate is a loaded gun for
future callers.

## Underlying separation (measured)

| Case | Similarity |
|---|---:|
| Same person, brighter / half-size / blurred / JPEG q30 | 0.66 – 0.92 |
| **Different person, same photo** | **0.038** |
| **Different person, another photo** | **0.119** |

Threshold 0.363. The gap between same-person and different-person is very
large, which is what makes false identification unlikely.

## Tracking and loss (Part 21)

| State | Meaning | Spoken |
|---|---|---|
| `candidate` | seen, unconfirmed | nothing |
| `confirmed` | may be reported | *"Ravi is to your right"* |
| `occluded` | >1.5s unseen | *"The crowd is blocking my view of Ravi right now."* |
| `lost` | >6s unseen | *"I've lost visual contact with Ravi."* |

Movement is reported; intent never is:

| Signal | Phrase |
|---|---|
| growing, not lateral | "moving toward you" |
| shrinking | "moving away from you" |
| vx > 0.05 | "moving toward your right" |

**Never** "coming to meet you".

## Privacy

Only people the user deliberately enrolled can ever be named. Unenrolled faces
are counted and referred to as "someone" — never described, never identified.
Enrolment stores a 128-d embedding, never a photograph.

---

# Haptics

## What the hardware here can actually do

| Provider | Real? | Direction? |
|---|---|---|
| `MockHapticProvider` | dashboard only, no buzz | no |
| `BrowserHapticProvider` | **yes on Android Chrome** | **no — one motor** |
| `GamepadHapticProvider` | yes with a pad attached | yes — dual motor |

Part 40 forbids claiming directional haptics a device cannot deliver. The
browser provider reports `supportsDirection: false`, and any turn pattern sent
to it carries:

> *"Provider has a single motor and cannot convey direction; this pattern
> signals a turn but not which way. The spoken instruction carries the side."*

This laptop has a Bluetooth radio but no controller, and desktop Chrome exposes
no vibration API — so the honest default here is the mock, with every decision
visible in the dashboard.

## Tactile language (Part 40)

| Shape | Meaning |
|---|---|
| one short pulse | informational |
| double pulse | important |
| rapid burst | urgent |
| long pulse | navigation / attention |
| long-short | left |
| short-long | right |

14 event types, each with a pattern and priority.

## Cooldowns (Part 42)

An obstacle that stays in view must **not** vibrate every frame:

```
obstacle 1st:                True
immediately again (×10):     False
after cooldown:              True
```

Cooldown is keyed per event *instance* (`friend:Ravi` vs `friend:Ankit`), so
two different friends are not treated as one repeating event.

## User control (Part 43)

| Setting | Effect |
|---|---|
| `haptics_enabled: false` | everything off — **except safety** |
| `haptics_safety_only: true` | "only vibrate for important things" |
| `haptics_friend / haptics_crowd: false` | per-category |
| `intensity` | scales pulse length (the only knob one motor has) |

Safety patterns fire regardless of every one of these.
