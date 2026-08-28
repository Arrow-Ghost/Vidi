# Doors, Exits and Obstacle Alerts

Three capabilities: warn about things in the way, find and track doors, and
locate an exit.

---

## Proximity alerts — "chair directly ahead of you"

Runs on **every frame**, entirely locally. It reads the world model the
detector already fills, so it costs nothing extra.

### What earns an interrupt

Four gates, all of which must pass:

| Gate | Rule |
|---|---|
| **Close** | fused depth says `very_close` or `close` |
| **In the way** | direction is centre-left, centre, or centre-right — the corridor the user would actually walk into |
| **Certain** | detection ≥ 0.55 **and** depth confidence ≥ 0.40 |
| **New** | not already announced, unless it has moved into a closer band |

Something close but off to the far left is not on a collision course, and is
deliberately silent. A "close" reading from a depth estimate that isn't
confident is a guess, and guesses become false alarms that train the user to
ignore the assistant.

### Not nagging

The same chair is announced once, then held for 12s. But **getting closer
bypasses the cooldown** — `nearby → close → very_close` is new information and
re-alerts immediately. An object that leaves view and returns alerts again.

Severity is `SAFETY` when something is very close *and* directly ahead, which
routes it past every cooldown, preference and quiet-mode setting in the
attention manager. Nothing a user configures can silence a hazard.

### The rule that matters most

**It never reports that the path is clear.** Detecting nothing means nothing
*recognised* was found in a narrow field of view. Glass doors, steps down, low
bollards, and anything outside COCO's vocabulary are all invisible to it.

That warning is attached to the tool result itself, not left to the prompt:

> *"This does NOT mean the way ahead is clear or safe… Never tell the user a
> path is clear, safe, or free of obstacles."*

People are called **"someone"**, never "person" — and never described.

---

## Doors and exits

### Why these don't use YOLO

COCO has no door class. Pointed at a corridor containing three doors, the
local detector returns **nothing at all** — verified, and pinned by a test so
nobody wastes time re-discovering it.

So doors come from Gemini vision. Measured on a corridor fixture with known
ground truth (closed left, open centre with an EXIT sign, ajar right):

| Model | Latency | Result |
|---|---:|---|
| **gemini-3.1-flash-lite** | **3.4s** | 3/3 doors, correct sides, exit correctly identified |
| gemini-3.6-flash | 45.6s | 3/3 including "ajar" — far too slow to be useful |

`flash-lite` is used. Its one measured weakness is calling a slightly-ajar door
"closed"; **open vs closed is reliable**, and that is the distinction that
matters for walking through one.

**Negative control:** on a photograph of a football pitch it reported **zero**
doors. It is not inventing them to satisfy the question.

### Opening and closing

`DoorTracker` matches doors between sweeps by horizontal position — crude, but
appropriate, since sweeps are seconds apart and a door does not move. A state
change produces an event:

> *"The door to your left has opened."*

A read of `unknown` never becomes a transition. Two doors closer together than
18% of frame width would be conflated; that is an accepted limit.

### Exits

A door is only an exit with **visible evidence** — an EXIT sign, or daylight
through the opening. With none, `find_exit` refuses to pick one:

> *"I haven't confirmed an exit in what the camera has seen. I can see other
> doors, but nothing marks them as an exit."*

Guessing here could send someone into a fire escape or a stairwell.

### Cost control

A sweep is a billed API call, so:

- **On demand** — `scan_for_doors` forces a fresh look.
- **Background sweep every 12s**, but gated on the scene actually changing.

The gate is a mean-absolute-difference over a 96×96 greyscale pair, well under
a millisecond:

| Comparison | Sweeps? |
|---|---|
| Identical frame | no |
| Slight brightness noise | no |
| Camera panned | **yes** |
| Entirely different scene | **yes** |
| No previous frame | **yes** |

The threshold is deliberately low: missing a door because we were being thrifty
is much worse than an occasional wasted call.

---

## Tools

| Tool | Cost | Question |
|---|---|---|
| `check_path` | instant, free | "Is anything in front of me?" |
| `describe_doors` | instant | "Are there any doors?" "Is it open?" |
| `find_exit` | instant | "Where's the exit?" |
| `scan_for_doors` | ~3–8s | "Scan the room", "look again" |

Three read state the pipeline already gathered, so the user can ask repeatedly
while walking without paying for it.

---

## Verified end to end

Real Gemini, real YOLO + depth, real Live API:

```
DOOR SWEEP — 3 found
   door  closed  exit=False  to your left
   exit  open    exit=True   directly ahead of you
   door  closed  exit=False  to your right

USER : Are there any doors around me?
AGENT: I see three doors. There's an open exit directly ahead of you,
       an open door to your left, and a closed door to your right.

USER : Where is the exit?
AGENT: The exit is directly ahead of you, and it's open. I can see an EXIT sign.

USER : Is anything in front of me?
AGENT: Someone is directly ahead of you, approximately 1 to 2 meters away.
```

State change, from a second sweep: *"The door to your left has opened."*

## Known limits

1. **Ajar reads as closed.** Open vs closed is reliable; partially-open is not.
2. **Doors need network.** Offline, obstacle alerts and sign reading still work;
   door detection reports itself unavailable rather than guessing.
3. **Two adjacent doors may be treated as one** if within 18% of frame width.
4. **Obstacles are limited to COCO classes.** Steps, kerbs, glass and overhead
   hazards are not detected at all — which is precisely why the system never
   claims a path is clear.
5. **Distances are coarse ranges**, not measurements. There is no calibrated
   depth hardware here.
