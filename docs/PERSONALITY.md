# Personality Engine

Parts 4–16, 33, 56 of the spec.

## Personality is a vector, not a switch

A user is not "an extrovert". They are a point in an 18-dimensional preference
space, and behaviour is derived from the whole vector:

```json
{ "nature_lover": 0.90, "extrovert": 0.85, "explorer": 0.80 }
```

Unset traits sit at **0.5 (neutral)**, not 0. Opposing pairs stay consistent —
raising `extrovert` lowers `introvert`, so a profile can never become
incoherent.

| Opposing pairs |
|---|
| extrovert ↔ introvert |
| chatterbox ↔ minimalist |
| cautious ↔ adventurer |
| routine_oriented ↔ spontaneous |

All 18 traits: extrovert, introvert, explorer, adventurer, social, curious,
calm, practical, chatterbox, minimalist, independent, cautious, nature_lover,
routine_oriented, spontaneous, achievement_oriented, family_oriented,
friend_oriented.

## This is a preference model, not an assessment

Traits move **only** from explicit statements by the user. Nothing infers
personality from behaviour, speech patterns, or appearance. Clinical
vocabulary is refused outright rather than stored:

```
User:  "Remember that I'm clinically depressed."
Tool:  { "saved": false, "refused": true }
Agent: offers to note a communication preference instead.
```

`CLINICAL_TERMS` in `profile_tools.py` is the refusal list. Test:
`test_clinical_labels_are_refused_not_stored`.

## Derived behaviour, not a prompt line

`PersonalityEngine.behaviour()` computes real numbers the rest of the system
obeys. Personality is not "told" to the LLM and hoped for — it changes
deterministic gating.

| Knob | Effect |
|---|---|
| `interruption_threshold` | score an event must beat to be spoken (0.15–0.9) |
| `proactive_cooldown_s` | minimum gap between unprompted remarks (6–90s) |
| `sentence_budget` | how long a spoken reply may be |
| `announce_friends` | surface known people unprompted |
| `suggest_social` | nudge toward talking to people |
| `suggest_exploration` | volunteer places |
| `environmental_detail` | how much optional scene commentary |

Measured example — the same friend-detected event:

| Profile | score | threshold | outcome |
|---|---:|---:|---|
| extrovert 0.9 + social 0.85 | 0.81 | 0.42 | **speaks** |
| introvert 0.9 + minimalist 0.85 | 0.53 | 0.73 | stays quiet |
| practical 0.9 | 0.61 | 0.56 | speaks (briefly) |

And the same park:

| Profile | outcome |
|---|---|
| explorer + nature_lover | *"There's a sign reading 'City Park' slightly to your left. Want to go and take a look?"* |
| introvert + calm | ranks the **library** above the park |
| practical + minimalist | says nothing unprompted |

## The safety boundary (Part 56)

Personality shapes **style, proactivity and suggestions**. It can never touch
safety, privacy, identity confidence, or consent.

This is structural, not a prompt instruction. In `AttentionManager.evaluate()`
the safety branch returns **before any personality value is read**:

```python
if event.severity == Severity.SAFETY:
    return self._commit(Decision(True, True, 1.0, 0.0, "safety_override", event), now)
```

Verified with every suppression mechanism engaged at once — quiet mode on, all
alert categories disabled, haptics off, the most reserved possible profile, and
the agent mid-sentence. The hazard still speaks
(`test_no_preference_or_trait_can_silence_a_hazard`).

## Preferences stop volunteering, never answering

A subtle bug worth recording: after *"don't tell me about places any more"*,
asking *"is there anywhere interesting nearby?"* was **refused**. That is wrong
— a preference not to be interrupted is not a refusal to answer.

Fixed in two places:
- `RecommendationEngine.recommend(proactive=False)` skips the gate when asked.
- The tool description and system prompt state the distinction explicitly.

Now:

```
User:  "Don't tell me about places any more."
Agent: "I won't suggest places unless you ask me to, Shrish."
User:  "Is there anywhere interesting nearby?"
Agent: "There's a sign reading 'City Park' slightly to your left, Shrish."
```

## What the user can say

| Utterance | Effect |
|---|---|
| "Remember my name is Shrish" | `set_name` |
| "I'm an extrovert" / "I like exploring" | `set_personality` (+0.25, opposite −0.15) |
| "I'm not really an extrovert" | `set_personality(strength="not")` |
| "Be more/less talkative" | `set_communication(verbosity=…)` |
| "Don't interrupt me so much" | `set_communication(proactive_level="low")` |
| "Tell me when my friends are nearby" | `set_preference(announce_friends, true)` |
| "Don't push me to talk to people" | `set_preference(suggest_social, false)` |
| "Only vibrate for important things" | `set_preference(haptics_safety_only, true)` |
| "Ravi and Ankit are my college friends" | `create_friend_group` |

Profile and personality live in their own `profile` table, separate from
`memories` (Part 46) — a memory is a fact the user told us; the profile is how
the assistant should behave.

## Storage

```
profile(key, value, updated_at)     name · personality · communication · preferences
friend_groups(name, members, …)     Part 52
```

Survives restarts. `reset_profile` erases it entirely.
