# Privacy

This system continuously perceives a user's environment on their behalf. It
is designed so that capability never becomes surveillance of the user or of
the people around them.

## Defaults (from Segment 2 onward, when camera/mic exist)

- **No permanent camera recording.** Frames are processed in memory for
  perception and discarded; the pipeline is designed against unbounded
  frame queues (see the Segment 2 acceptance test in `ARCHITECTURE.md`).
- **No permanent microphone recording.** Audio is streamed for speech
  recognition, not archived.
- **No unnecessary image storage.** Nothing is written to disk from the
  camera pipeline unless the user explicitly saves something (e.g. "remember
  this as my backpack" stores a description/embedding, not a photo archive).
- **No unnecessary face/person data.** Ordinary people in view are detected
  and tracked as anonymous entities (`person_1`, `person_2`, ...) with no
  identity, ever, unless they are an explicitly enrolled trusted person.

## Trusted-person system (Segment 8)

- A person is only ever identified by name if the user explicitly enrolled
  them ("Remember Rahul as my friend.").
- The user can enroll, rename, delete, re-enroll, inspect, and revoke any
  trusted person at any time.
- The system never attempts unrestricted identification of strangers, and
  never infers emotion, health, personality, intent, gender, or diagnosis
  from appearance, for anyone.

## Memory (Segment 16)

Three tiers, all user-controllable:

- **Short-term** — current scene state, cleared continuously.
- **Session** — current conversation/task context, cleared at session end.
- **Long-term** — only what the user explicitly asked to be remembered
  (`save_memory`), inspectable and deletable via `search_memory` /
  `delete_memory` and a visible memory-management UI.

## User controls

The user must always be able to, from the UI or by voice:

- delete any saved memory item
- delete any trusted person
- clear the conversation
- clear saved objects
- disable the camera
- disable the microphone

## What this document is not

This is a working privacy design for a prototype, not a compliance
certification. If this system is ever deployed beyond personal/prototype
use, it needs a real legal/privacy review (biometric data laws vary
significantly by jurisdiction, especially for face enrollment).
