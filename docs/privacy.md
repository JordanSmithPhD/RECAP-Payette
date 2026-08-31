# Privacy and responsible use

The public repository is deliberately data-free. Camera media, row-level annotations, trained weights, exact site coordinates, archive manifests, and detailed result tables can create privacy, security, or stewardship risks even when each item appears harmless in isolation.

## Before collecting or processing images

- Obtain the land manager's authority and any required institutional review.
- Document the monitoring purpose, field of view, capture interval, retention period, access roles, and deletion process.
- Minimize collection of faces, license plates, private property, and areas unrelated to the research question.
- Use notice or other transparency measures appropriate to the setting and governing requirements.
- Confirm that collection, transfer, cloud processing, model use, and publication comply with applicable law, policy, permits, and agreements.

This repository provides analytical code, not legal or ethics approval.

## Storage and access

- Keep primary media in access-controlled storage outside the Git clone.
- Use pseudonymous site IDs and archive-relative paths.
- Encrypt sensitive storage and transfers where required.
- Separate the re-identification key and precise coordinates from analytical tables.
- Grant least-privilege access and keep an access log for governed artifacts.
- Define retention and verified deletion for media, annotation exports, caches, temporary copies, and model checkpoints.

Model weights and embeddings can retain information about training data. Govern them as derived sensitive artifacts rather than assuming they are anonymous.

## Annotation and quality review

Give annotators only the minimum media needed for the task. Train them on handling incidental people, vehicles, and identifying details. Record ambiguous-object rules, reviewer disagreement, and adjudication without copying identifying content into free-text notes.

## Results and disclosure

Prefer site-hour or broader aggregation. Before release, remove:

- absolute archive paths;
- precise coordinates and camera orientation;
- image IDs that can be joined to a private lookup;
- row-level boxes or timestamps when they expose individual movement;
- low-count combinations that reveal uncommon events; and
- operational details that could facilitate equipment interference.

Relative activity patterns may still reveal peak times or vulnerable locations. Review aggregate outputs with the responsible land manager before publication.

## Repository hygiene

`.gitignore` reduces accidents but is not a security boundary. Before every push:

```bash
git status --short
git diff --cached --name-only
git diff --cached
```

Confirm that no committed file contains primary media, site coordinates, credentials, tokens, private archive paths, governed labels, checkpoints, or result extracts. If sensitive content enters Git history, removing it in a later commit is insufficient; stop distribution, rotate exposed credentials if relevant, and follow an approved history-remediation process.

## Applying the code elsewhere

Do not assume that a model or threshold transfers across camera angle, season, weather, illumination, object mix, or recreation context. Validate on locally governed data, preserve missingness, document the intended use, and provide a human review path for consequential decisions.
