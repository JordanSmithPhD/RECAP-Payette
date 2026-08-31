# Contributing

Issues that improve portability, validation, documentation, or responsible
interpretation are welcome. Before opening a pull request:

1. Do not add primary imagery, photographs, labels, model weights, credentials,
   local absolute paths, sensitive site details, or generated scientific runs.
2. Preserve the separation between `historical_replay`, `recommended`, and
   post-hoc benchmark behavior.
3. Add or update synthetic tests for scientific-contract changes.
4. Run `make test`, `make lint`, and `make audit`.
5. Explain whether a change alters an estimator, schema, default, or supported
   interpretation.

By contributing, you agree that your contribution is licensed under
AGPL-3.0-only, consistent with the repository license. If a proposed
contribution has another copyright holder, confirm that you are authorized to
submit it.
