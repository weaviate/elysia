# Tests that require environment variables

To run these tests, you need the following:

```bash
WCD_URL=...
WCD_API_KEY=...
```
and
```bash
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
```
in your local `.env` file.

These tests deal with various items, such as checking API keys are correctly configured on running decision trees and similar functions.
They also deal with _running LLMs_, in a non-trivial amount. **Running these tests WILL cost you credits**.

You are not required to be able to run these tests to contribute to Elysia. Instead, ensure that the tests in `no_reqs/` pass instead.