# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | ✅        |
| Older   | ❌        |

We only actively patch the latest released version of **nanoevals**.

---

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

If you discover a security issue, report it privately by emailing the maintainers at [codebasics.io](https://codebasics.io).

Include as much of the following as possible:

- A description of the vulnerability and its potential impact.
- Steps to reproduce or a proof-of-concept (if applicable).
- Any suggested fix or mitigation.

You will receive a response within **72 hours**. We aim to release a patch within **14 days** of confirmation, depending on severity.

---

## Scope

**nanoevals** is a pure-Python evaluation library with zero mandatory runtime dependencies. Most security considerations relate to:

- **LLM prompt injection** — user-controlled data passed into judge prompts.
- **Third-party LLM clients** — security of your API keys and HTTP clients is your responsibility; nanoevals does not store or transmit credentials.
- **Dependency vulnerabilities** — we keep the optional `dev` dependency set minimal and regularly updated.

---

## Disclosure Policy

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure). We ask that you give us reasonable time to address the issue before public disclosure.
