# Security

ASpec is an early source release for trusted training environments, not a multi-tenant security
boundary or a service intended for direct public-internet exposure.

- Treat model/dataset/config input as code-bearing or resource-bearing. Use `trust_remote_code`
  only for sources/revisions you have reviewed and trust.
- Load trusted checkpoints only. Exact resume includes Python/PyTorch serialization; a completion
  marker or matching hash does not make an untrusted checkpoint safe.
- Restrict Ray, Mooncake endpoints and storage to trusted members. Authentication is not a promise
  of end-to-end encryption or safe public exposure.
- Keep credentials, Ray authentication files, private prompts, weights and data out of issues/commits.
  Prefer dedicated venvs and least-privilege filesystem/network access.
- Do not use host-wide process cleanup or change system drivers to troubleshoot an untrusted job.

Use the repository owner's private security channel if one is available. This file does not establish
a contact address, response SLA or bounty program. If no private channel is advertised, request one
without disclosing exploit details or credentials publicly. Ordinary install/correctness bugs can use
the issue channel with redacted reproductions.
