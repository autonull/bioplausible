# Improve Codebase Organization and Documentation

- I want to rely on README.md as the primary (and only necessary) documentation and guide/index to all codebase functionality.  Focused on the various algorithms/architectures/models/optimizers (the "Zoo"), plus all related tools (tests, hyperparameter optimization, AutoScientist, GUIs, etc).  I suspect a better categorization exists than what is present now.  I also want to verify the index is complete and accurate by comparing against the actual implemented code.  Each "Zoo" component should be associated with at least 1 main source code file, in which more specific documentation can be included in the comments.  The goal is a minimal but totally complete documentation strategy, focused/centered on the README.md as the index.
- Consider code reorganization.  For example, `mep/` likely should be a sibling of other related algorithms and not its own top-level folder.  Consider other inconsistencies like this that can be resolved for clearer codebase organization.
- Identify any functional redundancies that should be unified.  For example, suppose two algorithms are similar (or equivalent): consider whether more code can be shared between them, resulting in reduced codebase size.
- Consider renaming things and identifiers for clarity and accuracy.
- Consider safely archiving non-README.md documentation, clarifying and focusing documentation into the top-level README.md.
- Consider other opportunities for valuable codebase enhancement can be applied in this process.
- Thoroughly study README.md and the codebase before anything else.  You may safely ignore any other documentation files, which we'll likely end up archiving. 
- No backwards compatibility is necessary; full-speed ahead
- Don't consider line numbers/counts.

Don't perform any actual changes yet; instead, create a plan `REFACTOR.md` that we can iteratively develop before executing it.
