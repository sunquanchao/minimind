# Design: Japanese Documentation

## Context

The MiniMind project currently provides bilingual documentation:
- `PROJECT_INTRODUCTION.cn.md` - Chinese documentation
- `PROJECT_INTRODUCTION.en.md` - English documentation

The project is gaining international attention, and Japanese-speaking developers represent a significant portion of the open-source ML/AI community. Adding Japanese documentation will lower the barrier for Japanese contributors and users.

**Current State:**
- Documentation structure uses `.cn.md` and `.en.md` suffixes
- Comprehensive content covering project overview, architecture, and usage
- Well-maintained English and Chinese versions

**Constraints:**
- Must maintain consistency with existing documentation structure
- Technical terminology must be accurate in Japanese
- Should follow existing formatting and organization

## Goals / Non-Goals

**Goals:**
- Add comprehensive Japanese documentation (`PROJECT_INTRODUCTION.ja.md`)
- Cover all sections present in English/Chinese versions
- Ensure accurate technical translation
- Maintain consistent documentation structure

**Non-Goals:**
- Translating code comments or inline documentation
- Translating other project documentation files (README, etc.)
- Changing the documentation structure or naming convention
- Automated translation - this will be manual translation for quality

## Decisions

### 1. File Naming Convention
**Decision:** Use `.ja.md` suffix for Japanese documentation

**Rationale:**
- Follows existing pattern (`.cn.md`, `.en.md`)
- ISO 639-1 language code `ja` for Japanese
- Consistent with current project structure
- Alternative `.jp.md` was considered but `.ja.md` is more standard

### 2. Translation Scope
**Decision:** Translate `PROJECT_INTRODUCTION.en.md` comprehensively

**Rationale:**
- English documentation is the most complete and up-to-date
- Japanese version should match English version in scope
- Ensures feature parity between languages
- Easier to maintain synchronization

### 3. Technical Terminology Approach
**Decision:** Keep technical terms in English where appropriate, with Japanese explanations

**Rationale:**
- ML/AI terminology is often used in English form in Japanese technical contexts
- Terms like "Transformer", "Fine-Tuning", "RoPE" are commonly understood
- Improves readability for Japanese developers
- Follows conventions in Japanese technical documentation

### 4. Document Structure
**Decision:** Mirror English/Chinese documentation structure exactly

**Rationale:**
- Maintains consistency across language versions
- Easier for users to compare versions
- Simplifies maintenance and updates
- Preserves all sections: Overview, Architecture, Core Modules, Quick Start, etc.

## Risks / Trade-offs

### Risk 1: Translation Quality
**Risk:** Technical nuances may be lost in translation
**Mitigation:**
- Careful review of technical terminology
- Reference to Japanese ML/AI documentation conventions
- Community feedback loop for improvements

### Risk 2: Maintenance Overhead
**Risk:** Future updates need to be propagated to three languages
**Mitigation:**
- Keep translations synchronized with English version
- Clear documentation of update process
- Community contributions welcomed

### Risk 3: Inconsistent Updates
**Risk:** Japanese version may become outdated if not updated with English changes
**Mitigation:**
- Document synchronization process
- Consider adding translation status indicators
- Encourage community contributions for updates

## Migration Plan

**Steps to deploy:**
1. Create `PROJECT_INTRODUCTION.ja.md` with complete translation
2. Review translation for technical accuracy
3. Update project documentation to mention Japanese version
4. (Optional) Update README to link to all language versions

**Rollback strategy:**
- Simply remove `PROJECT_INTRODUCTION.ja.md` if needed
- No code changes, so rollback is trivial

## Open Questions

- **Q1:** Should we translate other documentation files (README, CONTRIBUTING, etc.)?
  - **A:** Out of scope for this change, can be considered separately

- **Q2:** Should we add automated translation checks?
  - **A:** Not necessary initially, can be added if maintenance becomes burdensome

- **Q3:** Should we include Japanese in CLAUDE.md for AI assistants?
  - **A:** Consider after initial documentation is complete and community feedback is received
