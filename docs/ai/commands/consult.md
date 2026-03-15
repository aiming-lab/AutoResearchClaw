# /consult - Strategic Planning Oracle

Activate the PYTHIA agent for strategic planning, research, and specification before implementation.

## Agent Invocation

**IMPORTANT:** Before proceeding, read and adopt the PYTHIA agent persona:

**Agent Definition:** [../agents/pythia.md](../agents/pythia.md)

Read the agent file above, then follow its guidelines for research, synthesis, and specification creation.

## Usage

```
/consult [topic or question]
```

## When to Use

Use `/consult` when you need to:
- Research unfamiliar technologies or approaches
- Plan complex features with unclear requirements
- Explore architectural trade-offs
- Synthesize information from multiple sources
- Transform vague ideas into actionable specifications
- Get strategic guidance before diving into implementation

## Workflow

```
/consult [research/plan topic]
    |
PYTHIA researches, synthesizes, produces spec
    |
User reviews and approves spec
    |
/orchestrate [implement approved spec]
    |
ARCHON delegates to specialists
```

## Behavior

When this skill is invoked, read the PYTHIA agent definition and adopt its persona to:

1. **Clarify scope** - Ask targeted questions to bound the problem
2. **Research deeply** - Use web search to gather current information
3. **Synthesize findings** - Connect information into coherent insights
4. **Produce specification** - Create actionable document for implementation
5. **Prepare handoff** - Structure output for ARCHON orchestration

## Examples

### Technology Research
```
/consult What's the best approach for real-time sync between browser tabs?
```
Result: Research on BroadcastChannel, SharedWorker, localStorage events, with recommendation

### Feature Planning
```
/consult Plan a webhook system that can handle 10k events/minute with retry logic
```
Result: Architecture spec with queue design, retry strategy, monitoring approach

### Integration Research
```
/consult How should we integrate Stripe for subscription billing?
```
Result: API research, webhook requirements, data model recommendations, security considerations

### Architectural Decision
```
/consult Should we use GraphQL or REST for our new API layer?
```
Result: Trade-off analysis, recommendation based on project context, migration path if applicable

## Output Types

### Research Summary
For "what/how" questions about technologies or approaches:
- Key findings with sources
- Recommended approach with rationale
- Alternatives considered
- Open questions

### Implementation Specification
For feature planning:
- Requirements (functional & non-functional)
- Architecture decisions
- Component breakdown
- Data flow
- Edge cases
- Ready-for-ARCHON task list

### Decision Record
For architectural decisions:
- Context and problem statement
- Options evaluated
- Decision and rationale
- Consequences and trade-offs

## PYTHIA vs ARCHON

| Aspect | PYTHIA (/consult) | ARCHON (/orchestrate) |
|--------|-------------------|----------------------|
| **When** | Before implementation | During implementation |
| **Focus** | Research & planning | Delegation & coordination |
| **Output** | Specifications | Working code |
| **Thinking** | Deep, exploratory | Efficient, directive |
| **Context** | Can be heavy (research) | Should stay clean |

## Tips

1. **Be specific** - "How do we add auth?" vs "How should we implement JWT refresh token rotation with Redis session storage?"

2. **Provide context** - Mention existing tech stack, constraints, and preferences

3. **Iterate** - PYTHIA will ask clarifying questions; engage with them

4. **Approve before orchestrating** - Review the spec before handing off to ARCHON

## Integration with Orchestrator

After PYTHIA produces an approved specification:

```
User: /consult Plan a notification system with email and push support

PYTHIA: [Produces detailed specification]

User: Looks good, let's build it

User: /orchestrate Implement the notification system per the PYTHIA spec above
```

ARCHON receives clear requirements and can delegate efficiently without heavy thinking.
