You annotate the argument structure of televised political debates (BBC Question Time), the way the expert annotators of the QT30 corpus do, following Inference Anchoring Theory.

# Input
- Turns T1, T2, ... in the order they were spoken, each as "Speaker: words".
- Under each turn, the proposition(s) it expresses, P1, P2, ... A proposition restates what was said so that it stands on its own: pronouns are resolved, and a question may use "xxx" for the unknown part.
- Propositions that no turn could be matched to are listed at the end.
- "Links to type" lists the turn-proposition pairs whose illocution you must give.

# Task 1: illocutions
For every listed link, say how the turn puts its proposition forward:
- Asserting: states it as true. This is by far the most common type (about 90%).
- Pure Questioning: a genuine question, with no commitment to an answer.
- Assertive Questioning: question form, but the speaker also conveys their own position.
- Rhetorical Questioning: a question asked to make a point, not to get an answer.
- Agreeing: the turn agrees with an earlier claim ("yes", "exactly", "you're right"); the proposition is the claim agreed with.
- Challenging: the turn challenges an earlier claim and demands a justification.
- Default Illocuting: rare; use it only when the turn plainly fits none of the others.

# Task 2: relations between propositions
- Inference: "from" is a premise, "to" is the conclusion it supports (a reason, evidence or explanation for it). The premise may come before or after the conclusion in the dialogue ("X because Y" and "Y, so X" both give Y -> X).
- Conflict: "from" attacks "to": it contradicts, rebuts or undermines it.
- Rephrase: "from" restates, reformulates or specifies "to". An answer rephrases the question it answers.
- Conflict and Rephrase almost always point from the later proposition to the earlier one.
- Give at most one relation per pair of propositions, and never relate a proposition to itself.
- A conclusion may have several separate premises, and one premise may support several conclusions. Premises that only work together are each listed as an Inference to the conclusion.

# How the annotators work (measured on 1,250 annotated maps)
- Relations mostly join neighbouring turns: 69% are one turn apart and 84% at most two.
- 94% of Inferences join propositions by the same speaker. Rephrases are mostly same-speaker too, apart from answers to questions.
- Conflict is rare: most maps have none. Mark it only when one proposition clearly opposes the other.
- A map has about one relation for every two propositions, and about 80% of propositions take part in at least one relation.

# Confidence
Give each relation a confidence that the annotators drew it: "high" when it is clear, "medium" when it is likely, "low" when it is plausible but uncertain. List the uncertain relations too, marked "low".

# How to work
Before you write the answer, think the whole map through. For each proposition, ask which other propositions it supports, attacks or restates. For every Inference, check which side is the reason and which is the claim. For every candidate Conflict, check that one proposition really opposes the other. Then compare the candidate relations across the map, keep the ones the annotators would draw, and only then write the JSON.

# Output
Answer only with JSON in the required schema: "illocutions" (one entry per listed link), then "relations".
