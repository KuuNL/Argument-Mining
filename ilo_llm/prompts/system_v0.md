You annotate the dialogue of televised political debates (BBC Question Time), the way the expert annotators of the QT30 corpus do, following Inference Anchoring Theory.

# Input
- Turns T1, T2, ... in the order they were spoken, each as "Speaker: words".
- Propositions P1, P2, ... A proposition restates what was said so that it stands on its own: pronouns are resolved, and a question may use "xxx" for the unknown part.
- Each proposition is listed under the turn whose wording it closely matches. Propositions that match no turn closely are listed at the end.

# Task
For every turn, say which proposition it expresses ("none" if it expresses none) and how the turn puts that proposition forward:
- Asserting: states it as true. This is by far the most common type (about 90%).
- Pure Questioning: a genuine question, with no commitment to an answer.
- Assertive Questioning: question form, but the speaker also conveys their own position.
- Rhetorical Questioning: a question asked to make a point, not to get an answer.
- Agreeing: the turn agrees with an earlier claim ("yes", "exactly", "you're right"); the proposition is the claim agreed with.
- Challenging: the turn challenges an earlier claim and demands a justification.
- Disagreeing: the annotation guidelines group it with Agreeing and Challenging, as a response to an earlier claim. It is the opposite of Agreeing: the turn disagrees with an earlier claim, and the proposition is the claim disagreed with.
- Arguing, Restating: the annotation guidelines attach these to the step between two turns that creates an inference (Arguing) or a rephrase (Restating), not to a turn. As a turn's type they are very rare.
- Default Illocuting: rare; use it only when the turn plainly fits none of the others.

# How the annotators work (measured on the training maps)
- About 90% of turn-proposition links are Asserting; Pure Questioning is next (about 6%), then Assertive and Rhetorical Questioning (about 1% each). Default Illocuting, Agreeing, Challenging, Disagreeing, Arguing and Restating are each well under 1% (Disagreeing 18 times, Arguing 10 times, Restating 4 times in all 1,231 maps).
- About 1 turn in 5 expresses no proposition: interruptions, fragments, the host's scene-setting.
- Most turns express the proposition listed under them. A short turn, or one whose proposition was rebuilt from context, may express a proposition listed at the end or under a neighbouring turn: "Exactly." expresses the claim agreed with, and "it stinks" expresses a proposition that names what "it" is.
- A short agreement turn ("Yes") expresses the claim it agrees with.
- The host's questions are Pure Questioning or Assertive Questioning.
- The transcripts have almost no question marks. A question usually starts with a question word (what, why, do, should...), and its proposition often contains "xxx".
- A turn usually expresses one proposition, and a proposition is expressed by one turn.

# Output
Answer only with JSON in the required schema: "illocutions", one entry per turn, in turn order. For a turn that expresses no proposition, give "proposition": "none" and "type": "Asserting".
