# Speaker Notes — 5-Minute Spotlight Talk

**Target: 5 minutes sharp. Practice with a timer.**

---

## Slide 1: Title (5s)

Hi everyone. I'm Rishi, and my project asks: can we make language agents learn more efficiently by giving them a caregiver relationship, the way human children learn from their parents?

---

## Slide 2: Motivation & Research Question (40s)

Human learning is deeply social. From natural pedagogy---Csibra and Gergely's theory that we covered in class---we know caregivers don't just present information randomly; they curate it. They build a mental model of what the child knows, which connects to Theory of Mind and Gweon's inferential social learning framework. And they scaffold within Vygotsky's Zone of Proximal Development.

Current AI agent training has none of this. So I asked: if we give a language agent a persistent caregiver relationship, does it learn better?

This touches several course topics: Theory of Mind, cooperative communication, social learning, and mental state attribution.

---

## Slide 3: Computational Framework (50s)

Here's the setup. I built 197 household tasks across 8 difficulty levels. The caregiver is a frozen 235-billion-parameter model with access to the task solution. The child is an 8B model with a LoRA adapter that gets updated during training.

The memory architecture has four components inspired by Tulving's memory taxonomy: instinct buffer, working memory, long-term episodic memory, and habits implemented as LoRA weights.

The key mechanism is the salience signal. It combines novelty, prediction error using a Rescorla-Wagner model, and a teaching signal that captures productive struggle. Critically, gamma---the teaching signal weight---is zero without a caregiver.

I compare four conditions: Solo, Symmetric Peer, Role-Labeled with a caregiver but no teaching salience, and our full Relational condition. Each ran with 3 seeds, 12 total concurrent runs.

---

## Slide 4: Hypotheses (15s)

Three hypotheses. H1: the relational condition should produce better transfer to novel tasks. H2: habits should form faster with caregiver guidance. And H3: the child should develop a theory of mind of the caregiver. Let's see what happened.

---

## Slide 5: Training Results (50s)

Training results are dramatic. Left: learning curves as the curriculum gets harder---Solo and Peer degrade, caregiver conditions stay near-perfect. Right: cumulative completion and total LoRA updates---caregiver conditions hit 100 percent success with all 160 episodes producing LoRA updates.

The table: caregiver conditions also need 25 percent fewer turns. H2, habit acceleration, is strongly supported.

---

## Slide 6: The Surprise — No Transfer Advantage (55s)

Now the surprise. When we test the child ALONE on held-out tasks---no caregiver---all four conditions perform the same. About 0.68 to 0.71, differences within noise.

Despite the massive training advantage, caregiver-trained agents do NOT transfer better. H1 is not supported.

Why? Scaffolding dependency. Wood, Bruner, and Ross showed that learners become dependent on external support when it isn't faded. Our LoRA weights encode behaviors for the assisted context. Remove the caregiver, and performance reverts.

Also: Role-Labeled and Relational are nearly identical---prompt asymmetry alone captures most of the benefit.

---

## Slide 7: Takeaway & Next Steps (45s)

Key takeaway: asymmetric caregiver relationships are powerful for training, but without explicit scaffolding fading, the agent becomes dependent. This parallels human developmental psychology.

Course connections: natural pedagogy explains why the caregiver's framing changes what gets encoded. Theory of Mind is needed for adaptive teaching. Cooperative communication drives efficiency. And the scaffolding dependency result shows social context fundamentally changes the learner's representations.

Next steps: implement scaffolding fading, replace BM25 with semantic similarity for ToM, and run full-length training. Thank you!

---

## Slide 8: Thank You / Q&A

Happy to take questions!

---

**Total speaking time: ~4:20, leaving buffer for transitions and natural pauses.**
