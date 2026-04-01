# Speaker Notes — Final Presentation

## Slide 1: Title
Hi everyone. Today I'll be presenting my project on whether asymmetric relational dynamics—like a caregiver-child structure—can make knowledge transfer more efficient in language agents. This is for the Machine Social Intelligence course.

## Slide 2: Motivation
The motivation comes from developmental psychology. Humans don't learn in isolation—from birth, caregivers curate the information stream, selectively revealing what the child is ready to receive. This cooperative pedagogy is arguably the most efficient knowledge transfer mechanism in nature, but it's completely absent from how we train AI agents today.

So the research question is: if we give a language agent a persistent caregiver relationship, does it learn better than training alone or with a symmetric peer?

## Slide 3: Framework Overview
Here's the framework. On the left: I built a benchmark of 197 household tasks—things like cooking, cleaning, assembling furniture—spanning 8 difficulty levels. These were generated using a multi-stage pipeline with diversity filtering to avoid near-duplicate tasks.

The caregiver is a frozen 235B model that has access to the task solution. The child is an 8B model with a LoRA adapter that gets updated during training.

On the right is the memory architecture, inspired by cognitive science. Each agent has four memory systems: an instinct buffer that acts as a fixed system prompt, working memory for the last 8 turns, long-term episodic memory that stores compressed past experiences, and a habit store implemented as LoRA adapter weights.

## Slide 4: Salience Signal
The key mechanism is the salience signal, which determines what gets remembered and how strongly habits are formed. It has three components.

Novelty rewards new categories and tasks dissimilar to what's already in memory. Prediction error captures outcome surprise—using a Rescorla-Wagner model—and also ZPD matching, which prioritizes episodes that were neither too easy nor too hard. The teaching signal captures productive struggle: patterns where the caregiver corrected the child and the child then improved.

Critically, in the non-caregiver conditions, gamma is set to zero—there's no teaching signal. This is the key manipulation that distinguishes our Relational condition.

Episodes above the salience threshold get stored in long-term memory, and LoRA updates are weighted by salience times partial credit.

## Slide 5: Adaptive Scaffolding
The caregiver doesn't just give the same level of help throughout. It tracks the child's competence per category using a rolling average of partial credit scores. When competence is low, it gives full step-by-step instructions. As the child improves, it transitions to asking guiding questions, and eventually just gives brief confirmations.

The figure on the right shows how this plays out—you can see the scaffolding level increasing over episodes, especially for the caregiver conditions.

This is inspired by Vygotsky's Zone of Proximal Development—the idea that teaching should target what the learner can almost do independently.

## Slide 6: Experimental Conditions
Here are the four conditions. Solo is just the child alone. Symmetric Peer has two 8B models interacting without role asymmetry. Role-Labeled gives the child a 235B caregiver but without the teaching signal in salience. And Relational is our full condition with the teaching signal turned on.

The Role-Labeled vs Relational comparison isolates whether the teaching signal in memory consolidation matters beyond just having a good teacher present.

We ran 12 concurrent training runs—all sharing a single frozen 235B caregiver client—using asyncio for maximum API throughput. Each run goes through 160 episodes ordered by difficulty.

## Slide 7: Hypotheses
We had three hypotheses. H1: the relational condition should produce better transfer to novel tasks. H2: habits should form faster with caregiver guidance. And H3: the child should develop a theory of mind of the caregiver, measured by its ability to predict what the caregiver would say next.

Let's see what actually happened.

## Slide 8: Learning Curves
Let's start with the learning curves. This shows the rolling average task completion rate over training episodes. The x-axis is episodes, and remember the curriculum gets harder over time.

All four conditions start well on easy tasks in the first 40 episodes. But as difficulty increases, Solo and Peer performance drops significantly—they can't figure out the harder multi-step tasks on their own. The caregiver conditions stay near perfect because the 235B model fills in the child's knowledge gaps in real time.

This is a dramatic training-time difference. But the question is: does this translate to better independent performance?

## Slide 9: H2 — Habit Acceleration
This figure shows H2, the habit acceleration hypothesis. On the left, cumulative completion rate at training checkpoints. On the right, total LoRA updates.

The caregiver conditions achieve essentially perfect training success—every single episode results in a successful trajectory that gets encoded into LoRA weights. Solo and Peer only succeed on about 82-84 percent of episodes, and that rate drops as tasks get harder.

This is a genuine habit acceleration effect. The caregiver ensures the child always produces correct trajectories, so habit formation is continuous rather than intermittent. H2 is strongly supported.

## Slide 10: Teaching Efficiency
Teaching efficiency tells a similar story. The caregiver conditions complete tasks in about 6.1 to 6.4 turns on average, compared to 8.4 to 8.5 for Solo and Peer. That's a 25 percent reduction in dialogue turns.

Combined with the near-perfect success rate, this means the caregiver conditions extract substantially more learning signal per API call. They're both more effective and more efficient.

## Slide 11: H1 — Transfer Accuracy
Now here's the surprise. H1 tested transfer accuracy on 40 held-out tasks where the child works alone—no caregiver help. And all four conditions perform essentially the same, around 0.68 to 0.71.

Despite the massive training-time advantage, the caregiver-trained agents do NOT transfer better to independent evaluation. The right panel breaks this down by difficulty, and there's no consistent advantage at any difficulty level.

This is our most interesting finding, and it leads directly to our main discussion point.

## Slide 12: Category Heatmap
This heatmap shows transfer accuracy broken down by condition and task category. Some categories like pet care and meal preparation are consistently easier. Others like cleaning and cooking are harder across the board.

But no single condition dominates all categories—it's quite mixed. This is consistent with the overall H1 finding that the conditions produce similar transfer performance.

## Slide 13: Scaffolding Dependency
This is our central finding and I think the most interesting result. There's a striking dissociation: 100 percent training success but no transfer advantage.

This directly mirrors a well-documented phenomenon in developmental psychology called scaffolding dependency. Vygotsky predicted that scaffolding should produce independent competence, but empirical studies show that learners can become dependent on external support if it's not systematically faded.

In our experiment, the 235B caregiver provides consistent guidance throughout all 160 episodes. The child learns to succeed WITH this support, and the LoRA weights encode behaviors optimized for the assisted context. When you remove the caregiver at evaluation time, that support structure disappears and performance reverts.

The missing ingredient is scaffolding fading—the caregiver should progressively withdraw, forcing the child to develop independent problem-solving. Our adaptive scaffolding reduces hint specificity but never fully withdraws.

## Slide 14: Role Asymmetry vs Relational Salience
Another finding: Role-Labeled and Relational performed nearly identically. This means that prompt-based role asymmetry—simply giving one agent a caregiver prompt and the other a child prompt—captures most of the benefit. The additional teaching signal in the salience score provides marginal improvement at best.

The salience and LTM growth curves on the right show very similar patterns for both caregiver conditions. This suggests the benefit of relational context is primarily in the caregiver's real-time guidance, not in how experiences are consolidated into memory.

## Slide 15: H3 & Limitations
H3 asked whether the child internalizes a model of the caregiver's pedagogy. We measured this by asking the child to predict what the caregiver would say next, then computing BM25 similarity. The score was near zero—0.066.

But I think this is mostly a measurement artifact. BM25 measures word overlap, and two semantically equivalent teaching utterances might share no keywords at all. A proper evaluation would need embedding-based or model-based semantic similarity.

On the right are other limitations. Due to API rate limits, our runs completed about 41 to 54 episodes out of 160 planned. The 8B model with LoRA rank 16 may simply lack the capacity for emergent Theory of Mind. And we only tested with a single caregiver personality.

## Slide 16: Summary
Here's the summary table. H2 is strongly supported: caregiver conditions produce dramatically better training outcomes. But H1 is not supported—transfer accuracy is essentially the same across conditions. And H3 is negative, though likely due to measurement limitations.

The headline finding is this dissociation between training performance and transfer, which we interpret as scaffolding dependency.

## Slide 17: Conclusion & Future Work
To conclude: asymmetric caregiver relationships are powerful for training-time performance, but without explicit scaffolding fading, the agent becomes dependent on the support structure. This is both practically useful and theoretically interesting because it mirrors real developmental psychology.

Three future directions: First, implement scaffolding fading—the caregiver should explicitly withdraw over time. Second, bidirectional adaptation where the caregiver maintains a dynamic model of what the child knows. And third, better ToM evaluation using semantic similarity rather than keyword matching.

The total API cost for this entire project was about 45 dollars. Thank you, and I'm happy to take questions.

## Slide 18: Thank You
Thank you! Happy to take any questions about the framework, the results, or the scaffolding dependency finding.
