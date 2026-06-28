---
marp: true
theme: default
paginate: true
---

<!--
LoRAForge - GENERAL deck (outcome-focused, non-technical). Slides separated by "---".
  Marp:   marp presentation-general.md --pdf
  Slidev: npx slidev presentation-general.md
Plain language, value and story over internals. No code, no jargon.
-->

# LoRAForge

### I built and own a custom AI, instead of renting one

I taught a small open AI model to be a customer-support agent, shrank it to run on a normal
laptop, and proved it got better. All on free hardware.

**Hamza El Sousi**

---

## The big idea

Almost everyone using AI today is **renting** it: paying a company, per message, for a model
they do not control.

I wanted to **own** one.

LoRAForge takes a free, open AI model and teaches it a specific job, then runs it privately.

---

## What it actually does

I took a general-purpose AI and trained it on thousands of real customer-support conversations.

The result is an assistant that answers support questions in the right tone and style, and it
runs **on a laptop**, with no monthly bill and no data leaving the machine.

> Think of it as hiring a generalist and giving them a week of on-the-job training for one role.

---

## Why this matters

- **Cost:** you own the model. No per-message fees, no subscription.
- **Privacy:** it runs locally, so customer conversations never leave your machine.
- **Control:** you decide how it behaves, and you can change it any time.

Most AI projects just call someone else's model. This one **creates** the model.

---

## The story: it did not go to plan

Two real roadblocks, and the lesson was in how I handled them.

- The first AI model I chose **would not cooperate** with the training tools. Instead of fighting
  it for days, I picked a better-suited model and moved on.
- The "free GPU" I was given **was not the one I asked for** and could not run my code. I found a
  workaround and forced the right hardware.

Good engineering is often choosing the right path, not brute-forcing the wrong one.

---

## The results

I measured the model **before and after** training, on questions it had never seen.

- Answer quality on support questions: **roughly 60% to 90%**
- It got **much better** at sounding like a real support agent
- The improvement showed up on **fresh, held-out questions**, not memorized ones

The training clearly worked.

---

## Being honest about the limits

Good work means reporting what did **not** hit the target, too.

- It still answers off-topic questions it should politely decline. That is a **data** gap, and I
  know exactly how to close it.
- On a laptop it replies in a few seconds, not instantly. That is the cost of running locally
  without a graphics card.

Saying this out loud is the difference between a demo and engineering.

---

## What this shows

- I can **build and adapt** AI, not just call an API
- I make **practical trade-offs** under real constraints (free hardware, broken tools)
- I **measure honestly** and report the misses
- I **ship the whole thing**, end to end

---

## It is real, and public

Anyone can download the model and run it.

- The model: on **Hugging Face**
- The full project and instructions: on **GitHub**

huggingface.co/HamzaElSousi/loraforge-qwen3-4b-gguf
github.com/HamzaElSousi/LoraForge-fine-tuning

---

# Thank you

**LoRAForge**: a custom AI you own, trained and served on free hardware.

Questions?
