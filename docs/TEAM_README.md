# Our possession model, explained simply

This explains what we built for the More Than A Hack 2027 possession challenge, how it got better
version by version, and what is left to do. You don't need to know any coding or data science to
follow it.

---

## 1. The task in one paragraph

We were given one football match: **Barcelona vs Athletic Club**. For every moment of the video
(25 moments, called *frames*, per second, about 199,000 in total), we have to say one of three
things:

- **H**: Barcelona (home) has the ball
- **A**: Athletic (away) has the ball
- **D**: dead ball, meaning play has stopped (throw-in, free kick, goal kick, injury...)

We write the answer as a long list, one line per frame. The organisers compare it with the true
answer and give a score.

## 2. What we're given

- **Tracking data**: for every frame, where each player and the ball are on the pitch, worked out
  automatically by a computer from the video. It's messy: **the ball is missing in about 4 out of
  10 frames** (hidden behind players, in the air, and so on), and players' labels get mixed up.
- **Smart Tagging**: automatic notes such as "Barcelona building up" or "set piece here". Useful
  hints, but they also come from a computer and are sometimes wrong.
- **The video**, which we don't use.

## 3. How we're scored, and why that matters

The main score doesn't care about single frames. It looks at **whole possessions**: one
continuous spell of one team having the ball. If we flicker "Barça, Athletic, Barça" over three
frames, we have invented two extra possessions that never happened, and the score punishes that
heavily. So our model must make **calm, stable decisions**: only switch teams when the evidence
is strong and lasts.

The score goes from **0 to 1** (1 = perfect). The simple example program the organisers gave us
scores about **0.24** on the full match.

## 4. The golden rule: we never "peek" at the answers

The rules say we may **not** tune our model using the answers for this match, may not label
anything by hand, and may not upload the match data to AI tools. So how do we know if a change
helps?

We test on **other, public matches** where the answers are known:

- **2 matches from Metrica Sports** (a free sample dataset)
- **7 Bundesliga matches** from a published research dataset (Bassek et al., 2025), with
  official "who has the ball" labels from the German league's own data provider

To make the test fair, we first make those matches as messy as ours (we hide the ball about 40%
of the time). And when the model *learns* from these matches, we always score it on a match it
**didn't** learn from. That's like testing a student on questions they haven't seen.

Every number below is from that fair test. **We won't know our real score on Barça vs Athletic
until the results come out.**

---

## 5. The journey, version by version

The two numbers are our score on the **Metrica** matches and on the **Bundesliga** matches
(0 to 1, higher is better).

### v1: the starting point (Metrica 0.33 · Bundesliga 0.19)
**Idea:** whoever is **closest to the ball** probably has it. Add the Smart Tagging hints. Then
use a "smoother" that charges a small penalty every time it switches team, so it only switches
when the evidence lasts. This alone turns flickering guesses into proper possessions.
**Weakness:** it hardly ever said "dead ball". When the ball disappeared, it just assumed play
went on.

### v2: spotting stoppages (0.42 · 0.23)
**Idea:** two signs that play has stopped:
1. The ball has been **missing for a long time** (it's off the pitch, or in someone's hands).
2. **Players are walking**, not running.

### v3: who really controls the ball (0.44 · 0.25)
**Idea:** being near the ball isn't the same as having it.
- A player **within 1 metre of a slow ball** almost certainly has it, so that counts extra.
- A ball **flying fast** (a pass, a clearance) just passes near people, so that counts less.

The "within 1 metre" rule gave most of the gain.

### v4: Rayhan's "own half" rule (0.46 · 0.26)
**Idea (from playing football):** when it's unclear who has the ball, like a loose pass or a
clearance in the air, the team **defending that half** usually ends up with it, because it has
more players there. We only use this on unclear moments. Using it all the time made things worse.

### v5: the ball sitting still (0.49 · 0.38)
**Idea:** in live play the ball is almost never still. At a free kick, corner or throw-in it's
**placed or held still**. So "ball not moving for about a second" counts as a sign of a stoppage.
This was a **big jump** on the Bundesliga matches.

### v6: let the computer learn the weights (0.53 · 0.42)
Up to v5, *we* decided how much each clue counts, by football sense and a little trial and error.
In v6 a simple learning program (a "classifier") works out **how much each clue should count** by
studying the public matches. It's still trained **only on outside matches**.

### v7: ball touches (0.62 · 0.53)
**Idea:** possession only changes when someone **touches** the ball. A touch shows up as the ball
**suddenly changing speed or direction** with a player close by. For every frame we tell the
model which team **touched the ball last**, which team **touches it next**, and how long ago or
how soon that is. (Looking ahead is allowed: the match is already recorded.)
This was the **biggest single jump**.

### More data (v7 retested: 0.61 · 0.54)
We added 4 more Bundesliga matches, so 9 outside matches in total. Scores this time are averaged
over more matches, so they're more reliable. The small change from the line above is just the
bigger, fairer test.

### v8: players around the ball (0.61 · 0.54)
**Idea:** count how many players of each team are **within 3 m and 8 m** of the ball. With only 5
matches this didn't help reliably; with 9 it did, but only a little.

### v9: a smarter learner, our final model (0.63 · 0.55)
**Idea:** swap the simple classifier for a **small neural network**. A simple classifier can only
add clues up. A neural network can learn **combinations**, like "close AND slow" (that's control)
versus "close AND fast" (the ball is just flying past). With 9 matches to learn from, it beat
v8 on **8 of the 9** matches. It's also simpler to run than v8.

### Summary

| version | main idea | Metrica | Bundesliga |
|---|---|---|---|
| v1 | nearest player + calm switching | 0.33 | 0.19 |
| v2 | stoppages: missing ball, walking players | 0.42 | 0.23 |
| v3 | close control counts more | 0.44 | 0.25 |
| v4 | "own half" rule (Rayhan) | 0.46 | 0.26 |
| v5 | ball sitting still = stoppage | 0.49 | 0.38 |
| v6 | computer learns the weights | 0.53 | 0.42 |
| v7 | ball touches | 0.62 | 0.53 |
| v8 | players around the ball | 0.61 | 0.54 |
| **v9** | **small neural network (final)** | **0.63** | **0.55** |

(v1–v7 were scored with 5 outside matches, v8–v9 with 9.)

**In short: about twice as good as where we started, on matches the model has never seen.**

---

## 6. Things we tried that did NOT help

Being honest about these also counts in the technical document:

- Changing how often the model is allowed to switch teams: no difference.
- Counting "passes in a row" and "time on the ball" per team: made it worse.
- Training on more variations of the same matches: no difference.
- Training on the Bundesliga matches only: worse. More varied data helps.

## 7. What we hand in

All through **one Google Form**, whose link the organisers email to every team by **14 Nov**.
Deadline: **17 Nov, 23:59 CET** (18 Nov, 4:29 AM India time).

1. **Team name**, exactly as registered.
2. **`<team name>.csv`**: the answer file. It comes from one command on Rayhan's PC:
   `python solution/make_submission.py --team "<team name>"`
3. **A public link to our GitHub repository**: the code, so the organisers can re-run it and get
   exactly the same file. (We checked: it gives the same file every time.)
4. **`<team name>.pdf` or `.md`**: the technical document explaining all of the above.

## 8. Rules to remember

- Don't upload the match data (tracking, tagging, video) to ChatGPT, Claude or any other AI or
  cloud tool.
- Don't label anything by hand or "fix" the data by hand.
- Don't tune anything using the match's own answers. The 5-minute public sample is only for
  checking the file works.
- Don't put the match data in the GitHub repository (it's already blocked there).
