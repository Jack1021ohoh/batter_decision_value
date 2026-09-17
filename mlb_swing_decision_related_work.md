# Related Work: Valuing MLB Hitters' Swing Decisions

**Project focus:** Pitch-level valuation of MLB hitters' swing/take decisions  
**Related project components:** SOTO+-style scaling, individualized nitro zones, convex-hull zone construction, and reliability/stabilization analysis  
**Last updated:** September 17, 2026

## 1. Research objective

The central problem is to estimate the value of a hitter's decision to swing or take each pitch, independently—as far as possible—from the realized outcome of that decision.

At the moment of decision, a hitter faces two counterfactual action values:

$$
Q_{\text{swing}}(s) = E[R \mid A=\text{swing},S=s]
$$

and

$$
Q_{\text{take}}(s) = E[R \mid A=\text{take},S=s],
$$

where $s$ is the pre-decision state and $R$ is an offensive-value target such as change in run expectancy. The difference

$$
\Delta(s)=Q_{\text{swing}}(s)-Q_{\text{take}}(s)
$$

describes which action has greater expected value. A pitch-level decision-regret measure can then be defined as

$$
L_i=\max\{Q_{\text{swing},i},Q_{\text{take},i}\}-Q_{\text{chosen},i}.
$$

This formulation distinguishes decision quality from realized results: a hitter can make the correct decision and still record an out, or make a poor decision and obtain a hit.

## 2. Summary of closely related work

| Work | Type | Main method | Primary output | Availability |
|---|---|---|---|---|
| Yee & Deshpande (2024) | Peer-reviewed paper | Bayesian additive regression trees for strike, contact, and run-expectancy models | Pitch-level optimal swing/take decision with uncertainty | Open paper and code |
| Mould & Anderson (2022), EAGLE | Public sabermetric research | Gradient-boosted event-probability models plus run expectancy | Expected additional runs from swinging or looking | Articles; methodology public |
| Vock & Vock (2018) | Peer-reviewed paper | Potential outcomes and G-computation | Counterfactual offensive performance under alternative swing policies | Abstract public; journal access may be required |
| Orr (2023), SEAGER | Public sabermetric research | Count/location swing and take values plus called-strike probability | Selective aggression metric | Baseball Prospectus articles |
| Haugen, SwRV | Public sabermetric research | Expected swing and take run values | Pitch-decision value and hitter aggregates | Public article |
| Nestico (2023) | Public project | Separate XGBoost models for observed swings and takes | In-zone awareness, out-of-zone awareness, and Decision Value | Article, notebook, and output data |
| Salorio (2024), SOTO | Public project | Separate XGBoost swing/take models with standardized aggregation | SOTO and SOTO+ | Public article |
| Salorio, SOTO v2/retrospective | Public methodological review | Nitro-zone extension and bias analysis | Revised assessment of SOTO, SEAGER, and SwRV | Public article |
| Creally (2026), weighted decision value | Public project | Count-by-attack-zone linear weights | Weighted Decision Value and wDV/100 | Public article |
| Driveline (2019) | Applied research | Individualized favorable swing regions | Hitter-specific decision maps | Public article |
| Douglas et al. (2021) | Research paper/preprint | Stochastic zero-sum game and neural outcome prediction | Optimal pitcher/batter strategies | Open preprint |

## 3. Detailed review

### 3.1 Yee and Deshpande: Bayesian plate-discipline evaluation

Yee and Deshpande develop the most directly relevant academic framework. Rather than treating every pitch inside the rule-book strike zone as a required swing, they estimate:

1. the probability that a taken pitch is called a strike;
2. the probability of contact conditional on a swing; and
3. expected runs after each possible pitch outcome.

The models incorporate pitch location, count, game state, player information, handedness, and umpire-related context. Bayesian additive regression trees (BART) allow nonlinear effects and propagate uncertainty from the intermediate models into the final swing/take comparison.

The result is a posterior distribution for the expected-value difference between swinging and taking. Therefore, the framework can distinguish a clear decision from a borderline one rather than forcing every pitch into an equally confident binary label.

**Strengths**

- Provides a formal pitch-level counterfactual comparison.
- Includes game-state context and hitter/pitcher quality.
- Quantifies uncertainty, not only the preferred action.
- Validates intermediate probability models out of sample.
- Provides public code for downloading data, fitting models, and calculating summary metrics.

**Limitations relevant to this project**

- The published analysis uses one MLB season, so temporal generalization requires additional testing.
- The authors warn that hitter-level comparisons are affected by the distribution of pitches each hitter faces.
- Contact probability is not the same as contact quality; a richer swing-value model should distinguish weak and damaging contact.
- The model is computationally heavier than an XGBoost baseline.

**Project relevance:** This should be the main statistical foundation for a pitch-level expected-regret model.

### 3.2 EAGLE: Expected Additional runs Gained by Looking/swinging Estimate

Mould and Anderson's EAGLE model independently developed a closely related tree-structured framework. It models called-strike and contact probabilities using flexible gradient-boosted methods, combines those probabilities with run expectancy, and compares the expected value of swinging with the expected value of taking.

**Strengths**

- Directly evaluates both actions instead of relying on zone/chase percentages.
- Uses accessible gradient-boosted models suitable for a reproducible Statcast pipeline.
- Provides an applied bridge between academic decision theory and baseball-operations interpretation.

**Project relevance:** EAGLE is a useful implementation benchmark for reproducing the structure of the Bayesian paper with faster machine-learning models.

### 3.3 Vock and Vock: causal evaluation of plate discipline

Vock and Vock frame plate discipline using potential outcomes and G-computation. Their main question is not simply whether an individual pitch should be swung at; instead, they estimate how a hitter's performance would change under another hitter's decision policy. Their example estimates how Starlin Castro might have performed had he adopted Andrew McCutchen's plate discipline.

This framework explicitly recognizes three correlated components:

- the pitches a hitter receives;
- the hitter's swing/take choices; and
- the hitter's ability after choosing to swing.

**Strengths**

- Establishes the causal distinction between approach and hitting ability.
- Supports policy-level counterfactual questions.
- Highlights why two separately fitted outcome models do not automatically identify causal swing/take effects.

**Limitations**

- The original analysis uses 2012–2014 PITCHf/x data.
- Its goal is evaluating alternative policies, not assigning a simple optimal-action value to every pitch.

**Project relevance:** This is the most important reference for addressing confounding, overlap, and counterfactual-policy evaluation.

### 3.4 SEAGER: selective aggression

Robert Orr's SEAGER (Selective Aggression Engagement Rate) evaluates whether hitters combine aggression on favorable pitches with selectivity on unfavorable pitches. The method estimates the average run values of swings and takes across count-location combinations and incorporates the probability that a taken pitch will be called a strike.

At a high level, SEAGER rewards swings when swinging has positive expected value and takes when taking avoids negative expected value. It produces an interpretable season-level description of approach.

**Strengths**

- Makes count and precise location central to valuation.
- Has a relatively intuitive baseball interpretation.
- Provides a stronger benchmark than O-Swing%, Z-Swing%, or Z-O-Swing%.

**Limitations**

- Primarily an aggregate selective-aggression measure rather than a fully individualized counterfactual action-value model.
- A league-average swing value may not represent what a specific hitter can do with the pitch.

**Project relevance:** A valuable leaderboard benchmark and an interpretability reference.

### 3.5 SwRV: expected run value of the swing decision

Drew Haugen's SwRV estimates expected swing run value and expected take run value, then scores the hitter's observed action. Public comparisons have reported that it predicts future wOBA more effectively than basic plate-discipline statistics while retaining a relationship with walk rate.

**Project relevance:** SwRV is conceptually close to pitch-level expected regret and should be included in any empirical benchmark set when its implementation details can be reproduced.

### 3.6 Nestico: XGBoost Decision Value

Nestico fits two XGBoost regressors:

- a take model; and
- a swing model.

The features are `plate_x`, `plate_z`, balls, and strikes. The target is an average change in run expectancy assigned to the observed outcome at the relevant count. Models trained on 2020–2022 data are applied to 2023 pitches. The outputs include In-Zone Awareness, Out-of-Zone Awareness, and an overall Decision Value per 100 pitches.

**Strengths**

- Fully reproducible public notebook.
- Simple feature set makes it an excellent baseline.
- Clearly demonstrates that the swing-value problem is harder than the take-value problem.

**Limitations**

- The swing model's reported RMSE is substantially higher than the take model's RMSE.
- Pitch location and count provide little information about the hitter's contact quality.
- Separate training on observed actions requires counterfactual extrapolation when comparing both actions on the same pitch.
- Averaging actual-pitch values can reward or punish hitters for the pitch distribution they face.

**Project relevance:** This should be reproduced as the primary machine-learning baseline before adding individualized or causal components.

### 3.7 SOTO and SOTO+

Salorio's Swing Or Take Only (SOTO) model uses separate XGBoost regressors for swings and takes. Its initial features include pitch location, count, and batter/pitcher handedness. Expected run values for a hitter's swings and takes are combined into xRV per 100 pitches.

SOTO+ places the result on a familiar plus scale:

- 100 represents league average;
- 10 points represent one standard deviation.

The original analysis compared SOTO with O-Swing% and Z-O-Swing% using descriptive and following-season relationships with BB%, wOBA, and ISO. It also examined year-to-year stickiness.

The original model identified the absence of individualized nitro zones as a limitation. A later version incorporated a hitter-specific nitro-zone indicator.

**Project relevance:** SOTO+ supplies a useful presentation scale, while the nitro-zone extension directly motivates individualized action values.

### 3.8 Later SOTO critique and lessons from model retirement

Salorio later reported that Zone% explained approximately 23% of variation in SOTO v2. Hitters who received fewer pitches in the strike zone could receive better grades because their pitch distributions created more valuable take opportunities. The author consequently retired SOTO.

The retrospective also emphasizes that:

- the value of taking is comparatively stable across hitters;
- swing value depends heavily on the hitter's physical and contact abilities;
- bat speed and other bat-tracking variables may improve swing-value estimation; and
- pitch-level interpretability is preferable to an aggregate-only framework.

**Project relevance:** This is an unusually useful negative result. It establishes the need to standardize the pitch distribution and distinguish decision quality from the opportunities a hitter receives.

### 3.9 Weighted Decision Value using linear weights

Creally proposes a transparent alternative using observed run values by count and five location regions:

1. heart;
2. inner shadow;
3. outer shadow;
4. chase; and
5. waste.

Average swing and take values are computed within every count-zone combination. A hitter's decisions can then be aggregated into Weighted Decision Value or a per-100-pitch rate.

**Strengths**

- Highly explainable.
- Easy to reproduce and audit.
- Separating inner and outer shadow pitches improves upon a simple in-zone/out-of-zone split.

**Limitations**

- Coarse location bins discard continuous spatial information.
- Estimates may be noisy in sparse count-zone cells.
- Does not fully personalize swing value.

**Project relevance:** This is the best non-ML baseline and should be implemented before more complex models.

### 3.10 Driveline's individualized approach

Driveline emphasizes that hitters should not receive equal credit for all in-zone swings because hitters differ in the locations they can damage. The analysis builds on the idea of comparing expected swing value with expected take value at a given location while making favorable swing regions hitter-specific.

**Project relevance:** This supports using an individualized nitro zone or a smooth hitter-location interaction rather than a universal ideal-swing region.

### 3.11 Game-theoretic and sequential approaches

Douglas et al. model the pitcher-batter interaction as a stochastic zero-sum game. Their system predicts pitch-location distributions and outcomes conditional on a swing, then solves for strategies intended to maximize or minimize on-base probability.

Related Markov decision process work treats the count and pitch sequence as a sequence of states rather than independent pitches. These approaches are valuable if the objective expands from retrospective decision valuation to optimal strategy against a particular pitcher.

**Project relevance:** This is a longer-term extension. The initial model can treat each pitch's state as sufficient, but future work could account for strategic adaptation and pitch sequencing.

## 4. Comparison of modeling philosophies

### 4.1 Classification-based metrics

Examples: O-Swing%, Z-Swing%, Z-O-Swing%, attack-zone swing rates.

These metrics are interpretable but assume that official-zone membership adequately describes pitch desirability. They ignore continuous location, count leverage, called-strike probability, and individualized damage ability.

### 4.2 Outcome-regression metrics

Examples: Nestico Decision Value and SOTO.

These methods estimate expected run value separately among observed swings and observed takes. They are flexible and relatively easy to implement but face action-selection bias and extrapolation when applied counterfactually.

### 4.3 Event-probability decomposition

Examples: EAGLE and Yee–Deshpande.

These frameworks model interpretable intermediate events—called strike, contact, miss, foul, and in-play outcomes—then combine them with run expectancy. They are easier to diagnose than a single direct run-value regression and can propagate uncertainty.

### 4.4 Causal policy evaluation

Example: Vock and Vock.

This approach asks what would happen under an alternative decision policy while adjusting for pitch distribution and hitter ability. It is the most appropriate framework for causal claims but requires stronger assumptions and careful overlap analysis.

### 4.5 Game-theoretic or sequential optimization

These methods model how pitchers and hitters adapt their strategies. They are theoretically appealing but substantially more complicated than a one-step decision-value model.

## 5. Proposed model direction

### 5.1 Take-value model

For ordinary non-swing pitches:

$$
Q_T(s)=P(CS\mid s)RE(s_{CS}) + [1-P(CS\mid s)]RE(s_B),
$$

where $CS$ is a called strike and $B$ is a called ball. Hit-by-pitch and other unusual events can be modeled separately or excluded under explicit rules.

Useful predictors include:

- normalized horizontal and vertical location;
- count;
- batter and pitcher handedness;
- catcher and umpire effects when available;
- pitch trajectory and movement variables;
- batter height or normalized strike-zone bounds; and
- game state if the target is contextual run expectancy.

### 5.2 Swing-value model

A decomposed version is:

$$
\begin{aligned}
Q_S(s)={}&P(Whiff\mid s)RE(s_{Whiff})\\
&+P(Foul\mid s)RE(s_{Foul})\\
&+P(BIP\mid s)E[RE(s_{BIP})\mid BIP,s].
\end{aligned}
$$

The in-play term can be further decomposed into expected contact quality or event probabilities. Relevant predictors include:

- pitch location, velocity, movement, and approach angles;
- count;
- hitter and pitcher handedness;
- hitter-level rolling contact, whiff, damage, and bat-speed estimates;
- pitch-type or raw-pitch-characteristic interactions;
- nitro-zone membership or continuous distance from the nitro-zone boundary; and
- hierarchical hitter-location effects.

### 5.3 Two versions of decision value

To separate two valid interpretations, report both:

1. **Generic decision value:** $Q_S$ uses league-average swing ability. This measures whether the decision would generally be advisable.
2. **Personalized decision value:** $Q_S$ uses the hitter's estimated abilities. This measures whether the action was appropriate for that particular hitter.

A powerful hitter may correctly swing at pitches that are negative-value opportunities for an average hitter. Reporting both versions makes that distinction visible.

### 5.4 Aggregation

Potential outputs include:

- expected decision regret per 100 pitches;
- correct-decision rate;
- regret-weighted correct-decision rate;
- chase regret;
- missed-opportunity regret;
- performance by count, pitch type, or location;
- generic versus personalized decision value; and
- standardized SOTO+-style scores, with 100 as league average and 10 points as one standard deviation.

Regret per 100 pitches should be the primary value measure. Correct-decision percentage treats a nearly indifferent decision and an obvious mistake as equally important.

## 6. Nitro-zone integration

An individualized nitro zone can be built from locations where the hitter has demonstrated high-quality contact. A convex hull provides an interpretable first representation, but it introduces several issues:

- sensitivity to outliers;
- instability for hitters with limited batted-ball samples;
- binary discontinuity at the hull boundary;
- possible inclusion of low-density interior space; and
- selection bias because the zone is estimated only from contacted pitches.

Possible refinements include:

- robust or trimmed convex hulls;
- kernel-density level sets;
- Gaussian-mixture contours;
- alpha shapes for non-convex regions;
- shrinkage toward a league-average zone; and
- a continuous distance-to-zone or expected-damage surface instead of a binary indicator.

The nitro zone should be estimated using training data only. Re-estimating it using validation or test-season batted balls would leak future information.

## 7. Major methodological risks

### 7.1 Missing counterfactual outcomes

For each pitch, only the outcome of the chosen action is observed. A swing model learns from pitches hitters selected for swings, and a take model learns from pitches selected for takes. The counterfactual application of either model may occur outside its observed support.

Recommended diagnostics and extensions:

- estimate the swing propensity $P(A=\text{swing}\mid s)$;
- inspect action overlap across location/count regions;
- trim or flag pitches with extreme propensities;
- compare direct outcome regression with inverse-propensity weighting; and
- consider augmented inverse-propensity weighting or another doubly robust estimator.

### 7.2 Opportunity-distribution bias

A hitter seeing many obvious balls has more easy take opportunities. A hitter seeing many competitive strikes faces harder decisions. Raw averages over actual pitches therefore combine decision ability with pitch opportunities.

Report both:

- **observed-opportunity value**, calculated over pitches actually faced; and
- **standardized-opportunity value**, calculated over a common league reference distribution.

### 7.3 Execution versus decision

Observed bat speed, contact point, attack angle, exit velocity, and launch angle occur during or after execution. Including the realized values can leak execution quality into the decision model. Prefer rolling or pre-pitch estimates of the hitter's distributions when the intended construct is pure decision quality.

### 7.4 Target construction

`delta_run_exp` is convenient but mixes decision quality with the realized result. For training labels, event-level run values or modeled future run expectancy should be constructed carefully and consistently across seasons.

Decide whether the model targets:

- context-neutral offensive talent;
- expected runs given base/out state; or
- win probability given score, inning, and leverage.

These are different estimands and may recommend different actions.

### 7.5 Data leakage and validation

Random pitch-level cross-validation can place pitches from the same hitter, game, or season in both training and validation sets. Prefer:

- chronological train/validation/test splits;
- held-out seasons;
- grouped splits by game or hitter where appropriate;
- calibration curves for called-strike, contact, and action-value probabilities; and
- evaluation by count, location, handedness, and pitch family.

## 8. Evaluation plan

### 8.1 Intermediate-model performance

- Brier score and log loss for called-strike probability.
- Brier score and log loss for contact/whiff probabilities.
- Calibration slope and reliability diagrams.
- RMSE or MAE for expected run-value components.
- Performance within spatial and count strata.

### 8.2 Decision-metric validity

- Year-to-year correlation.
- First-half to second-half reliability.
- Relationship with future BB%, K%, wOBA, xwOBA, and ISO.
- Incremental predictive value beyond O-Swing%, Z-Swing%, and Z-O-Swing%.
- Sensitivity to the distribution of pitches faced.
- Stability when the reference pitch distribution changes.
- Agreement and disagreement with SEAGER, SwRV, SOTO, and EAGLE-style decisions.

### 8.3 Stabilization

Cronbach's alpha can be used by dividing a hitter's chronological pitches into comparable samples and treating sample-level decision values as repeated measurements. However, the design must respect that pitchers, counts, and locations are not exchangeable.

Complement alpha with:

- split-half reliability with repeated random or stratified splits;
- intraclass correlation;
- bootstrap confidence intervals by hitter;
- reliability curves against number of pitches; and
- year-to-year or rolling-window reliability.

The stabilization threshold should be reported as a distribution or reliability curve rather than a single universal number whenever possible.

## 9. Recommended implementation sequence

1. Reproduce the five-zone, count-specific linear-weight baseline.
2. Reproduce Nestico's two-model XGBoost baseline.
3. Implement called-strike, contact, and contact-quality submodels.
4. Produce pitch-level $Q_S$, $Q_T$, value difference, confidence, and regret.
5. Add generic and personalized hitter-ability versions.
6. Add the individualized nitro-zone representation using training-only data.
7. Standardize all hitters over a common reference pitch distribution.
8. Add propensity and overlap diagnostics.
9. Compare observed-opportunity and standardized leaderboards.
10. Evaluate calibration, predictive validity, reliability, and stabilization.

## 10. Research gap and potential contribution

A useful contribution would combine elements that currently appear separately in public work:

- pitch-level counterfactual swing/take values;
- hitter-specific damage ability and nitro zones;
- uncertainty-aware regret rather than a binary correct/incorrect label;
- standardized pitch opportunities for fair hitter comparisons;
- explicit overlap or causal diagnostics;
- chronological out-of-sample evaluation; and
- transparent reliability/stabilization analysis.

The resulting metric could answer two distinct questions:

1. **Evaluation:** How much expected value did a hitter preserve or lose through his decisions on the pitches he actually faced?
2. **Skill estimation:** How good is the hitter's underlying decision ability after adjusting for the difficulty and distribution of those opportunities?

Keeping these questions separate would address one of the central weaknesses identified in the later SOTO analysis.

## 11. Recommended reading order

1. Yee and Deshpande — complete pitch-level statistical framework.
2. Mould and Anderson's EAGLE — applied event-tree implementation.
3. Vock and Vock — causal and counterfactual-policy framing.
4. Nestico — reproducible XGBoost baseline.
5. Salorio's original SOTO article — standardized metric and initial validation.
6. Salorio's later critique — opportunity bias and swing-value limitations.
7. Orr's SEAGER and Haugen's SwRV — public benchmark metrics.
8. Creally — transparent linear-weight baseline.
9. Driveline — individualized favorable zones.
10. Douglas et al. — game-theoretic extension.

## References

Creally, M. (2026). *Using Linear Weights to Evaluate Swing Decisions*. Medium.  
https://medium.com/@mattjcreally/using-linear-weights-to-evaluate-swing-decisions-956ea6c14105

Douglas, C., Witt, E., Bendy, M., & Vorobeychik, Y. (2021). *Computing an Optimal Pitching Strategy in a Baseball At-Bat*. arXiv:2110.04321.  
https://arxiv.org/abs/2110.04321

Driveline Baseball. (2019). *Quantifying Swing Decisions: An Individualized Approach*.  
https://drivelinebaseball.com/2019/07/quantifying-swing-decisions-an-individualized-approach/

Haugen, D. *SwRV swing-decision model*. Down on the Farm. The model and its relationship to SEAGER and SOTO are also summarized in Salorio's review below.  
https://downonthefarm.substack.com/

Major League Baseball. *Statcast Search CSV Documentation*.  
https://baseballsavant.mlb.com/csv-docs

Major League Baseball. *Baseball Savant: Statcast*.  
https://baseballsavant.mlb.com/

Mould, J., & Anderson, D. (2022). *Quantifying Hitter Plate Discipline with EAGLE: Part 1*. Baseball Prospectus.  
https://www.baseballprospectus.com/news/article/74173/quantifying-hitter-plate-discipline-with-eagle-part-1/

Mould, J., & Anderson, D. (2022). *Quantifying Hitter Plate Discipline with EAGLE: Part 2*. Baseball Prospectus.  
https://www.baseballprospectus.com/news/article/74214/quantifying-hitter-plate-discipline-with-eagle-part-2/

Nestico, T. (2023). *Modelling Batter Decision Value*. Medium.  
https://medium.com/@thomasjamesnestico/modelling-batter-decision-value-dac74c55e20a

Nestico, T. (2023). *decision_value: Modelling Batter Decision Value* [Jupyter notebook]. GitHub.  
https://github.com/tnestico/decision_value

Orr, R. (2023). *Quantifying the Corey Seager Approach*. Baseball Prospectus.  
https://www.baseballprospectus.com/news/article/86572/the-crooked-inning-corey-seager-rangers/

Orr, R. (2023). *SEAGER at the Team Level*. Baseball Prospectus.  
https://www.baseballprospectus.com/news/article/86926/the-crooked-inning-seager-at-the-team-level/

Salorio, A. (2024). *Introducing My Swing Decision Model*. Medium.  
https://medium.com/@adamsalorio/introducing-my-swing-decision-model-d0851ab37fb6

Salorio, A. *A Closer Look at Swing Decision Metrics: Should We Consider Bat Speed When Evaluating Swing Decisions?* Substack.  
https://adamsalorio.substack.com/p/a-closer-look-at-swing-decision-metrics

Vock, D. M., & Vock, L. F. B. (2018). Estimating the effect of plate discipline using a causal inference framework: An application of the G-computation algorithm. *Journal of Quantitative Analysis in Sports, 14*(2), 37–56.  
https://doi.org/10.1515/jqas-2016-0029

Yee, R., & Deshpande, S. K. (2024). Evaluating plate discipline in Major League Baseball with Bayesian additive regression trees. *Journal of Quantitative Analysis in Sports, 20*(1), 5–20.  
https://doi.org/10.1515/jqas-2023-0048

Open preprint: https://arxiv.org/abs/2305.05752  
Code: https://github.com/ryanyee3/plate_discipline_code

## Reference-management notes

- The peer-reviewed sources are Yee and Deshpande (2024) and Vock and Vock (2018).
- Douglas et al. is cited as an open preprint.
- EAGLE, SEAGER, SwRV, SOTO, Nestico's Decision Value, Driveline's approach, and Creally's weighted metric are public sabermetric or applied research rather than peer-reviewed journal articles.
- Verify publication dates and archived URLs before submitting this review as part of an academic manuscript.
