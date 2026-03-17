# Case B: RLHF with Curriculum Reward Shaping — Image Generation Prompt

## Prompt

A premium, modern data visualization infographic on a clean white background with subtle light-gray grid lines. The chart is a **line chart with square markers** showing progressive performance improvement across 5 data points on the X-axis (labeled "Self-Iteration Round").

**Overall title** at the top in bold dark navy sans-serif font: "Case B: RLHF with Curriculum-Based Reward Shaping for LLM Alignment"

**Y-axis:** "LLM Alignment Score (%)" ranging from 15% to 105%. **X-axis:** "Self-Iteration Round" with 5 labeled tick marks.

**Data points and line:**
- Point 0 (Baseline): 35.6% — large square marker, colored **slate gray** (#757575). X-label below: "Baseline" with a small gray play-button icon, subtitle "(Vanilla PPO)".
- Point 1 (Iter 1): 35.6% — large square marker, colored **slate gray** (#757575). X-label: "Iter 1" with a small gray pause icon, subtitle "(No Change)".
- Point 2 (Iter 2): 61.6% — large square marker, colored **emerald green** (#2E7D32). X-label: "Iter 2" with a small green sparkle/star icon, subtitle "(+Reward Model +Curriculum)".
- Point 3 (Iter 3): 63.0% — large square marker, colored **emerald green** (#2E7D32). X-label: "Iter 3" with a small green chart-trending-up icon, subtitle "(+Rank-Norm +Policy EMA)".
- Point 4 (Iter 4): 66.6% — large square marker, colored **emerald green** (#2E7D32). X-label: "Iter 4" with a small green shield-check icon, subtitle "(+Confidence Gating)".

**Connecting line:** Thick (3px) solid line in **deep purple** (#6A1B9A) connecting all 5 points in order. The area below the line (above the baseline value of 35.6%) is filled with a very light semi-transparent purple wash (#6A1B9A at 8% opacity).

**Annotations with callout arrows:**
- Near Point 1: A gray italic callout "No improvement (minor code fix)" with a thin gray arrow pointing down to Point 1. Include a small minus-circle icon.
- Near Point 2: A green callout box with text "+26.0 pts" in bold green, below it "+Learned reward model" and "+Curriculum scheduling" in smaller green text. A thin green arrow points from the callout to Point 2. Include a small upward-arrow icon and a tiny brain icon.
- Near Point 3: A smaller green callout with text "+1.4 pts" in green, below it "+Rank-norm +Policy EMA" in smaller text. A thin green arrow points to Point 3. Include a small upward-arrow icon.
- Near Point 4: A green callout box with text "+3.6 pts" in bold green, below it "+Confidence gating" and "+Mini-batch RM" in smaller green text. A thin green arrow points to Point 4. Include a small upward-arrow icon and a tiny lock/shield icon.

**Summary stats box:** Upper-left corner, a rounded rectangle with light purple background (#F3E5F5) and purple border (#6A1B9A), containing monospace text:
```
Baseline: 35.6%  →  Best: 66.6%
Improvement: +31.0 pts (87% rel.)
```

**Legend** at the bottom center with three items, each with a colored square swatch:
- Green square: "Improved"
- Red square: "Regressed (auto-recovered)"
- Gray square: "No change / Baseline"

**Style:** Clean, professional, tech-forward aesthetic. Use a modern sans-serif font (like Inter, SF Pro, or Helvetica Neue). Subtle drop shadows on the summary box and annotation callouts. Smooth anti-aliased lines. The overall feel should be suitable for a top-tier AI research company's product page or investor deck — polished, data-rich, and visually compelling. High contrast text. No 3D effects. Flat design with depth through subtle shadows and layering.

**Dimensions:** 1200 x 900 pixels, 2x retina resolution.
