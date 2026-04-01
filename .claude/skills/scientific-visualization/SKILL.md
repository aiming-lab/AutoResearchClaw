---
name: scientific-visualization
description: Publication-ready scientific figure design with matplotlib and seaborn. Use when creating journal submission figures with proper formatting, accessibility, and statistical annotations.
metadata:
  category: writing
  trigger-keywords: "figure,plot,chart,visualization,matplotlib,seaborn,colorblind,publication"
  applicable-stages: "14,17,22"
  priority: "3"
  version: "1.0"
  author: researchclaw
  references: "adapted from K-Dense-AI/claude-scientific-skills"
---

## Scientific Visualization Best Practice

### Figure Design Principles
1. Every figure must have a clear, self-contained message
2. Minimize chartjunk: remove gridlines, background shading, and 3D effects
3. Use direct labeling instead of legends when possible
4. Remove top and right spines for cleaner appearance
5. Ensure all text is readable at final print size (minimum 6pt font)

### Journal Figure Sizing
1. **Single column**: 3.3-3.5 inches (85-89 mm) wide
2. **1.5 column**: 4.5-5.5 inches (114-140 mm) wide
3. **Double column / full width**: 6.5-7.1 inches (165-180 mm) wide
4. Resolution: 300 DPI minimum for raster; prefer vector formats (PDF, EPS, SVG)
5. Check target journal author guidelines for exact specifications

### Colorblind-Safe Design
1. Use colorblind-friendly palettes: seaborn "colorblind", Okabe-Ito, viridis, cividis
2. NEVER rely on color alone — combine with shape, pattern, or line style
3. Avoid red-green combinations; prefer blue-orange or blue-yellow contrasts
4. Test figures with a colorblind simulator before submission
5. Ensure figures work in grayscale for print journals

### Multi-Panel Layouts
1. Label panels with uppercase letters: (A), (B), (C) in bold, top-left corner
2. Use consistent axis scales across panels when comparing related data
3. Share axes where appropriate to reduce redundancy
4. Maintain consistent font sizes and line widths across all panels
5. Use `plt.subplots()` with `constrained_layout=True` for automatic spacing

### Statistical Annotations on Figures
1. Show individual data points alongside summary statistics (box + strip plots)
2. Always include error bars; specify type in caption (SEM, SD, 95% CI)
3. Use significance brackets with stars: * p<.05, ** p<.01, *** p<.001
4. Annotate effect sizes or key statistics directly on the figure when helpful
5. Never use bar charts for small-n data — use dot plots or box plots instead

### Export and Quality Checklist
1. Save in vector format (PDF/SVG) for line art; TIFF/PNG for photographs
2. Embed fonts or convert text to outlines for cross-platform consistency
3. Verify axis labels include units in parentheses: "Time (s)", "Force (N)"
4. Ensure figure caption fully explains all symbols, abbreviations, and panels
5. Check that color-coded elements match between figure and caption
