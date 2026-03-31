---
name: statistical-reporting
description: Statistical test selection, assumption checking, and APA-formatted reporting. Use when analyzing experimental results or writing results sections.
metadata:
  category: writing
  trigger-keywords: "statistic,hypothesis test,p-value,regression,ANOVA,t-test,effect size,confidence interval"
  applicable-stages: "14,17"
  priority: "3"
  version: "1.0"
  author: researchclaw
  references: "adapted from K-Dense-AI/claude-scientific-skills"
---

## Statistical Reporting Best Practice

### Test Selection Quick Reference
1. **Comparing two groups (independent, normal)**: Independent t-test
2. **Comparing two groups (independent, non-normal)**: Mann-Whitney U test
3. **Comparing two groups (paired, normal)**: Paired t-test
4. **Comparing two groups (paired, non-normal)**: Wilcoxon signed-rank test
5. **Comparing 3+ groups (independent, normal)**: One-way ANOVA + post-hoc
6. **Comparing 3+ groups (non-normal)**: Kruskal-Wallis test
7. **Relationship between continuous variables**: Pearson or Spearman correlation
8. **Categorical outcomes**: Chi-square or Fisher's exact test
9. **Predicting continuous outcome**: Linear regression
10. **Predicting binary outcome**: Logistic regression

### Assumption Checking
1. **Normality**: Shapiro-Wilk test (n < 50) or visual Q-Q plots
2. **Homogeneity of variance**: Levene's test before t-tests and ANOVA
3. **Independence**: Verify study design ensures independent observations
4. **Linearity**: Scatter plots and residual plots for regression
5. **Multicollinearity**: VIF < 5 for multiple regression predictors
6. When assumptions are violated, use non-parametric alternatives or robust methods

### APA Reporting Format
1. **t-test**: t(df) = X.XX, p = .XXX, d = X.XX
2. **ANOVA**: F(df_between, df_within) = X.XX, p = .XXX, eta-squared = .XX
3. **Correlation**: r(df) = .XX, p = .XXX [95% CI: .XX, .XX]
4. **Chi-square**: chi-square(df, N = XXX) = X.XX, p = .XXX
5. **Regression**: beta = X.XX, SE = X.XX, t = X.XX, p = .XXX
6. Always report exact p-values (not "p < .05") unless p < .001
7. Use leading zero for values that can exceed 1 (e.g., t = 0.50) but not for those bounded by 1 (e.g., p = .032, r = .45)

### Effect Sizes
1. ALWAYS report effect sizes alongside p-values
2. Cohen's d for group comparisons: small = 0.2, medium = 0.5, large = 0.8
3. Eta-squared for ANOVA: small = .01, medium = .06, large = .14
4. R-squared for regression: report adjusted R-squared for multiple predictors
5. Odds ratios for logistic regression with 95% confidence intervals
6. Distinguish statistical significance from practical significance

### Common Mistakes to Avoid
1. Never say "the results were not significant, therefore there is no effect"
2. Do not confuse correlation with causation in observational data
3. Apply multiple comparison corrections (Bonferroni, FDR) when running many tests
4. Report confidence intervals, not just point estimates
5. State whether tests are one-tailed or two-tailed and justify the choice
