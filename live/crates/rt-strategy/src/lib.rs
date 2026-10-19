use rt_domain::require_positive;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const FEATURE_VERSION: &str = "v2_numeric_cross_section";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortfolioRules {
    pub universe_size: usize,
    pub tail_count: usize,
    pub minimum_universe: usize,
    pub inverse_volatility: bool,
    pub dollar_neutral: bool,
    pub btc_beta_neutral: bool,
    pub smoothing_days: usize,
    /// Blend weight of the book that is also neutral to the alt index (and size); 0 disables it.
    #[serde(default)]
    pub alt_beta_neutral: f64,
    /// Also neutralize the cross-sectional size z-score inside the alt-neutral book.
    #[serde(default)]
    pub size_neutral: bool,
}

/// Per-asset exposures, known at the decision close, for the alt-neutral book.
#[derive(Debug, Clone, Copy)]
pub struct FactorExposures<'a> {
    pub alt_beta: &'a [f64],
    pub size: &'a [f64],
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OverlayRules {
    pub enabled: bool,
    pub median_lookback: usize,
    pub median_min_periods: usize,
    pub low_vol_multiplier: f64,
    pub high_vol_multiplier: f64,
}

impl Default for OverlayRules {
    fn default() -> Self {
        Self {
            enabled: true,
            median_lookback: 180,
            median_min_periods: 90,
            low_vol_multiplier: 1.10,
            high_vol_multiplier: 0.75,
        }
    }
}

impl Default for PortfolioRules {
    fn default() -> Self {
        Self {
            universe_size: 75,
            tail_count: 6,
            minimum_universe: 20,
            inverse_volatility: true,
            dollar_neutral: true,
            btc_beta_neutral: true,
            smoothing_days: 21,
            alt_beta_neutral: 0.5,
            size_neutral: true,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StrategyError {
    #[error("all input arrays must have an equal length")]
    LengthMismatch,
    #[error("invalid portfolio rule: {0}")]
    InvalidRule(String),
}

pub fn airok_feature_names() -> Vec<String> {
    let raw = [
        "ret_1",
        "ret_2",
        "ret_3",
        "ret_5",
        "ret_7",
        "ret_14",
        "ret_21",
        "ret_30",
        "ret_60",
        "ret_90",
        "ret_120",
        "vol_5",
        "vol_10",
        "vol_20",
        "vol_60",
        "downvol_20",
        "downvol_60",
        "range_1",
        "range_5",
        "range_20",
        "close_loc_1",
        "close_loc_5",
        "body_1",
        "body_5",
        "volume_5_20",
        "volume_20_60",
        "dvol_level",
        "amihud_20",
        "dist_high_20",
        "dist_high_60",
        "dist_high_120",
        "beta_60",
        "resmom_21",
        "resmom_60",
        "short_reversal",
        "mom_21_ex_3",
        "mom_60_ex_7",
    ];
    let mut names = raw
        .iter()
        .map(|name| format!("{name}_xrank"))
        .collect::<Vec<_>>();
    names.extend(
        [
            "mkt_btc_ret7",
            "mkt_btc_ret21",
            "mkt_btc_vol20",
            "mkt_breadth",
            "mkt_dispersion",
        ]
        .iter()
        .map(ToString::to_string),
    );
    names
}

/// Match pandas rank(method="average", pct=True) for finite eligible observations.
pub fn cross_section_rank_percentile(
    values: &[f64],
    eligible: &[bool],
) -> Result<Vec<f64>, StrategyError> {
    if values.len() != eligible.len() {
        return Err(StrategyError::LengthMismatch);
    }
    let mut indexed = values
        .iter()
        .enumerate()
        .filter(|(index, value)| eligible[*index] && value.is_finite())
        .map(|(index, value)| (index, *value))
        .collect::<Vec<_>>();
    indexed.sort_by(|left, right| left.1.total_cmp(&right.1));
    let count = indexed.len();
    let mut output = vec![f64::NAN; values.len()];
    let mut start = 0;
    while start < count {
        let mut end = start + 1;
        while end < count && indexed[end].1.total_cmp(&indexed[start].1) == Ordering::Equal {
            end += 1;
        }
        let rank = (start + 1 + end) as f64 / 2.0;
        for (index, _) in &indexed[start..end] {
            output[*index] = rank / count as f64;
        }
        start = end;
    }
    Ok(output)
}

/// Aggregate per-seed raw model scores exactly as the frozen `median_rank` ensemble.
pub fn median_rank_ensemble(
    seed_scores: &[Vec<f64>],
    eligible: &[bool],
) -> Result<Vec<f64>, StrategyError> {
    if seed_scores.is_empty()
        || seed_scores
            .iter()
            .any(|scores| scores.len() != eligible.len())
    {
        return Err(StrategyError::LengthMismatch);
    }
    let ranks = seed_scores
        .iter()
        .map(|scores| cross_section_rank_percentile(scores, eligible))
        .collect::<Result<Vec<_>, _>>()?;
    // The research ensemble stores seed ranks as float32 and takes numpy's float32 nanmedian.
    // Matching that rounding matters: equal true ranks must tie exactly so that the stable
    // sort orders them by column, as numpy's stable argsort does.
    let mut result = vec![f64::NAN; eligible.len()];
    for index in 0..eligible.len() {
        let mut values = ranks
            .iter()
            .map(|rank| rank[index] as f32)
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if values.is_empty() {
            continue;
        }
        values.sort_by(f32::total_cmp);
        let middle = values.len() / 2;
        result[index] = f64::from(if values.len().is_multiple_of(2) {
            (values[middle - 1] + values[middle]) / 2.0
        } else {
            values[middle]
        });
    }
    Ok(result)
}

/// Cross-sectional z-score over eligible finite values (population std); missing results are zero.
pub fn cross_sectional_z(values: &[f64], eligible: &[bool]) -> Vec<f64> {
    let members = values
        .iter()
        .zip(eligible)
        .filter(|(value, eligible)| **eligible && value.is_finite())
        .map(|(value, _)| *value)
        .collect::<Vec<_>>();
    if members.is_empty() {
        return vec![0.0; values.len()];
    }
    let mean = members.iter().sum::<f64>() / members.len() as f64;
    let std = (members
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / members.len() as f64)
        .sqrt();
    let scale = if std > 1e-12 { std } else { 1.0 };
    values
        .iter()
        .map(|value| {
            let z = (value - mean) / scale;
            if z.is_finite() { z } else { 0.0 }
        })
        .collect()
}

/// Build one decision's raw long/short weights.
///
/// `trade_mask` restricts the tradable names (e.g. Copy Trading); the liquidity-ranked
/// universe is then formed among the allowed names only.
pub fn build_daily_weights(
    scores: &[f64],
    eligible: &[bool],
    liquidity_rank: &[f64],
    beta: &[f64],
    volatility: &[f64],
    exposures: Option<FactorExposures<'_>>,
    trade_mask: Option<&[bool]>,
    rules: &PortfolioRules,
) -> Result<Vec<f64>, StrategyError> {
    let count = scores.len();
    let mut lengths = vec![
        eligible.len(),
        liquidity_rank.len(),
        beta.len(),
        volatility.len(),
    ];
    if let Some(exposures) = exposures {
        lengths.extend([exposures.alt_beta.len(), exposures.size.len()]);
    }
    if let Some(mask) = trade_mask {
        lengths.push(mask.len());
    }
    if lengths.iter().any(|length| *length != count) {
        return Err(StrategyError::LengthMismatch);
    }
    if rules.tail_count == 0 || rules.universe_size == 0 || rules.smoothing_days == 0 {
        return Err(StrategyError::InvalidRule(
            "positive sizes are required".to_owned(),
        ));
    }
    if !(0.0..=1.0).contains(&rules.alt_beta_neutral) {
        return Err(StrategyError::InvalidRule(
            "alt_beta_neutral must be within [0, 1]".to_owned(),
        ));
    }
    let alt_exposures = if rules.alt_beta_neutral > 0.0 {
        Some(exposures.ok_or_else(|| {
            StrategyError::InvalidRule("alt-index neutralization needs factor exposures".to_owned())
        })?)
    } else {
        None
    };
    let valid = |index: &usize| {
        eligible[*index]
            && liquidity_rank[*index].is_finite()
            && scores[*index].is_finite()
            && beta[*index].is_finite()
            && volatility[*index].is_finite()
            && volatility[*index] > 1e-5
    };
    let mut candidates = match trade_mask {
        None => (0..count)
            .filter(|index| valid(index) && liquidity_rank[*index] <= rules.universe_size as f64)
            .collect::<Vec<_>>(),
        Some(mask) => {
            let mut allowed = (0..count)
                .filter(|index| valid(index) && mask[*index])
                .collect::<Vec<_>>();
            allowed.sort_by(|left, right| liquidity_rank[*left].total_cmp(&liquidity_rank[*right]));
            allowed.truncate(rules.universe_size);
            allowed.sort_unstable();
            allowed
        }
    };
    if candidates.len() < rules.minimum_universe {
        return Ok(vec![0.0; count]);
    }
    candidates.sort_by(|left, right| scores[*left].total_cmp(&scores[*right]));
    let tail = rules.tail_count.min(candidates.len() / 2);
    if tail == 0 {
        return Ok(vec![0.0; count]);
    }
    let shorts = &candidates[..tail];
    let longs = &candidates[candidates.len() - tail..];
    let mut weights = vec![0.0; count];
    let long_side = side_weights(longs, volatility, rules.inverse_volatility)?;
    let short_side = side_weights(shorts, volatility, rules.inverse_volatility)?;
    for (index, side_weight) in longs.iter().zip(long_side) {
        weights[*index] = 0.5 * side_weight;
    }
    for (index, side_weight) in shorts.iter().zip(short_side) {
        weights[*index] = -0.5 * side_weight;
    }
    if rules.dollar_neutral || rules.btc_beta_neutral {
        let raw = weights.clone();
        project_dollar_and_beta(&mut weights, beta);
        if let Some(exposures) = alt_exposures {
            let mut factors = vec![exposures.alt_beta];
            if rules.size_neutral {
                factors.push(exposures.size);
            }
            let full = project_onto_complement(&raw, beta, &factors);
            let strength = rules.alt_beta_neutral;
            for (weight, neutral) in weights.iter_mut().zip(full) {
                *weight = (1.0 - strength) * *weight + strength * neutral;
            }
            let gross = weights.iter().map(|value| value.abs()).sum::<f64>();
            if gross > 1e-8 {
                for value in &mut weights {
                    *value /= gross;
                }
            }
        }
    }
    Ok(weights)
}

pub fn smooth_fixed_window(signals: &[Vec<f64>], days: usize) -> Result<Vec<f64>, StrategyError> {
    if signals.is_empty() {
        return Ok(Vec::new());
    }
    let width = signals[0].len();
    if days == 0 || signals.iter().any(|signal| signal.len() != width) {
        return Err(StrategyError::LengthMismatch);
    }
    let start = signals.len().saturating_sub(days);
    let mut output = vec![0.0; width];
    for signal in &signals[start..] {
        for (weight, value) in output.iter_mut().zip(signal) {
            *weight += value / days as f64;
        }
    }
    Ok(output)
}

/// Return the next-open target after applying the source simulation's volatility overlay.
/// `btc_volatility` must end with the volatility for the just-confirmed decision candle.
pub fn apply_volatility_overlay(
    weights: &mut [f64],
    btc_volatility: &[f64],
    rules: OverlayRules,
) -> Result<f64, StrategyError> {
    if !rules.enabled {
        return Ok(1.0);
    }
    if rules.median_lookback == 0 || rules.median_min_periods == 0 {
        return Err(StrategyError::InvalidRule(
            "overlay lookback and minimum periods must be positive".to_owned(),
        ));
    }
    let Some(current) = btc_volatility
        .last()
        .copied()
        .filter(|value| value.is_finite())
    else {
        return Ok(1.0);
    };
    let end = btc_volatility.len().saturating_sub(1);
    let start = end.saturating_sub(rules.median_lookback);
    let mut history = btc_volatility[start..end]
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if history.len() < rules.median_min_periods {
        return Ok(1.0);
    }
    history.sort_by(f64::total_cmp);
    let middle = history.len() / 2;
    let median = if history.len().is_multiple_of(2) {
        (history[middle - 1] + history[middle]) / 2.0
    } else {
        history[middle]
    };
    let multiplier = if current < median {
        rules.low_vol_multiplier
    } else {
        rules.high_vol_multiplier
    };
    for weight in weights {
        *weight *= multiplier;
    }
    Ok(multiplier)
}

fn side_weights(
    indices: &[usize],
    volatility: &[f64],
    inverse_volatility: bool,
) -> Result<Vec<f64>, StrategyError> {
    if !inverse_volatility {
        return Ok(vec![1.0 / indices.len() as f64; indices.len()]);
    }
    let mut values = indices
        .iter()
        .map(|index| volatility[*index])
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let low = quantile_linear(&values, 0.10);
    let high = quantile_linear(&values, 0.90);
    let clipped_low = low.max(1e-5);
    let clipped_high = high.max(clipped_low + 1e-5);
    let mut weights = indices
        .iter()
        .map(|index| 1.0 / volatility[*index].clamp(clipped_low, clipped_high))
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    require_positive(sum, "inverse volatility side weight sum")
        .map_err(|error| StrategyError::InvalidRule(error.to_string()))?;
    for value in &mut weights {
        *value /= sum;
    }
    Ok(weights)
}

fn quantile_linear(values: &[f64], percentile: f64) -> f64 {
    let index = percentile * (values.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    values[lower] + (values[upper] - values[lower]) * (index - lower as f64)
}

/// Remove the ridge projection of the selected weights onto [1, beta, factors...] and
/// normalize gross to one, exactly as the research `_project_dollar_and_beta` with extra betas.
fn project_onto_complement(weights: &[f64], beta: &[f64], factors: &[&[f64]]) -> Vec<f64> {
    let selected = weights
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0.0).then_some(index))
        .collect::<Vec<_>>();
    let mut output = vec![0.0; weights.len()];
    if selected.len() < 4 {
        return output;
    }
    let finite_or_zero = |value: f64| if value.is_finite() { value } else { 0.0 };
    let mut rows = vec![
        vec![1.0; selected.len()],
        selected
            .iter()
            .map(|index| beta[*index])
            .collect::<Vec<_>>(),
    ];
    for factor in factors {
        rows.push(
            selected
                .iter()
                .map(|index| finite_or_zero(factor[*index]))
                .collect(),
        );
    }
    let raw = selected
        .iter()
        .map(|index| weights[*index])
        .collect::<Vec<_>>();
    let k = rows.len();
    let dot = |left: &[f64], right: &[f64]| left.iter().zip(right).map(|(a, b)| a * b).sum::<f64>();
    let mut gram = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            gram[i][j] = dot(&rows[i], &rows[j]) + if i == j { 1e-6 } else { 0.0 };
        }
    }
    let rhs = rows.iter().map(|row| dot(row, &raw)).collect::<Vec<_>>();
    let Some(coefficients) = solve_linear(gram, rhs) else {
        return output;
    };
    let neutral = (0..selected.len())
        .map(|position| {
            raw[position]
                - (0..k)
                    .map(|factor| rows[factor][position] * coefficients[factor])
                    .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let gross = neutral.iter().map(|value| value.abs()).sum::<f64>();
    if gross > 1e-8 && gross.is_finite() {
        for (index, value) in selected.iter().zip(neutral) {
            output[*index] = value / gross;
        }
    }
    output
}

/// Gaussian elimination with partial pivoting (the LU scheme numpy's `solve` uses).
fn solve_linear(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for column in 0..n {
        let pivot = (column..n).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right][column].abs())
        })?;
        if !matrix[pivot][column].is_finite() || matrix[pivot][column].abs() <= 1e-300 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        for row in column + 1..n {
            let factor = matrix[row][column] / matrix[column][column];
            let (upper, lower) = matrix.split_at_mut(row);
            for (target, source) in lower[0][column..].iter_mut().zip(&upper[column][column..]) {
                *target -= factor * source;
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let tail = (row + 1..n)
            .map(|inner| matrix[row][inner] * solution[inner])
            .sum::<f64>();
        solution[row] = (rhs[row] - tail) / matrix[row][row];
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

fn project_dollar_and_beta(weights: &mut [f64], beta: &[f64]) {
    let selected = weights
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0.0).then_some(index))
        .collect::<Vec<_>>();
    if selected.len() < 4 {
        weights.fill(0.0);
        return;
    }
    let n = selected.len() as f64;
    let beta_sum = selected.iter().map(|index| beta[*index]).sum::<f64>();
    let beta_squared_sum = selected
        .iter()
        .map(|index| beta[*index].powi(2))
        .sum::<f64>();
    let raw_sum = selected.iter().map(|index| weights[*index]).sum::<f64>();
    let raw_beta_sum = selected
        .iter()
        .map(|index| beta[*index] * weights[*index])
        .sum::<f64>();
    let a = n + 1e-6;
    let b = beta_sum;
    let d = beta_squared_sum + 1e-6;
    let determinant = a * d - b * b;
    if !determinant.is_finite() || determinant.abs() <= 1e-12 {
        weights.fill(0.0);
        return;
    }
    let correction_constant = (d * raw_sum - b * raw_beta_sum) / determinant;
    let correction_beta = (-b * raw_sum + a * raw_beta_sum) / determinant;
    for index in selected {
        weights[index] -= correction_constant + correction_beta * beta[index];
    }
    let gross = weights.iter().map(|value| value.abs()).sum::<f64>();
    if gross <= 1e-8 || !gross.is_finite() {
        weights.fill(0.0);
        return;
    }
    for value in weights {
        *value /= gross;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FactorExposures, OverlayRules, PortfolioRules, airok_feature_names,
        apply_volatility_overlay, build_daily_weights, cross_section_rank_percentile,
        cross_sectional_z, median_rank_ensemble,
    };

    fn sample(count: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let scores = (0..count)
            .map(|value| ((value * 7) % count) as f64)
            .collect();
        let beta = (0..count).map(|value| 0.5 + value as f64 / 20.0).collect();
        let alt = (0..count)
            .map(|value| 1.0 + ((value * 3) % 5) as f64 / 4.0)
            .collect();
        let size = (0..count)
            .map(|value| (value as f64 - 10.0) / 6.0)
            .collect();
        (scores, beta, alt, size)
    }

    #[test]
    fn full_alt_neutral_book_has_no_alt_or_size_exposure() {
        let (scores, beta, alt, size) = sample(24);
        let weights = build_daily_weights(
            &scores,
            &[true; 24],
            &[1.0; 24],
            &beta,
            &[0.02; 24],
            Some(FactorExposures {
                alt_beta: &alt,
                size: &size,
            }),
            None,
            &PortfolioRules {
                tail_count: 4,
                alt_beta_neutral: 1.0,
                ..PortfolioRules::default()
            },
        )
        .expect("weights");
        let exposure = |factor: &[f64]| {
            weights
                .iter()
                .zip(factor)
                .map(|(w, f)| w * f)
                .sum::<f64>()
                .abs()
        };
        assert!(weights.iter().sum::<f64>().abs() < 1e-5);
        assert!(exposure(&beta) < 1e-5);
        assert!(exposure(&alt) < 1e-5);
        assert!(exposure(&size) < 1e-5);
        assert!((weights.iter().map(|value| value.abs()).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn partial_neutralization_blends_both_books_at_unit_gross() {
        let (scores, beta, alt, size) = sample(24);
        let build = |strength: f64| {
            build_daily_weights(
                &scores,
                &[true; 24],
                &[1.0; 24],
                &beta,
                &[0.02; 24],
                Some(FactorExposures {
                    alt_beta: &alt,
                    size: &size,
                }),
                None,
                &PortfolioRules {
                    tail_count: 4,
                    alt_beta_neutral: strength,
                    ..PortfolioRules::default()
                },
            )
            .expect("weights")
        };
        let (btc_only, full, half) = (build(0.0), build(1.0), build(0.5));
        let blended = btc_only
            .iter()
            .zip(&full)
            .map(|(a, b)| 0.5 * a + 0.5 * b)
            .collect::<Vec<_>>();
        let gross = blended.iter().map(|value| value.abs()).sum::<f64>();
        for (expected, actual) in blended.iter().zip(&half) {
            assert!((expected / gross - actual).abs() < 1e-12);
        }
    }

    #[test]
    fn trade_mask_forms_the_liquid_universe_after_filtering() {
        let count = 30;
        let scores = (0..count).map(|value| value as f64).collect::<Vec<_>>();
        let rank = (0..count)
            .map(|value| value as f64 + 1.0)
            .collect::<Vec<_>>();
        let mask = (0..count).map(|value| value % 2 == 0).collect::<Vec<_>>();
        let weights = build_daily_weights(
            &scores,
            &vec![true; count],
            &rank,
            &vec![1.0; count],
            &vec![0.02; count],
            None,
            Some(&mask),
            &PortfolioRules {
                universe_size: 10,
                tail_count: 2,
                minimum_universe: 4,
                alt_beta_neutral: 0.0,
                btc_beta_neutral: false,
                dollar_neutral: false,
                ..PortfolioRules::default()
            },
        )
        .expect("weights");
        // Allowed names are the even columns; the 10 most liquid of them are 0, 2, ..., 18.
        let held = (0..count)
            .filter(|index| weights[*index] != 0.0)
            .collect::<Vec<_>>();
        assert_eq!(held, vec![0, 2, 16, 18]);
    }

    #[test]
    fn z_score_uses_eligible_names_and_zeroes_missing_values() {
        let z = cross_sectional_z(&[1.0, 3.0, 100.0, f64::NAN], &[true, true, false, true]);
        assert_eq!(z[0], -1.0);
        assert_eq!(z[1], 1.0);
        assert_eq!(z[2], 98.0);
        assert_eq!(z[3], 0.0);
    }

    #[test]
    fn feature_contract_matches_current_airok_width() {
        let names = airok_feature_names();
        assert_eq!(names.len(), 42);
        assert_eq!(names.first().expect("first"), "ret_1_xrank");
        assert_eq!(names.last().expect("last"), "mkt_dispersion");
    }

    #[test]
    fn average_rank_ties_match_pandas_semantics() {
        let ranks =
            cross_section_rank_percentile(&[1.0, 2.0, 2.0, 4.0], &[true; 4]).expect("ranks");
        assert_eq!(ranks, vec![0.25, 0.625, 0.625, 1.0]);
    }

    #[test]
    fn weights_are_dollar_and_beta_neutral() {
        let scores = (0..20).map(|value| value as f64).collect::<Vec<_>>();
        let beta = (0..20).map(|value| value as f64 / 10.0).collect::<Vec<_>>();
        let weights = build_daily_weights(
            &scores,
            &[true; 20],
            &[1.0; 20],
            &beta,
            &[0.02; 20],
            None,
            None,
            &PortfolioRules {
                tail_count: 3,
                alt_beta_neutral: 0.0,
                ..PortfolioRules::default()
            },
        )
        .expect("weights");
        // The source strategy uses a 1e-6 ridge in the two-factor projection,
        // so residual dollar/beta exposure is intentionally small rather than exact zero.
        assert!(weights.iter().sum::<f64>().abs() < 3e-6);
        assert!(
            weights
                .iter()
                .zip(beta)
                .map(|(weight, beta)| weight * beta)
                .sum::<f64>()
                .abs()
                < 3e-6
        );
        assert!((weights.iter().map(|value| value.abs()).sum::<f64>() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn overlay_excludes_the_current_day_from_its_median() {
        let mut weights = vec![0.5, -0.5];
        let multiplier = apply_volatility_overlay(
            &mut weights,
            &[0.01, 0.02, 0.03],
            OverlayRules {
                median_lookback: 2,
                median_min_periods: 2,
                low_vol_multiplier: 1.1,
                high_vol_multiplier: 0.75,
                ..OverlayRules::default()
            },
        )
        .expect("overlay computes");
        assert_eq!(multiplier, 0.75);
        assert_eq!(weights, vec![0.375, -0.375]);
    }

    #[test]
    fn ensemble_takes_the_median_of_each_seed_rank() {
        let result = median_rank_ensemble(
            &[
                vec![1.0, 2.0, 3.0],
                vec![3.0, 2.0, 1.0],
                vec![1.0, 3.0, 2.0],
            ],
            &[true, true, true],
        )
        .expect("ensemble");
        let expected = [1.0_f32 / 3.0, 2.0 / 3.0, 2.0 / 3.0].map(f64::from);
        assert_eq!(result, expected);
    }

    #[test]
    fn even_seed_medians_round_like_float32_numpy() {
        // True medians of both assets are 0.06; float32 rounding must make them tie exactly.
        let eligible = vec![true; 150];
        let seed = |first: usize, second: usize| {
            (0..150)
                .map(|index| match index {
                    0 => first as f64,
                    1 => second as f64,
                    other => 1000.0 + other as f64,
                })
                .collect::<Vec<_>>()
        };
        let result = median_rank_ensemble(&[seed(8, 9), seed(10, 9)], &eligible).expect("ensemble");
        assert_eq!(result[0], result[1]);
    }
}
