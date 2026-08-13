use anyhow::{Context, Result, bail};
use lightgbm3::Booster;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub bundle_id: String,
    pub backend: String,
    pub horizon_days: u32,
    pub aggregation: String,
    pub cutoff_date: String,
    pub feature_schema_sha256: String,
    pub universe_sha256: String,
    pub models: Vec<ModelFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub seed: u64,
    pub file: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureContract {
    pub version: String,
    pub names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniverseContract {
    pub symbols: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BundleMetadata {
    pub root: PathBuf,
    pub manifest: ModelManifest,
    pub features: FeatureContract,
    pub universe: UniverseContract,
}

pub fn inspect_bundle(path: impl AsRef<Path>) -> Result<BundleMetadata> {
    let root = path
        .as_ref()
        .canonicalize()
        .with_context(|| format!("resolve model bundle {}", path.as_ref().display()))?;
    let manifest: ModelManifest = read_json(root.join("manifest.json"))?;
    let features: FeatureContract = read_json(root.join("feature_contract.json"))?;
    let universe: UniverseContract = read_json(root.join("universe.json"))?;
    if manifest.backend != "lightgbm" {
        bail!(
            "unsupported model backend {:?}; expected lightgbm",
            manifest.backend
        );
    }
    if manifest.aggregation != "median_rank" {
        bail!(
            "unsupported seed aggregation {:?}; expected median_rank",
            manifest.aggregation
        );
    }
    if manifest.models.is_empty() {
        bail!("model bundle contains no seed models");
    }
    if sha256_json(&features)? != manifest.feature_schema_sha256 {
        bail!("feature contract hash does not match manifest");
    }
    if sha256_json(&universe)? != manifest.universe_sha256 {
        bail!("universe contract hash does not match manifest");
    }
    for model in &manifest.models {
        let source = root.join(&model.file);
        if sha256_file(&source)? != model.sha256 {
            bail!("model hash mismatch for {}", source.display());
        }
    }
    Ok(BundleMetadata {
        root,
        manifest,
        features,
        universe,
    })
}

/// A statically linked `lightgbm3` runtime for production inference.
/// Training and bundle creation are intentionally owned by the Python research protocol.
#[derive(Debug, Clone, Copy, Default)]
pub struct LightGbmLibrary;

impl LightGbmLibrary {
    pub fn linked() -> Self {
        Self
    }

    pub fn load_booster(&self, model_path: impl AsRef<Path>) -> Result<NativeBooster> {
        let model_path = model_path.as_ref();
        let source = model_path
            .to_str()
            .with_context(|| format!("non-UTF8 model path {}", model_path.display()))?;
        let booster = Booster::from_file(source)
            .with_context(|| format!("load LightGBM model {}", model_path.display()))?;
        Ok(NativeBooster {
            booster,
            feature_count: None,
        })
    }
}

pub struct NativeBooster {
    booster: Booster,
    feature_count: Option<usize>,
}

impl NativeBooster {
    pub fn predict_rows(&mut self, rows: &[f32], feature_count: usize) -> Result<Vec<f64>> {
        if feature_count == 0 || !rows.len().is_multiple_of(feature_count) {
            bail!("input matrix is not a whole number of non-empty feature rows");
        }
        if let Some(expected) = self.feature_count {
            if expected != feature_count {
                bail!("feature count changed from {expected} to {feature_count}");
            }
        } else {
            self.feature_count = Some(feature_count);
        }
        let output = self
            .booster
            .predict(
                rows,
                i32::try_from(feature_count).context("feature count exceeds LightGBM range")?,
                true,
            )
            .context("predict LightGBM scores")?;
        if output.len() != rows.len() / feature_count {
            bail!(
                "LightGBM produced {} predictions for {} rows",
                output.len(),
                rows.len() / feature_count
            );
        }
        Ok(output)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: PathBuf) -> Result<T> {
    let source = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&source).with_context(|| format!("parse {}", path.display()))
}

pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let bytes =
        fs::read(path.as_ref()).with_context(|| format!("read {}", path.as_ref().display()))?;
    Ok(hex_digest(&bytes))
}

pub fn sha256_json<T: Serialize>(value: &T) -> Result<String> {
    Ok(hex_digest(&serde_json::to_vec(value)?))
}
fn hex_digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::{FeatureContract, LightGbmLibrary, UniverseContract, sha256_json};

    #[test]
    fn stable_contract_hashes_change_when_schema_changes() {
        let left = FeatureContract {
            version: "v1".to_owned(),
            names: vec!["feature_a".to_owned()],
        };
        let right = FeatureContract {
            version: "v1".to_owned(),
            names: vec!["feature_b".to_owned()],
        };
        assert_ne!(
            sha256_json(&left).expect("hash"),
            sha256_json(&right).expect("hash")
        );
        assert_eq!(
            sha256_json(&UniverseContract {
                symbols: vec!["BTC".to_owned()]
            })
            .expect("hash"),
            sha256_json(&UniverseContract {
                symbols: vec!["BTC".to_owned()]
            })
            .expect("hash")
        );
        let _ = LightGbmLibrary::linked();
    }
}
