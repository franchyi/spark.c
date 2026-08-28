//! Strict GLM5Next layer topology derived from the locked GGUF tensor inventory.
//! The runtime does not infer KDA/DSA or dense/MoE placement from a marketing
//! model name; every layer must expose exactly one recognized tensor family.

use std::fmt::{Display, Formatter};

use crate::gguf::{GgufMetadataValue, GgufSet};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmAttentionKind {
    Kda,
    DsaMla,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmFeedForwardKind {
    Dense,
    RoutedMoe,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlmLayerSpec {
    pub layer: u16,
    pub attention: GlmAttentionKind,
    pub feed_forward: GlmFeedForwardKind,
    pub mtp: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlmTopology {
    layers: Vec<GlmLayerSpec>,
    leading_dense_layers: u16,
    trunk_layers: u16,
    mtp_layers: u16,
}

impl GlmTopology {
    pub fn from_gguf(set: &GgufSet) -> Result<Self, GlmTopologyError> {
        if set.architecture != "glm5next" {
            return Err(GlmTopologyError::WrongArchitecture);
        }
        let blocks = metadata_u16(set, "glm5next.block_count")?;
        let mtp_layers = metadata_u16(set, "glm5next.nextn_predict_layers")?;
        let trunk_layers = blocks
            .checked_sub(mtp_layers)
            .ok_or(GlmTopologyError::InvalidMetadata)?;
        let leading_dense_layers = metadata_u16(set, "glm5next.leading_dense_block_count")?;
        if trunk_layers == 0 || mtp_layers >= blocks || leading_dense_layers >= trunk_layers {
            return Err(GlmTopologyError::InvalidMetadata);
        }
        let mut layers = Vec::with_capacity(usize::from(blocks));
        for layer in 0..blocks {
            let has_kda = set.tensors.contains_key(&format!("blk.{layer}.ssm_a"));
            let has_dsa = set
                .tensors
                .contains_key(&format!("blk.{layer}.indexer.proj.weight"));
            let attention = match (has_kda, has_dsa) {
                (true, false) => GlmAttentionKind::Kda,
                (false, true) => GlmAttentionKind::DsaMla,
                _ => return Err(GlmTopologyError::AmbiguousAttention(layer)),
            };
            let dense_name = format!("blk.{layer}.ffn_gate.weight");
            let routed_name = format!("blk.{layer}.ffn_gate_exps.weight");
            let has_dense = set.tensors.contains_key(&dense_name);
            let has_routed = set.tensors.contains_key(&routed_name);
            let feed_forward = match (has_dense, has_routed, layer < leading_dense_layers) {
                (true, false, true) => GlmFeedForwardKind::Dense,
                (false, true, false) => GlmFeedForwardKind::RoutedMoe,
                _ => return Err(GlmTopologyError::InvalidFeedForward(layer)),
            };
            layers.push(GlmLayerSpec {
                layer,
                attention,
                feed_forward,
                mtp: layer >= trunk_layers,
            });
        }
        Ok(Self {
            layers,
            leading_dense_layers,
            trunk_layers,
            mtp_layers,
        })
    }

    pub fn layers(&self) -> &[GlmLayerSpec] {
        &self.layers
    }

    pub fn layer(&self, layer: u16) -> Option<GlmLayerSpec> {
        self.layers.get(usize::from(layer)).copied()
    }

    pub fn leading_dense_layers(&self) -> u16 {
        self.leading_dense_layers
    }

    pub fn trunk_layers(&self) -> &[GlmLayerSpec] {
        &self.layers[..usize::from(self.trunk_layers)]
    }

    pub fn mtp_layers(&self) -> &[GlmLayerSpec] {
        &self.layers[usize::from(self.trunk_layers)..]
    }

    pub fn trunk_layer_count(&self) -> u16 {
        self.trunk_layers
    }

    pub fn mtp_layer_count(&self) -> u16 {
        self.mtp_layers
    }

    pub fn kda_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.attention == GlmAttentionKind::Kda)
            .count()
    }

    pub fn dsa_layers(&self) -> impl Iterator<Item = u16> + '_ {
        self.layers.iter().filter_map(|layer| {
            (layer.attention == GlmAttentionKind::DsaMla).then_some(layer.layer)
        })
    }

    pub fn trunk_dsa_layers(&self) -> impl Iterator<Item = u16> + '_ {
        self.trunk_layers().iter().filter_map(|layer| {
            (layer.attention == GlmAttentionKind::DsaMla).then_some(layer.layer)
        })
    }
}

fn metadata_u16(set: &GgufSet, key: &'static str) -> Result<u16, GlmTopologyError> {
    set.shards
        .first()
        .and_then(|shard| shard.metadata(key))
        .and_then(GgufMetadataValue::as_u64)
        .and_then(|value| u16::try_from(value).ok())
        .ok_or(GlmTopologyError::InvalidMetadata)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlmTopologyError {
    WrongArchitecture,
    InvalidMetadata,
    AmbiguousAttention(u16),
    InvalidFeedForward(u16),
}

impl Display for GlmTopologyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongArchitecture => formatter.write_str("GGUF architecture is not glm5next"),
            Self::InvalidMetadata => formatter.write_str("invalid GLM topology metadata"),
            Self::AmbiguousAttention(layer) => write!(
                formatter,
                "GLM layer {layer} does not identify exactly one KDA or DSA/MLA family"
            ),
            Self::InvalidFeedForward(layer) => write!(
                formatter,
                "GLM layer {layer} disagrees with the dense/MoE metadata boundary"
            ),
        }
    }
}

impl std::error::Error for GlmTopologyError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use super::*;
    use crate::gguf::{
        GgmlTensorType, GgufMetadataEntry, GgufShard, GgufTensorLocation, GgufValueType,
    };

    fn metadata(value: u64) -> GgufMetadataEntry {
        GgufMetadataEntry {
            value_type: GgufValueType::Uint64,
            value: GgufMetadataValue::Unsigned(value),
        }
    }

    fn placeholder() -> GgufTensorLocation {
        GgufTensorLocation {
            shard: 0,
            absolute_offset: 0,
            data_bytes: 4,
            dimensions: vec![1],
            tensor_type: GgmlTensorType::F32,
        }
    }

    fn set() -> GgufSet {
        let mut metadata_map = BTreeMap::new();
        metadata_map.insert("glm5next.block_count".into(), metadata(5));
        metadata_map.insert("glm5next.nextn_predict_layers".into(), metadata(1));
        metadata_map.insert("glm5next.leading_dense_block_count".into(), metadata(2));
        let shard = GgufShard {
            path: PathBuf::from("fixture.gguf"),
            version: 3,
            alignment: 32,
            data_offset: 0,
            file_bytes: 0,
            metadata: metadata_map,
            tensors: Vec::new(),
        };
        let mut tensors = BTreeMap::new();
        for layer in 0..5 {
            let attention = if layer == 3 {
                format!("blk.{layer}.indexer.proj.weight")
            } else {
                format!("blk.{layer}.ssm_a")
            };
            let ffn = if layer < 2 {
                format!("blk.{layer}.ffn_gate.weight")
            } else {
                format!("blk.{layer}.ffn_gate_exps.weight")
            };
            tensors.insert(attention, placeholder());
            tensors.insert(ffn, placeholder());
        }
        GgufSet {
            architecture: "glm5next".into(),
            shards: vec![shard],
            tensors,
        }
    }

    #[test]
    fn freezes_attention_and_dense_boundaries_from_tensor_families() {
        let topology = GlmTopology::from_gguf(&set()).expect("topology");
        assert_eq!(topology.leading_dense_layers(), 2);
        assert_eq!(topology.trunk_layer_count(), 4);
        assert_eq!(topology.mtp_layer_count(), 1);
        assert_eq!(topology.trunk_layers().len(), 4);
        assert_eq!(topology.mtp_layers().len(), 1);
        assert_eq!(topology.kda_layers(), 4);
        assert_eq!(topology.dsa_layers().collect::<Vec<_>>(), vec![3]);
        assert_eq!(topology.trunk_dsa_layers().collect::<Vec<_>>(), vec![3]);
        assert_eq!(
            topology.layer(4),
            Some(GlmLayerSpec {
                layer: 4,
                attention: GlmAttentionKind::Kda,
                feed_forward: GlmFeedForwardKind::RoutedMoe,
                mtp: true,
            })
        );
    }
}
