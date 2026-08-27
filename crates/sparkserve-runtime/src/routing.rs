use std::fmt::{Display, Formatter};

use crate::kernel::{GroupedExpertLayout, KernelContractError};

pub const PADDED_ROUTE: u32 = u32::MAX;

/// Host-side reference and graph metadata for one routed MoE batch. Arithmetic
/// kernels consume expert-contiguous rows; the scheduler retains both maps so
/// dispatch and weighted reduction never infer ordering independently.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoutePlan {
    pub num_tokens: u32,
    pub top_k: u32,
    pub num_experts: u32,
    pub expert_rows: Vec<u32>,
    pub grouped: GroupedExpertLayout,
    /// Natural route index `(token * top_k + rank)` to padded packed row.
    pub route_to_packed_row: Vec<u32>,
    /// Padded packed row to natural route index; padding is `PADDED_ROUTE`.
    pub packed_row_to_route: Vec<u32>,
}

impl RoutePlan {
    pub fn build(
        num_tokens: u32,
        top_k: u32,
        num_experts: u32,
        expert_ids: &[u32],
    ) -> Result<Self, RouteError> {
        if num_tokens == 0 || top_k == 0 || num_experts == 0 || num_experts > 512 {
            return Err(RouteError::InvalidShape);
        }
        if top_k > num_experts {
            return Err(RouteError::InvalidShape);
        }
        let routes = usize::try_from(
            u64::from(num_tokens)
                .checked_mul(u64::from(top_k))
                .ok_or(RouteError::IntegerOverflow)?,
        )
        .map_err(|_| RouteError::IntegerOverflow)?;
        if expert_ids.len() != routes {
            return Err(RouteError::RouteCount {
                expected: routes,
                actual: expert_ids.len(),
            });
        }

        let mut expert_rows = vec![0_u32; num_experts as usize];
        for token in 0..num_tokens as usize {
            let begin = token * top_k as usize;
            let end = begin + top_k as usize;
            let token_experts = &expert_ids[begin..end];
            for (rank, expert) in token_experts.iter().copied().enumerate() {
                if expert >= num_experts {
                    return Err(RouteError::ExpertOutOfRange {
                        route: begin + rank,
                        expert,
                        num_experts,
                    });
                }
                if token_experts[..rank].contains(&expert) {
                    return Err(RouteError::DuplicateExpert { token, expert });
                }
                expert_rows[expert as usize] = expert_rows[expert as usize]
                    .checked_add(1)
                    .ok_or(RouteError::IntegerOverflow)?;
            }
        }

        let grouped = GroupedExpertLayout::from_expert_rows(&expert_rows)?;
        let packed_rows =
            usize::try_from(grouped.total_rows).map_err(|_| RouteError::IntegerOverflow)?;
        let mut route_to_packed_row = vec![0_u32; routes];
        let mut packed_row_to_route = vec![PADDED_ROUTE; packed_rows];
        let mut cursors: Vec<u32> = grouped.m_indptr[..num_experts as usize]
            .iter()
            .map(|offset| u32::try_from(*offset).map_err(|_| RouteError::IntegerOverflow))
            .collect::<Result<_, _>>()?;
        for (route, expert) in expert_ids.iter().copied().enumerate() {
            let cursor = &mut cursors[expert as usize];
            let packed_row = *cursor;
            *cursor = cursor.checked_add(1).ok_or(RouteError::IntegerOverflow)?;
            route_to_packed_row[route] = packed_row;
            packed_row_to_route[packed_row as usize] =
                u32::try_from(route).map_err(|_| RouteError::IntegerOverflow)?;
        }

        Ok(Self {
            num_tokens,
            top_k,
            num_experts,
            expert_rows,
            grouped,
            route_to_packed_row,
            packed_row_to_route,
        })
    }

    pub fn active_experts(&self) -> usize {
        self.expert_rows.iter().filter(|rows| **rows != 0).count()
    }

    pub fn useful_rows(&self) -> u64 {
        u64::from(self.num_tokens) * u64::from(self.top_k)
    }

    pub fn padded_rows(&self) -> u64 {
        self.grouped.total_rows - self.useful_rows()
    }

    pub fn dispatch_bytes(&self, packed_row_bytes: u64) -> Result<u64, RouteError> {
        self.grouped
            .total_rows
            .checked_mul(packed_row_bytes)
            .ok_or(RouteError::IntegerOverflow)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RouteError {
    InvalidShape,
    IntegerOverflow,
    RouteCount {
        expected: usize,
        actual: usize,
    },
    ExpertOutOfRange {
        route: usize,
        expert: u32,
        num_experts: u32,
    },
    DuplicateExpert {
        token: usize,
        expert: u32,
    },
    KernelContract(KernelContractError),
}

impl From<KernelContractError> for RouteError {
    fn from(error: KernelContractError) -> Self {
        Self::KernelContract(error)
    }
}

impl Display for RouteError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidShape => formatter.write_str("invalid MoE routing shape"),
            Self::IntegerOverflow => formatter.write_str("MoE routing index overflow"),
            Self::RouteCount { expected, actual } => {
                write!(formatter, "expected {expected} routes, got {actual}")
            }
            Self::ExpertOutOfRange {
                route,
                expert,
                num_experts,
            } => write!(
                formatter,
                "route {route} selects expert {expert}, but only {num_experts} exist"
            ),
            Self::DuplicateExpert { token, expert } => {
                write!(formatter, "token {token} selects expert {expert} twice")
            }
            Self::KernelContract(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for RouteError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_decode_routes_ten_experts_with_stable_inverse_map() {
        let ids = [9, 1, 20, 3, 7, 31, 2, 15, 8, 4];
        let plan = RoutePlan::build(1, 10, 512, &ids).expect("decode route");
        assert_eq!(plan.active_experts(), 10);
        assert_eq!(plan.useful_rows(), 10);
        assert_eq!(plan.grouped.total_rows, 40);
        assert_eq!(plan.padded_rows(), 30);
        for (route, row) in plan.route_to_packed_row.iter().copied().enumerate() {
            assert_eq!(plan.packed_row_to_route[row as usize], route as u32);
        }
        assert_eq!(plan.dispatch_bytes(1280), Ok(51_200));
    }

    #[test]
    fn multiple_tokens_share_expert_segments_without_losing_token_order() {
        let ids = [2, 0, 1, 2, 0, 3];
        let plan = RoutePlan::build(2, 3, 4, &ids).expect("prefill route");
        assert_eq!(plan.expert_rows, vec![2, 1, 2, 1]);
        assert_eq!(plan.grouped.m_indptr, vec![0, 4, 8, 12, 16]);
        assert_eq!(plan.route_to_packed_row, vec![8, 0, 4, 9, 1, 12]);
        assert_eq!(plan.packed_row_to_route[0], 1);
        assert_eq!(plan.packed_row_to_route[1], 4);
        assert_eq!(plan.packed_row_to_route[2], PADDED_ROUTE);
    }

    #[test]
    fn rejects_duplicate_or_out_of_range_experts() {
        assert!(matches!(
            RoutePlan::build(1, 2, 4, &[1, 1]),
            Err(RouteError::DuplicateExpert { .. })
        ));
        assert!(matches!(
            RoutePlan::build(1, 2, 4, &[1, 4]),
            Err(RouteError::ExpertOutOfRange { .. })
        ));
    }
}
