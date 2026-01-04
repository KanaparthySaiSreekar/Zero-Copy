use crate::distance::{cosine_distance, l2_distance, DistanceMetric};
use crate::vector_store::VectorStore;
use parking_lot::RwLock;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;
use ordered_float::OrderedFloat;

/// A neighbor in the HNSW graph
#[derive(Debug, Clone)]
struct Neighbor {
    id: usize,
    distance: f32,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

/// HNSW index for approximate nearest neighbor search
pub struct HNSWIndex {
    vector_store: Arc<VectorStore>,
    layers: Vec<Vec<Vec<usize>>>, // layers[node][layer] = neighbors
    entry_point: usize,
    max_layers: usize,
    ef_construction: usize,
    m: usize,          // Max number of connections per layer
    m_max_0: usize,    // Max connections for layer 0
    metric: DistanceMetric,
}

impl HNSWIndex {
    /// Create a new HNSW index
    pub fn new(
        vector_store: Arc<VectorStore>,
        m: usize,
        ef_construction: usize,
        metric: DistanceMetric,
    ) -> Self {
        let num_vectors = vector_store.len();
        let max_layers = (num_vectors as f64).log2().ceil() as usize;

        HNSWIndex {
            vector_store,
            layers: Vec::new(),
            entry_point: 0,
            max_layers,
            ef_construction,
            m,
            m_max_0: m * 2,
            metric,
        }
    }

    /// Build the HNSW index
    pub fn build(&mut self) {
        let num_vectors = self.vector_store.len();
        if num_vectors == 0 {
            return;
        }

        // Initialize layers
        self.layers = vec![vec![Vec::new(); self.max_layers]; num_vectors];

        // Insert all vectors
        for i in 0..num_vectors {
            if i == 0 {
                self.entry_point = 0;
                continue;
            }

            self.insert(i);
        }
    }

    /// Insert a vector into the index
    fn insert(&mut self, id: usize) {
        let vector = self.vector_store.get(id).unwrap();
        let level = self.random_level();

        // Find nearest neighbors at each layer
        let mut nearest = vec![Vec::new(); level + 1];
        let mut current = self.entry_point;

        // Search from top to target level
        for lc in (level + 1..self.max_layers).rev() {
            current = self.search_layer(vector, current, 1, lc)[0].id;
        }

        // Insert at each level
        for lc in (0..=level).rev() {
            let candidates = self.search_layer(vector, current, self.ef_construction, lc);

            // Select M neighbors
            let m = if lc == 0 { self.m_max_0 } else { self.m };
            nearest[lc] = self.select_neighbors(vector, candidates, m);

            // Add bidirectional links
            for neighbor in &nearest[lc] {
                self.layers[id][lc].push(neighbor.id);
                self.layers[neighbor.id][lc].push(id);

                // Prune neighbors if needed
                let max_conn = if lc == 0 { self.m_max_0 } else { self.m };
                if self.layers[neighbor.id][lc].len() > max_conn {
                    self.prune_neighbors(neighbor.id, lc, max_conn);
                }
            }

            if !nearest[lc].is_empty() {
                current = nearest[lc][0].id;
            }
        }

        // Update entry point if needed
        if level > self.get_level(self.entry_point) {
            self.entry_point = id;
        }
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.vector_store.is_empty() {
            return Vec::new();
        }

        let mut current = self.entry_point;

        // Search from top to layer 0
        for lc in (1..self.max_layers).rev() {
            let results = self.search_layer(query, current, 1, lc);
            if !results.is_empty() {
                current = results[0].id;
            }
        }

        // Search at layer 0
        let mut results = self.search_layer(query, current, ef.max(k), 0);
        results.truncate(k);

        results.into_iter().map(|n| (n.id, n.distance)).collect()
    }

    /// Search within a single layer
    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, layer: usize) -> Vec<Neighbor> {
        let mut visited = vec![false; self.vector_store.len()];
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let entry_dist = self.distance(query, entry);
        let entry_neighbor = Neighbor {
            id: entry,
            distance: entry_dist,
        };

        candidates.push(entry_neighbor.clone());
        results.push(entry_neighbor);
        visited[entry] = true;

        while let Some(current) = candidates.pop() {
            // Check if current is farther than worst result
            if let Some(worst) = results.peek() {
                if current.distance > worst.distance {
                    break;
                }
            }

            // Check neighbors
            if layer < self.layers[current.id].len() {
                for &neighbor_id in &self.layers[current.id][layer] {
                    if !visited[neighbor_id] {
                        visited[neighbor_id] = true;

                        let dist = self.distance(query, neighbor_id);
                        let neighbor = Neighbor {
                            id: neighbor_id,
                            distance: dist,
                        };

                        if results.len() < ef || dist < results.peek().unwrap().distance {
                            candidates.push(neighbor.clone());
                            results.push(neighbor);

                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut sorted_results: Vec<_> = results.into_iter().collect();
        sorted_results.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal)
        });
        sorted_results
    }

    /// Calculate distance between query and a vector in the store
    #[inline]
    fn distance(&self, query: &[f32], id: usize) -> f32 {
        let vector = self.vector_store.get(id).unwrap();
        match self.metric {
            DistanceMetric::L2 => l2_distance(query, vector),
            DistanceMetric::Cosine => cosine_distance(query, vector),
            DistanceMetric::DotProduct => {
                -query.iter().zip(vector.iter()).map(|(a, b)| a * b).sum::<f32>()
            }
        }
    }

    /// Select best neighbors from candidates
    fn select_neighbors(&self, query: &[f32], candidates: Vec<Neighbor>, m: usize) -> Vec<Neighbor> {
        let mut selected = Vec::new();

        for candidate in candidates.iter().take(m) {
            selected.push(candidate.clone());
        }

        selected
    }

    /// Prune neighbors to fit max connections
    fn prune_neighbors(&mut self, id: usize, layer: usize, max_conn: usize) {
        let neighbors = &self.layers[id][layer];
        if neighbors.len() <= max_conn {
            return;
        }

        let vector = self.vector_store.get(id).unwrap();
        let mut neighbor_dists: Vec<_> = neighbors
            .iter()
            .map(|&n| {
                let dist = self.distance(vector, n);
                (n, dist)
            })
            .collect();

        neighbor_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        self.layers[id][layer] = neighbor_dists
            .into_iter()
            .take(max_conn)
            .map(|(n, _)| n)
            .collect();
    }

    /// Get the level of a node
    fn get_level(&self, id: usize) -> usize {
        self.layers[id]
            .iter()
            .rposition(|neighbors| !neighbors.is_empty())
            .unwrap_or(0)
    }

    /// Generate random level for new node
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let ml = 1.0 / (self.m as f64).ln();
        let mut level = 0;

        while rng.gen::<f64>() < 0.5 && level < self.max_layers - 1 {
            level += 1;
        }

        level
    }

    /// Get statistics about the index
    pub fn stats(&self) -> IndexStats {
        let mut total_connections = 0;
        let mut max_connections = 0;

        for node_layers in &self.layers {
            for layer_connections in node_layers {
                let count = layer_connections.len();
                total_connections += count;
                max_connections = max_connections.max(count);
            }
        }

        IndexStats {
            num_vectors: self.vector_store.len(),
            dimension: self.vector_store.dimension(),
            num_layers: self.max_layers,
            total_connections,
            max_connections,
            avg_connections: if self.vector_store.len() > 0 {
                total_connections as f64 / self.vector_store.len() as f64
            } else {
                0.0
            },
        }
    }
}

/// Statistics about the HNSW index
#[derive(Debug)]
pub struct IndexStats {
    pub num_vectors: usize,
    pub dimension: usize,
    pub num_layers: usize,
    pub total_connections: usize,
    pub max_connections: usize,
    pub avg_connections: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::VectorStore;

    #[test]
    fn test_hnsw_build_and_search() {
        // Create test data
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
        ];

        let path = "/tmp/test_hnsw_vectors.vec";
        VectorStore::create(path, &vectors).unwrap();
        let store = Arc::new(VectorStore::open(path).unwrap());

        // Build index
        let mut index = HNSWIndex::new(store, 4, 10, DistanceMetric::L2);
        index.build();

        // Search
        let query = vec![1.0, 0.1, 0.1];
        let results = index.search(&query, 2, 10);

        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0); // Should find [1,0,0] as nearest

        std::fs::remove_file(path).ok();
    }
}
