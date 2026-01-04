use std::arch::x86_64::*;

/// Distance metric types
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    L2,
    Cosine,
    DotProduct,
}

/// Calculate L2 (Euclidean) distance between two vectors using SIMD
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { l2_distance_avx2(a, b) }
        } else {
            l2_distance_scalar(a, b)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        l2_distance_scalar(a, b)
    }
}

/// Calculate cosine distance (1 - cosine similarity) using SIMD
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { cosine_distance_avx2(a, b) }
        } else {
            cosine_distance_scalar(a, b)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        cosine_distance_scalar(a, b)
    }
}

/// Scalar implementation of L2 distance (fallback)
#[inline]
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

/// Scalar implementation of cosine distance (fallback)
#[inline]
fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// AVX2 implementation of L2 distance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let mut result = [0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total = result.iter().sum::<f32>();

    // Handle remaining elements
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total.sqrt()
}

/// AVX2 implementation of cosine distance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();

    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));

        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
    }

    // Horizontal sum
    let mut dot_arr = [0f32; 8];
    let mut norm_a_arr = [0f32; 8];
    let mut norm_b_arr = [0f32; 8];

    _mm256_storeu_ps(dot_arr.as_mut_ptr(), dot_sum);
    _mm256_storeu_ps(norm_a_arr.as_mut_ptr(), norm_a_sum);
    _mm256_storeu_ps(norm_b_arr.as_mut_ptr(), norm_b_sum);

    let mut dot: f32 = dot_arr.iter().sum();
    let mut norm_a: f32 = norm_a_arr.iter().sum();
    let mut norm_b: f32 = norm_b_arr.iter().sum();

    // Handle remaining elements
    for i in (chunks * 8)..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let dist = l2_distance(&a, &b);
        assert!((dist - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-5); // Orthogonal vectors
    }
}
