const RIEMANN_DIVISIONS: u32 = 100;

fn riemann_sum(f: impl Fn(f64) -> f64, a: f64, b: f64, n: u32) -> f64 {
    let h = (b - a) / n as f64;
    let sum = (1..n).map(|i| f(a + i as f64 * h)).sum::<f64>();
    h * ((f(a) + f(b)) / 2.0 + sum)
}

fn false_positive_area(threshold: f64, b: u32, r: u32) -> f64 {
    let proba = |s: f64| -> f64 { 1.0 - (1.0 - s.powi(r as i32)).powi(b as i32) };
    riemann_sum(proba, 0.0, threshold, RIEMANN_DIVISIONS)
}

fn false_negative_area(threshold: f64, b: u32, r: u32) -> f64 {
    let proba = |s: f64| -> f64 { 1.0 - (1.0 - (1.0 - s.powi(r as i32)).powi(b as i32)) };
    riemann_sum(proba, threshold, 1.0, RIEMANN_DIVISIONS)
}

pub fn optimal_param(
    threshold: f64,
    num_perm: u32,
    false_positive_weight: f64,
    false_negative_weight: f64,
) -> (u32, u32) {
    let mut min_error: f64 = f64::INFINITY;

    let mut opt: (u32, u32) = (0, 0);

    for b in 1..(num_perm + 1) {
        let max_r: u32 = num_perm / b;
        for r in 1..max_r + 1 {
            let false_positive = false_positive_area(threshold, b, r);
            let false_negative = false_negative_area(threshold, b, r);
            let error =
                false_positive_weight * false_positive + false_negative_weight * false_negative;
            if error < min_error {
                min_error = error;
                opt = (b, r);
            }
        }
    }
    opt
}
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_riemann_sum() {
        fn test_function(x: f64) -> f64 {
            x * x // A simple function, f(x) = x^2
        }
        let a = 0.0;
        let b = 1.0;
        let n = 1000;
        let result = riemann_sum(test_function, a, b, n);
        let expected = 1.0 / 3.0; // The integral of x^2 from 0 to 1 is 1/3

        // Assert that the result is close to the expected value
        let tolerance = 0.001;
        println!("Result: {}, Expected: {}", result, expected);
        assert!(
            (result - expected).abs() < tolerance,
            "The calculated integral was not within the expected tolerance"
        );
    }
    #[test]
    fn test_optimal_param() {
        let threshold = 0.5;
        let num_perm = 128;
        let false_positive_weight = 1.0;
        let false_negative_weight = 1.0;
        let (b, r) = optimal_param(
            threshold,
            num_perm,
            false_positive_weight,
            false_negative_weight,
        );
        println!("Optimal parameters: b = {}, r = {}", b, r);
        assert_eq!(b, 25);
        assert_eq!(r, 5);
    }
}
