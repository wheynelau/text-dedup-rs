const RIEMANN_DIVISIONS: u32 = 100;

fn riemann_sum(f: impl Fn(f64) -> f64, a: f64, b: f64, n: u32) -> f64 {
    let h = (b - a) / n as f64;
    let sum = (1..n)
        .map(|i| f(a + i as f64 * h))
        .sum::<f64>();
    h * ((f(a) + f(b)) / 2.0 + sum)
}

fn false_positive_area(threshold: f64, b:i32, r: i32) -> f64 {
    let proba = |s: f64| -> f64 {
        1.0 - (1.0 - s.powi(r)).powi(b)
    };
    riemann_sum(proba, 0.0, threshold, RIEMANN_DIVISIONS)
}

fn false_negative_area(threshold: f64, b: i32, r: i32) -> f64 {
    let proba = |s: f64| -> f64 {
        1.0 - (1.0 - (1.0 - s.powi(r)).powi(b))
    };
    riemann_sum(proba, threshold, 1.0, RIEMANN_DIVISIONS)
}

pub fn optimal_param(threshold: f64, 
    num_perm: i32, 
    false_positive_weight:f64,
    false_negative_weight: f64) -> (i32, i32) {

    let mut min_error:f64 = f64::INFINITY;

    let mut opt: (i32, i32) = (0,0);

    for b in 1..(num_perm + 1) as i32 {
        let max_r: i32 = num_perm / b;
        for r in 1..max_r + 1 as i32 {
            let false_positive = false_positive_area(threshold, b, r);
            let false_negative = false_negative_area(threshold, b, r);
            let error = false_positive_weight * false_positive + false_negative_weight * false_negative;
            if error < min_error {
                min_error = error;
                opt = (b, r);
            }
        }
    }
    opt
}

mod tests {
    

    #[test]
    fn test_riemann_sum() {
        fn test_function(x: f64) -> f64 {
            x * x  // A simple function, f(x) = x^2
        }
        let a = 0.0;
        let b = 1.0;
        let n = 1000;
        let result = riemann_sum(test_function, a, b, n);
        let expected = 1.0 / 3.0;  // The integral of x^2 from 0 to 1 is 1/3

        // Assert that the result is close to the expected value
        let tolerance = 0.001;
        println!("Result: {}, Expected: {}", result, expected);
        assert!((result - expected).abs() < tolerance, "The calculated integral was not within the expected tolerance");
    }
}