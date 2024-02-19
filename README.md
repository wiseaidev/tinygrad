# ✨️ tinygrad

[![Crates.io](https://img.shields.io/crates/v/tinygrad.svg)](https://crates.io/crates/tinygrad)
[![docs](https://docs.rs/tinygrad/badge.svg)](https://docs.rs/tinygrad/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust crate for building and training neural networks. `tinygrad` provides a simple interface for defining tensors, performing forward and backward passes, and implementing basic operations such as dot products and summation.

## 🚀 Quick Start

Get started with the `tinygrad` library by following these simple steps:

1. Install the `tinygrad` crate by adding the following line to your `Cargo.toml` file:

```toml
[dependencies]
tinygrad = "0.1.0"
```

1. Use the `Tensor` and `ForwardBackward` traits to create and work with tensors:

```rust
use ndarray::{array, Array1};
use tinygrad::{Tensor, Context, TensorTrait};

// Create a tensor
let value = array![1.0, 2.0, 3.0];
let tensor = Tensor::new(value);

// Perform forward and backward passes
let mut ctx = Context::new();
let result = tensor.forward(&mut ctx, vec![tensor.get_value()]);
tensor.backward(&mut ctx, array![1.0, 1.0, 1.0].view());
```

3. Implement custom operations by defining structs that implement the `ForwardBackward` trait:

```rust
use ndarray::ArrayView1;
use tinygrad::{ForwardBackward, Context, TensorTrait};

// Example operation: Dot product
struct Dot;

impl ForwardBackward for Dot {
    fn forward(&self, _ctx: &mut Context, inputs: Vec<ArrayView1<f64>>) -> f64 {
        let input = &inputs[0];
        let weight = &inputs[1];
        input.dot(weight)
    }

    fn backward(&self, ctx: &mut Context, grad_output: ArrayView1<f64>) {
        // Implement backward pass
        // ...
    }
}
```

# 🔧 Usage Example

```rust
use ndarray::{array, Array1};
use tinygrad::{Tensor, Context, TensorTrait};

fn main() {
    let input = array![1.0, 2.0, 3.0];
    let weight = array![4.0, 5.0, 6.0];

    let input_tensor = Box::new(Tensor::new(input));
    let weight_tensor = Box::new(Tensor::new(weight));

    let dot_fn = Dot;
    let mut ctx = Context::new();

    let inputs = vec![
        input_tensor.get_value(),
        weight_tensor.get_value(),
    ];
    let output = dot_fn.forward(&mut ctx, inputs);

    println!("Dot product: {:?}", output);

    let grad_output = array![1.0, 1.0, 1.0];
    dot_fn.backward(&mut ctx, grad_output.view());

    let grad_input = &input_tensor.grad.clone();
    let grad_weight = &weight_tensor.grad.clone();

    println!("Gradient for input: {:?}", grad_input);
    println!("Gradient for weight: {:?}", grad_weight);
}
```

# 🧪 Testing

Run tests for the `tinygrad` crate using:

```bash
cargo test
```

## 🌐 GitHub Repository

You can access the source code for the `tinygrad` crate on [GitHub](https://github.com/wiseaidev/tinygrad).

## 🤝 Contributing

Contributions and feedback are welcome! If you'd like to contribute, report an issue, or suggest an enhancement, please engage with the project on [GitHub](https://github.com/wiseaidev/tinygrad). Your contributions help improve this crate for the community.

# 📘 Documentation

Full documentation for `tinygrad` is available on [docs.rs](https://docs.rs/tinygrad/).

# 📄 License

This project is licensed under the [MIT License](LICENSE).
