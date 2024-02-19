//! # tinygrad
//!
//! `tinygrad` is a crate for building and training neural networks in Rust. It provides a simple interface for defining tensors,
//! performing forward and backward passes, and implementing basic operations such as dot products and summation.
//!
//! ## Quick Start
//!
//! Get started with the `tinygrad` library by following these simple steps:
//!
//! 1. Install the `tinygrad` crate by adding the following line to your `Cargo.toml` file:
//!
//! ```toml
//! [dependencies]
//! tinygrad = "0.1.0"
//! ```
//!
//! 2. Use the `Tensor` and `ForwardBackward` traits to create and work with tensors:
//!
//! ```rust
//! use ndarray::{array, Array1};
//! use tinygrad::{Tensor, Context, TensorTrait};
//!
//! // Create a tensor
//! let value = array![1.0, 2.0, 3.0];
//! let tensor = Tensor::new(value);
//!
//! // Perform forward and backward passes
//! let mut ctx = Context::new();
//! let result = tensor.forward(&mut ctx, vec![tensor.get_value()]);
//! tensor.backward(&mut ctx, array![1.0, 1.0, 1.0].view());
//! ```
//!
//! 3. Implement custom operations by defining structs that implement the `ForwardBackward` trait:
//!
//! ```rust
//! use ndarray::ArrayView1;
//! use tinygrad::{ForwardBackward, Context, TensorTrait};
//!
//! // Example operation: Dot product
//! struct Dot;
//!
//! impl ForwardBackward for Dot {
//!     fn forward(&self, _ctx: &mut Context, inputs: Vec<ArrayView1<f64>>) -> f64 {
//!         let input = &inputs[0];
//!         let weight = &inputs[1];
//!         input.dot(weight)
//!     }
//!
//!     fn backward(&self, ctx: &mut Context, grad_output: ArrayView1<f64>) {
//!         // Implement backward pass
//!         // ...
//!     }
//! }
//! ```
//!
//! ## GitHub Repository
//!
//! You can access the source code for the `tinygrad` crate on [GitHub](https://github.com/wiseaidev/tinygrad).
//!
//! ## Contributing
//!
//! Contributions and feedback are welcome! If you'd like to contribute, report an issue, or suggest an enhancement,
//! please engage with the project on [GitHub](https://github.com/wiseaidev/tinygrad).
//! Your contributions help improve this crate for the community.

use ndarray::{array, Array1, ArrayView1};

type Gradient = Array1<f64>;

/// This trait defines the common interface for tensors in a computational graph.
pub trait TensorTrait {
    /// Computes the forward pass of the tensor.
    ///
    /// # Arguments
    /// * `ctx` - A mutable reference to the computation context.
    /// * `inputs` - A vector of input arrays for the forward pass.
    ///
    /// # Returns
    /// (`f64`): The result of the forward pass.
    fn forward(&self, ctx: &mut Context, inputs: Vec<ArrayView1<f64>>) -> f64;

    /// Computes the backward pass of the tensor to calculate gradients.
    ///
    /// # Arguments
    /// * `ctx` - A mutable reference to the computation context.
    /// * `grad_output` - The gradient of the loss with respect to the output.
    fn backward(&self, ctx: &mut Context, grad_output: ArrayView1<f64>);

    /// Gets the value of the tensor.
    ///
    /// # Returns
    /// (`ArrayView1<f64>`): The view of the tensor's value.
    fn get_value(&self) -> ArrayView1<f64>;

    /// Gets the gradient of the tensor.
    ///
    /// # Returns
    /// (`Option<Gradient>`): The option containing the gradient if available.
    fn get_grad(&self) -> Option<Gradient>;
}

/// Represents a basic implementation of a tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub value: Array1<f64>,
    pub grad: Option<Gradient>,
}

impl Tensor {
    /// Creates a new Tensor with the given value.
    ///
    /// # Arguments
    ///
    /// * `value` - The array representing the value of the tensor.
    ///
    /// # Returns
    ///
    /// (`Tensor`): A new Tensor instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, Array1};
    /// use tinygrad::Tensor;
    ///
    /// let value = array![1.0, 2.0, 3.0];
    /// let tensor = Tensor::new(value);
    /// ```
    pub fn new(value: Array1<f64>) -> Tensor {
        Tensor { value, grad: None }
    }
}

impl TensorTrait for Tensor {
    /// Implements the forward pass for the Tensor.
    ///
    /// # Arguments
    ///
    /// * `_ctx` - A mutable reference to the computation context (not used in this example).
    /// * `_inputs` - A vector of ArrayView1<f64> representing the input values (not used in this example).
    ///
    /// # Returns
    ///
    /// (`f64`): The result of the forward pass (not implemented in this example, returns 0.0).
    fn forward(&self, _ctx: &mut Context, _inputs: Vec<ArrayView1<f64>>) -> f64 {
        // TODO: Implement forward function for Tensor
        0.0
    }

    /// Implements the backward pass for the Tensor.
    ///
    /// # Arguments
    ///
    /// * `_ctx` - A mutable reference to the computation context (not used in this example).
    /// * `_grad_output` - An ArrayView1<f64> representing the gradient of the output.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, Array1};
    /// use tinygrad::{Tensor, Context, TensorTrait};
    ///
    /// let mut ctx = Context::new();
    /// let tensor = Tensor::new(array![1.0, 2.0, 3.0]);
    /// tensor.backward(&mut ctx, array![1.0, 1.0, 1.0].view());
    /// ```
    fn backward(&self, _ctx: &mut Context, _grad_output: ArrayView1<f64>) {
        // TODO: Implement backward function for Tensor
    }

    /// Returns the value of the Tensor.
    ///
    /// # Returns
    ///
    /// (`ArrayView1<f64>`): The view of the array representing the value of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, Array1};
    /// use tinygrad::{Tensor, TensorTrait};
    ///
    /// let tensor = Tensor::new(array![1.0, 2.0, 3.0]);
    /// let value = tensor.get_value();
    /// ```
    fn get_value(&self) -> ArrayView1<f64> {
        self.value.view()
    }

    /// Returns the gradient of the Tensor if available.
    ///
    /// # Returns
    ///
    /// (`Option<Gradient>`): The optional gradient of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, Array1};
    /// use tinygrad::{Tensor, TensorTrait};
    ///
    /// let tensor = Tensor::new(array![1.0, 2.0, 3.0]);
    /// let grad = tensor.get_grad();
    /// ```
    fn get_grad(&self) -> Option<Gradient> {
        self.grad.clone()
    }
}

/// This trait defines the interface for operations that have both forward and backward passes.
pub trait ForwardBackward {
    /// Computes the forward pass of the operation.
    ///
    /// # Arguments
    /// * `ctx` - A mutable reference to the computation context.
    /// * `inputs` - A vector of input arrays for the forward pass.
    ///
    /// # Returns
    /// (`f64`): The result of the forward pass.
    fn forward(&self, ctx: &mut Context, inputs: Vec<ArrayView1<f64>>) -> f64;

    /// Computes the backward pass of the operation to calculate gradients.
    ///
    /// # Arguments
    /// * `ctx` - A mutable reference to the computation context.
    /// * `grad_output` - The gradient of the loss with respect to the output.
    fn backward(&self, ctx: &mut Context, grad_output: ArrayView1<f64>);
}

/// Represents the Dot operation.
struct Dot;

/// Represents the Sum operation.
struct Sum;

impl ForwardBackward for Dot {
    fn forward(&self, _ctx: &mut Context, inputs: Vec<ArrayView1<f64>>) -> f64 {
        let input = &inputs[0];
        let weight = &inputs[1];

        input.dot(weight)
    }

    fn backward(&self, ctx: &mut Context, grad_output: ArrayView1<f64>) {
        if ctx.saved_tensors.is_empty() {
            println!("Warning: saved_tensors is empty. Unable to compute gradients.");
            return;
        }

        let mut input = ctx.saved_tensors[0].clone();
        let mut weight = ctx.saved_tensors[1].clone();

        let grad_input = grad_output.dot(&input.get_value().t());
        let grad_weight = input.get_value().t().dot(&grad_output);

        input.grad = Some(array![grad_input]);
        weight.grad = Some(array![grad_weight]);

        ctx.save_for_backward(vec![Box::new(*input), Box::new(*weight)]);
    }
}

impl ForwardBackward for Sum {
    fn forward(&self, _ctx: &mut Context, inputs: Vec<ArrayView1<f64>>) -> f64 {
        let input = &inputs[0];

        input.sum()
    }

    fn backward(&self, ctx: &mut Context, grad_output: ArrayView1<f64>) {
        let mut input = ctx.saved_tensors[0].clone();

        input.grad = Some(Array1::from(grad_output.map(|x| x * 1.0)));

        ctx.save_for_backward(vec![Box::new(*input)]);
    }
}

/// Represents the computation context, storing tensors for backward pass computations.
pub struct Context {
    pub saved_tensors: Vec<Box<Tensor>>,
}

impl Context {
    /// Creates a new Context instance.
    ///
    /// # Returns
    ///
    /// (`Context`): A new Context instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use tinygrad::Context;
    ///
    /// let context = Context::new();
    /// ```
    pub fn new() -> Context {
        Context {
            saved_tensors: Vec::new(),
        }
    }

    /// Saves tensors for backward pass.
    ///
    /// # Arguments
    ///
    /// * `tensors` - A vector of Boxed Tensors to be saved.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::{array, Array1};
    /// use tinygrad::{Tensor, Context};
    ///
    /// let mut ctx = Context::new();
    /// let tensor = Tensor::new(array![1.0, 2.0, 3.0]);
    /// ctx.save_for_backward(vec![Box::new(tensor)]);
    /// ```
    pub fn save_for_backward(&mut self, tensors: Vec<Box<Tensor>>) {
        self.saved_tensors.extend(tensors);
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
