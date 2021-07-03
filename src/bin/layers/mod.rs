use precod_compdag::mat_mul;
use precod_compdag::DataWrapper;
use precod_compdag::DiffComp;
use precod_compdag::ExpAvgPreCod;
use precod_compdag::InputWithErrorBackProp;
use precod_compdag::LeafNode;
use precod_compdag::OutputWithErrorBackProp;
use precod_compdag::PreCodDagComms;
use precod_compdag::PreCodDagNode;
use precod_compdag::PredictiveCoding;
use rand;
use rand::prelude::*;
use rand_distr::Uniform;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use thiserror::Error;

pub const VEC_DIM: usize = 784; // IMAGE_DIM * IMAGE_DIM
pub const MNIST_CLASSES: usize = 10;
const HIDDEN_LAYERS: usize = 128;
const ERR_LEARNING_RATE: f64 = 0.5;
const WEIGHT_LEARNING_RATE: f64 = 0.01;
const PREDICTION_LEARNING_THRESHOLD: f64 = 0.2;
pub const IMAGE_DIM: usize = 28;
const MAX_U8_AS_F64: f64 = 256.0;
pub const EMPTY_VEC: Vec<usize> = Vec::<usize>::new();
const GRADIENT_CUTOFF: f64 = 1.0;
const TOL: f64 = 1.0e-7;

//todo remove pub value
#[derive(Debug, Clone)]
pub struct MNISTImage(pub mat_mul::Matrix<u8, IMAGE_DIM, IMAGE_DIM>);

//todo remove pub value
#[derive(Debug, Clone)]
pub struct MNISTClass(pub mat_mul::Vector<bool, MNIST_CLASSES>);
impl MNISTClass {
    pub fn smooth(&self) -> MNISTClassSmoother {
        let MNISTClass(data) = self;
        MNISTClassSmoother(data.map(|x| (*x as i32) as f64))
    }

    pub fn get_index(&self) -> usize {
        let MNISTClass(data) = self;
        data.iter()
            .enumerate()
            .find(|(_, val)| **val)
            .map(|(index, _)| index)
            .expect("Should've had one non-zero entry")
    }
}

fn u8_to_grey(u: &u8) -> &str {
    match *u / 64 {
        3 => "▓▓",
        2 => "▒▒",
        1 => "░░",
        _ => "  ",
    }
}

impl Display for MNISTImage {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "\n")?;
        self.0
            .row_iter()
            .map(|row| -> Result<(), std::fmt::Error> {
                row.iter()
                    .map(|col| -> Result<(), std::fmt::Error> {
                        let pixel = u8_to_grey(col);
                        write!(f, "{}", pixel)?;
                        Ok(())
                    })
                    .collect::<Result<Vec<()>, std::fmt::Error>>()?;
                write!(f, "\n")?;
                Ok(())
            })
            .collect::<Result<Vec<()>, std::fmt::Error>>()?;
        Ok(())
    }
}

impl Default for MNISTClass {
    fn default() -> Self {
        let mut data = [false; MNIST_CLASSES];
        data[0] = true; // 0 < MNIST_CLASSES
        MNISTClass(mat_mul::Vector::from_iter(data.into_iter().map(|x| *x as i8)).map(|x| *x > 0))
    }
}

impl MNISTImage {
    pub fn to_f32_arr(&self) -> MNISTVector {
        let MNISTImage(data) = self;
        MNISTVector(mat_mul::Vector::from_iter(
            data.iter().map(|datum| *datum as f64),
        ))
    }
}

#[derive(Debug, Clone)]
pub struct MNISTVector(pub mat_mul::Vector<f64, VEC_DIM>);

// todo: make this derivable as a macro?
impl DataWrapper<mat_mul::Vector<f64, VEC_DIM>> for MNISTVector {
    fn from_data(data: &mat_mul::Vector<f64, VEC_DIM>) -> Self {
        MNISTVector(data.clone())
    }
    fn data(&self) -> &mat_mul::Vector<f64, VEC_DIM> {
        let MNISTVector(data) = self;
        data
    }
    fn data_mut(&mut self) -> &mut mat_mul::Vector<f64, VEC_DIM> {
        let MNISTVector(data) = self;
        data
    }
}

#[derive(Debug)]
pub struct Hidden(mat_mul::Vector<f64, HIDDEN_LAYERS>);

impl DataWrapper<mat_mul::Vector<f64, HIDDEN_LAYERS>> for Hidden {
    fn from_data(data: &mat_mul::Vector<f64, HIDDEN_LAYERS>) -> Self {
        Hidden(data.clone())
    }
    fn data(&self) -> &mat_mul::Vector<f64, HIDDEN_LAYERS> {
        let Hidden(data) = self;
        data
    }
    fn data_mut(&mut self) -> &mut mat_mul::Vector<f64, HIDDEN_LAYERS> {
        let Hidden(data) = self;
        data
    }
}

#[derive(Debug)]
pub struct MNISTClassSmoother(pub mat_mul::Vector<f64, MNIST_CLASSES>);

impl DataWrapper<mat_mul::Vector<f64, MNIST_CLASSES>> for MNISTClassSmoother {
    fn from_data(data: &mat_mul::Vector<f64, MNIST_CLASSES>) -> Self {
        MNISTClassSmoother(data.clone())
    }
    fn data(&self) -> &mat_mul::Vector<f64, MNIST_CLASSES> {
        let MNISTClassSmoother(data) = self;
        data
    }
    fn data_mut(&mut self) -> &mut mat_mul::Vector<f64, MNIST_CLASSES> {
        let MNISTClassSmoother(data) = self;
        data
    }
}

pub struct InputLayerCalculator {
    weights: mat_mul::Matrix<f64, HIDDEN_LAYERS, VEC_DIM>,
}
impl InputLayerCalculator {
    fn new<Rng: rand::Rng>(rng: &mut Rng) -> InputLayerCalculator {
        let weights = mat_mul::Matrix::<f64, HIDDEN_LAYERS, VEC_DIM>::from_distribution(
            rng,
            &Uniform::new(0., 1.0 / ((VEC_DIM * HIDDEN_LAYERS) as f64).sqrt()),
        );

        InputLayerCalculator { weights }
    }
}
impl ExpAvgPreCod<f64> for InputLayerCalculator {
    type InputType = mat_mul::Vector<f64, VEC_DIM>;
    type OutputType = mat_mul::Vector<f64, HIDDEN_LAYERS>;
    const INPUT_RETENTION: f64 = 0.9;
    const ACTIVATION_RETENTION: f64 = 0.9;
}

impl DiffComp for InputLayerCalculator {
    type InputType = mat_mul::Vector<f64, VEC_DIM>;
    type OutputType = mat_mul::Vector<f64, HIDDEN_LAYERS>;
    fn forward(
        &self,
        input: &Self::InputType,
        activation: &mut Self::OutputType,
    ) -> Self::OutputType {
        let mut logits = &self.weights * input;
        self.smooth_activation(activation, &logits);
        logits.map_inplace(|x| x.tanh());
        logits
    }

    fn backward(
        &self,
        pred_err_from_next: &Self::OutputType,
        activation: &Self::OutputType,
    ) -> Self::OutputType {
        activation.map(|x| x.cosh().powi(-2))
    }

    fn update(
        &mut self,
        input: &Self::InputType,
        error_from_next: &Self::OutputType,
        activation: &Self::OutputType,
        prediction_error: &Self::InputType,
        prediction: &mut Self::InputType,
    ) {
        let mut backprop = self.backward(error_from_next, activation);
        backprop.map_inplace(|x| x.min(GRADIENT_CUTOFF).max(-GRADIENT_CUTOFF));

        let node_adjust = &backprop * &self.weights;
        let mut delta = prediction_error.clone();
        delta -= &node_adjust;
        let mut weight_update_row = input.clone();
        weight_update_row
            .iter_mut()
            .zip(delta.iter_mut())
            .zip(prediction_error.iter())
            .for_each(|((input_entry, delta_entry), prediction_error_entry)| {
                // We only update the weights if the update_to_the_prediction is convergent
                // It may be worth rethinking this, and have the backprop only get sent
                // when the prediction is converged.
                if delta_entry.abs() > PREDICTION_LEARNING_THRESHOLD * prediction_error_entry.abs()
                {
                    *input_entry = 0.0;
                    *delta_entry *= -ERR_LEARNING_RATE;
                } else {
                    *delta_entry = 0.0;
                    *input_entry *= WEIGHT_LEARNING_RATE;
                }
            });

        *prediction += delta;
        self.weights
            .row_iter_mut()
            .zip(error_from_next.iter())
            .zip(backprop.iter())
            .for_each(|((row, err_from_next_entry), backprop_entry)| {
                *row += &weight_update_row * *err_from_next_entry * *backprop_entry;
            });
    }
}

pub struct MNISTInputLayer(
    pub  PreCodDagNode<
        f64,
        MNISTVector,
        mat_mul::Vector<f64, VEC_DIM>,
        Hidden,
        mat_mul::Vector<f64, HIDDEN_LAYERS>,
        InputLayerCalculator,
    >,
);

impl MNISTInputLayer {
    pub fn new<Rng: rand::Rng>(
        input_forward: crossbeam::channel::Receiver<MNISTVector>,
        output_forward: crossbeam::channel::Sender<Hidden>,
        output_error: crossbeam::channel::Receiver<Hidden>,
        rng: &mut Rng,
    ) -> MNISTInputLayer {
        let calculator = InputLayerCalculator::new(rng);
        let input_init = MNISTVector(mat_mul::Vector::<f64, VEC_DIM>::new());
        let activation_init = Hidden(mat_mul::Vector::<f64, HIDDEN_LAYERS>::new());
        let comms = PreCodDagComms::Root {
            input_forward,
            output_forward,
            output_error,
        };
        let dag_node = PreCodDagNode::<
            f64,
            MNISTVector,
            mat_mul::Vector<f64, VEC_DIM>,
            Hidden,
            mat_mul::Vector<f64, HIDDEN_LAYERS>,
            InputLayerCalculator,
        >::new(comms, input_init, activation_init, calculator);

        MNISTInputLayer(dag_node)
    }
}

pub struct HiddenLayerCalculator {
    weights: mat_mul::Matrix<f64, MNIST_CLASSES, HIDDEN_LAYERS>,
}

impl HiddenLayerCalculator {
    fn new<Rng: rand::Rng>(rng: &mut Rng) -> HiddenLayerCalculator {
        let weights = mat_mul::Matrix::<f64, MNIST_CLASSES, HIDDEN_LAYERS>::from_distribution(
            rng,
            &Uniform::new(0., 1.0 / ((HIDDEN_LAYERS * MNIST_CLASSES) as f64).sqrt()),
        );

        HiddenLayerCalculator { weights }
    }
}
impl DiffComp for HiddenLayerCalculator {
    type InputType = mat_mul::Vector<f64, HIDDEN_LAYERS>;
    type OutputType = mat_mul::Vector<f64, MNIST_CLASSES>;
    fn forward(
        &self,
        input: &Self::InputType,
        activation: &mut Self::OutputType,
    ) -> Self::OutputType {
        let logits = &self.weights * input;
        self.smooth_activation(activation, &logits);
        let mut total = 0.0;
        let exps = logits.map(|logit| {
            let exp = logit.exp();
            total += exp;
            exp
        });
        exps.map(|x| x / total)
    }

    fn backward(
        &self,
        pred_err_from_next: &Self::OutputType,
        activation: &Self::OutputType,
    ) -> Self::OutputType {
        activation + (&*activation * ((&*activation * pred_err_from_next) * -1.0))
    }

    fn update(
        &mut self,
        input: &Self::InputType,
        error_from_next: &Self::OutputType,
        activation: &Self::OutputType,
        prediction_error: &Self::InputType,
        prediction: &mut Self::InputType,
    ) {
        let mut backprop = self.backward(error_from_next, activation);
        backprop.map_inplace(|x| x.min(GRADIENT_CUTOFF).max(-GRADIENT_CUTOFF));

        let node_adjust = &backprop * &self.weights;
        let mut delta = prediction_error.clone();
        delta -= &node_adjust;
        let mut weight_update_row = input.clone();
        weight_update_row
            .iter_mut()
            .zip(delta.iter_mut())
            .zip(prediction_error.iter())
            .for_each(|((input_entry, delta_entry), prediction_error_entry)| {
                // We only update the weights if the update_to_the_prediction is convergent
                // It may be worth rethinking this, and have the backprop only get sent
                // when the prediction is converged.
                if delta_entry.abs() > PREDICTION_LEARNING_THRESHOLD * prediction_error_entry.abs()
                {
                    *input_entry = 0.0;
                    *delta_entry *= -ERR_LEARNING_RATE;
                } else {
                    *delta_entry = 0.0;
                    *input_entry *= WEIGHT_LEARNING_RATE;
                }
            });
        *prediction += delta;
        self.weights
            .row_iter_mut()
            .zip(error_from_next.iter())
            .zip(backprop.iter())
            .for_each(|((row, err_from_next_entry), backprop_entry)| {
                *row += &weight_update_row * *err_from_next_entry * *backprop_entry;
            });
    }
}

impl ExpAvgPreCod<f64> for HiddenLayerCalculator {
    type InputType = mat_mul::Vector<f64, HIDDEN_LAYERS>;
    type OutputType = mat_mul::Vector<f64, MNIST_CLASSES>;
    const INPUT_RETENTION: f64 = 0.9;
    const ACTIVATION_RETENTION: f64 = 0.9;
}

pub struct MNISTHiddenLayer(
    pub  PreCodDagNode<
        f64,
        Hidden,
        mat_mul::Vector<f64, HIDDEN_LAYERS>,
        MNISTClassSmoother,
        mat_mul::Vector<f64, MNIST_CLASSES>,
        HiddenLayerCalculator,
    >,
);

impl MNISTHiddenLayer {
    pub fn new<Rng: rand::Rng>(
        input_forward: crossbeam::channel::Receiver<Hidden>,
        input_error: crossbeam::channel::Sender<Hidden>,
        output_forward: crossbeam::channel::Sender<MNISTClassSmoother>,
        output_error: crossbeam::channel::Receiver<MNISTClassSmoother>,
        rng: &mut Rng,
    ) -> MNISTHiddenLayer {
        let calculator = HiddenLayerCalculator::new(rng);
        let input_init = Hidden(mat_mul::Vector::<f64, HIDDEN_LAYERS>::new());
        let activation_init = MNISTClassSmoother(mat_mul::Vector::<f64, MNIST_CLASSES>::new());
        let comms = PreCodDagComms::Branch {
            input_forward,
            input_error,
            output_forward,
            output_error,
        };
        let dag_node = PreCodDagNode::<
            f64,
            Hidden,
            mat_mul::Vector<f64, HIDDEN_LAYERS>,
            MNISTClassSmoother,
            mat_mul::Vector<f64, MNIST_CLASSES>,
            HiddenLayerCalculator,
        >::new(comms, input_init, activation_init, calculator);

        MNISTHiddenLayer(dag_node)
    }
}
