use precod_compdag::mat_mul;
use precod_compdag::DataWrapper;
use precod_compdag::DiffComp;
use precod_compdag::ExpAvgPreCod;
use precod_compdag::InputWithErrorBackProp;
use precod_compdag::LeafNode;
use precod_compdag::OutputWithErrorBackProp;
use precod_compdag::PreCodDagComms;
use precod_compdag::PreCodDagNode;
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

const VEC_DIM: usize = 784; // IMAGE_DIM * IMAGE_DIM
pub const MNIST_CLASSES: usize = 10;
const HIDDEN_LAYERS: usize = 128;
const ERR_LEARNING_RATE: f64 = 0.2;
const WEIGHT_LEARNING_RATE: f64 = 0.2;
const PREDICTION_LEARNING_THRESHOLD: f64 = 0.2;
const IMAGE_DIM: usize = 28;
const MAX_U8_AS_F64: f64 = 256.0;
const EMPTY_VEC: Vec<usize> = Vec::<usize>::new();
const GRADIENT_CUTOFF: f64 = 1.0;
const TOL: f64 = 1.0e-7;

#[derive(Debug, Clone)]
pub struct MNISTImage(mat_mul::Matrix<u8, IMAGE_DIM, IMAGE_DIM>);

#[derive(Debug, Clone)]
pub struct MNISTClass(mat_mul::Vector<bool, MNIST_CLASSES>);
impl MNISTClass {
    fn smooth(&self) -> MNISTClassSmoother {
        let MNISTClass(data) = self;
        MNISTClassSmoother(data.map(|x| (*x as i32) as f64))
    }

    fn get_index(&self) -> usize {
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
    fn to_f32_arr(&self) -> MNISTVector {
        let MNISTImage(data) = self;
        MNISTVector(mat_mul::Vector::from_iter(
            data.iter().map(|datum| *datum as f64),
        ))
    }
}

#[derive(Debug, Clone)]
struct MNISTVector(mat_mul::Vector<f64, VEC_DIM>);

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
struct Hidden(mat_mul::Vector<f64, HIDDEN_LAYERS>);

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
struct MNISTClassSmoother(mat_mul::Vector<f64, MNIST_CLASSES>);

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

struct HiddenLayerCalculator {
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
    fn forward(&self, input: &Self::InputType) -> Self::OutputType {
        let logits = &self.weights * input;
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
    ) -> Self::InputType {
        self.weights.transpose()
            * (activation.component_mul(pred_err_from_next)
                + (&*activation * ((&*activation * pred_err_from_next) * -1.0)))
    }

    fn update(
        &mut self,
        input: &Self::InputType,
        error_from_next: &Self::OutputType,
        activation: &Self::OutputType,
        prediction_error: &Self::InputType,
        prediction: &mut Self::InputType,
    ) {
        let backprop = self.backward(error_from_next, activation);
        let delta = &*prediction_error + (&backprop * (-1.0));
        delta
            .iter()
            .zip(prediction_error.iter())
            .zip(prediction.iter_mut())
            .zip(backprop.iter())
            .zip(self.weights.row_iter_mut())
            .for_each(
                |((((node_delta, prediction_error), prediction), backprop_row), row)| {
                    if node_delta.abs() > PREDICTION_LEARNING_THRESHOLD * prediction_error.abs() {
                        *prediction -= ERR_LEARNING_RATE * node_delta
                    } else {
                        let scale = -1.0
                            * backprop_row.min(GRADIENT_CUTOFF).max(-GRADIENT_CUTOFF)
                            * WEIGHT_LEARNING_RATE
                            * *prediction_error;
                        let boost = &*input * scale;
                        // println!("UPDATING WEIGHTS!");
                        // println!("{:?}, {:?}", row, boost);
                        *row += &boost;
                    }
                },
            );
    }
}

impl ExpAvgPreCod<f64> for HiddenLayerCalculator {
    type InputType = mat_mul::Vector<f64, HIDDEN_LAYERS>;
    type OutputType = mat_mul::Vector<f64, MNIST_CLASSES>;
    const INPUT_RETENTION: f64 = 0.9;
    const ACTIVATION_RETENTION: f64 = 0.9;
}

struct MNISTHiddenLayer(
    PreCodDagNode<
        f64,
        Hidden,
        mat_mul::Vector<f64, HIDDEN_LAYERS>,
        MNISTClassSmoother,
        mat_mul::Vector<f64, MNIST_CLASSES>,
        HiddenLayerCalculator,
    >,
);

impl MNISTHiddenLayer {
    fn new<Rng: rand::Rng>(
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
