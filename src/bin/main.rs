extern crate precod_compdag;
use precod_compdag::mat_mul;
use precod_compdag::DataWrapper;
use precod_compdag::DiffComp;
use precod_compdag::ExpAvgPreCod;
use precod_compdag::InputWithErrorBackProp;
use precod_compdag::LeafNode;
use precod_compdag::OutputWithErrorBackProp;
use precod_compdag::PreCodDagNode;
use precod_compdag::PreCodDagComms;
use rand;
use rand::prelude::*;
use rand_distr::Uniform;
use std::borrow::BorrowMut;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fs::File;
use std::io::prelude::*;
use std::io::Read;
use std::ops::Mul;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use thiserror::Error;

const IMAGE_DIM: usize = 28;
const MAX_U8_AS_F64: f64 = 256.0;
const VEC_DIM: usize = 784; // IMAGE_DIM * IMAGE_DIM
const MNIST_CLASSES: usize = 10;
const HIDDEN_LAYERS: usize = 128;
const EMPTY_VEC: Vec<usize> = Vec::<usize>::new();
const GRADIENT_CUTOFF: f64 = 1.0;
const TOL: f64 = 1.0e-7;
const ERR_LEARNING_RATE: f64 = 0.2;
const WEIGHT_LEARNING_RATE: f64 = 0.2;
const PREDICTION_LEARNING_THRESHOLD: f64 = 0.2;
#[derive(Error, Debug)]
enum Error {
    #[error("No error at index")]
    NoData(usize),
}

#[derive(Debug, Clone)]
struct MNISTImage(mat_mul::Matrix<u8, IMAGE_DIM, IMAGE_DIM>);
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

#[derive(Debug, Clone)]
struct MNISTClass(mat_mul::Vector<bool, MNIST_CLASSES>);

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
fn main() {
    let path = Path::new("data/train-images-idx3-ubyte");
    let mut file = File::open(path).expect("Missing train set");
    let mut byte_vec = Vec::<u8>::new();
    file.read_to_end(&mut byte_vec).unwrap();
    let images = byte_vec[16..]
        .chunks(VEC_DIM)
        .map(|image_data| {
            let matrix_data: mat_mul::Matrix<u8, IMAGE_DIM, IMAGE_DIM> =
                mat_mul::Matrix::from_iter(image_data.iter().map(|x| *x));

            MNISTImage(matrix_data)git push --set-upstream origin refactor-branch-node
        })
        .collect::<Vec<_>>();
    let mut rng = rand::thread_rng();

    let test_label_path = Path::new("data/train-labels-idx1-ubyte");
    let file = File::open(test_label_path).expect("Missing train labels");
    let labels = file
        .bytes()
        .skip(8)
        .map(|b| {
            let mut class: [bool; MNIST_CLASSES] = [false; MNIST_CLASSES];
            let index = b.unwrap();
            let value = class.get_mut(index as usize).unwrap();
            *value = true;
            MNISTClass(mat_mul::Vector::from_iter(class.iter().map(|x| *x as i8)).map(|x| *x > 0))
        })
        .collect::<Vec<_>>();

    println!("Hello, world! {:?}", &byte_vec[0..4]);
    for _ in 0..4 {
        let index = (&mut rng).gen_range(0..images.len());
        println!("{}", &images[index]);
        println!("LABEL: {:?}", (&labels)[index].get_index());
    }

    let mut indexes_by_class = [EMPTY_VEC; VEC_DIM];
    for (i, class) in labels.iter().enumerate() {
        let class_index = class.get_index();
        let class_list = (&mut indexes_by_class)
            .get_mut(class_index)
            .expect("Def need to have an index less than VEC_DIM");
        class_list.push(i)
    }

    let (mnist_input_send, root_inp_rcv) = crossbeam::channel::unbounded::<MNISTVector>();
    let (hidden_err_sender, root_err_rcv) = crossbeam::channel::unbounded::<Hidden>();
    let (root_out_send, hidden_input_rcv) = crossbeam::channel::unbounded::<Hidden>();
    let (leaf_err_sender, hidde_err_rcv) = crossbeam::channel::unbounded::<MNISTClassSmoother>();
    let (hidden_output_sender, leaf_input_rcv) =
        crossbeam::channel::unbounded::<MNISTClassSmoother>();
    let (ground_truth_sender, leaf_ground_truth_rcv) =
        crossbeam::channel::unbounded::<MNISTClass>();
    let (progress_sender, progress_receiver) =
        crossbeam::channel::unbounded::<(f64, MNISTClassSmoother, MNISTClass)>();

    let mnist_input = RootNode {
        forward_input: root_inp_rcv,
        forward_output: OutputWithErrorBackProp {
            forward_data: root_out_send,
            reverse_error_prop: root_err_rcv,
        },
    };

    let mnist_output = LeafNode {
        node_data: MNISTClassSmoother(mat_mul::Vector::new()),
        input_channels: InputWithErrorBackProp {
            forward_data: leaf_input_rcv,
            reverse_error_prop: leaf_err_sender,
        },
        ground_truth_channel: leaf_ground_truth_rcv,
    };

    let mnist_input_thr = thread::spawn(move || {
        let mut rng = thread_rng();
        let scale = (0.25 / ((VEC_DIM * HIDDEN_LAYERS) as i32 as f64)).sqrt();
        let RootNode {
            forward_input,
            forward_output:
                OutputWithErrorBackProp {
                    forward_data,
                    reverse_error_prop,
                },
        } = mnist_input;

        let input_layer_weights = mat_mul::Matrix::<f64, HIDDEN_LAYERS, VEC_DIM>::from_distribution(
            &mut rng,
            &Uniform::new(-scale, scale),
        );

        struct BackwardOwned {
            weights: mat_mul::Matrix<f64, HIDDEN_LAYERS, VEC_DIM>,
            predictions: mat_mul::Vector<f64, VEC_DIM>,
        }

        struct ForwardOwned {
            input: mat_mul::Vector<f64, VEC_DIM>,
            activation: mat_mul::Vector<f64, HIDDEN_LAYERS>,
            prediction_errors: mat_mul::Vector<f64, VEC_DIM>,
        }

        let forward_owned_data = Arc::new(Mutex::new(ForwardOwned {
            input: mat_mul::Vector::new(),
            activation: mat_mul::Vector::new(),
            prediction_errors: mat_mul::Vector::new(),
        }));

        let read_forward_data_for_back = Arc::clone(&forward_owned_data);

        let backward_owned_data = Arc::new(Mutex::new(BackwardOwned {
            weights: input_layer_weights,
            predictions: mat_mul::Vector::new(),
        }));

        let read_backward_data_for_fore = Arc::clone(&backward_owned_data);

        let input_mutex = Arc::new(Mutex::new(mat_mul::Vector::<f64, VEC_DIM>::new()));

        let predictions_mutex = Arc::new(Mutex::new(mat_mul::Vector::<f64, VEC_DIM>::new()));

        let predictions_errors_mutex = Arc::new(Mutex::new(mat_mul::Vector::<f64, VEC_DIM>::new()));

        let activations_mutex = Arc::new(Mutex::new(mat_mul::Vector::<f64, HIDDEN_LAYERS>::new()));

        // backprop thread
        thread::spawn(move || {
            reverse_error_prop.iter().for_each(|error_from_next| {
                let Hidden(data) = error_from_next;
                let ForwardOwned {
                    input,
                    activation,
                    prediction_errors,
                } = &*read_forward_data_for_back.lock().unwrap();
                let BackwardOwned {
                    weights,
                    predictions,
                } = &mut *backward_owned_data.lock().unwrap();
                let backprop = weights.transpose()
                    * (activation.component_mul(&data)
                        + (&*activation * ((&*activation * &data) * -1.0)));
                // todo impl sub ops
                let delta = &*prediction_errors + (&backprop * (-1.0));
                delta
                    .iter()
                    .zip(prediction_errors.iter())
                    .zip(predictions.iter_mut())
                    .zip(backprop.iter())
                    .zip(weights.row_iter_mut())
                    .for_each(
                        |((((node_delta, prediction_error), prediction), backprop_row), row)| {
                            if node_delta.abs()
                                > PREDICTION_LEARNING_THRESHOLD * prediction_error.abs()
                            {
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
            });
        });

        forward_input.iter().for_each(|mnist_vec| {
            let ForwardOwned {
                input,
                activation,
                prediction_errors,
            } = &mut *forward_owned_data.lock().unwrap();

            let BackwardOwned {
                weights,
                predictions,
            } = &*read_backward_data_for_fore.lock().unwrap();
            let MNISTVector(data) = mnist_vec;

            *input += &data;
            *input *= 0.5;

            let logits = weights * &data;
            let output_vector = logits.map(|logit| logit.tanh());
            *prediction_errors = predictions + (&data * (-1.0));
            *activation = output_vector.clone();
            forward_data.send(Hidden(output_vector)).unwrap();
        });
    });

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
                        if node_delta.abs() > PREDICTION_LEARNING_THRESHOLD * prediction_error.abs()
                        {
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
                HiddenLayerCalculator
            >::new(
                comms,
                input_init,
                activation_init,
                calculator
            );

            MNISTHiddenLayer(dag_node)
        }
    }

    let hidden_layer_thr = thread::spawn(move || {
        let mut rng = thread_rng();
        let MNISTHiddenLayer(hidden_layer_branch_node) = MNISTHiddenLayer::new(
            hidden_input_rcv,
            hidden_err_sender,
            hidden_output_sender,
            hidde_err_rcv,
            &mut rng,
        );
        hidden_layer_branch_node.run();
    });

    let mnist_output_thr =
        thread::spawn(move || {
            let rng = thread_rng();
            let current_label = Arc::new(Mutex::new(MNISTClass::default()));
            let update_label = Arc::clone(&current_label);
            let LeafNode {
                input_channels,
                ground_truth_channel,
                node_data,
            } = mnist_output;

            // Label update thready .. updates from main training routine
            thread::spawn(move || {
                ground_truth_channel.iter().for_each(|new_ground_truth| {
                    let mut updateable = update_label.lock().expect("can't update networkt output");
                    *updateable = new_ground_truth;
                });
            });

            input_channels.forward_data.iter().for_each(|class_input| {
                let MNISTClassSmoother(data) = class_input;
                let guard = current_label.lock().expect("Couldn't get current_label");
                let MNISTClass(current_label_data) = &*guard;
                let backprop_iter = data
                    .iter()
                    .zip(current_label_data.iter())
                    .map(|(x, label)| if *label { -1.0 / *x } else { 1.0 / *x });
                // println!("LOSS: {:?}", loss);
                // println!("Index label: {:?}", index);
                // println!("prediction: {:?}", data);
                // println!("Updating progress");
                let mut back_prop_data = [0.0; MNIST_CLASSES];
                let loss = data
                    .iter()
                    .zip(current_label_data.iter())
                    .map(|(pred, label)| if *label { -pred.ln() } else { pred.ln() })
                    .sum();
                input_channels.reverse_error_prop.send(MNISTClassSmoother(
                    mat_mul::Vector::from_iter(backprop_iter),
                ));
                progress_sender
                    .send((
                        loss,
                        MNISTClassSmoother(data),
                        MNISTClass(current_label_data.clone()),
                    ))
                    .expect("Can't update progress");
            });
        });

    let mut counter = 0;
    let first_batch_size = 600;
    let epochs = 600;
    for epoch in (1..(epochs + 1)).rev() {
        let i = (&mut rng).gen_range(0..10);
        let indexes = indexes_by_class.get(i).unwrap();
        for _ in 0..(first_batch_size / epoch) {
            let sample = (&mut rng).gen_range(0..indexes.len());
            let index = *(&indexes).get(sample).unwrap();
            mnist_input_send.send(images[index].to_f32_arr()).unwrap();
            ground_truth_sender.send(labels[index].clone()).unwrap();
            counter += 1;
            if counter % 100 == 0 {
                println!("label: {:?}", labels[index].clone());
            }
            //println!("{}", &images[index]);
            //println!("LABEL: {:?}", (&labels)[index].get_index());
        }
    }

    let mut processed = 0;
    progress_receiver
        .iter()
        .take(counter)
        .for_each(|(loss, pred, label)| {
            processed += 1;
            if processed % 100 == 0 {
                println!("processed {:?}. Loss: {:?}", processed, loss);
                println!("Prediction: {:?}", pred);
                println!("sum: {:?}", pred.0.iter().map(|x| *x).sum::<f64>());
                println!("label: {:?}", label);
            }
        });
}
