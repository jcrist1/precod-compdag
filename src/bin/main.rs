extern crate precod_compdag;
use precod_compdag::mat_mul;
use precod_compdag::BranchNode;
use precod_compdag::InputWithErrorBackProp;
use precod_compdag::LeafNode;
use precod_compdag::OutputWithErrorBackProp;
use precod_compdag::RootNode;
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
const TOL: f64 = 1.0e-7;
const ERR_LEARNING_RATE: f64 = 0.4;
const WEIGHT_LEARNING_RATE: f64 = 0.1;

#[derive(Error, Debug)]
enum Error {
    #[error("No error at index")]
    NoData(usize),
}

#[derive(Debug, Clone)]
struct MNISTImage(mat_mul::Matrix<u8, IMAGE_DIM, IMAGE_DIM>);
#[derive(Debug, Clone)]
struct MNISTVector(mat_mul::Vector<f64, VEC_DIM>);

#[derive(Debug, Clone)]
struct MNISTClass(mat_mul::Vector<bool, MNIST_CLASSES>);
#[derive(Debug)]
struct MNISTClassSmoother(mat_mul::Vector<f64, MNIST_CLASSES>);

#[derive(Debug)]
struct Hidden(mat_mul::Vector<f64, HIDDEN_LAYERS>);

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

            MNISTImage(matrix_data)
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
    let (progress_sender, progress_receiver) = crossbeam::channel::unbounded::<f64>();

    let mnist_input = RootNode {
        forward_input: root_inp_rcv,
        forward_output: OutputWithErrorBackProp {
            forward_data: root_out_send,
            reverse_error_prop: root_err_rcv,
        },
    };

    let mnist_hidden = BranchNode {
        input_channels: InputWithErrorBackProp {
            forward_data: hidden_input_rcv,
            reverse_error_prop: hidden_err_sender,
        },
        output_channels: OutputWithErrorBackProp {
            forward_data: hidden_output_sender,
            reverse_error_prop: hidde_err_rcv,
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
        let back_prop_input = Arc::clone(&input_mutex);

        let predictions_mutex = Arc::new(Mutex::new(mat_mul::Vector::<f64, VEC_DIM>::new()));
        let back_prop_predictions = Arc::clone(&predictions_mutex);

        let predictions_errors_mutex = Arc::new(Mutex::new(mat_mul::Vector::<f64, VEC_DIM>::new()));
        let back_prop_pred_err = Arc::clone(&predictions_errors_mutex);

        let activations_mutex = Arc::new(Mutex::new(mat_mul::Vector::<f64, HIDDEN_LAYERS>::new()));
        let back_prop_activations = Arc::clone(&activations_mutex);

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
                            if node_delta.abs() > 0.1 * prediction_error.abs() {
                                *prediction -= ERR_LEARNING_RATE * node_delta
                            } else {
                                let scale =
                                    -1.0 * *backprop_row * WEIGHT_LEARNING_RATE * *prediction_error;
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

    let hidden_layer_thr = thread::spawn(move || {
        let mut rng = thread_rng();

        println!("initialised next weights");
        let hidden_layer_data =
            mat_mul::Matrix::<f64, MNIST_CLASSES, HIDDEN_LAYERS>::from_distribution(
                &mut rng,
                &Uniform::new(0., 1.0 / ((HIDDEN_LAYERS * MNIST_CLASSES) as f64).sqrt()),
            );

        let hidden_layer_preds: mat_mul::Vector<f64, HIDDEN_LAYERS> =
            mat_mul::Vector::from_distribution(
                &mut rng,
                &Uniform::new(0., 1.0 / (HIDDEN_LAYERS as f64)),
            );

        // This should be doable with left_right.  Probably need to implemnt two ops
        // ```
        // AddAssignWeights(mat_mul::Matrxic<f64, HIDDEN_LAYERS, MNIST_CLASSES>),
        // AddAssignPredictions(mat_mul::Vector<f64, HIDDEN_LAYERS>)
        // ```
        struct HiddenLayerBackOwned {
            weights: mat_mul::Matrix<f64, MNIST_CLASSES, HIDDEN_LAYERS>,
            predictions: mat_mul::Vector<f64, HIDDEN_LAYERS>,
        }

        struct HiddenLayerForwardOwned {
            input: mat_mul::Vector<f64, HIDDEN_LAYERS>,
            activation: mat_mul::Vector<f64, MNIST_CLASSES>,
            prediction_errors: mat_mul::Vector<f64, HIDDEN_LAYERS>,
        }

        let back_prop_data = Arc::new(Mutex::new(HiddenLayerBackOwned {
            weights: hidden_layer_data,
            predictions: hidden_layer_preds,
        }));

        let read_back_prop_for_forward = Arc::clone(&back_prop_data);

        let forward_data = Arc::new(Mutex::new(HiddenLayerForwardOwned {
            input: mat_mul::Vector::new(),
            activation: mat_mul::Vector::new(),
            prediction_errors: mat_mul::Vector::new(),
        }));

        let read_forward_data_for_back_prop = Arc::clone(&forward_data);

        let BranchNode {
            input_channels,
            output_channels,
        } = mnist_hidden;

        let OutputWithErrorBackProp {
            forward_data: output_forward,
            reverse_error_prop: output_backward,
        } = output_channels;

        let InputWithErrorBackProp {
            forward_data: input_forward,
            reverse_error_prop: input_backward,
        } = input_channels;

        // Error propagation thread
        thread::spawn(move || {
            output_backward.iter().for_each(|error_from_next| {
                let MNISTClassSmoother(data) = error_from_next;

                let HiddenLayerForwardOwned {
                    input,
                    activation,
                    prediction_errors,
                } = &*read_forward_data_for_back_prop.lock().unwrap();

                let HiddenLayerBackOwned {
                    weights,
                    predictions,
                } = &mut *back_prop_data.lock().unwrap();

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
                            if node_delta.abs() > 0.1 * prediction_error.abs() {
                                *prediction -= ERR_LEARNING_RATE * node_delta
                            } else {
                                let scale =
                                    -1.0 * *backprop_row * WEIGHT_LEARNING_RATE * *prediction_error;
                                let boost = &*input * scale;
                                // println!("UPDATING WEIGHTS!");
                                // println!("{:?}, {:?}", row, boost);
                                *row += &boost;
                            }
                        },
                    );
                // This is so bad
            });
        });

        // forward propagation
        input_forward.iter().for_each(|hidden_input| {
            let HiddenLayerForwardOwned {
                input,
                activation,
                prediction_errors,
            } = &mut *forward_data.lock().unwrap();

            let HiddenLayerBackOwned {
                weights,
                predictions,
            } = &*read_back_prop_for_forward.lock().unwrap();
            let Hidden(data) = hidden_input;
            *input += &data; // we exponentially average the stored input over time, to smooth out the weight updates
            *input *= 0.5;
            let new_prediction_errors = predictions + (&data * (-1.0));
            *prediction_errors = new_prediction_errors.clone();
            input_backward.send(Hidden(new_prediction_errors)).unwrap();

            // todo: implement vec * matrix to get transpose mult
            let logits = weights * &data;
            let mut total = 0.0;
            let exps = logits.map(|logit| {
                let exp = logit.exp();
                total += exp;
                exp
            });
            let output_data = exps.map(|x| x / total);
            *activation = output_data.clone();
            output_forward
                .send(MNISTClassSmoother(output_data))
                .expect("Should do something");
        });
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
                let (index, pred) = data
                    .iter()
                    .zip(current_label_data.iter())
                    .enumerate()
                    .find_map(|(index, (pred, bool_index))| {
                        if *bool_index {
                            Some((index, pred))
                        } else {
                            None
                        }
                    })
                    .unwrap();
                let loss = -pred.ln();
                //println!("LOSS: {:?}", loss);
                // println!("Index label: {:?}", index);
                // println!("prediction: {:?}", data);
                // println!("Updating progress");
                let mut back_prop_data = [0.0; MNIST_CLASSES];
                back_prop_data[index] = -1.0 / pred;
                input_channels.reverse_error_prop.send(MNISTClassSmoother(
                    mat_mul::Vector::from_iter(back_prop_data.iter().map(|x| *x)),
                ));
                progress_sender.send(loss).expect("Can't update progress");
            });
        });

    let threes = indexes_by_class.get(3).unwrap();
    let mut counter = 0;
    for _ in 0..40000 {
        let threes_index = (&mut rng).gen_range(0..threes.len());
        let index = *(&threes).get(threes_index).unwrap();
        mnist_input_send.send(images[index].to_f32_arr()).unwrap();
        ground_truth_sender.send(labels[index].clone()).unwrap();
        counter += 1;
        //println!("{}", &images[index]);
        //println!("LABEL: {:?}", (&labels)[index].get_index());
    }

    let mut processed = 0;
    progress_receiver.iter().take(counter).for_each(|loss| {
        processed += 1;
        if processed % 100 == 0 {
            println!("processed {:?}. Loss: {:?}", processed, loss);
        }
    });
}
