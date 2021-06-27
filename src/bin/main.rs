extern crate precod_compdag;
mod layers;
use crate::layers::{MNISTClassSmoother, MNIST_CLASSES}
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
use std::fmt::Display;
use std::fmt::Formatter;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use thiserror::Error;

#[derive(Error, Debug)]
enum Error {
    #[error("No error at index")]
    NoData(usize),
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
