use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use rand::Rng;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Sparkline},
};
use std::io;
use std::time::Duration;

// Activation function: Sigmoid
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Derivative of sigmoid for backpropagation
fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

// Simple feedforward neural network with one hidden layer
struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,

    // Weights and biases
    weights_input_hidden: Vec<Vec<f64>>,
    bias_hidden: Vec<f64>,
    weights_hidden_output: Vec<Vec<f64>>,
    bias_output: Vec<f64>,

    // Learning rate
    learning_rate: f64,
}

impl NeuralNetwork {
    // Initialize network with random weights
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize weights with random values between -1 and 1
        let weights_input_hidden = (0..input_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let bias_hidden = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let weights_hidden_output = (0..hidden_size)
            .map(|_| (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let bias_output = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output,
            learning_rate,
        }
    }

    // Forward pass through the network
    fn forward(&self, inputs: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Calculate hidden layer activations
        let mut hidden_raw = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            for i in 0..self.input_size {
                hidden_raw[j] += inputs[i] * self.weights_input_hidden[i][j];
            }
            hidden_raw[j] += self.bias_hidden[j];
        }

        let hidden_activated: Vec<f64> = hidden_raw.iter().map(|&x| sigmoid(x)).collect();

        // Calculate output layer activations
        let mut output_raw = vec![0.0; self.output_size];
        for k in 0..self.output_size {
            for j in 0..self.hidden_size {
                output_raw[k] += hidden_activated[j] * self.weights_hidden_output[j][k];
            }
            output_raw[k] += self.bias_output[k];
        }

        let output_activated: Vec<f64> = output_raw.iter().map(|&x| sigmoid(x)).collect();

        (hidden_raw, hidden_activated, output_activated)
    }

    // Backpropagation algorithm to update weights
    fn backward(&mut self, inputs: &[f64], targets: &[f64]) -> f64 {
        // Forward pass
        let (hidden_raw, hidden_activated, outputs) = self.forward(inputs);

        // Calculate output layer error
        let mut output_errors = vec![0.0; self.output_size];
        let mut total_error = 0.0;
        for k in 0..self.output_size {
            output_errors[k] = targets[k] - outputs[k];
            total_error += output_errors[k].powi(2);
        }

        // Calculate output layer gradients
        let mut output_gradients = vec![0.0; self.output_size];
        for k in 0..self.output_size {
            output_gradients[k] = output_errors[k] * sigmoid_derivative(outputs[k]);
        }

        // Calculate hidden layer error
        let mut hidden_errors = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            for k in 0..self.output_size {
                hidden_errors[j] += output_gradients[k] * self.weights_hidden_output[j][k];
            }
        }

        // Calculate hidden layer gradients
        let mut hidden_gradients = vec![0.0; self.hidden_size];
        for j in 0..self.hidden_size {
            hidden_gradients[j] = hidden_errors[j] * sigmoid_derivative(hidden_raw[j]);
        }

        // Update weights and biases for output layer
        for j in 0..self.hidden_size {
            for k in 0..self.output_size {
                self.weights_hidden_output[j][k] +=
                    self.learning_rate * output_gradients[k] * hidden_activated[j];
            }
        }
        for k in 0..self.output_size {
            self.bias_output[k] += self.learning_rate * output_gradients[k];
        }

        // Update weights and biases for hidden layer
        for i in 0..self.input_size {
            for j in 0..self.hidden_size {
                self.weights_input_hidden[i][j] +=
                    self.learning_rate * hidden_gradients[j] * inputs[i];
            }
        }
        for j in 0..self.hidden_size {
            self.bias_hidden[j] += self.learning_rate * hidden_gradients[j];
        }

        // Return MSE
        total_error / self.output_size as f64
    }

    // Predict output for given input
    fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        let (_, _, outputs) = self.forward(inputs);
        outputs
    }
}

// Training data generator for f(x) = x^2 (normalized)
fn generate_training_data(samples: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = rand::thread_rng();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..samples {
        let x = rng.gen_range(-1.0..1.0);
        let y = x * x; // f(x) = x^2

        inputs.push(vec![x]);
        targets.push(vec![y]);
    }

    (inputs, targets)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Initialize neural network
    let mut nn = NeuralNetwork::new(1, 8, 1, 0.1);

    let epochs = 5000;
    let batch_size = 50;
    let mut error_history: Vec<u64> = Vec::new();

    // Training loop
    for epoch in 0..epochs {
        let (inputs, targets) = generate_training_data(batch_size);
        let mut epoch_error = 0.0;

        for i in 0..batch_size {
            let error = nn.backward(&inputs[i], &targets[i]);
            epoch_error += error;
        }

        let avg_error = epoch_error / batch_size as f64;

        // Store error for visualization (scaled)
        error_history.push((avg_error * 1000.0) as u64);
        if error_history.len() > 100 {
            error_history.remove(0);
        }

        // Update UI every 10 epochs
        if epoch % 10 == 0 {
            terminal.draw(|f| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(3),
                        Constraint::Length(8),
                        Constraint::Min(10),
                    ])
                    .split(f.size());

                // Title
                let title = Paragraph::new("Mini Neural Network - Training f(x) = x²")
                    .style(
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )
                    .block(Block::default().borders(Borders::ALL));
                f.render_widget(title, chunks[0]);

                // Training info
                let info = vec![
                    Line::from(vec![
                        Span::styled("Epoch: ", Style::default().fg(Color::Yellow)),
                        Span::raw(format!("{}/{}", epoch, epochs)),
                    ]),
                    Line::from(vec![
                        Span::styled("MSE: ", Style::default().fg(Color::Yellow)),
                        Span::raw(format!("{:.6}", avg_error)),
                    ]),
                    Line::from(vec![
                        Span::styled("Architecture: ", Style::default().fg(Color::Yellow)),
                        Span::raw("1 → 8 → 1"),
                    ]),
                    Line::from(vec![
                        Span::styled("Learning Rate: ", Style::default().fg(Color::Yellow)),
                        Span::raw("0.1"),
                    ]),
                ];
                let info_widget = Paragraph::new(info).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Training Info"),
                );
                f.render_widget(info_widget, chunks[1]);

                // Error graph
                let sparkline = Sparkline::default()
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title("Error History (MSE × 1000)"),
                    )
                    .data(&error_history)
                    .style(Style::default().fg(Color::Green));
                f.render_widget(sparkline, chunks[2]);
            })?;

            // Check for user input to exit
            if event::poll(Duration::from_millis(10))? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('q') {
                        break;
                    }
                }
            }
        }
    }

    // Show final results
    terminal.draw(|f| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(10)])
            .split(f.size());

        let title = Paragraph::new("Training Complete! Testing predictions...")
            .style(
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(title, chunks[0]);

        // Test predictions
        let test_values = vec![-0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8];
        let mut results = Vec::new();

        results.push(Line::from(vec![
            Span::styled(
                "Input",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  | "),
            Span::styled(
                "Expected",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | "),
            Span::styled(
                "Predicted",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | "),
            Span::styled(
                "Error",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
        ]));
        results.push(Line::from("-".repeat(50)));

        for x in test_values {
            let prediction = nn.predict(&[x])[0];
            let expected = x * x;
            let error = (expected - prediction).abs();

            results.push(Line::from(format!(
                "{:6.2} | {:8.4} | {:9.4} | {:6.4}",
                x, expected, prediction, error
            )));
        }

        results.push(Line::from(""));
        results.push(Line::from(Span::styled(
            "Press 'q' to exit",
            Style::default().fg(Color::Gray),
        )));

        let results_widget = Paragraph::new(results)
            .block(Block::default().borders(Borders::ALL).title("Test Results"));
        f.render_widget(results_widget, chunks[1]);
    })?;

    // Wait for user to press 'q'
    loop {
        if let Event::Key(key) = event::read()? {
            if key.code == KeyCode::Char('q') {
                break;
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}
