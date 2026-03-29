use futures_util::{SinkExt, StreamExt};
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{RequestedFormat, RequestedFormatType, Resolution},
};

use std::process::Command;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[tokio::main]
async fn main() {
    let username = std::env::var("USERNAME").unwrap_or("anonymous".to_string());
    let help = std::env::var("HELPER").unwrap_or("true".to_string()) == "true";
    let un = username.clone();
    let username = Arc::new(username);

    println!("default: {}", help);
    // making camera
    // let mut camera = find_working_camera();

    let url = "ws://127.0.0.1:8080/ws";
    let (ws_stream, _) = connect_async(url).await.expect("Failed to connect");
    println!("Connected as {username}! Type messages below:");

    let (mut sender, mut receiver) = ws_stream.split();

    // Task: print incoming messages
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                if !text.starts_with(un.as_str()) {
                    // \r clears the current input line for cleaner output
                    print!("\rrecv {text}\n> ");
                    // flush so the prompt reappears immediately
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();
                }
            }
        }
    });

    // Task: read stdin and send
    let send_task = tokio::spawn(async move {
        let stdin = BufReader::new(tokio::io::stdin());
        let mut lines = stdin.lines();

        print!("Sending > ");
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        while let Ok(Some(line)) = lines.next_line().await {
            if line.eq_ignore_ascii_case("quit") {
                break;
            }
            // save cap
            let imgtag = process_frame(format!("{}", line.to_string()));
            let msg = format!("{}: {} [{}]", username, line, imgtag);
            if sender.send(Message::Text(msg.into())).await.is_err() {
                eprintln!("Failed to send message");
                break;
            }
            print!("> ");
            std::io::stdout().flush().unwrap();
        }
    });

    // Exit when either task finishes
    tokio::select! {
        _ = recv_task => {},
        _ = send_task => {},
    }
}

// pub fn process_frame(camera: &mut Camera, strname: String) {
pub fn process_frame(strname: String) -> String {
    let mut camera = find_working_camera();

    let frame = camera.frame().unwrap();
    let decoded = frame.decode_image::<RgbFormat>().unwrap();

    let path = format!("./frames/frame_{}.png", strname);
    decoded.save(&path).unwrap();
    // println!("-> Saved {}", path);

    // println!("recognition!");

    // let output = Command::new("ls")
    // let output = Command::new("./deps/predict ../../image.png meow ./deps/models")
    let output = Command::new("./deps/predict")
        // .arg("-la")
        .arg(path)
        .arg("meow")
        .arg("./deps/models/")
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // let stderr = String::from_utf8_lossy(&output.stderr);

    camera.stop_stream().ok();
    stdout.to_string()
}

pub fn find_working_camera() -> Camera {
    let cameras = nokhwa::query(nokhwa::utils::ApiBackend::Auto).unwrap();

    if cameras.is_empty() {
        panic!("No cameras found!");
    }

    for cam_info in cameras {
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestResolution(
            Resolution::new(1280, 720),
        ));

        let mut camera = match Camera::new(cam_info.index().clone(), requested) {
            Ok(cam) => cam,
            Err(_e) => {
                // println!("Skipping camera {}: {}", cam_info.index(), e);
                continue;
            }
        };

        if camera.open_stream().is_err() {
            // println!("Skipping camera {}: couldn't open stream", cam_info.index());
            continue;
        }

        // Try to grab a frame and check it's not empty
        match camera.frame() {
            Ok(frame) => match frame.decode_image::<RgbFormat>() {
                Ok(decoded) if decoded.width() > 0 && decoded.height() > 0 => {
                    println!("Using camera: {}", cam_info.human_name());
                    return camera; // stream is already open
                }
                _ => {
                    // println!(
                    //     "Skipping camera {}: empty or invalid frame",
                    //     cam_info.index()
                    // );
                    camera.stop_stream().ok();
                }
            },
            Err(_e) => {
                // println!("Skipping camera {}: {}", cam_info.index(), e);
                camera.stop_stream().ok();
            }
        }
    }

    panic!("No working camera found!");
}
