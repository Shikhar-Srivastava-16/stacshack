use futures_util::SinkExt;
use futures_util::stream::StreamExt;
use std::sync::Arc;
use tokio::sync::broadcast;
use warp::Filter;
use warp::ws::{Message, WebSocket};

#[tokio::main]
async fn main() {
    // Sender is Clone, so no Mutex needed — just wrap in Arc
    let (tx, _) = broadcast::channel::<String>(100);
    let tx = Arc::new(tx);
    let tx_ws = tx.clone();

    let ws_route = warp::path("ws")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let tx = tx_ws.clone();
            ws.on_upgrade(move |websocket| handle_connection(websocket, tx))
        });

    warp::serve(ws_route).run(([127, 0, 0, 1], 8080)).await;
}

pub async fn handle_connection(ws: WebSocket, tx: Arc<broadcast::Sender<String>>) {
    let (mut ws_sender, mut ws_receiver) = ws.split();
    let mut rx = tx.subscribe();

    // Forward broadcast messages to this WebSocket client
    let send_task = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    if ws_sender.send(Message::text(msg)).await.is_err() {
                        break; // Client disconnected
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    eprintln!("Client lagged, skipped {n} messages");
                    // Continue — or break if you want to disconnect laggy clients
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // Receive messages from this client and broadcast them
    while let Some(result) = ws_receiver.next().await {
        match result {
            Ok(message) if message.is_text() => {
                if let Ok(text) = message.to_str() {
                    if tx.send(text.to_string()).is_err() {
                        break; // No receivers left
                    }
                }
            }
            Ok(message) if message.is_close() => break,
            Ok(_) => {} // Ignore binary/ping/pong
            Err(e) => {
                eprintln!("WebSocket error: {e}");
                break;
            }
        }
    }

    // Clean up the sender task when the receiver loop exits
    send_task.abort();
}
