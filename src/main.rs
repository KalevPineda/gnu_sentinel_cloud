


use axum::{
    extract::Multipart, 
    routing::{get, post}, // Ahora usamos ambos
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Serialize, Deserialize)]
struct TurbineEvent {
    turbine_token: String,
    capture_timestamp: u64,
    angle_position: f32,
    max_temp_detected: f32,
}

#[tokio::main]
async fn main() {
    // Inicializar tracing para ver logs en la consola
    tracing_subscriber::fmt::init();

    let app = Router::new()
        // Endpoint para saber si el servidor vive (Usa GET)
        .route("/health", get(health_handler))
        // Endpoints de ingesta de datos (Usan POST)
        .route("/ingest/telemetry", post(telemetry_handler))
        .route("/ingest/event", post(event_handler))
        .route("/ingest/upload", post(upload_handler));

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("☁️ GSU Cloud escuchando en {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// --- HANDLERS ---

// Nuevo: Health Check para monitoreo
async fn health_handler() -> &'static str {
    "🟢 GSU Cloud Online"
}

async fn telemetry_handler(Json(payload): Json<serde_json::Value>) -> Json<&'static str> {
    // Aquí puedes imprimir lo que llega para debug
    println!("📡 Telemetría: {:?}", payload);
    Json("ack")
}

async fn event_handler(Json(event): Json<TurbineEvent>) -> Json<&'static str> {
    println!("🚨 ALERTA Turbina {}: Temp {}°C (Ángulo {})", 
        event.turbine_token, event.max_temp_detected, event.angle_position);
    Json("event_recorded")
}

async fn upload_handler(mut multipart: Multipart) -> Json<&'static str> {
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        if name == "dataset_file" {
            // Si el archivo es muy grande, esto lo carga en RAM. 
            // En producción usaríamos streams para guardarlo directo a disco.
            let data = field.bytes().await.unwrap();
            println!("💾 Dataset recibido: {} bytes", data.len());
            
            // Guardar archivo (asegúrate de crear la carpeta 'cloud_storage' antes)
            let _ = tokio::fs::write("cloud_storage/last_capture.npz", data).await;
        }
    }
    Json("upload_success")
}