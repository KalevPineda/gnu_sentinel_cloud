use axum::{
    extract::{Multipart, Path, State},
    routing::{get, post},
    Json, Router,
};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    fs::File,
    net::SocketAddr,
    sync::{Arc, RwLock},
};
use tower_http::cors::CorsLayer;

// --- ESTRUCTURAS DE DATOS ---

// Configuración Controlable desde la Web
#[derive(Serialize, Deserialize, Clone, Debug)]
struct RemoteConfig {
    max_temp_trigger: f32,
    scan_wait_time_sec: u64,
    system_enabled: bool, // Switch maestro ON/OFF
}

// Estado "Vivo" del robot (Telemetría)
#[derive(Serialize, Deserialize, Clone, Debug)]
struct LiveStatus {
    last_update: u64, // Unix Timestamp
    turbine_token: String,
    mode: String,
    current_angle: f32,
    current_max_temp: f32,
    is_online: bool,
}

// Alerta histórica
#[derive(Serialize, Deserialize, Clone, Debug)]
struct AlertRecord {
    id: String,
    timestamp: u64,
    turbine_token: String,
    max_temp: f32,
    angle: f32,
    dataset_path: String, // Ruta al archivo .npz para análisis detallado
}

// Estado Global en Memoria RAM
struct AppState {
    // Configuración que el Core descargará
    config: Arc<RwLock<RemoteConfig>>,
    // Último estado conocido (para el Dashboard en vivo)
    live_status: Arc<RwLock<LiveStatus>>,
    // Historial de alertas (en prod usarías una Base de Datos SQL)
    alerts: Arc<RwLock<VecDeque<AlertRecord>>>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // Estado inicial por defecto
    let shared_state = Arc::new(AppState {
        config: Arc::new(RwLock::new(RemoteConfig {
            max_temp_trigger: 50.0,
            scan_wait_time_sec: 5,
            system_enabled: true,
        })),
        live_status: Arc::new(RwLock::new(LiveStatus {
            last_update: 0,
            turbine_token: "unknown".into(),
            mode: "Offline".into(),
            current_angle: 0.0,
            current_max_temp: 0.0,
            is_online: false,
        })),
        alerts: Arc::new(RwLock::new(VecDeque::new())),
    });

    // Crear carpeta de almacenamiento
    std::fs::create_dir_all("cloud_storage").unwrap();

    let app = Router::new()
        // --- API PARA LA INTERFAZ WEB / MÓVIL ---
        .route("/api/live", get(get_live_status))          // Ver estado actual
        .route("/api/config", get(get_config).post(update_config)) // Leer/Escribir config
        .route("/api/alerts", get(get_alerts))             // Listar alertas
        .route("/api/evolution/:filename", get(get_evolution_data)) // Gráfica detallada
        
        // --- API PARA EL ROBOT (CORE) ---
        .route("/ingest/heartbeat", post(heartbeat_handler)) // Telemetría ligera
        .route("/ingest/upload", post(upload_handler))       // Subida de archivos pesados
        
        // Permitir CORS (para que tu frontend React/Vue/HTML pueda conectarse)
        .layer(CorsLayer::permissive())
        .with_state(shared_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("☁️ GSU Cloud API lista en {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// --- HANDLERS PARA WEB/APP ---

// 1. Dashboard en Vivo: Retorna JSON ligero para pintar agujas/textos
async fn get_live_status(State(state): State<Arc<AppState>>) -> Json<LiveStatus> {
    let status = state.live_status.read().unwrap();
    // Aquí podrías lógica para marcar is_online = false si last_update es muy viejo
    Json(status.clone())
}

// 2. Control: Obtener y Actualizar configuración
async fn get_config(State(state): State<Arc<AppState>>) -> Json<RemoteConfig> {
    Json(state.config.read().unwrap().clone())
}

async fn update_config(State(state): State<Arc<AppState>>, Json(new_conf): Json<RemoteConfig>) -> Json<&'static str> {
    let mut conf = state.config.write().unwrap();
    *conf = new_conf;
    println!("⚙️ Configuración actualizada desde Web: Temp Trigger > {}", conf.max_temp_trigger);
    Json("updated")
}

// 3. Notificaciones: Listar alertas recientes
async fn get_alerts(State(state): State<Arc<AppState>>) -> Json<Vec<AlertRecord>> {
    let alerts = state.alerts.read().unwrap();
    Json(alerts.iter().cloned().collect())
}

// 4. Evolución: Abre el .npz y extrae la curva de temperatura
#[derive(Serialize)]
struct EvolutionPoint {
    frame_index: usize,
    max_temp: f32,
    avg_temp: f32,
}

async fn get_evolution_data(Path(filename): Path<String>) -> Json<Vec<EvolutionPoint>> {
    let path = format!("cloud_storage/{}", filename);
    let mut points = Vec::new();

    // Intenta abrir el archivo .npz (Esto bloquea un poco, en prod usar spawn_blocking)
    if let Ok(file) = File::open(path) {
        // Leemos la matriz guardada. Asumimos que guardamos un array 3D o una lista de arrays.
        // NOTA: Para simplificar, asumiremos que el Core guardó un solo Array2 caliente, 
        // pero idealmente guardarías un stack de frames.
        // Aquí simulamos lectura de un frame para el ejemplo:
        if let Ok(matrix) = Array2::<f32>::read_npy(file) {
            // Generamos un punto único (en realidad iterarías sobre frames temporales)
            //let max = matrix.fold(0./0., |a, &b| a.max(b));
            // Usamos f32::NEG_INFINITY como valor inicial, que es explícitamente f32 
            // y es la forma correcta de iniciar una búsqueda de máximo.
            let max = matrix.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = matrix.sum();
            let avg = sum / matrix.len() as f32;
            
            points.push(EvolutionPoint { frame_index: 0, max_temp: max, avg_temp: avg });
        }
    }
    Json(points)
}

// --- HANDLERS PARA GSU CORE ---

// Recibe datos cada segundo
async fn heartbeat_handler(
    State(state): State<Arc<AppState>>, 
    Json(payload): Json<LiveStatus>
) -> Json<RemoteConfig> {
    // 1. Actualizar estado "vivo"
    {
        let mut status = state.live_status.write().unwrap();
        *status = payload;
        status.is_online = true;
        status.last_update = chrono::Utc::now().timestamp() as u64;
    }

    // 2. Responder con la configuración actual (así el robot se sincroniza)
    let config = state.config.read().unwrap().clone();
    Json(config)
}

async fn upload_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart
) -> Json<&'static str> {
    let mut turbine_token = String::new();
    let mut angle = 0.0;
    let mut filename = String::new();
    
    // Procesar campos
    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        
        if name == "turbine_token" {
            turbine_token = field.text().await.unwrap();
        } else if name == "angle" {
            if let Ok(val) = field.text().await.unwrap().parse::<f32>() { angle = val; }
        } else if name == "dataset_file" {
            let data = field.bytes().await.unwrap();
            let ts = chrono::Utc::now().timestamp();
            filename = format!("capture_{}_{}.npz", turbine_token, ts);
            let path = format!("cloud_storage/{}", filename);
            let _ = tokio::fs::write(&path, data).await;
            println!("💾 Archivo guardado: {}", path);
        }
    }

    // Registrar Alerta
    if !filename.is_empty() {
        let alert = AlertRecord {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            turbine_token,
            angle,
            max_temp: 0.0, // Se actualizaría procesando el archivo o enviándolo en el form
            dataset_path: filename,
        };
        state.alerts.write().unwrap().push_front(alert);
    }

    Json("upload_success")
}