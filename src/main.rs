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
use tower_http::cors::Any;


// --- ESTRUCTURAS DE DATOS ---

// 1. Configuración (Se envía al Core)
#[derive(Serialize, Deserialize, Clone, Debug)]
struct RemoteConfig {
    max_temp_trigger: f32,   // Umbral de alerta
    scan_wait_time_sec: u64, // Tiempo de espera en bordes
    system_enabled: bool,    // Interruptor maestro
    pan_step_degrees: f32,   // Velocidad
}

// 2. Estado en Vivo (Viene del Core)
#[derive(Serialize, Deserialize, Clone, Debug)]
struct LiveStatus {
    last_update: u64,
    turbine_token: String,
    mode: String,
    current_angle: f32,
    current_max_temp: f32,
    is_online: bool,
}

// 3. Registro de Alerta (Historial para la Web)
#[derive(Serialize, Deserialize, Clone, Debug)]
struct AlertRecord {
    id: String,
    timestamp: u64,
    turbine_token: String,
    max_temp: f32,
    angle: f32,
    dataset_path: String, // Nombre del archivo .npz
}

// 4. Punto de datos para graficar evolución
#[derive(Serialize)]
struct EvolutionPoint {
    frame_index: usize,
    max_temp: f32,
    avg_temp: f32,
}

// 5. Estado Global de la Aplicación (Memoria RAM)
struct AppState {
    config: Arc<RwLock<RemoteConfig>>,
    live_status: Arc<RwLock<LiveStatus>>,
    alerts: Arc<RwLock<VecDeque<AlertRecord>>>,
}

#[tokio::main]
async fn main() {
    // Iniciar logs
    tracing_subscriber::fmt::init();
    let cors = CorsLayer::new()
        .allow_origin(Any)      // Acepta localhost:5173, localhost:3000, 192.168...
        .allow_methods(Any)     // Acepta GET, POST, OPTIONS
        .allow_headers(Any);    // Acepta Content-Type json


    // Crear carpeta para guardar archivos si no existe
    if let Err(e) = std::fs::create_dir_all("cloud_storage") {
        eprintln!("⚠️ Advertencia: No se pudo crear carpeta cloud_storage: {}", e);
    }

    // Estado Inicial
    let shared_state = Arc::new(AppState {
        config: Arc::new(RwLock::new(RemoteConfig {
            max_temp_trigger: 50.0,
            scan_wait_time_sec: 5,
            system_enabled: true,
            pan_step_degrees: 0.5,
        })),
        live_status: Arc::new(RwLock::new(LiveStatus {
            last_update: 0,
            turbine_token: "Esperando conexión...".into(),
            mode: "Offline".into(),
            current_angle: 0.0,
            current_max_temp: 0.0,
            is_online: false,
        })),
        alerts: Arc::new(RwLock::new(VecDeque::new())),
    });

    // Definición de Rutas
    let app = Router::new()
        // --- API PARA LA INTERFAZ WEB / MÓVIL ---
        .route("/api/live", get(get_live_status))                  // Datos tiempo real
        .route("/api/config", get(get_config).post(update_config)) // Leer/Escribir config
        .route("/api/alerts", get(get_alerts))                     // Historial
        .route("/api/evolution/:filename", get(get_evolution_data)) // Datos para gráficas
        
        // --- API PARA EL ROBOT (CORE) ---
        .route("/ingest/heartbeat", post(heartbeat_handler)) // Ping cada 1s
        .route("/ingest/upload", post(upload_handler))       // Subida de archivos
        
        // Middleware: CORS (Permite que React/Vue/HTML accedan a la API)
        //.layer(CorsLayer::permissive())
        .layer(cors)
        .with_state(shared_state);

    let addr = SocketAddr::from(([192,168,0,4], 8080));
    println!("☁️ GSU Sentinel Cloud escuchando en http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// --- HANDLERS (Lógica del Servidor) ---

// [WEB] Obtener estado en vivo
async fn get_live_status(State(state): State<Arc<AppState>>) -> Json<LiveStatus> {
    let mut status = state.live_status.read().unwrap().clone();
    
    // Si no hemos recibido datos en 5 segundos, marcar como Offline
    let now = chrono::Utc::now().timestamp() as u64;
    if now > status.last_update + 5 {
        status.is_online = false;
        status.mode = "Lost Connection".to_string();
    }
    
    Json(status)
}

// [WEB] Obtener Configuración
async fn get_config(State(state): State<Arc<AppState>>) -> Json<RemoteConfig> {
    Json(state.config.read().unwrap().clone())
}

// [WEB] Actualizar Configuración
async fn update_config(
    State(state): State<Arc<AppState>>, 
    Json(new_conf): Json<RemoteConfig>
) -> Json<&'static str> {
    let mut conf = state.config.write().unwrap();
    *conf = new_conf;
    println!("⚙️ Configuración actualizada vía Web: Trigger={}°C", conf.max_temp_trigger);
    Json("Config updated successfully")
}

// [WEB] Ver historial de alertas
async fn get_alerts(State(state): State<Arc<AppState>>) -> Json<Vec<AlertRecord>> {
    let alerts = state.alerts.read().unwrap();
    Json(alerts.iter().cloned().collect())
}

// [WEB] Analizar archivo NPZ para gráficas
async fn get_evolution_data(Path(filename): Path<String>) -> Json<Vec<EvolutionPoint>> {
    let path = format!("cloud_storage/{}", filename);
    let mut points = Vec::new();

    // Intentamos abrir el archivo
    if let Ok(file) = File::open(&path) {
        // Leemos la matriz guardada por el Core
        if let Ok(matrix) = Array2::<f32>::read_npy(file) {
            
            // CORRECCIÓN: Usamos f32::NEG_INFINITY para evitar error de tipo ambiguo
            let max_val = matrix.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            let sum: f32 = matrix.sum();
            let count = matrix.len() as f32;
            let avg_val = if count > 0.0 { sum / count } else { 0.0 };

            // En un sistema real, el archivo podría tener múltiples frames (tiempo).
            // Aquí asumimos que es una "foto" del momento más caliente.
            points.push(EvolutionPoint { 
                frame_index: 1, 
                max_temp: max_val, 
                avg_temp: avg_val 
            });
        }
    } else {
        println!("⚠️ Error: No se encontró el archivo {}", path);
    }
    
    Json(points)
}

// [CORE] Heartbeat: Recibe estado, devuelve config
async fn heartbeat_handler(
    State(state): State<Arc<AppState>>, 
    Json(payload): Json<LiveStatus>
) -> Json<RemoteConfig> {
    // 1. Guardar estado del robot
    {
        let mut status = state.live_status.write().unwrap();
        *status = payload;
        status.last_update = chrono::Utc::now().timestamp() as u64;
        status.is_online = true;
    }

    // 2. Responder con la configuración actual
    let config = state.config.read().unwrap().clone();
    Json(config)
}

// [CORE] Upload: Recibe archivos pesados
async fn upload_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart
) -> Json<&'static str> {
    
    let mut turbine_token = String::new();
    let mut angle = 0.0;
    let mut file_saved_name = String::new();
    let mut temp_max_detected = 0.0; // Idealmente el Core debería enviar esto en un campo de texto también

    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        
        if name == "turbine_token" {
            if let Ok(txt) = field.text().await { turbine_token = txt; }
        } else if name == "angle" {
            if let Ok(txt) = field.text().await {
                angle = txt.parse().unwrap_or(0.0);
            }
        } else if name == "dataset_file" {
            // Guardar archivo
            let data = field.bytes().await.unwrap();
            let timestamp = chrono::Utc::now().timestamp();
            // Nombre seguro para el archivo
            file_saved_name = format!("capture_{}_{}.npz", turbine_token, timestamp);
            let filepath = format!("cloud_storage/{}", file_saved_name);
            
            if let Err(e) = tokio::fs::write(&filepath, &data).await {
                eprintln!("Error escribiendo archivo: {}", e);
                return Json("write_error");
            }
            println!("💾 Archivo recibido y guardado: {}", filepath);
            
            // ANALISIS RÁPIDO: Abrimos el archivo en memoria para sacar la temperatura máxima
            // y guardarla en la alerta sin esperar al usuario.
            if let Ok(matrix) = Array2::<f32>::read_npy(std::io::Cursor::new(&data)) {
                 temp_max_detected = matrix.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            }
        }
    }

    // Registrar Alerta si hubo archivo
    if !file_saved_name.is_empty() {
        let alert = AlertRecord {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            turbine_token,
            max_temp: temp_max_detected,
            angle,
            dataset_path: file_saved_name,
        };
        
        state.alerts.write().unwrap().push_front(alert);
        
        // Limitar historial a 50 alertas para no llenar RAM
        if state.alerts.read().unwrap().len() > 50 {
            state.alerts.write().unwrap().pop_back();
        }
    }

    Json("upload_success")
}