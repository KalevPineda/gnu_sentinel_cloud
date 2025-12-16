use axum::{
    body::Body,
    extract::{Multipart, Path, State},
    http::{header, StatusCode},
    response::IntoResponse,
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
    path::PathBuf,
    sync::{Arc, RwLock},
};
use tower_http::cors::{Any, CorsLayer};

// --- ESTRUCTURAS DE DATOS ---

// 1. Configuración (Actualizada con Gemini API Key)
#[derive(Serialize, Deserialize, Clone, Debug)]
struct RemoteConfig {
    max_temp_trigger: f32,
    scan_wait_time_sec: u64,
    system_enabled: bool,
    pan_step_degrees: f32,
    // Campo opcional para la API Key de Gemini
    pub gemini_api_key: Option<String>,
}

// 2. Estado en Vivo
#[derive(Serialize, Deserialize, Clone, Debug)]
struct LiveStatus {
    last_update: u64,
    turbine_token: String,
    mode: String,
    current_angle: f32,
    current_max_temp: f32,
    is_online: bool,
}

// 3. Registro de Alerta
#[derive(Serialize, Deserialize, Clone, Debug)]
struct AlertRecord {
    id: String,
    timestamp: u64,
    turbine_token: String,
    max_temp: f32,
    angle: f32,
    dataset_path: String,
}

// 4. Punto de datos para evolución
#[derive(Serialize)]
struct EvolutionPoint {
    frame_index: usize,
    max_temp: f32,
    avg_temp: f32,
}

// 5. Estructura para listar archivos
#[derive(Serialize)]
struct FileEntry {
    name: String,
    size_kb: u64,
    date: String,
    #[serde(rename = "type")]
    file_type: String,
}

// 6. NUEVA: Estructura para devolver la Matriz Cruda (Heatmap)
#[derive(Serialize)]
struct ThermalFrameData {
    width: usize,
    height: usize,
    min_temp: f32,
    max_temp: f32,
    // Aplanamos la matriz 2D a un vector 1D para enviarla fácil por JSON
    pixels: Vec<f32>,
}

// 7. Estado Global
struct AppState {
    config: Arc<RwLock<RemoteConfig>>,
    live_status: Arc<RwLock<LiveStatus>>,
    alerts: Arc<RwLock<VecDeque<AlertRecord>>>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    // CORS Permisivo
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let storage_folder = "cloud_storage";
    if let Err(e) = std::fs::create_dir_all(storage_folder) {
        eprintln!("⚠️ Error creando carpeta {}: {}", storage_folder, e);
    } else {
        println!("📂 Carpeta '{}' lista.", storage_folder);
    }

    // Estado Inicial
    let shared_state = Arc::new(AppState {
        config: Arc::new(RwLock::new(RemoteConfig {
            max_temp_trigger: 50.0,
            scan_wait_time_sec: 5,
            system_enabled: true,
            pan_step_degrees: 0.5,
            gemini_api_key: Some("".to_string()), // Inicializar vacío
        })),
        live_status: Arc::new(RwLock::new(LiveStatus {
            last_update: 0,
            turbine_token: "Waiting...".into(),
            mode: "Offline".into(),
            current_angle: 0.0,
            current_max_temp: 0.0,
            is_online: false,
        })),
        alerts: Arc::new(RwLock::new(VecDeque::new())),
    });

    let app = Router::new()
        // --- API WEB ---
        .route("/api/live", get(get_live_status))
        .route("/api/config", get(get_config).post(update_config))
        .route("/api/alerts", get(get_alerts))
        .route("/api/files", get(list_files_handler))
        .route("/api/evolution/:filename", get(get_evolution_data))
        
        // --- NUEVOS ENDPOINTS SOLICITADOS ---
        // Descarga de archivos forzada
        .route("/api/download/:filename", get(download_file_handler)) 
        // Obtención de matriz cruda para visualización térmica
        .route("/api/matrix/:filename/:frame_index", get(get_matrix_handler))
        
        // --- API ROBOT (CORE) ---
        .route("/ingest/heartbeat", post(heartbeat_handler))
        .route("/ingest/upload", post(upload_handler))
        
        .layer(cors)
        .with_state(shared_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("☁️ GSU Sentinel Cloud escuchando en http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// --- HANDLERS NUEVOS Y MODIFICADOS ---

// 1. NUEVO: Descarga forzada de archivos .npz
async fn download_file_handler(Path(filename): Path<String>) -> impl IntoResponse {
    let mut path = PathBuf::from("cloud_storage");
    path.push(&filename);

    // Verificación básica de seguridad (evitar ../)
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
         return (StatusCode::BAD_REQUEST, "Invalid filename").into_response();
    }

    // Leemos el archivo asíncronamente
    match tokio::fs::read(&path).await {
        Ok(file_bytes) => {
            // Convertimos bytes a Body de Axum
            let body = Body::from(file_bytes);

            // Configuramos headers para forzar descarga
            let headers = [
                (header::CONTENT_TYPE, "application/octet-stream"),
                (header::CONTENT_DISPOSITION, &format!("attachment; filename=\"{}\"", filename)),
            ];

            (headers, body).into_response()
        },
        Err(_) => (StatusCode::NOT_FOUND, "File not found").into_response(),
    }
}

// 2. NUEVO: Obtener Matriz Cruda (JSON)
// Devuelve los datos necesarios para que el frontend dibuje el mapa de calor
async fn get_matrix_handler(
    Path((filename, frame_index)): Path<(String, usize)>
) -> Result<Json<ThermalFrameData>, StatusCode> {
    
    let mut path = PathBuf::from("cloud_storage");
    path.push(&filename);

    // 1. Abrir archivo
    let file = File::open(&path).map_err(|_| StatusCode::NOT_FOUND)?;

    // 2. Leer .npz
    // Nota: Actualmente el Core guarda una única Array2<f32>.
    // Si en el futuro guardas una pila (Array3), aquí deberías lógica para seleccionar el frame.
    // Por ahora, ignoramos frame_index si es 0, o devolvemos error si piden > 0 en archivo simple.
    
    let matrix: Array2<f32> = Array2::<f32>::read_npy(file).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if frame_index > 0 {
        // Como el formato actual es solo 1 frame por archivo, si piden el index 1, 2... devolvemos error
        // O podrías devolver el único frame que hay si prefieres ser permisivo.
        return Err(StatusCode::BAD_REQUEST); 
    }

    let (rows, cols) = matrix.dim();
    
    // Estadísticas rápidas para normalización en frontend
    let min_temp = matrix.fold(f32::INFINITY, |a, &b| a.min(b));
    let max_temp = matrix.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Aplanar datos (convertir [[1,2],[3,4]] a [1,2,3,4])
    // as_standard_layout asegura que estén ordenados fila por fila
    let pixels = matrix.as_standard_layout().into_owned().into_raw_vec();

    Ok(Json(ThermalFrameData {
        width: cols,
        height: rows,
        min_temp,
        max_temp,
        pixels,
    }))
}

// --- HANDLERS EXISTENTES ---

async fn list_files_handler() -> Json<Vec<FileEntry>> {
    let mut files = Vec::new();
    let path = "cloud_storage";

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                if metadata.is_file() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.ends_with(".npz") || name.ends_with(".txt") {
                        let date: chrono::DateTime<chrono::Utc> = metadata.modified()
                            .unwrap_or(std::time::SystemTime::now())
                            .into();

                        files.push(FileEntry {
                            name: name.clone(),
                            size_kb: metadata.len() / 1024,
                            date: date.format("%Y-%m-%d %H:%M:%S").to_string(),
                            file_type: if name.contains("log") { "log".to_string() } else { "capture".to_string() },
                        });
                    }
                }
            }
        }
    }
    files.sort_by(|a, b| b.date.cmp(&a.date));
    Json(files)
}

async fn get_live_status(State(state): State<Arc<AppState>>) -> Json<LiveStatus> {
    let mut status = state.live_status.read().unwrap().clone();
    let now = chrono::Utc::now().timestamp() as u64;
    if now > status.last_update.saturating_add(5) {
        status.is_online = false;
        status.mode = "Lost Connection".to_string();
    }
    Json(status)
}

async fn get_config(State(state): State<Arc<AppState>>) -> Json<RemoteConfig> {
    Json(state.config.read().unwrap().clone())
}

async fn update_config(
    State(state): State<Arc<AppState>>, 
    Json(new_conf): Json<RemoteConfig>
) -> Json<&'static str> {
    let mut conf = state.config.write().unwrap();
    *conf = new_conf;
    // Imprimir si se actualizó la Key
    if let Some(ref key) = conf.gemini_api_key {
        if !key.is_empty() {
             println!("🔑 Gemini API Key actualizada.");
        }
    }
    Json("Config updated successfully")
}

async fn get_alerts(State(state): State<Arc<AppState>>) -> Json<Vec<AlertRecord>> {
    let alerts = state.alerts.read().unwrap();
    Json(alerts.iter().cloned().collect())
}

async fn get_evolution_data(Path(filename): Path<String>) -> Json<Vec<EvolutionPoint>> {
    let mut path = PathBuf::from("cloud_storage");
    path.push(&filename);
    let mut points = Vec::new();

    if let Ok(file) = File::open(&path) {
        if let Ok(matrix) = Array2::<f32>::read_npy(file) {
            let max_val = matrix.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = matrix.sum();
            let count = matrix.len() as f32;
            let avg_val = if count > 0.0 { sum / count } else { 0.0 };

            points.push(EvolutionPoint { 
                frame_index: 0, 
                max_temp: max_val, 
                avg_temp: avg_val 
            });
        }
    }
    Json(points)
}

async fn heartbeat_handler(
    State(state): State<Arc<AppState>>, 
    Json(payload): Json<LiveStatus>
) -> Json<RemoteConfig> {
    {
        let mut status = state.live_status.write().unwrap();
        *status = payload;
        status.last_update = chrono::Utc::now().timestamp() as u64;
        status.is_online = true;
    }
    let config = state.config.read().unwrap().clone();
    Json(config)
}

async fn upload_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart
) -> Json<&'static str> {
    let mut turbine_token = String::new();
    let mut angle = 0.0;
    let mut file_saved_name = String::new();
    let mut temp_max_detected = 0.0; 

    while let Some(field) = multipart.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        
        if name == "turbine_token" {
            if let Ok(txt) = field.text().await { turbine_token = txt; }
        } else if name == "angle" {
            if let Ok(txt) = field.text().await { angle = txt.parse().unwrap_or(0.0); }
        } else if name == "dataset_file" {
            let data = field.bytes().await.unwrap();
            let timestamp = chrono::Utc::now().timestamp();
            
            file_saved_name = format!("capture_{}_{}.npz", turbine_token, timestamp);
            let mut filepath = PathBuf::from("cloud_storage");
            filepath.push(&file_saved_name);
            
            if let Err(e) = tokio::fs::write(&filepath, &data).await {
                eprintln!("❌ Error escribiendo archivo en {:?}: {}", filepath, e);
                return Json("write_error");
            }
            println!("💾 Archivo recibido y guardado: {:?}", filepath);
            
            if let Ok(matrix) = Array2::<f32>::read_npy(std::io::Cursor::new(&data)) {
                 temp_max_detected = matrix.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            }
        }
    }

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
        if state.alerts.read().unwrap().len() > 50 {
            state.alerts.write().unwrap().pop_back();
        }
    }
    Json("upload_success")
}