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
    path::PathBuf, // Importante para rutas seguras en Linux
    sync::{Arc, RwLock},
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir; // Importante para servir archivos estáticos

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
// 6. Estructura para listar archivos en el JSON
#[derive(Serialize)]
struct FileEntry {
    name: String,
    size_kb: u64,
    date: String,
    #[serde(rename = "type")] // Renombramos para que en JSON salga "type"
    file_type: String,
}

#[tokio::main]
async fn main() {
    // Iniciar logs
    tracing_subscriber::fmt::init();

    // Configuración CORS Permisiva (Para que tu Web pueda conectarse desde cualquier lugar)
    let cors = CorsLayer::new()
        .allow_origin(Any)      
        .allow_methods(Any)     
        .allow_headers(Any);    

    // Nombre de la carpeta de almacenamiento
    let storage_folder = "cloud_storage";

    // Crear carpeta para guardar archivos si no existe
    if let Err(e) = std::fs::create_dir_all(storage_folder) {
        eprintln!("⚠️ Advertencia: No se pudo crear carpeta {}: {}", storage_folder, e);
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
        .route("/api/live", get(get_live_status))                  
        .route("/api/config", get(get_config).post(update_config)) 
        .route("/api/alerts", get(get_alerts))                     
        .route("/api/evolution/:filename", get(get_evolution_data)) 
        
        // --- API PARA EL ROBOT (CORE) ---
        .route("/ingest/heartbeat", post(heartbeat_handler)) 
        .route("/ingest/upload", post(upload_handler))
        .route("/api/files", get(list_files_handler)) // <--- AGREGAR ESTA LÍNEA  
        
        // --- SERVICIO DE ARCHIVOS ESTÁTICOS (Corrección para descargar archivos) ---
        // Esto permite que tu web acceda a: http://TU_VPS:8080/files/archivo.npz
        //.nest_service("/files", ServeDir::new(storage_folder))

        // Middleware
        .layer(cors)
        .with_state(shared_state);

    // Escuchar en 0.0.0.0 es OBLIGATORIO para VPS
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("☁️ GSU Sentinel Cloud escuchando en http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// --- HANDLERS ---

// [WEB] Listar archivos en la carpeta de almacenamiento
async fn list_files_handler() -> Json<Vec<FileEntry>> {
    let mut files = Vec::new();
    let path = "cloud_storage";

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                if metadata.is_file() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    
                    // Solo listar archivos .npz o .txt relevantes
                    if name.ends_with(".npz") || name.ends_with(".txt") {
                        // Formatear fecha (requiere chrono)
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
    
    // Ordenar: más recientes primero
    files.sort_by(|a, b| b.date.cmp(&a.date));
    Json(files)
}

// [WEB] Obtener estado en vivo
async fn get_live_status(State(state): State<Arc<AppState>>) -> Json<LiveStatus> {
    let mut status = state.live_status.read().unwrap().clone();
    
    // Si no hemos recibido datos en 5 segundos, marcar como Offline
    let now = chrono::Utc::now().timestamp() as u64;
    // Chequeo de seguridad para evitar overflow en resta
    if now > status.last_update.saturating_add(5) {
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
    // Uso de PathBuf para evitar ataques de directorio (ej ../../)
    let mut path = PathBuf::from("cloud_storage");
    path.push(&filename);
    
    let mut points = Vec::new();

    // Intentamos abrir el archivo
    if let Ok(file) = File::open(&path) {
        // Leemos la matriz guardada
        if let Ok(matrix) = Array2::<f32>::read_npy(file) {
            
            // CORRECCIÓN: Usamos f32::NEG_INFINITY para evitar error de tipo ambiguo
            let max_val = matrix.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            let sum: f32 = matrix.sum();
            let count = matrix.len() as f32;
            let avg_val = if count > 0.0 { sum / count } else { 0.0 };

            points.push(EvolutionPoint { 
                frame_index: 1, 
                max_temp: max_val, 
                avg_temp: avg_val 
            });
        }
    } else {
        println!("⚠️ Error: No se encontró el archivo {:?}", path);
    }
    
    Json(points)
}

// [CORE] Heartbeat
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
    // Responder con la configuración
    let config = state.config.read().unwrap().clone();
    Json(config)
}

// [CORE] Upload
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
            if let Ok(txt) = field.text().await {
                angle = txt.parse().unwrap_or(0.0);
            }
        } else if name == "dataset_file" {
            // Obtener bytes
            let data = field.bytes().await.unwrap();
            let timestamp = chrono::Utc::now().timestamp();
            
            // Construir nombre y ruta segura
            file_saved_name = format!("capture_{}_{}.npz", turbine_token, timestamp);
            let mut filepath = PathBuf::from("cloud_storage");
            filepath.push(&file_saved_name);
            
            // Guardar archivo
            if let Err(e) = tokio::fs::write(&filepath, &data).await {
                eprintln!("❌ Error escribiendo archivo en {:?}: {}", filepath, e);
                return Json("write_error");
            }
            println!("💾 Archivo recibido y guardado: {:?}", filepath);
            
            // Análisis Rápido (sin leer de disco, usando los bytes en memoria)
            if let Ok(matrix) = Array2::<f32>::read_npy(std::io::Cursor::new(&data)) {
                 temp_max_detected = matrix.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            }
        }
    }

    // Registrar Alerta
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
        
        // Mantener solo las últimas 50 alertas
        if state.alerts.read().unwrap().len() > 50 {
            state.alerts.write().unwrap().pop_back();
        }
    }

    Json("upload_success")
}